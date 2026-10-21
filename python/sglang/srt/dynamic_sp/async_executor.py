# Copyright 2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""Independent group execution with coordinator-owned worker leases.

Only startup/shutdown involve the full world. Each operation leases the union
of old and new KV owners; overlapping operations cannot execute concurrently.
A worker has one execution thread, so overlapping communicators are entered in
coordinator ticket order. Group creation is local to members, and cached.
"""

import contextvars
import os
import socket
import threading
import time
from dataclasses import dataclass
from multiprocessing.connection import Client, Listener, wait

import torch
import torch.distributed as dist
from sglang.srt.dynamic_sp.communication import GroupCommunication, rendezvous_store
from sglang.srt.dynamic_sp.engine import DynamicSPPrefillEngine, PlanRejected, _State
from sglang.srt.dynamic_sp.plan import PrefillPlan, PromptStep
from sglang.srt.utils.monotonic_clock import monotonic_clock_domain


def clock_domain():
    """Linux CLOCK_MONOTONIC is shared only within a boot and time namespace.

    Reject unsupported timing domains instead of estimating clock offsets with
    network round trips. perf_counter uses CLOCK_MONOTONIC on this platform.
    """
    domain = monotonic_clock_domain()
    if domain is None:
        raise RuntimeError("Async measurement requires Linux CLOCK_MONOTONIC")
    return domain


def _nodelay(connection):
    sock = socket.socket(fileno=connection.fileno())
    try:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    finally:
        sock.detach()


@dataclass(frozen=True)
class Operation:
    ticket: int
    members: tuple[int, ...]
    plan: PrefillPlan
    previous: tuple[tuple[object, tuple[int, ...], int], ...]


class GroupWorker:
    def __init__(
        self, model, options, *, store, verify=False, atol=1e-4, rtol=1e-4, timeout=600
    ):
        self.model = model
        self.options = options
        self.verify, self.atol, self.rtol = verify, atol, rtol
        self.timeout = timeout
        self.store = store
        self.rank = dist.get_rank()
        self.states = {}
        self.groups = {}

    def engine(self, operation):
        members = operation.members
        if members not in self.groups:
            communication = GroupCommunication(
                self.store, members, self.model.device, self.timeout
            )
            engine = DynamicSPPrefillEngine(
                self.model,
                control_group=None,
                communication=communication,
                **self.options,
            )
            self.groups[members] = (operation.ticket, communication, engine)
        return self.groups[members][-1]

    @torch.inference_mode()
    def execute(self, operation):
        started = time.perf_counter()
        engine = self.engine(operation)
        members = operation.members
        local = {rank: index for index, rank in enumerate(members)}
        tail = (self.model.num_layers, 2, self.model.num_kv_heads, self.model.head_dim)

        def restore():
            states = {}
            for prompt, ranks, end in operation.previous:
                saved = self.states.get(prompt.request_id)
                if self.rank in ranks:
                    if saved is None or (saved.prompt, saved.ranks, saved.end) != (
                        prompt,
                        ranks,
                        end,
                    ):
                        raise ValueError(
                            "Committed KV metadata differs from coordinator"
                        )
                    kv = saved.kv
                else:
                    if saved is not None:
                        raise ValueError("Nonowner retains unexpected KV state")
                    kv = torch.empty(
                        (0, *tail), device=self.model.device, dtype=self.model.dtype
                    )
                states[prompt.request_id] = _State(
                    prompt, tuple(local[r] for r in ranks), end, kv
                )
            return states

        engine._states = engine._checked("restore operation state", restore)
        engine.version = 0
        plan = PrefillPlan(
            0,
            tuple(
                PromptStep(s.request_id, tuple(local[r] for r in s.ranks), s.end)
                for s in operation.plan.steps
            ),
            operation.plan.admit,
            operation.plan.release,
        )
        try:
            outputs = engine.execute(plan if engine.rank == 0 else None)

            def check_and_stage():
                errors = {}
                if self.verify and engine.rank == 0:
                    from sglang.srt.dynamic_sp.__main__ import reference_logits

                    for rid, actual in outputs.items():
                        state = engine._states[rid]
                        expected = reference_logits(
                            self.model, state.prompt.token_ids[: state.end]
                        )
                        torch.testing.assert_close(
                            actual, expected, atol=self.atol, rtol=self.rtol
                        )
                        errors[rid] = float(
                            (actual.float() - expected.float()).abs().max()
                        )
                staged = dict(self.states)
                for rid in operation.plan.release:
                    staged.pop(rid, None)
                for step in operation.plan.steps:
                    state = engine._states[step.request_id]
                    if self.rank in step.ranks:
                        staged[step.request_id] = _State(
                            state.prompt, step.ranks, state.end, state.kv
                        )
                    else:
                        staged.pop(step.request_id, None)
                tokens = {}
                if engine.rank == 0:
                    for rid, logits in outputs.items():
                        state = engine._states[rid]
                        if state.end == len(state.prompt.token_ids):
                            tokens[rid] = int(logits.argmax().item())
                return staged, tokens, errors

            staged, tokens, errors = engine._checked(
                "verify, sample and stage publication", check_and_stage
            )
            engine._agree(operation.ticket, "Operation tickets differ at commit")
            self.states = staged
            ready = time.perf_counter()
            return {
                "ticket": operation.ticket,
                "rank": self.rank,
                "status": "committed",
                "output_ids": tokens,
                "ready_at": ready,
                "started_at": started,
                "seconds": ready - started,
                "metrics": dict(engine.metrics),
                "errors": errors,
            }
        finally:
            # Cached engines retain communicators, not stale staged/committed KV.
            engine._states = {}

    def serve(self, connection):
        if self.model.device.type == "cuda":
            torch.cuda.set_device(self.model.device)
        _nodelay(connection)
        connection.send({"rank": self.rank, "clock_domain": clock_domain()})
        while True:
            message = connection.recv()
            if message[0] == "stop":
                for _, communication, _ in sorted(
                    self.groups.values(), key=lambda entry: entry[0]
                ):
                    communication.close()
                connection.send({"status": "stopped"})
                connection.close()
                return
            elif message[0] == "execute":
                operation = message[1]
                try:
                    result = self.execute(operation)
                except PlanRejected as exc:
                    if any(entry[-1]._failed for entry in self.groups.values()):
                        connection.send(
                            {
                                "ticket": operation.ticket,
                                "rank": self.rank,
                                "status": "fatal",
                                "error": str(exc),
                            }
                        )
                        raise
                    result = {
                        "ticket": operation.ticket,
                        "rank": self.rank,
                        "status": "rejected",
                        "error": str(exc),
                    }
                except Exception as exc:
                    connection.send(
                        {
                            "ticket": operation.ticket,
                            "rank": self.rank,
                            "status": "fatal",
                            "error": repr(exc),
                        }
                    )
                    raise
                connection.send(result)
            else:
                raise ValueError("Unknown async worker command")


def _register_workers(listener, connections, world_size, domain, timeout, local_error):
    """Bound accept, authentication and registration, including local failure.

    multiprocessing Listener authentication itself can block, so a daemon
    performs the handshake while the coordinator enforces one total deadline.
    Startup failures are fail-stop; torchrun terminates the other workers.
    """
    finished = threading.Event()
    errors = []

    def accept_workers():
        try:
            for _ in range(world_size):
                connection = listener.accept()
                try:
                    _nodelay(connection)
                    registration = connection.recv()
                    rank = registration["rank"]
                    if registration["clock_domain"] != domain:
                        raise RuntimeError(
                            "Async worker clock domain changed during startup"
                        )
                    if (
                        type(rank) is not int
                        or rank not in range(world_size)
                        or rank in connections
                    ):
                        raise RuntimeError(
                            f"Invalid async worker registration: {rank!r}"
                        )
                    connections[rank] = connection
                except BaseException:
                    connection.close()
                    raise
        except BaseException as exc:  # noqa: BLE001 - propagate thread failure
            errors.append(exc)
        finally:
            finished.set()

    thread = threading.Thread(
        target=accept_workers, daemon=True, name="dynamic-sp-registration"
    )
    thread.start()
    deadline = time.monotonic() + timeout
    try:
        while not finished.wait(min(0.1, max(0.0, deadline - time.monotonic()))):
            error = local_error()
            if error is not None:
                raise RuntimeError(
                    f"Async rank 0 worker failed during startup: {error!r}"
                ) from error
            if time.monotonic() >= deadline:
                missing = sorted(set(range(world_size)) - set(connections))
                raise TimeoutError(
                    f"Async worker startup exceeded {timeout:g}s; missing ranks={missing}. "
                    "A worker failed to connect, authenticate, or register."
                )
        if errors:
            raise RuntimeError(
                f"Async worker registration failed: {errors[0]!r}"
            ) from errors[0]
    finally:
        listener.close()


class AsyncExecutor:
    """Rank-zero dispatch API. Other ranks run serve() until close().

    Call setup on every rank once. submit() is nonblocking with respect to GPU
    execution; poll() returns only operations acknowledged by every participant.
    Recoverable rejection leaves worker KV intact and returns a rejected event.
    Transport/process failures are fail-stop, with a bounded operation timeout.
    """

    def __init__(
        self,
        model,
        control_group,
        options,
        *,
        verify=False,
        atol=1e-4,
        rtol=1e-4,
        timeout=600,
    ):
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.timeout = timeout
        self.clock_domain = clock_domain()
        domains = [None] * self.world_size
        dist.all_gather_object(domains, self.clock_domain, group=control_group)
        if any(domain != self.clock_domain for domain in domains):
            raise RuntimeError(
                "Async measurement requires all workers to share a host boot and "
                "monotonic time namespace; cross-host clock estimation is unsupported"
            )
        self.store = rendezvous_store(control_group, timeout)
        self.worker = GroupWorker(
            model,
            options,
            store=self.store,
            verify=verify,
            atol=atol,
            rtol=rtol,
            timeout=timeout,
        )
        self.connections = {}
        self.inflight = {}
        self.busy = set()
        self.thread = None
        self.thread_error = None
        if self.rank == 0:
            authkey = os.urandom(32)
            # Clock-domain validation above requires a single host. Use IPv4
            # loopback instead of resolving torchrun's possibly external address.
            listener = Listener(
                ("127.0.0.1", 0),
                family="AF_INET",
                authkey=authkey,
                backlog=max(8, self.world_size),
            )
            endpoint = (
                "127.0.0.1",
                listener.address[1],
                authkey,
            )
        else:
            endpoint = None
        packet = [endpoint]
        dist.broadcast_object_list(packet, src=0, group=control_group)
        host, port, authkey = packet[0]
        self.endpoint = ((host, port), authkey)
        if self.rank != 0:
            return

        def local_worker():
            try:
                connection = Client((host, port), family="AF_INET", authkey=authkey)
                self.worker.serve(connection)
            except BaseException as exc:  # noqa: BLE001 - forward worker-thread termination
                self.thread_error = exc

        context = contextvars.copy_context()
        self.thread = threading.Thread(
            target=lambda: context.run(local_worker),
            daemon=True,
            name="dynamic-sp-worker-0",
        )
        self.thread.start()
        _register_workers(
            listener,
            self.connections,
            self.world_size,
            self.clock_domain,
            min(timeout, 60.0),
            lambda: self.thread_error,
        )

    def serve(self):
        address, key = self.endpoint
        connection = Client(address, family="AF_INET", authkey=key)
        self.worker.serve(connection)

    def submit(self, operation):
        if self.rank != 0 or operation.ticket in self.inflight:
            raise ValueError("Only coordinator may submit fresh operation tickets")
        if (
            not operation.members
            or tuple(sorted(set(operation.members))) != operation.members
        ):
            raise ValueError("Operation members must be sorted distinct ranks")
        if self.busy.intersection(operation.members):
            raise ValueError("Operation conflicts with leased workers")
        self.busy.update(operation.members)
        self.inflight[operation.ticket] = (operation, time.perf_counter(), {})
        for rank in operation.members:
            self.connections[rank].send(("execute", operation))

    def poll(self, timeout=0.0):
        if self.thread_error:
            raise RuntimeError("Local async worker failed") from self.thread_error
        events = []
        for connection in wait(list(self.connections.values()), timeout):
            message = connection.recv()
            if message["status"] == "fatal":
                raise RuntimeError(f"Async worker failed: {message}")
            operation, _start, responses = self.inflight[message["ticket"]]
            rank = message["rank"]
            if (
                connection is not self.connections[rank]
                or rank not in operation.members
                or rank in responses
            ):
                raise RuntimeError("Invalid or duplicate operation acknowledgement")
            responses[rank] = message
            if len(responses) == len(operation.members):
                statuses = {m["status"] for m in responses.values()}
                if len(statuses) != 1:
                    raise RuntimeError(
                        "Group commit outcomes diverged; restart workers"
                    )
                events.append((operation, list(responses.values())))
                self.busy.difference_update(operation.members)
                del self.inflight[operation.ticket]
        now = time.perf_counter()
        if any(now - start > self.timeout for _, start, _ in self.inflight.values()):
            raise TimeoutError("Async operation exceeded its completion timeout")
        return events

    def close(self):
        if self.inflight:
            raise RuntimeError("Cannot stop workers with operations in flight")
        for connection in self.connections.values():
            connection.send(("stop",))
        for connection in self.connections.values():
            if (
                not connection.poll(self.timeout)
                or connection.recv()["status"] != "stopped"
            ):
                raise TimeoutError("Async worker shutdown timed out")
            connection.close()
        if self.thread:
            self.thread.join(timeout=self.timeout)
            if self.thread.is_alive() or self.thread_error:
                raise RuntimeError("Local async worker did not stop cleanly")
