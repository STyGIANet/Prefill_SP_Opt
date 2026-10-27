# Copyright 2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""Collective, transactional execution of per-prompt SP placement plans.

One caller per worker enters execute(); only worker zero supplies a plan.
Communication uses the execution group, which may be the whole world or an
asynchronously leased subset. Only designated owners compute tokens and retain
KV. Nonowners participate in control and migration collectives.

Plans commit after every requested prefill and migration has succeeded. Old
KV remains live until commit, so recoverable validation/allocation/model
errors leave the previous version intact. A transport failure is fail-stop:
the process group must be restarted, not reused.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Protocol

import torch
import torch.distributed as dist
import torch.nn.functional as F
from sglang.srt.dynamic_sp.attention import AttentionAccumulator
from sglang.srt.dynamic_sp.kv_storage import (
    SlotKV,
    read_layer,
    write_layer,
)
from sglang.srt.dynamic_sp.plan import (
    PrefillPlan,
    Prompt,
    local_positions,
    validate_ranks,
)
from torch.nn.attention.bias import causal_lower_right


class PrefillModel(Protocol):
    device: torch.device
    dtype: torch.dtype
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    vocab_size: int
    context_length: int

    def embed(self, ids: torch.Tensor) -> torch.Tensor: ...

    def prepare(self, layer: int, x: torch.Tensor, positions: torch.Tensor): ...

    def complete(self, layer: int, residual: torch.Tensor, attention: torch.Tensor): ...

    def logits(self, x: torch.Tensor) -> torch.Tensor: ...


@dataclass
class _State:
    prompt: Prompt
    ranks: tuple[int, ...]
    end: int
    kv: torch.Tensor | SlotKV  # [local_tokens, layers, 2, kv_heads, head_dim]


@dataclass
class _AttentionWorkspace:
    send: torch.Tensor
    received: torch.Tensor
    indices: list[torch.Tensor]
    key_positions: list[torch.Tensor]
    packs: list[tuple[_State, int, int]]
    send_counts: list[int]
    receive_counts: list[int]
    shared: bool


class PlanRejected(RuntimeError):
    """Every worker rejected the same operation; committed state is unchanged."""


class DynamicSPPrefillEngine:
    def __init__(
        self,
        model: PrefillModel,
        *,
        control_group,
        data_group=None,
        communication=None,
        max_local_tokens: int,
        kv_block_size: int = 256,
        query_block_size: int = 128,
        attention_backend: str = "gathered",
        logits_mode: str = "all",
        kv_transport: str = "targeted",
        shared_attention: str = "auto",
        migration_overlap: bool = True,
        kv_pool=None,
    ):
        if not dist.is_initialized():
            raise RuntimeError("Initialize torch.distributed before the SP engine")
        self.communication = communication
        self._local_checks_only = False
        self.data_group = data_group
        self.control_group = control_group
        if communication is not None:
            self.rank = communication.rank
            self.world_size = communication.world_size
            self.global_ranks = communication.global_ranks
        else:
            self.rank = dist.get_rank(data_group)
            self.world_size = dist.get_world_size(data_group)
            self.global_ranks = tuple(
                dist.get_process_group_ranks(data_group or dist.group.WORLD)
            )
            if dist.get_world_size(control_group) != self.world_size:
                raise ValueError(
                    "Control and data groups must have identical membership"
                )
            if tuple(dist.get_process_group_ranks(control_group)) != self.global_ranks:
                raise ValueError("Control and data group rank orders differ")
        self.model = model
        self.kv_pool = kv_pool
        self.max_local_tokens = max_local_tokens
        self.kv_block_size = kv_block_size
        self.query_block_size = query_block_size
        self.attention_backend = attention_backend
        self.logits_mode = logits_mode
        self.kv_transport = kv_transport
        self.shared_attention = shared_attention
        self.migration_overlap = migration_overlap
        self._migration = None
        self._ring = None
        self._ring_initialized = False
        self.metrics = {}
        self._status = torch.zeros(2, dtype=torch.int32, device="cpu")
        self.version = 0
        self._states: dict[str, _State] = {}
        self._failed = False
        self._busy = False
        self._checked("configuration", self._validate_configuration)
        self._agree(
            (
                model.num_layers,
                model.num_heads,
                model.num_kv_heads,
                model.head_dim,
                model.vocab_size,
                model.context_length,
                str(model.dtype),
                max_local_tokens,
                kv_block_size,
                query_block_size,
                attention_backend,
                logits_mode,
                kv_transport,
                shared_attention,
                migration_overlap,
            ),
            "Worker model/configuration mismatch",
        )

    def _validate_configuration(self):
        if self.shared_attention not in ("auto", "sdpa"):
            raise ValueError("shared_attention must be auto or sdpa")
        if self.kv_transport not in ("targeted", "all_gather"):
            raise ValueError("kv_transport must be targeted or all_gather")
        if self.logits_mode not in ("all", "completed"):
            raise ValueError("logits_mode must be all or completed")
        m = self.model
        if self.attention_backend not in ("gathered", "streaming", "ring"):
            raise ValueError("attention_backend must be gathered, streaming or ring")
        for value in (
            self.max_local_tokens,
            self.kv_block_size,
            self.query_block_size,
            m.num_layers,
            m.num_heads,
            m.num_kv_heads,
            m.head_dim,
            m.vocab_size,
            m.context_length,
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(
                    "Dimensions, block sizes and capacity must be positive ints"
                )
        if m.num_heads % m.num_kv_heads:
            raise ValueError("Query head count must be divisible by KV head count")

    def _all_objects(self, value):
        if self.communication is not None:
            return self.communication.objects(value)
        results = [None] * self.world_size
        dist.all_gather_object(results, value, group=self.control_group)
        return results

    def _broadcast(self, tensor, owner):
        if self.communication is not None:
            self.communication.broadcast(tensor, owner)
        else:
            dist.broadcast(tensor, src=self.global_ranks[owner], group=self.data_group)

    def _exchange(self, output, value, receive_counts, send_counts):
        if self.communication is not None:
            self.communication.exchange(output, value, receive_counts, send_counts)
        else:
            dist.all_to_all_single(
                output, value, receive_counts, send_counts, group=self.data_group
            )

    def _agree(self, value, message):
        values = self._all_objects(value)
        if any(v != values[0] for v in values):
            raise PlanRejected(message)

    def _checked(self, label, fn):
        started = time.perf_counter()
        self.metrics["checked_phases"] = self.metrics.get("checked_phases", 0) + 1
        if self._local_checks_only:
            # Async workers report fatal errors through their coordinator socket.
            # Never enter a recovery collective after peers may have advanced.
            try:
                return fn()
            except BaseException as exc:
                self._failed = True
                raise RuntimeError(
                    f"Fatal prefill phase {label} on rank {self.rank}"
                ) from exc
            finally:
                self.metrics["local_work_and_device_wait_seconds"] = (
                    self.metrics.get("local_work_and_device_wait_seconds", 0.0)
                    + time.perf_counter()
                    - started
                )
                self.metrics["local_checked_phases"] = (
                    self.metrics.get("local_checked_phases", 0) + 1
                )
        result, error, fatal = None, None, False
        try:
            result = fn()
            # CUDA work stays asynchronous here. Cross-stream dependencies are
            # explicit events; device execution errors are checked at commit.
        except Exception as exc:  # noqa: BLE001 - every local failure needs consensus
            error = f"rank {self.rank}: {type(exc).__name__}: {exc}"
            # An asynchronous CUDA execution error can invalidate the context.
            # Allocation exhaustion is recoverable; a device-side fault is not.
            fatal = (
                self.model.device.type == "cuda"
                and isinstance(exc, RuntimeError)
                and not isinstance(exc, torch.OutOfMemoryError)
                and ("CUDA error" in str(exc) or "device-side assert" in str(exc))
            )
        self.metrics["local_work_and_device_wait_seconds"] = (
            self.metrics.get("local_work_and_device_wait_seconds", 0.0)
            + time.perf_counter()
            - started
        )
        started = time.perf_counter()
        # Only errors need strings or Python serialization. The same two-word
        # reduction is entered by every worker, including workers with no rows.
        self._status[0] = int(error is not None)
        self._status[1] = int(fatal)
        if self.communication is not None:
            self.communication.status(self._status)
        else:
            dist.all_reduce(
                self._status, op=dist.ReduceOp.MAX, group=self.control_group
            )
        self.metrics["status_consensus_seconds"] = (
            self.metrics.get("status_consensus_seconds", 0.0)
            + time.perf_counter()
            - started
        )
        if self._status[0].item():
            self._failed |= bool(self._status[1].item())
            statuses = self._all_objects(error)
            raise PlanRejected(f"{label}: " + "; ".join(s for s in statuses if s))
        return result

    def _complete_device_work(self):
        if self.model.device.type == "cuda":
            torch.cuda.current_stream(self.model.device).synchronize()

    def snapshot(self):
        """Metadata only; callers cannot mutate engine ownership or cached KV."""
        return {
            rid: {"ranks": s.ranks, "end": s.end, "local_tokens": len(s.kv)}
            for rid, s in self._states.items()
        }

    def _metadata_digest(self):
        metadata = (
            self.version,
            [
                (rid, s.prompt.token_ids, s.ranks, s.end)
                for rid, s in sorted(self._states.items())
            ],
        )
        return hashlib.sha256(repr(metadata).encode()).hexdigest()

    def _validate(self, plan):
        if not isinstance(plan, PrefillPlan) or type(plan.version) is not int:
            raise ValueError("Expected a PrefillPlan with an integer version")
        if plan.version != self.version:
            raise ValueError(f"Stale plan {plan.version}; expected {self.version}")
        admitted = [p.request_id for p in plan.admit]
        steps = [s.request_id for s in plan.steps]
        released = list(plan.release)
        for name, values in (
            ("admit", admitted),
            ("steps", steps),
            ("release", released),
        ):
            if any(type(v) is not str or not v for v in values):
                raise ValueError(f"{name}: request IDs must be nonempty strings")
            if len(set(values)) != len(values):
                raise ValueError(f"{name}: duplicate request ID")
        if set(admitted) & set(self._states):
            raise ValueError("Cannot admit an existing request ID")
        if set(released) - set(self._states):
            raise ValueError("Cannot release an unknown request")
        if set(released) & set(steps):
            raise ValueError("Cannot release and execute the same request")
        if set(admitted) - set(steps):
            raise ValueError("Every admission needs an initial placement step")
        prompts = {rid: s.prompt for rid, s in self._states.items()}
        for p in plan.admit:
            if not isinstance(p.token_ids, tuple) or not p.token_ids:
                raise ValueError("Prompt token_ids must be a nonempty tuple")
            if len(p.token_ids) > self.model.context_length:
                raise ValueError("Prompt exceeds the model context length")
            if any(
                type(t) is not int or not 0 <= t < self.model.vocab_size
                for t in p.token_ids
            ):
                raise ValueError("Prompt token ID outside model vocabulary")
            if p.prefix_request_id is not None:
                parent = self._states.get(p.prefix_request_id)
                if parent is None or p.prefix_request_id not in released:
                    raise ValueError(
                        "Prefix admission must atomically release its resident parent"
                    )
                if (
                    parent.end != len(parent.prompt.token_ids)
                    or p.prefix_length != parent.end
                    or p.token_ids[: parent.end] != parent.prompt.token_ids
                ):
                    raise ValueError(
                        "Prefix admission does not match complete parent KV"
                    )
                if (
                    sum(q.prefix_request_id == p.prefix_request_id for q in plan.admit)
                    != 1
                ):
                    raise ValueError("A retained prefix can transfer to only one child")
            prompts[p.request_id] = p
        layouts = {
            rid: (s.ranks, s.end)
            for rid, s in self._states.items()
            if rid not in released
        }
        for step in plan.steps:
            if step.request_id not in prompts:
                raise ValueError(f"Unknown request {step.request_id!r}")
            validate_ranks(step.ranks, self.world_size)
            start = layouts.get(
                step.request_id, ((), prompts[step.request_id].prefix_length)
            )[1]
            if type(step.end) is not int or not start <= step.end <= len(
                prompts[step.request_id].token_ids
            ):
                raise ValueError(
                    "Step end must be between the committed cursor and prompt length"
                )
            layouts[step.request_id] = (step.ranks, step.end)
        for rank in range(self.world_size):
            count = sum(
                len(local_positions(end, ranks, rank))
                for ranks, end in layouts.values()
            )
            if count > self.max_local_tokens:
                raise ValueError(
                    f"Rank {rank} needs {count} KV tokens; capacity is {self.max_local_tokens}"
                )
        return prompts

    @torch.inference_mode()
    def execute(self, plan: PrefillPlan | None = None) -> dict[str, torch.Tensor]:
        # Async workers retain ownership through their outer verification and
        # publication. Standalone engines own the same slot transaction here.
        owned = self.kv_pool is not None and self.kv_pool.allocated is None
        old = self._states
        if owned:
            self.kv_pool.begin()
        try:
            memory_before = (
                torch.cuda.memory_stats(self.model.device)
                if owned and self.model.device.type == "cuda"
                else None
            )
            outputs = self._execute(plan, memory_before=memory_before)
            if owned:
                self.kv_pool.commit(old, self._states)
            return outputs
        except BaseException:
            if owned and not self._failed:
                self._complete_device_work()
                self.kv_pool.rollback()
                self._states = old
            raise

    def _execute(
        self, plan: PrefillPlan | None = None, *, memory_before=None
    ) -> dict[str, torch.Tensor]:
        """Execute worker zero's plan; return last-prefix logits on every worker.

        Omitted requests remain resident and paused. A step with end equal to
        its cursor performs only migration. Release is explicit. Calls must
        be serialized and made by every worker, including idle workers.
        """
        if self._failed:
            raise RuntimeError("Transport failed; restart the worker group")
        if self._busy:
            raise RuntimeError("Concurrent execute() calls are unsupported")
        self._busy = True
        self.metrics = {
            "attention_backend": self.attention_backend,
            "migration_overlap": self.migration_overlap,
            "kv_collectives": 0,
            "migration_collectives": 0,
            "logit_collectives": 0,
            "logits_mode": self.logits_mode,
            "kv_transport": self.kv_transport,
            "shared_attention": self.shared_attention,
            "fused_attention_calls": 0,
        }
        try:
            command = [plan if self.rank == 0 else None]
            if self.communication is not None:
                command[0] = self._all_objects(command[0])[0]
            else:
                dist.broadcast_object_list(
                    command, src=self.global_ranks[0], group=self.control_group
                )
            plan = command[0]
            self._agree(self._metadata_digest(), "Worker committed states diverged")
            prompts = self._checked("plan validation", lambda: self._validate(plan))
            staged = {
                rid: s for rid, s in self._states.items() if rid not in plan.release
            }
            forwards = []
            if self.attention_backend == "ring" and not self._ring_initialized:
                # NCCL batched P2P may use a subset only after group-wide init.
                token = self._checked(
                    "ring transport allocation",
                    lambda: torch.zeros(
                        1, device=self.model.device, dtype=self.model.dtype
                    ),
                )
                if self.communication is not None:
                    self.communication.data.allreduce([token]).wait()
                else:
                    dist.all_reduce(token, group=self.data_group)
                self._checked("ring transport initialization", lambda: None)
                self._ring_initialized = True
            deferred = self.kv_pool is not None or (
                self.attention_backend in ("ring", "gathered")
                and self.migration_overlap
            )
            migrated = self._redistribute_batch(plan.steps, prompts, deferred=deferred)
            for step, state in zip(plan.steps, migrated):
                previous = self._states.get(step.request_id) or self._states.get(
                    prompts[step.request_id].prefix_request_id
                )
                start = previous.end if previous is not None else 0
                if step.end > start:
                    forwards.append((state, start))
                staged[step.request_id] = state
            if self._migration is not None and (
                not self.migration_overlap or self.attention_backend == "streaming"
            ):
                self._migration.wait(0)
                for layer in range(1, self.model.num_layers):
                    self._migration.launch(layer)
                    self._migration.wait(layer)
                self._migration.drain()
                self._migration = None
            # Preserve validation/allocation and completion consensus, but do not
            # synchronize hosts at each projection, attention and MLP phase.
            self._local_checks_only = self.communication is not None
            try:
                self._graph_requested = bool(plan.steps) and all(
                    s.use_graph for s in plan.steps
                )
                outputs = self._prefill_batch(forwards) if forwards else {}
            finally:
                self._local_checks_only = False
            if not forwards and self._migration is not None:
                self._migration.wait(0)
                for layer in range(1, self.model.num_layers):
                    self._migration.launch(layer)
                    self._migration.wait(layer)
                self._checked("migration-only completion", self._migration.drain)

            # One completion boundary per transaction, after communication
            # streams have joined the compute stream through their ready events.
            # Do not synchronize the entire device: independent work may overlap.
            def complete():
                self._complete_device_work()
                if memory_before is not None:
                    stats = torch.cuda.memory_stats(self.model.device)
                    if any(
                        stats.get(key, 0) != memory_before.get(key, 0)
                        for key in ("num_alloc_retries", "num_ooms")
                    ):
                        raise RuntimeError(
                            "Allocator retry/OOM invalidates this operation"
                        )

            self._checked("device completion", complete)
            self._agree(self.version, "Worker versions diverged before commit")
            self._states = staged
            self.version += 1
            return outputs
        except PlanRejected:
            raise
        except BaseException:
            self._failed = True
            raise
        finally:
            try:
                # Never free or reuse staging buffers with device work in flight,
                # including recoverable failures in model callbacks.
                # A failed async peer may never issue the matching collective.
                # Retain buffers and let process teardown reclaim them; draining
                # here could hide the fatal notification behind a NCCL timeout.
                if not self._failed:
                    if self._migration is not None:
                        self._migration.drain()
                    if self._ring is not None:
                        self._ring.drain()
            except BaseException:
                self._failed = True
                raise
            finally:
                self._migration = self._ring = None
                self._busy = False

    def _redistribute_batch(self, steps, prompts, deferred=False):
        if not steps:
            return []
        m = self.model

        def allocate():
            states, changes = [], []
            tail = (m.num_layers, 2, m.num_kv_heads, m.head_dim)
            for step in steps:
                shape = (len(local_positions(step.end, step.ranks, self.rank)), *tail)
                previous = self._states.get(step.request_id) or self._states.get(
                    prompts[step.request_id].prefix_request_id
                )
                reusable = (
                    previous.kv
                    if previous is not None
                    and previous.ranks == step.ranks
                    and isinstance(previous.kv, SlotKV)
                    else None
                )
                kv = (
                    self.kv_pool.allocate(shape[0], reusable)
                    if self.kv_pool is not None
                    else torch.empty(shape, device=m.device, dtype=m.dtype)
                )
                state = _State(prompts[step.request_id], step.ranks, step.end, kv)
                states.append(state)
                previous = self._states.get(step.request_id) or self._states.get(
                    prompts[step.request_id].prefix_request_id
                )
                if previous is None or previous.end == 0:
                    continue
                if previous.ranks == step.ranks:
                    if not isinstance(kv, SlotKV):
                        kv[: len(previous.kv)].copy_(previous.kv)
                else:
                    changes.append((state, previous))
            if not changes:
                return states, None

            # One destination-major transfer for all changing prompts. Within
            # each peer segment, requests occur in canonical plan order.
            sends, receives = [], []
            for peer in range(self.world_size):
                send_parts, recv_parts = [], []
                for state, previous in changes:
                    old = local_positions(previous.end, previous.ranks, self.rank)
                    send_parts.append(
                        [
                            i
                            for i, position in enumerate(old)
                            if state.ranks[position % len(state.ranks)] == peer
                        ]
                    )
                    recv_parts.append(
                        [
                            position
                            for position in local_positions(
                                previous.end, previous.ranks, peer
                            )
                            if state.ranks[position % len(state.ranks)] == self.rank
                        ]
                    )
                sends.append(send_parts)
                receives.append(recv_parts)
            send_counts = [sum(map(len, parts)) for parts in sends]
            recv_counts = [sum(map(len, parts)) for parts in receives]
            transfer_tail = tail[1:] if deferred else tail
            send = (
                self.kv_pool.buffer("migration_send", sum(send_counts))
                if self.kv_pool is not None
                else torch.empty(
                    (sum(send_counts), *transfer_tail), device=m.device, dtype=m.dtype
                )
            )
            recv = (
                self.kv_pool.buffer("migration_recv", sum(recv_counts))
                if self.kv_pool is not None
                else torch.empty(
                    (sum(recv_counts), *transfer_tail), device=m.device, dtype=m.dtype
                )
            )
            offset, pack = 0, []
            for parts in sends:
                for (_, previous), indices in zip(changes, parts):
                    if indices:
                        index = torch.tensor(indices, device=m.device, dtype=torch.long)
                        if deferred:
                            pack.append((previous.kv, index, offset, len(indices)))
                        else:
                            send[offset : offset + len(indices)].copy_(
                                previous.kv.index_select(0, index)
                            )
                    offset += len(indices)
            offset, unpack = 0, []
            for parts in receives:
                for (state, _), positions in zip(changes, parts):
                    if positions:
                        local_offset = state.ranks.index(self.rank)
                        indices = torch.tensor(
                            [
                                (position - local_offset) // len(state.ranks)
                                for position in positions
                            ],
                            device=m.device,
                            dtype=torch.long,
                        )
                        unpack.append((state.kv, indices, offset, len(positions)))
                    offset += len(positions)
            self.metrics["migration_workspace_bytes"] = (
                send.numel() + recv.numel()
            ) * send.element_size()
            transfer = (send, recv, send_counts, recv_counts, unpack)
            if deferred:
                from sglang.srt.dynamic_sp.ring import LayerMigration

                return states, LayerMigration(self, (*transfer, pack))
            return states, transfer

        states, transfer = self._checked("KV and migration allocation", allocate)
        if transfer is not None and deferred:
            self._migration = transfer
            self._migration.launch(0)
        elif transfer is not None:
            send, recv, send_counts, recv_counts, unpack = transfer
            self._exchange(recv, send, recv_counts, send_counts)
            self.metrics["migration_collectives"] = 1

            def complete():
                for kv, indices, offset, count in unpack:
                    kv.index_copy_(0, indices, recv[offset : offset + count])

            self._checked("migration completion", complete)
        return states

    def _prefill_batch(self, forwards):
        graph = getattr(self, "prefill_graph", None)
        if (
            graph is not None
            and getattr(self, "_graph_requested", False)
            and graph.shape(forwards) is not None
        ):
            if self._migration is not None:
                self._migration.wait(0)
                for layer in range(1, self.model.num_layers):
                    self._migration.launch(layer)
                    self._migration.wait(layer)
                self._migration.drain()
                self._migration = None
            outputs = graph.run(forwards, self.logits_mode)
            self.metrics["prefill_graph_replays"] = 1
            return outputs
        m = self.model
        layouts = []
        all_positions, all_ids = [], []
        for state, start in forwards:
            owned = list(local_positions(state.end, state.ranks, self.rank))
            prefix_count = len(local_positions(start, state.ranks, self.rank))
            new_positions = owned[prefix_count:]
            begin = len(all_positions)
            all_positions.extend(new_positions)
            all_ids.extend(state.prompt.token_ids[p] for p in new_positions)
            layouts.append((state, prefix_count, begin, len(all_positions)))

        def embed():
            positions = torch.tensor(all_positions, device=m.device, dtype=torch.long)
            ids = torch.tensor(all_ids, device=m.device, dtype=torch.long)
            x = m.embed(ids) if len(ids) else None
            return positions, x

        positions, x = self._checked("embedding", embed)
        workspace = (
            self._checked("gather workspace", lambda: self._gather_workspace(layouts))
            if self.attention_backend == "gathered"
            else None
        )
        if self.attention_backend == "ring":
            x = self._prefill_ring(x, positions, layouts)
        elif workspace is not None:
            x = self._prefill_gathered(x, positions, layouts, workspace)
        for layer in range(
            m.num_layers if self.attention_backend == "streaming" else 0
        ):

            def prepare(layer=layer, x=x):
                q = self._project_layer(layer, x, positions, layouts)
                return [
                    [
                        AttentionAccumulator(
                            q[i : min(i + self.query_block_size, end)],
                            positions[i : min(i + self.query_block_size, end)],
                            m.head_dim**-0.5,
                        )
                        for i in range(begin, end, self.query_block_size)
                    ]
                    for _, _, begin, end in layouts
                ]

            accumulators = self._checked(f"layer {layer} projections", prepare)
            # Buffers are bounded by kv_block_size, independent of prompt length.
            for (state, _, _, _), request_accumulators in zip(layouts, accumulators):
                for owner in state.ranks:
                    owner_positions = list(
                        local_positions(state.end, state.ranks, owner)
                    )
                    for offset in range(0, len(owner_positions), self.kv_block_size):
                        block_positions = owner_positions[
                            offset : offset + self.kv_block_size
                        ]

                        def make_block(
                            owner=owner,
                            state=state,
                            offset=offset,
                            block_positions=block_positions,
                            layer=layer,
                        ):
                            if self.rank == owner:
                                block = read_layer(
                                    state.kv,
                                    layer,
                                    slice(offset, offset + len(block_positions)),
                                ).contiguous()
                            else:
                                block = torch.empty(
                                    (
                                        len(block_positions),
                                        2,
                                        m.num_kv_heads,
                                        m.head_dim,
                                    ),
                                    device=m.device,
                                    dtype=m.dtype,
                                )
                            pos = torch.tensor(
                                block_positions, device=m.device, dtype=torch.long
                            )
                            return block, pos

                        block, key_positions = self._checked(
                            "attention workspace", make_block
                        )
                        self._broadcast(block, owner)
                        self.metrics["kv_collectives"] += 1

                        def accumulate(
                            request_accumulators=request_accumulators,
                            block=block,
                            key_positions=key_positions,
                        ):
                            for acc in request_accumulators:
                                acc.add(block[:, 0], block[:, 1], key_positions)

                        self._checked(f"layer {layer} attention", accumulate)

            def complete(x=x, accumulators=accumulators, layer=layer):
                if x is None:
                    return None
                output = torch.cat(
                    [a.finish() for group in accumulators for a in group], dim=0
                )
                return m.complete(layer, x, output)

            x = self._checked(f"layer {layer} output", complete)

        logit_layouts = [
            layout
            for layout in layouts
            if self.logits_mode == "all"
            or layout[0].end == len(layout[0].prompt.token_ids)
        ]
        self.metrics["logit_requests"] = len(logit_layouts)
        if not logit_layouts:
            return {}

        def make_logits():
            outputs = {}
            finite_checks = []
            for state, _, _, end in logit_layouts:
                owner = state.ranks[(state.end - 1) % len(state.ranks)]
                if self.rank == owner:
                    result = (
                        m.logits(x[end - 1 : end]).reshape(m.vocab_size).contiguous()
                    )
                    if result.dtype != m.dtype or result.device != m.device:
                        raise ValueError("Model logits changed dtype or device")
                    finite_checks.append(torch.isfinite(result).all())
                else:
                    result = torch.empty(m.vocab_size, device=m.device, dtype=m.dtype)
                outputs[state.prompt.request_id] = result
            if finite_checks and not bool(torch.stack(finite_checks).all()):
                raise ValueError("Model returned non-finite logits")
            return outputs

        outputs = self._checked("logits", make_logits)
        for state, _, _, _ in logit_layouts:
            owner = state.ranks[(state.end - 1) % len(state.ranks)]
            self._broadcast(outputs[state.prompt.request_id], owner)
            self.metrics["logit_collectives"] += 1
        self._checked("logits completion", lambda: None)
        return outputs

    def _prefill_ring(self, x, positions, layouts):
        from sglang.srt.dynamic_sp.ring import RingWorkspace

        self._ring = self._checked(
            "ring workspace", lambda: RingWorkspace(self, layouts)
        )
        for layer in range(self.model.num_layers):
            if self._migration is not None:
                self._migration.wait(layer)
            q = self._checked(
                "ring projections",
                lambda layer=layer, x=x: self._project_layer(
                    layer, x, positions, layouts
                ),
            )
            accumulators = self._checked(
                "ring preparation",
                lambda layer=layer, q=q: self._ring.prepare(
                    layer, q, positions, layouts
                ),
            )
            attention = self._ring.attention(q, layouts, accumulators)
            # Previous-layer attention no longer uses the communicator. Start
            # balancing the next layer, then enqueue this layer's output/MLP.
            # The next iteration waits for migrated history before reading it.
            if self._migration is not None and layer + 1 < self.model.num_layers:
                self._migration.launch(layer + 1)
            x = self._checked(
                "ring output",
                lambda layer=layer, x=x, attention=attention: (
                    self.model.complete(layer, x, attention) if x is not None else None
                ),
            )
        return x

    def _project_layer(self, layer, x, positions, layouts, workspace=None):
        m = self.model
        if x is None:
            q = torch.empty(
                (0, m.num_heads, m.head_dim), device=m.device, dtype=m.dtype
            )
        else:
            q, k, v = m.prepare(layer, x, positions)
            expected_q = (len(positions), m.num_heads, m.head_dim)
            expected_kv = (len(positions), m.num_kv_heads, m.head_dim)
            if (
                q.shape != expected_q
                or k.shape != expected_kv
                or v.shape != expected_kv
            ):
                raise ValueError(
                    "Model projection shapes do not match the declared head layout"
                )
            if any(t.dtype != m.dtype or t.device != m.device for t in (q, k, v)):
                raise ValueError("Model projections changed dtype or device")
            for state, prefix_count, begin, end in layouts:
                write_layer(state.kv, layer, prefix_count, k[begin:end], v[begin:end])
        if workspace is not None:
            for state, begin, end in workspace.packs:
                read_layer(state.kv, layer, out=workspace.send[begin:end])
        return q

    def _prefill_gathered(self, x, positions, layouts, workspace):
        if self._migration is not None:
            return self._prefill_gathered_migrating(x, positions, layouts, workspace)
        if not workspace.shared:
            # No transport occurs inside this region. A local failure can skip
            # subsequent layers safely; all ranks vote before logits or commit.
            def local_layers():
                hidden = x
                for layer in range(self.model.num_layers):
                    q = self._project_layer(
                        layer, hidden, positions, layouts, workspace
                    )
                    hidden = self._gathered_complete(
                        layer, hidden, q, positions, layouts, workspace
                    )
                return hidden

            return self._checked("private layers", local_layers)

        q = self._checked(
            "layer 0 projections",
            lambda: self._project_layer(0, x, positions, layouts, workspace),
        )
        for layer in range(self.model.num_layers):
            # Every allocation, pack and model callback precedes a successful
            # collective vote. Idle ranks enter this transport with zero splits.
            if self.kv_transport == "targeted":
                self._exchange(
                    workspace.received,
                    workspace.send,
                    workspace.receive_counts,
                    workspace.send_counts,
                )
            else:
                if self.communication is not None:
                    self.communication.gather(workspace.received, workspace.send)
                else:
                    dist.all_gather_single(
                        workspace.received, workspace.send, group=self.data_group
                    )
            self.metrics["kv_collectives"] += 1

            def complete_and_prepare(layer=layer, x=x, q=q):
                hidden = self._gathered_complete(
                    layer, x, q, positions, layouts, workspace
                )
                next_q = (
                    self._project_layer(
                        layer + 1, hidden, positions, layouts, workspace
                    )
                    if layer + 1 < self.model.num_layers
                    else None
                )
                return hidden, next_q

            # Merge adjacent local phases, retaining a rejection point before
            # every transport. A failed output or next projection cannot strand
            # peers in the following collective or modify committed KV.
            x, q = self._checked(
                f"layer {layer} output and next projections", complete_and_prepare
            )
        return x

    def _prefill_gathered_migrating(self, x, positions, layouts, workspace):
        """Use the same layer migration pipeline with targeted exchange/all-gather."""
        for layer in range(self.model.num_layers):
            self._migration.wait(layer)
            q = self._checked(
                "gathered projections",
                lambda layer=layer, x=x: self._project_layer(
                    layer, x, positions, layouts, workspace
                ),
            )
            if workspace.shared:
                if self.kv_transport == "targeted":
                    self._exchange(
                        workspace.received,
                        workspace.send,
                        workspace.receive_counts,
                        workspace.send_counts,
                    )
                elif self.communication is not None:
                    self.communication.gather(workspace.received, workspace.send)
                else:
                    dist.all_gather_single(
                        workspace.received, workspace.send, group=self.data_group
                    )
                self.metrics["kv_collectives"] += 1
            attention = self._checked(
                "gathered attention",
                lambda layer=layer, q=q: self._gathered_attention(
                    layer, q, positions, layouts, workspace
                ),
            )
            if layer + 1 < self.model.num_layers:
                self._migration.launch(layer + 1)
            x = self._checked(
                "gathered output",
                lambda layer=layer, x=x, attention=attention: (
                    self.model.complete(layer, x, attention) if x is not None else None
                ),
            )
        return x

    def _gather_workspace(self, layouts):
        """Pack shared KV only for destinations with new queries for that prompt."""
        m = self.model
        shared = [len(state.ranks) > 1 for state, _, _, _ in layouts]
        counts = [
            [
                len(local_positions(s.end, s.ranks, rank)) if shared[i] else 0
                for i, (s, _, _, _) in enumerate(layouts)
            ]
            for rank in range(self.world_size)
        ]
        consumers = []
        for state, _, _, _ in layouts:
            # An advancing strided suffix ends at state.end. Recover its global
            # start from this rank's prefix count only via the original state.
            old = self._states.get(state.prompt.request_id)
            start = old.end if old is not None else 0
            consumers.append(
                {
                    rank
                    for rank in state.ranks
                    if len(local_positions(state.end, state.ranks, rank))
                    > len(local_positions(start, state.ranks, rank))
                }
            )
        packs, indices, key_positions = [], [], []
        send_counts = [0] * self.world_size
        receive_counts = [0] * self.world_size
        offsets = [[0] * len(layouts) for _ in range(self.world_size)]
        if self.kv_transport == "targeted":
            cursor = 0
            for dest in range(self.world_size):
                for i, (state, _, _, _) in enumerate(layouts):
                    count = counts[self.rank][i] if dest in consumers[i] else 0
                    if count:
                        packs.append((state, cursor, cursor + count))
                        cursor += count
                        send_counts[dest] += count
            send_rows = cursor
            cursor = 0
            for source in range(self.world_size):
                for i in range(len(layouts)):
                    offsets[source][i] = cursor
                    count = counts[source][i] if self.rank in consumers[i] else 0
                    cursor += count
                    receive_counts[source] += count
            receive_rows = cursor
        else:
            stride = max(map(sum, counts))
            send_rows, receive_rows = stride, self.world_size * stride
            for source in range(self.world_size):
                cursor = source * stride
                for i, (state, _, _, _) in enumerate(layouts):
                    offsets[source][i] = cursor
                    count = counts[source][i]
                    if source == self.rank and count:
                        packs.append(
                            (
                                state,
                                cursor - source * stride,
                                cursor - source * stride + count,
                            )
                        )
                    cursor += count
        shape = (2, m.num_kv_heads, m.head_dim)
        send = (
            self.kv_pool.buffer("gather_send", send_rows)
            if self.kv_pool is not None
            else torch.empty((send_rows, *shape), device=m.device, dtype=m.dtype)
        )
        send.zero_()
        received = (
            self.kv_pool.buffer("gather_recv", receive_rows)
            if self.kv_pool is not None
            else torch.empty((receive_rows, *shape), device=m.device, dtype=m.dtype)
        )
        for i, (state, _, begin, end) in enumerate(layouts):
            mapping = (
                [
                    offsets[state.ranks[p % len(state.ranks)]][i]
                    + p // len(state.ranks)
                    for p in range(state.end)
                ]
                if begin != end and shared[i]
                else []
            )
            indices.append(torch.tensor(mapping, device=m.device, dtype=torch.long))
            key_positions.append(
                torch.arange(
                    state.end if begin != end and shared[i] else 0, device=m.device
                )
            )
        self.metrics["gather_workspace_bytes"] = (
            send.numel() + received.numel()
        ) * send.element_size()
        self.metrics["kv_send_bytes_per_layer"] = send.numel() * send.element_size()
        self.metrics["kv_receive_bytes_per_layer"] = (
            received.numel() * received.element_size()
        )
        self.metrics["shared_kv_rows"] = sum(map(sum, counts))
        self.metrics["private_kv_rows"] = sum(
            state.end for state, _, _, _ in layouts if len(state.ranks) == 1
        )
        return _AttentionWorkspace(
            send,
            received,
            indices,
            key_positions,
            packs,
            send_counts,
            receive_counts,
            any(shared),
        )

    def _gathered_complete(self, layer, x, q, positions, layouts, workspace):
        if x is None:
            return None
        return self.model.complete(
            layer, x, self._gathered_attention(layer, q, positions, layouts, workspace)
        )

    def _gathered_attention(self, layer, q, positions, layouts, workspace):
        m = self.model
        received, indices, key_positions = (
            workspace.received,
            workspace.indices,
            workspace.key_positions,
        )
        output = torch.empty_like(q)
        for request, (state, prefix_count, begin, end) in enumerate(layouts):
            if begin == end:
                continue
            private = len(state.ranks) == 1
            if (
                self.shared_attention == "auto"
                and q.device.type == "cuda"
                and q.dtype in (torch.float16, torch.bfloat16)
                and m.head_dim <= 256
            ):
                from sglang.srt.dynamic_sp.fused_attention import (
                    strided_causal_attention,
                )

                if private and isinstance(state.kv, SlotKV):
                    k = state.kv.pool.native.get_key_buffer(layer)
                    v = state.kv.pool.native.get_value_buffer(layer)
                    mapping = state.kv.indices
                elif private:
                    k, v = state.kv[:, layer, 0], state.kv[:, layer, 1]
                    mapping = None
                else:
                    # Read owner-ordered receive buffers through the logical
                    # position map, without materializing a reordered KV copy.
                    k, v = received[:, 0], received[:, 1]
                    mapping = indices[request]
                output[begin:end].copy_(
                    strided_causal_attention(
                        q[begin:end],
                        k,
                        v,
                        first_query=prefix_count * len(state.ranks)
                        + state.ranks.index(self.rank),
                        query_stride=len(state.ranks),
                        kv_indices=mapping,
                    )
                )
                self.metrics["fused_attention_calls"] += 1
                self.metrics["indexed_attention_calls"] = self.metrics.get(
                    "indexed_attention_calls", 0
                ) + int(mapping is not None)
                continue
            kv = (
                read_layer(state.kv, layer)
                if private
                else received.index_select(0, indices[request])
            )
            k = kv[:, 0].transpose(0, 1).unsqueeze(0)
            v = kv[:, 1].transpose(0, 1).unsqueeze(0)
            for offset in range(begin, end, self.query_block_size):
                stop = min(offset + self.query_block_size, end)
                if private:
                    # Queries form a contiguous suffix of this key prefix. A
                    # lower-right causal bias permits fused SDPA without a dense
                    # mask. Truncate keys to this query block's final position.
                    key_end = prefix_count + stop - begin
                    attended = F.scaled_dot_product_attention(
                        q[offset:stop].transpose(0, 1).unsqueeze(0),
                        k[:, :, :key_end],
                        v[:, :, :key_end],
                        attn_mask=causal_lower_right(stop - offset, key_end),
                        dropout_p=0.0,
                        enable_gqa=m.num_heads != m.num_kv_heads,
                    )
                    output[offset:stop].copy_(attended.squeeze(0).transpose(0, 1))
                    continue
                visible = (
                    key_positions[request][None, :] <= positions[offset:stop, None]
                )
                # Local query positions can be strided and start after a cached
                # prefix. is_causal=True would apply the wrong triangular mask.
                attended = F.scaled_dot_product_attention(
                    q[offset:stop].transpose(0, 1).unsqueeze(0),
                    k,
                    v,
                    attn_mask=visible,
                    dropout_p=0.0,
                    enable_gqa=m.num_heads != m.num_kv_heads,
                )
                output[offset:stop].copy_(attended.squeeze(0).transpose(0, 1))
        return output
