# Copyright 2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""Reusable shard-ring execution and layer-pipelined KV redistribution.

Network operations have a fixed, metadata-derived order. Recoverable attention
errors drain that order before the group votes to reject. Committed KV is never
modified. CUDA communication uses a separate stream; buffer reuse and reads
are ordered with stream events, not host estimates of completion.
"""

from contextlib import nullcontext

import torch
import torch.distributed as dist
from sglang.srt.dynamic_sp.attention import AttentionAccumulator
from sglang.srt.dynamic_sp.plan import local_positions


class LayerMigration:
    def __init__(self, engine, transfer):
        self.engine = engine
        (
            self.send,
            self.recv,
            self.send_counts,
            self.recv_counts,
            self.unpack,
            self.pack,
        ) = transfer
        self.stream = (
            torch.cuda.Stream(device=engine.model.device)
            if engine.model.device.type == "cuda"
            else None
        )
        self.ready = torch.cuda.Event() if self.stream is not None else None
        self.layer = -1

    def launch(self, layer):
        if layer != self.layer + 1:
            raise RuntimeError("Migration layers must be issued exactly once in order")

        def pack():
            for source, index, offset, count in self.pack:
                torch.index_select(
                    source[:, layer], 0, index, out=self.send[offset : offset + count]
                )

        # Vote on host-side packing failures before entering the collective.
        # Packing is asynchronous: transfer must wait for its producer stream.
        self.engine._checked("migration layer pack", pack)
        if self.stream is not None:
            self.stream.wait_stream(torch.cuda.current_stream(self.engine.model.device))
        context = (
            torch.cuda.stream(self.stream) if self.stream is not None else nullcontext()
        )
        with context:
            self.engine._exchange(
                self.recv, self.send, self.recv_counts, self.send_counts
            )
            for target, index, offset, count in self.unpack:
                target[:, layer].index_copy_(
                    0, index, self.recv[offset : offset + count]
                )
            if self.ready is not None:
                self.ready.record(self.stream)
        self.layer = layer
        self.engine.metrics["migration_collectives"] += 1

    def wait(self, layer):
        if layer != self.layer:
            raise RuntimeError("Attempted to consume an unmigrated KV layer")
        if self.ready is not None:
            torch.cuda.current_stream(self.engine.model.device).wait_event(self.ready)

    def drain(self):
        if self.stream is not None:
            self.stream.synchronize()


def rotate(engine, send, receive, following, preceding):
    """Post paired nonblocking P2P operations using execution-group local ranks."""
    if engine.communication is not None:
        backend = engine.communication.data
        if send.is_cuda:
            backend._start_coalescing()
            try:
                backend.send([send], following, 0)
                backend.recv([receive], preceding, 0)
            finally:
                work = backend._end_coalescing()
            return [work]
        return [
            backend.send([send], following, 0),
            backend.recv([receive], preceding, 0),
        ]
    return dist.batch_isend_irecv(
        [
            dist.P2POp(
                dist.isend, send, engine.global_ranks[following], engine.data_group
            ),
            dist.P2POp(
                dist.irecv, receive, engine.global_ranks[preceding], engine.data_group
            ),
        ]
    )


class RingWorkspace:
    def __init__(self, engine, layouts):
        self.engine = engine
        m = engine.model
        self.stream = (
            torch.cuda.Stream(device=m.device) if m.device.type == "cuda" else None
        )
        self.ready = torch.cuda.Event() if self.stream is not None else None
        self.buffers, self.positions = [], []
        for state, _, _, _ in layouts:
            rows = (state.end + len(state.ranks) - 1) // len(state.ranks)
            # Nonmembers take part in error votes, not in the ring itself.
            shape = (
                rows if engine.rank in state.ranks else 0,
                2,
                m.num_kv_heads,
                m.head_dim,
            )
            self.buffers.append(
                [torch.empty(shape, device=m.device, dtype=m.dtype) for _ in range(2)]
            )
            self.positions.append(
                [
                    torch.tensor(
                        list(local_positions(state.end, state.ranks, r)),
                        device=m.device,
                        dtype=torch.long,
                    )
                    for r in state.ranks
                ]
            )
        engine.metrics["ring_workspace_bytes"] = sum(
            b.numel() * b.element_size() for pair in self.buffers for b in pair
        )
        engine.metrics["ring_hops"] = 0

    def prepare(self, layer, q, positions, layouts):
        states = []
        m = self.engine.model
        fused = (
            m.device.type == "cuda"
            and m.dtype in (torch.float16, torch.bfloat16)
            and m.head_dim <= 256
            and self.engine.shared_attention == "auto"
        )
        for i, (state, prefix, begin, end) in enumerate(layouts):
            if self.engine.rank in state.ranks:
                self.buffers[i][0].zero_()
                self.buffers[i][0][: len(state.kv)].copy_(state.kv[:, layer])
            if fused:
                from sglang.srt.dynamic_sp.fused_attention import RingAccumulator

                acc = RingAccumulator(
                    q[begin:end],
                    prefix * len(state.ranks)
                    + (
                        state.ranks.index(self.engine.rank)
                        if self.engine.rank in state.ranks
                        else 0
                    ),
                    len(state.ranks),
                )
            else:
                acc = [
                    AttentionAccumulator(
                        q[j : min(j + self.engine.query_block_size, end)],
                        positions[j : min(j + self.engine.query_block_size, end)],
                        m.head_dim**-0.5,
                    )
                    for j in range(begin, end, self.engine.query_block_size)
                ]
            states.append(acc)
        return states

    def attention(self, q, layouts, accumulators):
        e = self.engine
        # Allocate before any P2P operation. A failed allocation rejects on all ranks.
        output = e._checked("ring output allocation", lambda: torch.empty_like(q))
        error = None
        for i, ((state, _, begin, end), accumulators_for_request) in enumerate(
            zip(layouts, accumulators)
        ):
            if e.rank not in state.ranks:
                continue
            degree = len(state.ranks)
            index = state.ranks.index(e.rank)
            current = self.buffers[i][0]
            for hop in range(degree):
                origin = (index - hop) % degree
                count = len(self.positions[i][origin])
                works = []
                if hop + 1 < degree:
                    following = self.buffers[i][(hop + 1) % 2]
                    context = (
                        torch.cuda.stream(self.stream)
                        if self.stream is not None
                        else nullcontext()
                    )
                    if self.stream is not None:
                        self.stream.wait_stream(
                            torch.cuda.current_stream(e.model.device)
                        )
                    with context:
                        works = rotate(
                            e,
                            current,
                            following,
                            state.ranks[(index + 1) % degree],
                            state.ranks[(index - 1) % degree],
                        )
                    e.metrics["ring_hops"] += 1
                # This work is independent of forwarding current and receiving next.
                # Synchronous local failures must not prevent subsequent P2P hops.
                try:
                    if error is None and end > begin:
                        if isinstance(accumulators_for_request, list):
                            for offset in range(0, count, e.kv_block_size):
                                stop = min(offset + e.kv_block_size, count)
                                for acc in accumulators_for_request:
                                    acc.add(
                                        current[offset:stop, 0],
                                        current[offset:stop, 1],
                                        self.positions[i][origin][offset:stop],
                                    )
                        else:
                            accumulators_for_request.add(
                                current[:count, 0], current[:count, 1], origin
                            )
                            e.metrics["fused_attention_calls"] += 1
                except Exception as exc:  # noqa: BLE001 - drain the ring before rejection
                    error = exc
                if works:
                    context = (
                        torch.cuda.stream(self.stream)
                        if self.stream is not None
                        else nullcontext()
                    )
                    with context:
                        for work in works:
                            work.wait()
                        if self.ready is not None:
                            self.ready.record(self.stream)
                    if self.ready is not None:
                        torch.cuda.current_stream(e.model.device).wait_event(self.ready)
                    current = following
            try:
                if error is None and end > begin:
                    if isinstance(accumulators_for_request, list):
                        output[begin:end].copy_(
                            torch.cat([a.finish() for a in accumulators_for_request])
                        )
                    else:
                        output[begin:end].copy_(accumulators_for_request.finish())
            except Exception as exc:  # noqa: BLE001 - peers must finish their rings
                error = exc

        def finish():
            if error is not None:
                raise error
            return output

        return e._checked("ring attention completion", finish)

    def drain(self):
        if self.stream is not None:
            self.stream.synchronize()
