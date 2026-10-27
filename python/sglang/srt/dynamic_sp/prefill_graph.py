# Copyright 2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""Reusable length/batch CUDA graphs over the native paged KV pool.

Single-worker graphs read historical KV directly from the pool. Padding never
writes a KV slot or changes a causal position. Shapes without a captured graph
use the ordinary engine. Graph capture occurs before benchmark arrivals.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _store(
    K,
    V,
    KP,
    VP,
    Slots,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    KS: tl.constexpr,
    VS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    slot = tl.load(Slots + row)
    x = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    k = tl.load(K + row * H * D + x, x < H * D, 0)
    v = tl.load(V + row * H * D + x, x < H * D, 0)
    tl.store(KP + slot * KS + x, k, (slot >= 0) & (x < H * D))
    tl.store(VP + slot * VS + x, v, (slot >= 0) & (x < H * D))


@triton.jit
def _attention(
    Q,
    K,
    V,
    Map,
    Starts,
    Ends,
    O,
    L: tl.constexpr,
    NH: tl.constexpr,
    NK: tl.constexpr,
    D: tl.constexpr,
    CAP: tl.constexpr,
    KS: tl.constexpr,
    VS: tl.constexpr,
    SCALE: tl.constexpr,
    BM: tl.constexpr = 16,
    BN: tl.constexpr = 64,
):
    b, h, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    qpos = tile * BM + tl.arange(0, BM)
    dim = tl.arange(0, D)
    start, end = tl.load(Starts + b), tl.load(Ends + b)
    valid = (qpos < L) & (qpos + start < end)
    q = tl.load(
        Q + ((b * L + qpos[:, None]) * NH + h) * D + dim[None, :], valid[:, None], 0
    )
    mx = tl.full((BM,), -float("inf"), tl.float32)
    den = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, D), tl.float32)
    for block in range(tl.cdiv(end, BN)):
        pos = block * BN + tl.arange(0, BN)
        slots = tl.load(Map + b * CAP + pos, pos < end, 0)
        kh = h // (NH // NK)
        k = tl.load(
            K + slots[None, :] * KS + kh * D + dim[:, None], pos[None, :] < end, 0
        )
        v = tl.load(
            V + slots[:, None] * VS + kh * D + dim[None, :], pos[:, None] < end, 0
        )
        scores = tl.dot(q, k).to(tl.float32) * SCALE
        scores = tl.where(
            valid[:, None]
            & (pos[None, :] < end)
            & (pos[None, :] <= start + qpos[:, None]),
            scores,
            -float("inf"),
        )
        new = tl.maximum(mx, tl.max(scores, 1))
        safe = tl.where(new == -float("inf"), 0.0, new)
        alpha = tl.exp(mx - safe)
        prob = tl.exp(scores - safe[:, None])
        acc = acc * alpha[:, None] + tl.dot(prob.to(v.dtype), v)
        den = den * alpha + tl.sum(prob, 1)
        mx = new
    result = acc / tl.maximum(den[:, None], 1.0e-20)
    tl.store(
        O + ((b * L + qpos[:, None]) * NH + h) * D + dim[None, :],
        result,
        qpos[:, None] < L,
    )


class PrefillGraphCache:
    def __init__(self, model, pool, lengths, batches, token_budget):
        self.model, self.pool = model, pool
        if model.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("Native prefill graphs require FP16 or BF16 weights")
        memory_before = torch.cuda.memory_stats(model.device)
        self.shapes = [
            (b, n)
            for b in batches
            for n in lengths
            if b * n <= min(token_budget, pool.capacity) and n <= model.context_length
        ]
        self.graphs = {}
        if not self.shapes:
            raise ValueError("No prefill graph shape fits the configured budget")
        if model.head_dim not in (32, 64, 128, 256):
            raise ValueError("Prefill graphs require head_dim 32, 64, 128 or 256")
        max_b = max(b for b, _ in self.shapes)
        hidden = model.num_heads * model.head_dim
        cfg = getattr(getattr(model, "model", None), "config", None)
        intermediate = getattr(cfg, "intermediate_size", 4 * hidden)
        estimate = (
            max(b * n for b, n in self.shapes)
            * (6 * intermediate + 12 * hidden)
            * model.dtype.itemsize
        )
        estimate += max_b * pool.capacity * 8 + sum(
            b * model.vocab_size * model.dtype.itemsize for b, _ in self.shapes
        )
        free, _ = torch.cuda.mem_get_info(model.device)
        if estimate + (1 << 30) > free:
            raise ValueError(
                "Insufficient graph workspace; reduce graph grid or KV capacity before capture"
            )
        self.maps = torch.zeros(
            (max_b, pool.capacity), device=model.device, dtype=torch.int64
        )
        self.starts = torch.zeros(max_b, device=model.device, dtype=torch.int64)
        self.ends = torch.ones(max_b, device=model.device, dtype=torch.int64)
        self.graph_pool = torch.cuda.graph_pool_handle()
        self.replays = 0
        self.capture()
        after = torch.cuda.memory_stats(model.device)
        if any(
            after.get(k, 0) != memory_before.get(k, 0)
            for k in ("num_alloc_retries", "num_ooms")
        ):
            raise RuntimeError(
                "Allocator retry during graph capture invalidates this run"
            )

    @torch.inference_mode()
    def capture(self):
        m = self.model
        stream = torch.cuda.Stream(device=m.device)
        stream.wait_stream(torch.cuda.current_stream(m.device))
        with torch.cuda.stream(stream):
            for b, length in sorted(
                self.shapes, key=lambda shape: shape[0] * shape[1], reverse=True
            ):
                ids = torch.zeros((b, length), device=m.device, dtype=torch.int64)
                slots = torch.full((b, length), -1, device=m.device, dtype=torch.int64)
                last = torch.zeros(b, device=m.device, dtype=torch.int64)
                offsets = torch.arange(length, device=m.device)
                row = torch.arange(b, device=m.device)

                def forward():
                    positions = (
                        (self.starts[:b, None] + offsets[None, :])
                        .clamp(max=m.context_length - 1)
                        .reshape(-1)
                    )
                    x = m.embed(ids.reshape(-1))
                    for layer in range(m.num_layers):
                        q, k, v = (
                            t.contiguous() for t in m.prepare(layer, x, positions)
                        )
                        kp, vp = (
                            self.pool.native.get_key_buffer(layer),
                            self.pool.native.get_value_buffer(layer),
                        )
                        _store[
                            (b * length, triton.cdiv(m.num_kv_heads * m.head_dim, 256))
                        ](
                            k,
                            v,
                            kp,
                            vp,
                            slots,
                            b * length,
                            m.num_kv_heads,
                            m.head_dim,
                            kp.stride(0),
                            vp.stride(0),
                            256,
                        )
                        output = torch.empty_like(q)
                        _attention[(b, m.num_heads, triton.cdiv(length, 16))](
                            q,
                            kp,
                            vp,
                            self.maps,
                            self.starts,
                            self.ends,
                            output,
                            length,
                            m.num_heads,
                            m.num_kv_heads,
                            m.head_dim,
                            self.pool.capacity,
                            kp.stride(0),
                            vp.stride(0),
                            m.head_dim**-0.5,
                        )
                        x = m.complete(layer, x, output)
                    selected = x.reshape(b, length, -1)[row, last]
                    return m.logits(selected)

                # Compile and initialize kernels outside stream capture.
                for _ in range(2):
                    forward()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=self.graph_pool, stream=stream):
                    logits = forward()
                self.graphs[b, length] = (graph, ids, slots, last, logits, offsets, row)
        torch.cuda.current_stream(m.device).wait_stream(stream)
        torch.cuda.current_stream(m.device).synchronize()

    def shape(self, forwards):
        count = len(forwards)
        length = max((state.end - start for state, start in forwards), default=0)
        return min(
            (
                shape
                for shape in self.shapes
                if shape[0] >= count and shape[1] >= length
            ),
            key=lambda shape: (shape[0] * shape[1], shape),
            default=None,
        )

    @torch.inference_mode()
    def run(self, forwards, logits_mode):
        shape = self.shape(forwards)
        if shape is None:
            return None
        graph, ids, slots, last, logits, _, _ = self.graphs[shape]
        b, length = shape
        ids.zero_()
        slots.fill_(-1)
        self.starts[:b].zero_()
        self.ends[:b].zero_()
        last.zero_()
        for row, (state, start) in enumerate(forwards):
            n = state.end - start
            ids[row, :n].copy_(
                torch.tensor(
                    state.prompt.token_ids[start : state.end], device=self.model.device
                )
            )
            self.maps[row, : state.end].copy_(state.kv.indices)
            slots[row, :n].copy_(state.kv.indices[start : state.end])
            self.starts[row] = start
            self.ends[row] = state.end
            last[row] = n - 1
        graph.replay()
        self.replays += 1
        return {
            state.prompt.request_id: logits[i]
            for i, (state, _) in enumerate(forwards)
            if logits_mode == "all" or state.end == len(state.prompt.token_ids)
        }
