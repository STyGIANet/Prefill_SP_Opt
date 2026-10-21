# Copyright 2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""Inference-only tiled causal attention for cyclic sequence shards.

Query i has absolute position first_query + i * query_stride. Gathered keys
are ordered by absolute position. The kernel fuses masking and online softmax,
uses FP32 accumulators, and never materializes the quadratic attention matrix.
"""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["nq", "nk", "first_query"])
def _strided_attention(
    Q,
    K,
    V,
    O,
    nq,
    nk,
    first_query,
    sq0: tl.constexpr,
    sq1: tl.constexpr,
    sq2: tl.constexpr,
    sk0: tl.constexpr,
    sk1: tl.constexpr,
    sk2: tl.constexpr,
    sv0: tl.constexpr,
    sv1: tl.constexpr,
    sv2: tl.constexpr,
    so0: tl.constexpr,
    so1: tl.constexpr,
    so2: tl.constexpr,
    QUERY_STRIDE: tl.constexpr,
    GQA: tl.constexpr,
    D: tl.constexpr,
    SCALE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BD: tl.constexpr,
):
    block = tl.program_id(0)
    head = tl.program_id(1)
    kv_head = head // GQA
    qi = block * BM + tl.arange(0, BM)
    di = tl.arange(0, BD)
    ki = tl.arange(0, BN)
    positions = first_query + qi * QUERY_STRIDE
    q = tl.load(
        Q + qi[:, None] * sq0 + head * sq1 + di[None, :] * sq2,
        mask=(qi[:, None] < nq) & (di[None, :] < D),
        other=0,
    )
    maximum = tl.full((BM,), -float("inf"), tl.float32)
    denominator = tl.zeros((BM,), tl.float32)
    numerator = tl.zeros((BM, BD), tl.float32)
    # The last valid query in this tile bounds the required key traversal.
    key_limit = tl.minimum(
        nk, first_query + (tl.minimum((block + 1) * BM, nq) - 1) * QUERY_STRIDE + 1
    )
    for start in range(0, key_limit, BN):
        keys = start + ki
        k = tl.load(
            K + keys[None, :] * sk0 + kv_head * sk1 + di[:, None] * sk2,
            mask=(keys[None, :] < nk) & (di[:, None] < D),
            other=0,
        )
        scores = tl.dot(q, k) * SCALE
        scores = tl.where(
            (keys[None, :] < nk) & (keys[None, :] <= positions[:, None]),
            scores,
            -float("inf"),
        )
        new_maximum = tl.maximum(maximum, tl.max(scores, axis=1))
        alpha = tl.exp2(maximum - new_maximum)
        weights = tl.exp2(scores - new_maximum[:, None])
        v = tl.load(
            V + keys[:, None] * sv0 + kv_head * sv1 + di[None, :] * sv2,
            mask=(keys[:, None] < nk) & (di[None, :] < D),
            other=0,
        )
        numerator = numerator * alpha[:, None] + tl.dot(weights.to(v.dtype), v)
        denominator = denominator * alpha + tl.sum(weights, axis=1)
        maximum = new_maximum
    output = numerator / denominator[:, None]
    tl.store(
        O + qi[:, None] * so0 + head * so1 + di[None, :] * so2,
        output,
        mask=(qi[:, None] < nq) & (di[None, :] < D),
    )


def strided_causal_attention(q, k, v, *, first_query: int, query_stride: int):
    """Return [queries, heads, dim] attention; k/v are [keys, KV heads, dim]."""
    if q.ndim != 3 or k.ndim != 3 or v.shape != k.shape:
        raise ValueError("Expected rank-three Q and matching K/V")
    if q.device.type != "cuda" or k.device != q.device or v.device != q.device:
        raise ValueError("Fused attention requires Q/K/V on the same CUDA device")
    if (
        q.dtype not in (torch.float16, torch.bfloat16)
        or k.dtype != q.dtype
        or v.dtype != q.dtype
    ):
        raise ValueError("Fused attention requires matching FP16 or BF16 inputs")
    nq, heads, dim = q.shape
    if (
        not heads
        or not k.shape[1]
        or heads % k.shape[1]
        or k.shape[2] != dim
        or not 0 < dim <= 256
    ):
        raise ValueError("Invalid head dimensions or GQA ratio")
    if (
        type(first_query) is not int
        or type(query_stride) is not int
        or first_query < 0
        or query_stride < 1
    ):
        raise ValueError("Invalid absolute query positions")
    if nq and first_query + (nq - 1) * query_stride >= len(k):
        raise ValueError("Query position exceeds the KV prefix")
    output = torch.empty_like(q)
    if nq:
        _strided_attention[(triton.cdiv(nq, 32), heads)](
            q,
            k,
            v,
            output,
            nq,
            len(k),
            first_query,
            *q.stride(),
            *k.stride(),
            *v.stride(),
            *output.stride(),
            query_stride,
            heads // k.shape[1],
            dim,
            dim**-0.5 * 1.4426950408889634,
            32,
            64 if dim <= 128 else 32,
            max(16, triton.next_power_of_2(dim)),
            num_warps=4,
            num_stages=2 if dim <= 128 else 1,
        )
    return output
