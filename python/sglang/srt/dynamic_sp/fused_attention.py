# Copyright 2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""Inference-only tiled causal attention for cyclic sequence shards.

Query i has absolute position first_query + i * query_stride. Optional device
slot maps translate logical key positions to physical pool/receive-buffer rows.
The kernel fuses indexed loads, masking and online softmax,
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
    Indices,
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
    INDEXED: tl.constexpr,
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
        rows = tl.load(Indices + keys, keys < nk, other=0) if INDEXED else keys
        k = tl.load(
            K + rows[None, :] * sk0 + kv_head * sk1 + di[:, None] * sk2,
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
            V + rows[:, None] * sv0 + kv_head * sv1 + di[None, :] * sv2,
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


def strided_causal_attention(
    q, k, v, *, first_query: int, query_stride: int, kv_indices=None
):
    """Return [queries, heads, dim]; optional indices map logical to physical KV.

    Indices must reference valid initialized rows. Ownership validation provides
    that invariant without a device-to-host bounds check in the attention path.
    """
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
    nk = len(k) if kv_indices is None else len(kv_indices)
    if kv_indices is not None and (
        kv_indices.device != q.device
        or kv_indices.dtype not in (torch.int32, torch.int64)
        or kv_indices.ndim != 1
        or not kv_indices.is_contiguous()
    ):
        raise ValueError("KV indices must be a contiguous device integer vector")
    if nq and first_query + (nq - 1) * query_stride >= nk:
        raise ValueError("Query position exceeds the KV prefix")
    output = torch.empty_like(q)
    if nq:
        _strided_attention[(triton.cdiv(nq, 32), heads)](
            q,
            k,
            v,
            kv_indices if kv_indices is not None else k,
            output,
            nq,
            nk,
            first_query,
            *q.stride(),
            *k.stride(),
            *v.stride(),
            *output.stride(),
            kv_indices is not None,
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


@triton.jit(do_not_specialize=["nq", "nk", "first_query", "first_key"])
def _ring_update(
    Q,
    K,
    V,
    Indices,
    M,
    Z,
    A,
    nq,
    nk,
    first_query,
    first_key,
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
    INDEXED: tl.constexpr,
    QUERY_STRIDE: tl.constexpr,
    KEY_STRIDE: tl.constexpr,
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
    heads = tl.num_programs(1)
    maximum = tl.load(M + qi * heads + head, qi < nq, other=-float("inf"))
    denominator = tl.load(Z + qi * heads + head, qi < nq, other=0)
    numerator = tl.load(
        A + qi[:, None] * so0 + head * so1 + di[None, :] * so2,
        (qi[:, None] < nq) & (di[None, :] < D),
        other=0,
    )
    key_limit = nk
    for start in range(0, key_limit, BN):
        keys = start + ki
        rows = tl.load(Indices + keys, keys < nk, other=0) if INDEXED else keys
        k = tl.load(
            K + rows[None, :] * sk0 + kv_head * sk1 + di[:, None] * sk2,
            mask=(keys[None, :] < nk) & (di[:, None] < D),
            other=0,
        )
        scores = tl.dot(q, k) * SCALE
        scores = tl.where(
            (keys[None, :] < nk)
            & (first_key + keys[None, :] * KEY_STRIDE <= positions[:, None]),
            scores,
            -float("inf"),
        )
        new_maximum = tl.maximum(maximum, tl.max(scores, axis=1))
        safe_maximum = tl.where(new_maximum == -float("inf"), 0.0, new_maximum)
        alpha = tl.exp2(maximum - safe_maximum)
        weights = tl.exp2(scores - safe_maximum[:, None])
        v = tl.load(
            V + rows[:, None] * sv0 + kv_head * sv1 + di[None, :] * sv2,
            mask=(keys[:, None] < nk) & (di[None, :] < D),
            other=0,
        )
        numerator = numerator * alpha[:, None] + tl.dot(weights.to(v.dtype), v)
        denominator = denominator * alpha + tl.sum(weights, axis=1)
        maximum = new_maximum
    tl.store(M + qi * heads + head, maximum, qi < nq)
    tl.store(Z + qi * heads + head, denominator, qi < nq)
    tl.store(
        A + qi[:, None] * so0 + head * so1 + di[None, :] * so2,
        numerator,
        (qi[:, None] < nq) & (di[None, :] < D),
    )


class RingAccumulator:
    """Flash-style online softmax across strided KV shards, without gathering.

    Statistics and the unnormalized numerator remain FP32 across ring hops.
    Both key and query absolute positions enforce causality, including shards
    that contain no visible keys. Scratch storage is linear in local queries.
    """

    def __init__(self, q, first_query, stride):
        self.q, self.first_query, self.stride = q, first_query, stride
        self.maximum = torch.full(
            q.shape[:2], -torch.inf, device=q.device, dtype=torch.float32
        )
        self.denominator = torch.zeros(
            q.shape[:2], device=q.device, dtype=torch.float32
        )
        self.numerator = torch.zeros(q.shape, device=q.device, dtype=torch.float32)

    def add(self, k, v, first_key, *, kv_indices=None):
        nk = len(k) if kv_indices is None else len(kv_indices)
        if kv_indices is not None and (
            kv_indices.device != self.q.device
            or kv_indices.dtype not in (torch.int32, torch.int64)
            or kv_indices.ndim != 1
            or not kv_indices.is_contiguous()
        ):
            raise ValueError("KV indices must be a contiguous device integer vector")
        if not len(self.q) or not nk:
            return
        nq, heads, dim = self.q.shape
        _ring_update[(triton.cdiv(nq, 32), heads)](
            self.q,
            k,
            v,
            kv_indices if kv_indices is not None else k,
            self.maximum,
            self.denominator,
            self.numerator,
            nq,
            nk,
            self.first_query,
            first_key,
            *self.q.stride(),
            *k.stride(),
            *v.stride(),
            *self.numerator.stride(),
            kv_indices is not None,
            self.stride,
            self.stride,
            heads // k.shape[1],
            dim,
            dim**-0.5 * 1.4426950408889634,
            32,
            64 if dim <= 128 else 32,
            max(16, triton.next_power_of_2(dim)),
            num_warps=4,
            num_stages=2 if dim <= 128 else 1,
        )

    def finish(self):
        return (self.numerator / self.denominator.unsqueeze(-1)).to(self.q.dtype)
