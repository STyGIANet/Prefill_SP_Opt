# Copyright 2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""Bounded-workspace causal attention over streamed, noncontiguous KV shards.

Absolute token positions define causality. Online softmax is accumulated in
FP32 (FP64 for FP64 inputs), including the all-masked block case. This path
uses PyTorch operations and supports GQA without a model-specific kernel.
"""

import torch


class AttentionAccumulator:
    def __init__(self, q: torch.Tensor, positions: torch.Tensor, scale: float):
        self.dtype = q.dtype
        self.q = q.transpose(0, 1).to(
            torch.float64 if q.dtype == torch.float64 else torch.float32
        )
        self.positions = positions
        self.scale = scale
        shape = self.q.shape[:-1]
        self.maximum = self.q.new_full(shape, -torch.inf)
        self.denominator = self.q.new_zeros(shape)
        self.numerator = torch.zeros_like(self.q)

    def add(self, k: torch.Tensor, v: torch.Tensor, positions: torch.Tensor):
        if not len(k) or not len(self.positions):
            return
        repeat = self.q.shape[0] // k.shape[1]
        k = k.transpose(0, 1).to(self.q.dtype).repeat_interleave(repeat, 0)
        v = v.transpose(0, 1).to(self.q.dtype).repeat_interleave(repeat, 0)
        scores = (self.q @ k.transpose(-1, -2)) * self.scale
        scores.masked_fill_(
            positions[None, None, :] > self.positions[None, :, None], -torch.inf
        )
        maximum = torch.maximum(self.maximum, scores.amax(-1))
        # All-masked blocks must contribute zero, never exp(-inf - -inf).
        safe_maximum = maximum.masked_fill(torch.isneginf(maximum), 0)
        alpha = torch.exp(self.maximum - safe_maximum)
        weights = torch.exp(scores - safe_maximum.unsqueeze(-1))
        self.numerator = self.numerator * alpha.unsqueeze(-1) + weights @ v
        self.denominator = self.denominator * alpha + weights.sum(-1)
        self.maximum = maximum

    def finish(self):
        if len(self.positions) and bool((self.denominator == 0).any()):
            raise RuntimeError("A query has no visible KV token")
        return (
            (self.numerator / self.denominator.unsqueeze(-1))
            .transpose(0, 1)
            .to(self.dtype)
        )
