# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0.
from __future__ import annotations
from copy import copy
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
import torch
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.compilation.compilation_config import register_split_op
from sglang.srt.utils.custom_op import register_custom_op
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    get_tc_piecewise_forward_context as get_forward_context,
)


def attention_forward(
    self,
    q,
    k,
    v,
    forward_batch: ForwardBatch,
    save_kv_cache: bool = True,
    **kwargs,
):
    if k is not None:
        # For cross-layer sharing, kv can be None
        assert v is not None
        if "k_rope" not in kwargs:
            k = k.view(-1, self.tp_k_head_num, self.qk_head_dim)
            v = v.view(-1, self.tp_v_head_num, self.v_head_dim)
        else:
            k = k.view(-1, self.tp_k_head_num, self.v_head_dim)

    if forward_batch.forward_mode.is_extend() and get_forward_context() is not None:
        if self.qk_head_dim != self.v_head_dim:
            output = q.new_empty((q.shape[0], self.tp_q_head_num * self.v_head_dim))
        else:
            output = torch.empty_like(q)
        laps_unified_attention_with_output(
            q, k, v, output, save_kv_cache, self.layer_id, **kwargs
        )
        return output
    else:
        return get_attn_backend().forward(
            q,
            k,
            v,
            self,
            forward_batch,
            save_kv_cache,
            **kwargs,
        )


@register_custom_op(mutates_args=["output"])
@register_split_op()
def laps_unified_attention_with_output(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    save_kv_cache: bool,
    layer_id: int,
    *,
    q_rope: Optional[torch.Tensor] = None,
    k_rope: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
) -> None:
    context = get_forward_context()
    forward_batch = context.forward_batch
    attention_layers = context.attention_layers
    attention_layer = attention_layers[layer_id]

    kwargs = {}
    if q_rope is not None:
        kwargs["q_rope"] = q_rope
    if k_rope is not None:
        kwargs["k_rope"] = k_rope
    if sinks is not None:
        kwargs["sinks"] = sinks

    # The released piecewise path pads tokenwise operations, but its attention
    # plan covers real query rows only. FlashInfer 0.7 checks that row count.
    # Rectangular batch graphs explicitly plan all padded rows and bypass this.
    real_rows = forward_batch.extend_num_tokens
    padded = (
        getattr(forward_batch, "padded_static_len", -1) == -1
        and real_rows < query.shape[0]
    )
    if padded:
        forward_batch = copy(forward_batch)
        forward_batch.out_cache_loc = forward_batch.out_cache_loc[:real_rows]
        query = query[:real_rows]
        key = key[:real_rows] if key is not None else None
        value = value[:real_rows] if value is not None else None

    ret = get_attn_backend().forward(
        query, key, value, attention_layer, forward_batch, save_kv_cache, **kwargs
    )
    if padded:
        output[:real_rows].view(ret.shape).copy_(ret)
        # Padding rows never enter attention or returned logits.
        return
    assert output.numel() == ret.numel(), (
        f"Output tensor element mismatch: {output.numel()} != {ret.numel()}"
    )

    output.view(ret.shape).copy_(ret)
    return
