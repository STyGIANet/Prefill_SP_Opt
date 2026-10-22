# Copyright 2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""Native SGLang dense Llama adapter for dynamically owned token rows."""

import torch.nn.functional as F


class SGLangLlamaPrefillModel:
    def __init__(self, model):
        from sglang.srt.models.llama import LlamaForCausalLM
        from sglang.srt.runtime_context import get_parallel

        if type(model) is not LlamaForCausalLM:
            raise ValueError(
                "Dynamic SP currently supports native LlamaForCausalLM only"
            )
        parallel = get_parallel()
        if parallel.tp_size != 1 or parallel.pp_size != 1:
            raise ValueError("Dynamic SP requires replicated TP1/PP1 weights")
        if model.quant_config is not None:
            raise ValueError("Dynamic SP does not support quantized weights")
        cfg = model.config
        if getattr(cfg, "sliding_window", None) is not None:
            raise ValueError("Sliding-window attention is not supported")
        if getattr(cfg, "attention_chunk_size", None) is not None:
            raise ValueError("Chunk-local attention is not supported")
        self.model = model.eval()
        self.device = model.model.embed_tokens.weight.device
        self.dtype = model.model.embed_tokens.weight.dtype
        self.num_layers = cfg.num_hidden_layers
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = model.model.layers[0].self_attn.head_dim
        self.vocab_size = cfg.vocab_size
        self.context_length = cfg.max_position_embeddings

    def embed(self, ids):
        return self.model.model.embed_tokens(ids)

    def prepare(self, layer, x, positions):
        layer = self.model.model.layers[layer]
        # Bypass stage-boundary residual plumbing: ownership is per prompt,
        # and residual rows remain colocated with their token across layers.
        normalized = layer.input_layernorm(x)
        q, k, v = layer.self_attn.forward_prepare_native(positions, normalized)
        return (
            q.reshape(-1, self.num_heads, self.head_dim),
            k.reshape(-1, self.num_kv_heads, self.head_dim),
            v.reshape(-1, self.num_kv_heads, self.head_dim),
        )

    def complete(self, layer, residual, attention):
        layer = self.model.model.layers[layer]
        projected, _ = layer.self_attn.o_proj(attention.flatten(1))
        normalized, residual = layer.post_attention_layernorm(projected, residual)
        return residual + layer.mlp(normalized)

    def logits(self, x):
        x = self.model.model.norm(x)
        # The vocabulary can be padded in native SGLang embedding/LM heads.
        return F.linear(x, self.model.lm_head.weight[: self.vocab_size])


def load_native_llama(model_path, *, dtype="bfloat16", device="cuda", local_rank=0):
    """Load native SGLang replicas into an already initialized torchrun world."""
    import torch.distributed as dist
    from sglang.srt.configs.device_config import DeviceConfig
    from sglang.srt.configs.load_config import LoadConfig
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from sglang.srt.model_loader import get_model
    from sglang.srt.runtime_context import SpawnRanks, publish
    from sglang.srt.server_args import ServerArgs

    args = ServerArgs(
        model_path=model_path,
        dtype=dtype,
        device=device,
        tp_size=1,
        pp_size=1,
        disable_cuda_graph=True,
        disable_radix_cache=True,
        disable_custom_all_reduce=True,
        attention_backend="torch_native",
    )
    # Logical model rank is zero on every replica; transport uses WORLD rank.
    publish(args, role="scheduler", ranks=SpawnRanks(world_rank=0, gpu_id=local_rank))
    config = ModelConfig.from_server_args(args)
    if config.hf_config.architectures != ["LlamaForCausalLM"]:
        raise ValueError("Use a dense LlamaForCausalLM checkpoint for dynamic SP")
    init_distributed_environment(
        world_size=dist.get_world_size(),
        rank=dist.get_rank(),
        local_rank=local_rank,
        backend=dist.get_backend(),
    )
    initialize_model_parallel(replicated_model_count=dist.get_world_size())
    model = get_model(
        model_config=config,
        load_config=LoadConfig(),
        device_config=DeviceConfig(device, gpu_id=local_rank),
    )
    return SGLangLlamaPrefillModel(model)
