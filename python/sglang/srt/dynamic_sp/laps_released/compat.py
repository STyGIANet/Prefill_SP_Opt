# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0.
"""API translations for the released LAPS runner on the current SGLang runtime."""

from contextlib import contextmanager, nullcontext
from dataclasses import fields
from types import SimpleNamespace

import torch

from sglang.srt.compilation.compile_phase import (
    enable_torch_compile_warmup as enable_piecewise_cuda_graph_compile,
    set_pcg_capture_stream,
)
from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch as NativeForwardBatch,
)
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    enable_tc_piecewise_cuda_graph as enable_piecewise_cuda_graph,
    set_tc_piecewise_forward_context,
)
from sglang.srt.runtime_context import get_parallel


def set_compiled(enabled):
    # The current compile trampoline dispatches on the enclosing piecewise
    # context. Every released set_compiled(True) scope already has that context.
    if not enabled:
        raise ValueError("LAPS never disables compilation inside a piecewise scope")
    return nullcontext()


def get_tensor_model_parallel_rank():
    return get_parallel().tp_group.rank_in_group


def get_attention_tp_rank():
    return get_parallel().attn_tp_rank


def get_attention_tp_size():
    return get_parallel().attn_tp_size


_FIELDS = {f.name for f in fields(NativeForwardBatch)}
_LEGACY = {
    "req_to_token_pool",
    "token_to_kv_pool",
    "attn_backend",
    "out_cache_loc_swa",
    "padded_static_len",
    "temp_scaled_logprobs",
    "temperature",
    "top_p_normalized_logprobs",
    "top_p",
}


def ForwardBatch(**kwargs):
    unknown = kwargs.keys() - _FIELDS - _LEGACY
    if unknown:
        raise TypeError(f"Unmapped released ForwardBatch fields: {sorted(unknown)}")
    batch = NativeForwardBatch(**{k: v for k, v in kwargs.items() if k in _FIELDS})
    for name in _LEGACY:
        setattr(
            batch, name, kwargs.get(name, -1 if name == "padded_static_len" else None)
        )
    backend = batch.attn_backend
    native = getattr(backend, "native", backend)
    native.kv_index_translator.bind_runner_slots(batch)
    batch.attn_tp_sequence_sharded = False
    return batch


@contextmanager
def set_forward_context(batch, attention_layers, quant_config, moe_layers, moe_fusions):
    backend = getattr(batch.attn_backend, "native", batch.attn_backend)
    with forward_context(ForwardContext(backend), forward_batch=batch):
        with set_tc_piecewise_forward_context(
            batch, attention_layers, quant_config, moe_layers, moe_fusions
        ):
            yield


class AttentionAdapter:
    """Expose released rectangular FlashInfer metadata entry points."""

    def __init__(self, native):
        self.native = native
        self.wrappers = {}

    def __getattr__(self, name):
        return getattr(self.native, name)

    def init_batch_prefill_cuda_graph_state(self, max_bs, max_seq_len):
        if self.native.num_wrappers != 1 or self.native.dispatch_reason is not None:
            raise ValueError("Released LAPS graphs require full-attention FlashInfer")
        # The native allocator supplies stable FlashInfer workspace and page
        # indices. The released graph retains its rectangular query layout.
        self.max_bs, self.max_seq_len = max_bs, max_seq_len
        self.native._create_full_cg_prefill_wrappers(max_bs, max_bs * max_seq_len)

    def _plan(self, bs, seq_len, req_pool_indices, seq_lens, seq_lens_cpu=None):
        from sglang.srt.layers.attention.flashinfer_backend import PrefillMetadata

        native = self.native
        key = bs, seq_len
        if seq_lens_cpu is None:
            seq_lens_cpu = seq_lens.cpu()
        plan = native.kv_index_translator.plan(
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            write_virtual=None,
        )
        native.indices_updater_prefill.update(
            req_pool_indices,
            seq_lens,
            seq_lens_cpu,
            int(seq_lens_cpu.sum()),
            prefix_lens=seq_lens - seq_len,
            prefill_wrappers=self.wrappers[key],
            use_ragged=False,
            encoder_lens=None,
            spec_info=None,
            plan=plan,
        )
        native.forward_metadata = PrefillMetadata(self.wrappers[key], False, False)

    def init_forward_metadata_capture_batch_prefill_cuda_graph(
        self, bs, seq_len, req_pool_indices, seq_lens
    ):
        key = bs, seq_len
        if key in self.wrappers:
            raise RuntimeError(f"Duplicate LAPS capture {key}")
        from flashinfer import BatchPrefillWithPagedKVCacheWrapper

        native = self.native
        self.wrappers[key] = [
            BatchPrefillWithPagedKVCacheWrapper(
                native.full_cg_prefill_workspace_buffer,
                "NHD",
                use_cuda_graph=True,
                backend=native.prefill_backend,
                qo_indptr_buf=native.full_cg_prefill_qo_indptr[0][: bs + 1],
                paged_kv_indptr_buf=native.full_cg_prefill_kv_indptr[0][: bs + 1],
                paged_kv_indices_buf=native.full_cg_prefill_kv_indices[0],
                paged_kv_last_page_len_buf=native.kv_last_page_len[:bs],
            )
        ]
        self._plan(bs, seq_len, req_pool_indices, seq_lens)

    def init_forward_metadata_replay_batch_prefill_cuda_graph(
        self, bs, seq_len, req_pool_indices, seq_lens, seq_lens_cpu=None
    ):
        self._plan(bs, seq_len, req_pool_indices, seq_lens, seq_lens_cpu)


class ModelRunnerAdapter:
    def __init__(self, runner, config):
        self.native = runner
        self.attn_backend = AttentionAdapter(runner.attn_backend)
        parallel = get_parallel()
        self.tp_group = parallel.tp_group
        self.server_args = SimpleNamespace(
            tp_size=parallel.tp_size,
            dp_size=parallel.dp_size,
            pp_size=parallel.pp_size,
            piecewise_cuda_graph_tokens=config["graph_token_buckets"],
            piecewise_cuda_graph_compiler=config["piecewise_compiler"],
            enable_torch_compile_debug_mode=False,
            enable_return_hidden_states=False,
            enable_lora=False,
            enable_mamba_extra_buffer=lambda: False,
            disable_radix_cache=True,
            enable_cudagraph_gc=False,
            enable_batch_prefill_cuda_graph=True,
            batch_prefill_batch_sizes=config["batch_prefill_batch_sizes"],
            batch_prefill_seq_lengths=config["batch_prefill_seq_lengths"],
        )

    def __getattr__(self, name):
        return getattr(self.native, name)


def install_compile_records():
    # Dynamo cannot construct msgspec.Struct during full-graph tracing. These
    # are worker-local residual bookkeeping records, not serialized messages.
    # Retain their fields and methods with a traceable Python constructor.
    from dataclasses import make_dataclass
    import sys
    from sglang.srt.layers.layer_boundary.residual import stream
    from sglang.srt.layers.layer_boundary import exit as exits

    for module, name in (
        (stream, "Contribution"),
        (stream, "DeclaredSum"),
        (stream, "OwedOutput"),
        (exits, "ExitDecision"),
    ):
        original = getattr(module, name)
        if not hasattr(original, "__struct_fields__"):
            continue
        names = original.__struct_fields__
        defaults = original.__struct_defaults__
        required = len(names) - len(defaults)
        specs = [
            (n, original.__annotations__[n])
            if i < required
            else (n, original.__annotations__[n], defaults[i - required])
            for i, n in enumerate(names)
        ]
        methods = {
            n: v
            for n, v in vars(original).items()
            if callable(v) and not n.startswith("__")
        }
        replacement = make_dataclass(
            name, specs, namespace=methods, frozen=name != "Contribution"
        )
        replacement.__module__ = __name__
        globals()[name] = replacement
        for loaded_name, loaded in tuple(sys.modules.items()):
            if (
                loaded_name.startswith("sglang.srt.layers.layer_boundary")
                and loaded is not None
            ):
                for attr, value in tuple(vars(loaded).items()):
                    if value is original:
                        setattr(loaded, attr, replacement)


class ReleasedLAPSGraphRunner:
    def __init__(self, model_runner):
        import json
        from sglang.srt.runtime_context import get_schedule
        from .graph import PiecewiseCudaGraphRunner
        from .flashinfer_compat import configure

        configure()
        config = json.loads(get_schedule().laps_graph_config)
        if (
            max(config["graph_token_buckets"] + config["batch_prefill_seq_lengths"])
            > model_runner.model_config.context_len
        ):
            raise ValueError("Released LAPS capture lengths exceed the model context")
        if (
            max(config["batch_prefill_batch_sizes"])
            > model_runner.req_to_token_pool.size
        ):
            raise ValueError("Released LAPS capture batches exceed the request pool")
        install_compile_records()
        if (
            model_runner.is_multimodal
            or model_runner.is_hybrid_swa
            or any(layer is not None for layer in model_runner.moe_layers)
        ):
            raise ValueError(
                "Released LAPS adapter requires a dense full-attention text model"
            )
        if (
            model_runner.lora_manager is not None
            or not model_runner.spec_algorithm.is_none()
        ):
            raise ValueError(
                "Released LAPS adapter requires ordinary prefill without LoRA or speculation"
            )
        from types import MethodType
        from .attention import attention_forward

        for layer in model_runner.attention_layers:
            if layer is not None:
                layer.forward = MethodType(attention_forward, layer)
        self.model_runner = model_runner
        self.adapter = ModelRunnerAdapter(model_runner, config)
        self.runner = PiecewiseCudaGraphRunner(self.adapter)

    def can_run_graph(self, batch):
        return self.runner.can_run_batch_prefill(batch) or self.runner.can_run(batch)

    def execute(self, batch, **kwargs):
        # These removed fields are inert on ordinary full-attention prefill;
        # the released replay_prepare still reads and forwards them.
        for name in _LEGACY:
            if not hasattr(batch, name):
                setattr(batch, name, -1 if name == "padded_static_len" else None)
        batch.attn_backend = self.adapter.attn_backend
        if self.runner.can_run_batch_prefill(batch):
            return self.runner.replay_batch_prefill(batch, **kwargs)
        return self.runner.replay(batch, **kwargs)
