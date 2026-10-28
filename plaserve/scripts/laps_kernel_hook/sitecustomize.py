"""Opt-in FlashInfer overrides and replay diagnostics for SGLang workers.

Activated by LAPS_FLASHINFER_KERNEL_ROOT or LAPS_REPLAY_CONTROL_DIR. No CUDA imports at interpreter
startup: hooks run when the targeted FlashInfer/SGLang modules are imported.
"""
import hashlib
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import sys


def instrument_logits(module):
    """Observe model outputs after execution, never inside graph capture."""
    import functools
    import inspect
    original = module.ModelRunner.forward
    signature = inspect.signature(original)

    @functools.wraps(original)
    def forward(self, *args, **kwargs):
        import torch
        bound = signature.bind(self, *args, **kwargs)
        batch = bound.arguments.get("forward_batch")
        if batch is None:
            batch = next((x for x in args if hasattr(x, "forward_mode")), None)
        if batch is None:
            raise RuntimeError("Cannot identify ForwardBatch for logits diagnostics")
        active = (batch.forward_mode.is_extend() or batch.forward_mode.is_decode()) and not torch.cuda.is_current_stream_capturing()
        metadata = None
        if active:
            label_path = Path(os.environ["LAPS_DIAGNOSTIC_CASE_FILE"])
            label = json.loads(label_path.read_text()) if label_path.exists() else {"phase": "startup"}
            if label.get("phase") == "generate":
                metadata = {"label": label,
                            "input_ids": batch.input_ids.detach().cpu().tolist(),
                            "seq_lens": batch.seq_lens.detach().cpu().tolist(),
                            "kind": "extend" if batch.forward_mode.is_extend() else "decode",
                            "req_pool_indices": batch.req_pool_indices.detach().cpu().tolist(),
                            "extend_lens": ([int(x) for x in batch.extend_seq_lens_cpu]
                                            if batch.forward_mode.is_extend() else [1] * batch.batch_size)}
        result = original(self, *args, **kwargs)
        if metadata is not None:
            def find_logits(value):
                logits = getattr(value, "next_token_logits", None)
                if logits is not None:
                    return logits
                child = getattr(value, "logits_output", None)
                if child is not None:
                    return find_logits(child)
                if isinstance(value, (tuple, list)):
                    for child in value:
                        logits = find_logits(child)
                        if logits is not None:
                            return logits
                return None
            logits = find_logits(result)
            if logits is None:
                raise RuntimeError(f"Cannot locate logits in ModelRunner result {type(result)}")
            values = logits.detach().float().cpu()
            if values.ndim != 2 or values.shape[0] != len(metadata["extend_lens"]):
                raise RuntimeError(f"Unexpected logits shape for diagnostics: {tuple(values.shape)}")
            metadata["tp_rank"] = getattr(self, "tp_rank", None)
            metadata["logits"] = values
            directory = Path(os.environ["LAPS_LOGITS_DIR"])
            directory.mkdir(parents=True, exist_ok=True)
            index = getattr(self, "_laps_logits_index", 0)
            self._laps_logits_index = index + 1
            torch.save(metadata, directory / f"{os.getpid()}-{index:05d}.pt")
        return result

    module.ModelRunner.forward = forward


def record(event):
    directory = Path(os.environ["LAPS_KERNEL_EVENT_DIR"])
    directory.mkdir(parents=True, exist_ok=True)
    event["pid"] = os.getpid()
    with (directory / f"{os.getpid()}.jsonl").open("a") as stream:
        stream.write(json.dumps(event) + "\n")


def configure(module, root):
    manifest = json.loads((root / "all-manifest.json").read_text())
    if manifest["status"] != "passed" or manifest["mode"] != "saturating":
        raise RuntimeError("Kernel override requires a passed saturating attention suite")
    relative = Path("flashinfer/attention/prefill.cuh")
    installed = Path(module.FLASHINFER_INCLUDE_DIR) / relative
    selected = root / "include" / relative
    for path, key in [(installed, "original_sha256"), (selected, "selected_sha256")]:
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest[key]:
            raise RuntimeError(f"Kernel header hash changed: {path}")
    if module.flashinfer_version != manifest["flashinfer"]:
        raise RuntimeError("FlashInfer version differs from attention validation")
    module.FLASHINFER_INCLUDE_DIR = root / "include"
    module.FLASHINFER_WORKSPACE_DIR = root
    module.FLASHINFER_JIT_DIR = root / "cached_ops"
    module.FLASHINFER_GEN_SRC_DIR = root / "generated"
    module.FLASHINFER_AOT_DIR = root / "empty_aot"
    record({"event": "configured", "root": str(root),
            "header_sha256": manifest["selected_sha256"]})


def instrument_load(module, root):
    original_load = module.JitSpec.load

    def load(spec, so_path):
        relevant = spec.name.startswith("batch_prefill")
        if relevant:
            path = Path(so_path).resolve()
            if not path.is_relative_to(root / "cached_ops"):
                raise RuntimeError(f"Attention kernel escaped isolated cache: {path}")
            ninja = path.parent / "build.ninja"
            if not ninja.is_file() or str(root / "include") not in ninja.read_text():
                raise RuntimeError(f"Cannot verify corrected attention build: {path}")
        result = original_load(spec, so_path)
        if relevant:
            record({"event": "kernel_loaded", "name": spec.name, "library": str(path)})
        return result

    module.JitSpec.load = load


class KernelLoader(importlib.abc.Loader):
    def __init__(self, original, fullname, root):
        self.original, self.fullname, self.root = original, fullname, root

    def create_module(self, spec):
        return self.original.create_module(spec)

    def exec_module(self, module):
        try:
            self.original.exec_module(module)
        except Exception as error:
            if self.fullname.endswith(".piecewise_cuda_graph_runner"):
                import replay_control
                replay_control.record({"event": "error", "error": f"Cannot import replay entry point: {error}"})
            raise
        if self.fullname.endswith(".piecewise_cuda_graph_runner"):
            if os.environ.get("LAPS_REPLAY_CONTROL_DIR"):
                import replay_control
                replay_control.instrument(module)
            if os.environ.get("LAPS_PD_COVERAGE_DIR"):
                import pd_coverage
                pd_coverage.instrument(module)
        elif self.fullname.endswith(".env"):
            configure(module, self.root)
        elif self.fullname.endswith(".model_runner"):
            instrument_logits(module)
        else:
            instrument_load(module, self.root)


class KernelFinder(importlib.abc.MetaPathFinder):
    def __init__(self, root):
        self.root = root

    def find_spec(self, fullname, path=None, target=None):
        targets = ["flashinfer.jit.env", "flashinfer.jit.core"] if self.root else []
        if os.environ.get("LAPS_REPLAY_CONTROL_DIR") or os.environ.get("LAPS_PD_COVERAGE_DIR"):
            targets.append("sglang.srt.model_executor.piecewise_cuda_graph_runner")
        if os.environ.get("LAPS_LOGITS_DIR"):
            targets.append("sglang.srt.model_executor.model_runner")
        if fullname not in targets:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            if fullname.endswith(".piecewise_cuda_graph_runner"):
                import replay_control
                replay_control.record({"event": "error", "error": f"Cannot locate replay module {fullname}"})
            raise ImportError(f"Cannot install FlashInfer hook for {fullname}")
        spec.loader = KernelLoader(spec.loader, fullname, self.root)
        return spec


if os.environ.get("LAPS_FLASHINFER_KERNEL_ROOT") or os.environ.get("LAPS_REPLAY_CONTROL_DIR"):
    root = os.environ.get("LAPS_FLASHINFER_KERNEL_ROOT")
    sys.meta_path.insert(0, KernelFinder(Path(root).resolve() if root else None))
