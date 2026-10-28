"""Small-model, first-turn prefill experiments on a single CUDA GPU.

This independent Transformers backend is deliberately narrower than SGLang:
one next-token prediction, no persistent KV cache, no decoding or PD transfer.
The official artifact workflow is provided separately in scripts/.
"""

from dataclasses import asdict
import gc
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import random
import time

from .model import GraphShape, LatencyModel, SimConfig
from .scheduler import POLICIES, Scheduler, batch_record
from .simulator import metrics


class QwenPrefill:
    def __init__(self, model_name: str, revision: str, device: int,
                 max_graphs: int = 18, graph_memory_mb: int = 2048,
                 require_graphs: bool = False):
        if max_graphs < 0 or graph_memory_mb <= 0:
            raise ValueError("max_graphs must be nonnegative; graph_memory_mb must be positive")
        try:
            import torch
            from transformers import AutoModelForCausalLM
        except ImportError as error:
            raise RuntimeError("GPU dependencies are missing. Run bash scripts/setup_single_gpu.sh on the GPU host.") from error
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Check nvidia-smi and install a Blackwell-compatible PyTorch CUDA build.")
        self.torch = torch
        torch.cuda.set_device(device)
        self.device = torch.device(f"cuda:{device}")
        self.dtype = torch.float32
        self.max_graphs = max_graphs
        self.graph_memory_mb = graph_memory_mb
        self.require_graphs = require_graphs
        self.graphs: dict[GraphShape, dict] = {}
        self.capture_log: list[dict] = []
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, revision=revision, torch_dtype=self.dtype,
            attn_implementation="sdpa", trust_remote_code=False).to(self.device).eval()
        if self.model.config.model_type != "qwen2":
            raise ValueError("The reference GPU backend supports Qwen2/Qwen2.5 causal models only.")
        self.pad_id = self.model.config.pad_token_id or self.model.config.eos_token_id or 0
        if isinstance(self.pad_id, list):
            self.pad_id = self.pad_id[0]
        self.vocab_size = self.model.config.vocab_size
        props = torch.cuda.get_device_properties(self.device)
        self.environment = {"python": platform.python_version(), "torch": torch.__version__,
                            "transformers": importlib.metadata.version("transformers"),
                            "cuda_runtime": torch.version.cuda, "gpu": props.name,
                            "compute_capability": list(torch.cuda.get_device_capability(self.device)),
                            "total_vram_bytes": props.total_memory,
                            "model": model_name, "requested_revision": revision,
                            "resolved_model_commit": getattr(self.model.config, "_commit_hash", None),
                            "dtype": "float32", "attention": "sdpa", "cached_history": False}

    def _inputs(self, rows: list[list[int]], shape: GraphShape):
        """Left-pad so every real lane's last position is its prediction token.

        Explicit 4D masks avoid data-dependent mask construction during capture.
        Dummy lanes have one valid pad token; padded queries attend themselves
        to avoid all-masked attention rows and NaNs.
        """
        t = self.torch
        if not rows or len(rows) > shape.depth or max(map(len, rows)) > shape.length:
            raise ValueError("input rows do not fit shape")
        ids = t.full((shape.depth, shape.length), self.pad_id, dtype=t.long)
        valid = t.zeros_like(ids, dtype=t.bool)
        for i, row in enumerate(rows):
            if not row or any(v < 0 or v >= self.vocab_size for v in row):
                raise ValueError("input tokens must be nonempty and inside the model vocabulary")
            ids[i, -len(row):] = t.tensor(row, dtype=t.long)
            valid[i, -len(row):] = True
        valid[len(rows):, -1] = True
        positions = (valid.long().cumsum(-1) - 1).clamp_min(0)
        causal = t.ones((shape.length, shape.length), dtype=t.bool).tril()
        allowed = causal[None, :, :] & valid[:, None, :]
        allowed |= (~valid[:, :, None]) & t.eye(shape.length, dtype=t.bool)[None, :, :]
        mask = t.zeros((shape.depth, 1, shape.length, shape.length), dtype=self.dtype)
        mask.masked_fill_(~allowed[:, None, :, :], t.finfo(self.dtype).min)
        return {"input_ids": ids, "position_ids": positions, "attention_mask": mask}

    def _forward(self, buffers):
        return self.model(**buffers, use_cache=False, logits_to_keep=1,
                          return_dict=True).logits[:, -1, :]

    def capture(self, shape: GraphShape) -> bool:
        t = self.torch
        baseline = t.cuda.memory_allocated(self.device)
        started = time.perf_counter()
        buffers = graph = output = reference = None
        try:
            rows = [[1 + (i + j) % (self.vocab_size - 1) for j in range(shape.length)]
                    for i in range(shape.depth)]
            buffers = {k: v.to(self.device) for k, v in self._inputs(rows, shape).items()}
            with t.inference_mode():
                stream = t.cuda.Stream(device=self.device)
                stream.wait_stream(t.cuda.current_stream(self.device))
                with t.cuda.stream(stream):
                    for _ in range(3):
                        reference = self._forward(buffers)
                t.cuda.current_stream(self.device).wait_stream(stream)
                t.cuda.synchronize(self.device)
                graph = t.cuda.CUDAGraph()
                with t.cuda.graph(graph):
                    output = self._forward(buffers)
                graph.replay()
                t.cuda.synchronize(self.device)
                t.testing.assert_close(output, reference, rtol=0.02, atol=0.02)
                # Replay must consume new data at the SAME input addresses.
                rows = [row[-max(1, shape.length // 2):] for row in rows]
                rows[0] = [(v + 17) % self.vocab_size for v in rows[0]]
                for key, value in self._inputs(rows, shape).items():
                    buffers[key].copy_(value)
                reference = self._forward(buffers)
                graph.replay()
                t.cuda.synchronize(self.device)
                t.testing.assert_close(output, reference, rtol=0.02, atol=0.02)
                # Also verify that left-padding preserves unpadded predictions.
                single_shape = GraphShape(len(rows[0]), 1)
                single = {k: v.to(self.device) for k, v in self._inputs([rows[0]], single_shape).items()}
                unpadded = self._forward(single)
                t.testing.assert_close(output[0], unpadded[0], rtol=0.02, atol=0.02)
            self.graphs[shape] = {"graph": graph, "buffers": buffers, "output": output}
            self.capture_log.append({"length": shape.length, "depth": shape.depth,
                                     "status": "captured_and_checked",
                                     "seconds": time.perf_counter() - started,
                                     "allocated_delta_bytes": max(0, t.cuda.memory_allocated(self.device) - baseline)})
            return True
        except (RuntimeError, AssertionError) as error:
            import   traceback; traceback.print_exc()
            self.capture_log.append({"length": shape.length, "depth": shape.depth,
                                     "status": "eager_fallback", "reason": str(error)})
            # Keep allocation references out of the next capture attempt.
            buffers = graph = output = reference = None
            gc.collect()
            t.cuda.empty_cache()
            if self.require_graphs:
                raise RuntimeError(f"CUDA Graph capture/check failed for {shape}: {error}") from error
            return False

    def prepare(self, shapes: list[GraphShape]) -> None:
        t = self.torch
        baseline = t.cuda.memory_allocated(self.device)
        limit = self.graph_memory_mb * 1024 * 1024
        # Prefer coverage at the largest batching depth, then cover smaller ones.
        for shape in sorted(shapes, key=lambda s: (-s.depth, s.length)):
            free, _ = t.cuda.mem_get_info(self.device)
            if (len(self.graphs) >= self.max_graphs or free < 1024 ** 3
                    or t.cuda.memory_allocated(self.device) - baseline >= limit):
                self.capture_log.append({"status": "capture_budget_reached"})
                break
            if not self.capture(shape):
                # Some capture failures invalidate a CUDA context; do not retry
                # an entire grid after the first error.
                break
        if self.require_graphs and not self.graphs:
            raise RuntimeError("No CUDA Graphs were captured within the configured limits")

    def execute(self, rows: list[list[int]], shape: GraphShape | None = None) -> list[int]:
        t = self.torch
        with t.inference_mode():
            if shape is not None:
                entry = self.graphs[shape]
                for key, value in self._inputs(rows, shape).items():
                    entry["buffers"][key].copy_(value)
                entry["graph"].replay()
                output = entry["output"][:len(rows)]
            else:
                eager_shape = GraphShape(max(map(len, rows)), len(rows))
                inputs = {k: v.to(self.device) for k, v in self._inputs(rows, eager_shape).items()}
                output = self._forward(inputs)
            next_tokens = output.argmax(-1).cpu().tolist()
            t.cuda.synchronize(self.device)
            return next_tokens


def gpu_config(path=None) -> SimConfig:
    from .cli import load_config
    config = (load_config(path, SimConfig) if path else
              SimConfig(workers=1, max_batch=4, kv_token_budget=4096,
                        long_chunk=4096, graph_depths=(1, 2, 4), eager_padding=True))
    if config.workers != 1 or not config.eager_padding:
        raise ValueError("reference GPU mode requires workers=1 and eager_padding=true")
    if config.long_chunk < config.kv_token_budget:
        raise ValueError("reference GPU mode has no persistent KV cache; long_chunk must cover the token budget")
    return config


def replay(workload, config, engine: QwenPrefill, policy: str, seed: int) -> dict:
    scheduler = Scheduler(config, LatencyModel(), policy)
    # CUDA Graph shapes that failed capture are removed from admission decisions.
    scheduler.catalog.shapes = [s for s in scheduler.catalog.shapes if s in engine.graphs]
    ordered = sorted(workload, key=lambda r: (r.arrival, r.id))
    rng = random.Random(seed)
    token_rows = {r.id: [rng.randrange(1, engine.vocab_size) for _ in range(r.tokens)] for r in ordered}
    batches, index = [], 0
    engine.torch.cuda.synchronize(engine.device)
    origin = time.perf_counter()
    while len(scheduler.completed) < len(ordered):
        now = time.perf_counter() - origin
        while index < len(ordered) and ordered[index].arrival <= now:
            scheduler.submit(ordered[index])
            index += 1
        batch, wake = scheduler.next_batch(0, now)
        if batch:
            rows = [token_rows[w.request.id] for w in batch.work]
            predictions = engine.execute(rows, batch.shape)
            finish = time.perf_counter() - origin
            record = batch_record(batch, finish)
            record["next_token_ids"] = predictions
            batches.append(record)
            scheduler.finish(batch, finish)
        else:
            next_arrival = ordered[index].arrival if index < len(ordered) else math.inf
            next_event = min(wake, next_arrival)
            if not math.isfinite(next_event):
                raise RuntimeError("GPU scheduler stalled")
            time.sleep(max(0.0, min(next_event - now, 0.01)))
    return {"schema_version": 1, "execution": "gpu_reference_first_turn", "policy": policy,
            "config": asdict(config), "environment": engine.environment,
            "capture_log": engine.capture_log,
            "metrics": metrics(scheduler.completed, batches, 1),
            "requests": sorted(scheduler.completed, key=lambda r: r["id"]),
            "batches": batches, "migrations": []}


def run_gpu(args):
    from .cli import get_workload, write_json
    from .workload import save_trace
    config = gpu_config(args.config)
    workload = get_workload(args, gpu=True)
    if not workload:
        raise ValueError("GPU benchmark requires at least one request")
    if any(r.history != 0 for r in workload):
        raise ValueError("reference GPU runner supports history=0 only; use the simulator or official SGLang artifact for cached re-prefill")
    if len({r.id for r in workload}) != len(workload):
        raise ValueError("duplicate request ids")
    if any(r.tokens > config.kv_token_budget for r in workload):
        raise ValueError("a request exceeds the GPU token budget")
    policies = POLICIES if args.policy == "all" else (args.policy,)
    plan = {"model": args.model, "revision": args.revision, "policies": policies,
            "requests": len(workload), "config": asdict(config),
            "max_graphs": args.max_graphs, "graph_memory_mb": args.graph_memory_mb,
            "output": args.output,
            "scope": "single-GPU first-turn prefill; capture/startup excluded from replay latency"}
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return None
    engine = QwenPrefill(args.model, args.revision, args.device, args.max_graphs,
                        args.graph_memory_mb, args.require_graphs)
    print(f"Loaded {engine.environment['gpu']}: {args.model}", flush=True)
    shapes = Scheduler(config, LatencyModel(), "full").catalog.shapes
    if any(policy in ("graphs", "full") for policy in policies):
        engine.prepare(shapes)
    # Warm eager execution separately; model download/capture costs are recorded
    # outside the timed replay, as they are not request service latency.
    engine.execute([[1] * min(config.kv_token_budget, max(r.tokens for r in workload))])
    output = Path(args.output)
    write_json(output / "plan.json", plan)
    write_json(output / "environment.json", engine.environment)
    write_json(output / "capture.json", engine.capture_log)
    save_trace(workload, output / "trace.jsonl")
    reports = []
    for policy in policies:
        print(f"Replaying {policy}...", flush=True)
        reports.append(replay(workload, config, engine, policy, args.seed))
    return reports
