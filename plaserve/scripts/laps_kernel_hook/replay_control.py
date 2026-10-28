"""Same-shape batch-prefill diagnostic; imported only when explicitly enabled.

Eager runs first. FlashInfer writes current-layer K/V before reading attention
(flashinfer_backend.py:848-859), so replay replaces every eager-written real slot
before consuming it, even when eager and replay differ numerically. Prefix slots
must be disjoint from all writes; slot zero is padding scratch and must not be
read by real rows. validate_slots checks these conditions against the plan.
This is restricted to the adapter's full-attention, page-size-one path.
"""
import functools
import inspect
import json
import os
from pathlib import Path


def record(event):
    directory = Path(os.environ["LAPS_REPLAY_CONTROL_DIR"])
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"{os.getpid()}.jsonl").open("a") as stream:
        stream.write(json.dumps(dict(event, pid=os.getpid()), allow_nan=False) + "\n")


def validate_slots(locations, pages, lengths, extends, width):
    """Prove writes touch only new suffix slots and unread padding scratch."""
    if len(locations) != len(lengths) * width or len(pages) != sum(lengths):
        raise RuntimeError("Replay control: inconsistent padded cache dimensions")
    writes, prefixes = [], []
    offset = 0
    for row, length in enumerate(lengths):
        table = pages[offset:offset + length]
        offset += length
        slot = locations[row * width:(row + 1) * width]
        if row >= len(extends):
            if any(slot):
                raise RuntimeError("Replay control: dummy row writes live cache")
            continue
        size = extends[row]
        if not 0 < size <= min(width, length) or any(x <= 0 for x in table):
            raise RuntimeError("Replay control: invalid real cache row")
        if any(slot[:width - size]) or slot[width - size:] != table[-size:]:
            raise RuntimeError("Replay control: writes differ from planned suffix")
        writes.extend(table[-size:])
        prefixes.extend(table[:-size])
    if len(writes) != len(set(writes)) or set(writes).intersection(prefixes):
        raise RuntimeError("Replay control: suffix writes alias live cache")


class ReplayProxy:
    def __init__(self, graph, callback):
        self.graph, self.callback, self.calls = graph, callback, 0

    def replay(self):
        self.calls += 1
        if self.calls != 1:
            raise RuntimeError("Replay control: graph replayed more than once")
        return self.callback(self.graph)


def compare(module, runner, batch, key, graph):
    import torch
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Replay control cannot run during capture")
    bs, width = key
    real = batch.batch_size
    fb = runner.batch_prefill_cuda_graph_fbs[key]
    backend = runner.model_runner.attn_backend
    if backend.batch_prefill_page_size != 1 or backend.dispatch_reason is not None:
        raise RuntimeError("Replay control requires full attention and page size one")
    lengths = fb.seq_lens.cpu().tolist()
    extends = [int(x) for x in batch.extend_seq_lens_cpu]
    validate_slots(fb.out_cache_loc.cpu().tolist(),
                   backend.batch_prefill_kv_indices[:sum(lengths)].cpu().tolist(),
                   lengths, extends, width)
    if fb.batch_size != bs or len(extends) != real:
        raise RuntimeError("Replay control: unexpected captured batch")
    # Mirror capture_one_batch_prefill.run_once, piecewise_cuda_graph_runner.py:940.
    # Reuse the actual captured FB, including its static CPU metadata. Do NOT
    # call ModelRunner.forward or init_forward_metadata: either changes the path.
    fb.dp_local_start_pos = fb.dp_local_num_tokens = None
    module.set_dp_buffer_len(None, bs * width, fb.dp_padding_mode.is_max_len())
    module.set_is_extend_in_batch(False)
    with torch.no_grad(), module.set_compiled(False), module.set_forward_context(
        fb, runner.attention_layers, runner.quant_config,
        runner.moe_layers, runner.moe_fusions,
    ):
        eager = runner.model_runner.model.forward(fb.input_ids, fb.positions, fb)
    # Copy before replay: graph output memory may be shared across captures.
    a = eager.next_token_logits[:real].detach().cpu().clone()
    graph.replay()
    b = runner.batch_prefill_outputs[key].next_token_logits[:real].detach().cpu().clone()
    if a.shape != b.shape or a.ndim != 2 or a.shape[0] != real or a.dtype != b.dtype:
        raise RuntimeError("Replay control: unexpected logits shape/dtype")
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise RuntimeError("Replay control: nonfinite real-row logits")
    delta = b.double() - a.double()
    ai, bi = a.argmax(dim=1), b.argmax(dim=1)
    label_path = Path(os.environ["LAPS_DIAGNOSTIC_CASE_FILE"])
    label = json.loads(label_path.read_text()) if label_path.exists() else {"phase": "startup"}
    index = getattr(runner, "_laps_replay_control_index", 0)
    runner._laps_replay_control_index = index + 1
    record({"event": "comparison", "index": index,
            "tp_rank": module.get_tensor_model_parallel_rank(), "label": label,
            "actual_bs": real, "target_bs": bs, "width": width,
            "extend_lens": extends, "seq_lens": lengths[:real],
            "input_ids": batch.input_ids.cpu().tolist(),
            "numel": a.numel(), "max_abs": delta.abs().max().item(),
            "rms": delta.square().mean().sqrt().item(),
            "bitwise_equal": torch.equal(a.contiguous().view(torch.uint8),
                                         b.contiguous().view(torch.uint8)),
            "argmax_agreement": int((ai == bi).sum().item()),
            "eager_argmax": ai.tolist(), "replay_argmax": bi.tolist(),
            "row_max_abs": delta.abs().amax(dim=1).tolist()})


def instrument(module):
    try:
        cls = module.PiecewiseCudaGraphRunner
        original = cls.replay_batch_prefill
        # Fail closed on an incompatible runner instead of installing a no-op.
        source = inspect.getsource(original)
        for name in ("batch_prefill_graphs", "init_forward_metadata_replay_batch_prefill_cuda_graph", ".replay()"):
            if name not in source:
                raise RuntimeError(f"Replay entry point missing {name}")
    except Exception as error:
        record({"event": "error", "error": f"Cannot locate batch-prefill replay entry point: {error}"})
        raise

    @functools.wraps(original)
    def replay(self, forward_batch, **kwargs):
        proxy = graph = key = None
        try:
            import torch
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Replay control invoked inside capture")
            key = self._find_batch_prefill_graph(
                forward_batch.batch_size, max(int(x) for x in forward_batch.extend_seq_lens_cpu))
            graph = self.batch_prefill_graphs[key]
            proxy = ReplayProxy(graph, lambda g: compare(module, self, forward_batch, key, g))
            # Original replay prepares buffers/plans, then calls this proxy at
            # piecewise_cuda_graph_runner.py:1058. No capture method is wrapped.
            self.batch_prefill_graphs[key] = proxy
            result = original(self, forward_batch, **kwargs)
            if proxy.calls != 1:
                raise RuntimeError("Replay entry point did not invoke control")
            return result
        except Exception as error:
            record({"event": "error", "error": str(error)})
            raise
        finally:
            if proxy is not None:
                self.batch_prefill_graphs[key] = graph

    cls.replay_batch_prefill = replay
    record({"event": "installed", "entry_point": f"{module.__name__}.PiecewiseCudaGraphRunner.replay_batch_prefill"})
