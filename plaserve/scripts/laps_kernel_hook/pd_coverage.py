"""Opt-in replay observation for PD correctness; no extra model executions."""
import functools
import json
import os
from pathlib import Path


def record(event):
    directory = Path(os.environ['LAPS_PD_COVERAGE_DIR'])
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f'{os.getpid()}.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(event, pid=os.getpid())) + '\n')


def instrument(module):
    cls = module.PiecewiseCudaGraphRunner
    original = cls.replay_batch_prefill

    @functools.wraps(original)
    def replay(self, forward_batch, **kwargs):
        result = original(self, forward_batch, **kwargs)
        extends = [int(x) for x in forward_batch.extend_seq_lens_cpu]
        bs, width = self._find_batch_prefill_graph(forward_batch.batch_size, max(extends))
        label = Path(os.environ['LAPS_DIAGNOSTIC_CASE_FILE'])
        record(dict(event='replay', tp_rank=module.get_tensor_model_parallel_rank(),
                    label=json.loads(label.read_text()) if label.exists() else {},
                    actual_bs=forward_batch.batch_size, target_bs=bs, width=width,
                    extend=extends, kv=forward_batch.seq_lens.cpu().tolist()))
        return result

    cls.replay_batch_prefill = replay
    record(dict(event='installed'))
