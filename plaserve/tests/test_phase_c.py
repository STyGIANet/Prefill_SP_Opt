import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def controller(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    return importlib.import_module('run_phase_c')


def test_foreign_user_on_allowed_gpu(controller, tmp_path):
    sample = dict(utc='now', gpus=dict(status=0, stdout='index, uuid, memory.used, utilization.gpu\n0, uuid0, 10 MiB, 20 %\n2, uuid2, 100 MiB, 99 %\n'),
        processes=dict(status=0, stdout='gpu_uuid, pid, process_name, used_memory\nuuid0, 10, python, 10 MiB\nuuid2, 20, python, 100 MiB\n'),
        owners=dict(stdout='PID USER COMMAND\n10 someone python\n20 colleague python\n'))
    p = tmp_path / 'samples.jsonl'
    p.write_text(json.dumps(sample) + '\n')
    result = controller.interference(p, 'apal')
    assert result['foreign_users'][0]['user'] == 'colleague'
    assert result['foreign_users'][0]['gpu'] == 2
    assert len(result['foreign_users']) == 1
    assert result['other_gpu_activity'][0]['gpu'] == 0


def test_ratio_uses_paired_repeats_and_sample_std(controller, tmp_path):
    roots = []
    for i, ratio in enumerate((1, 2, 3)):
        root = tmp_path / str(i)
        root.mkdir()
        records = [dict(policy=p, concurrency=cc, request_throughput_req_s=10 * (ratio if p == 'laps' else 1),
                        mean_ttft_ms=2, p90_ttft_ms=3) for p in controller.POLICIES for cc in controller.CCS]
        (root / 'validation.json').write_text(json.dumps(dict(status='passed', results=records)))
        roots.append(root)
    controller.aggregate(roots, tmp_path)
    report = json.loads((tmp_path / 'aggregate.json').read_text())
    assert report['laps_vanilla_paired_ratios']['4']['mean'] == 2
    assert report['laps_vanilla_paired_ratios']['4']['std'] == 1
