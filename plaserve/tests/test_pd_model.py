import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def pd(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    return importlib.import_module('check_pd_model')


def test_first_divergence_and_alignment(pd):
    a = [dict(case='cold-0', count=8, response=[dict(output_ids=[1, 2, 3])])]
    b = [dict(case='cold-0', count=8, response=[dict(output_ids=[1, 4, 5])])]
    assert pd.compare(a, a) == []
    report = pd.compare(a, b)
    assert report[0]['first_differing_token'] == 1
    assert report[0]['baseline'] == [1, 2, 3] and report[0]['pd'] == [1, 4, 5]
    b[0]['case'] = 'cold-1'
    with pytest.raises(RuntimeError, match='Misaligned'):
        pd.compare(a, b)


def test_occupied_gpus_refused(pd, tmp_path, monkeypatch):
    monkeypatch.setattr(pd.subprocess, 'check_output', lambda *a, **k: 'GPU-uuid, 123, 100 MiB\n')
    with pytest.raises(RuntimeError, match='occupied'):
        pd.gpu_check(tmp_path, 'test')
    assert (tmp_path / 'test-gpu-check.json').exists()


def test_role_isolation(pd, tmp_path, monkeypatch):
    monkeypatch.setenv('LAPS_REPLAY_CONTROL_DIR', '/stale')
    env = pd.env_for(tmp_path, tmp_path / 'kernel', 'prefill', '2,3', True)
    assert 'LAPS_REPLAY_CONTROL_DIR' not in env
    assert env['CUDA_VISIBLE_DEVICES'] == '2,3'
    assert env['LAPS_KERNEL_EVENT_DIR'].endswith('/prefill')
    assert '--enable-batch-prefill-cuda-graph' in pd.server_command(30300, 'prefill')
    assert '--enable-fp32-lm-head' in pd.server_command(30301, 'decode')
    assert '--disaggregation-mode' not in pd.server_command(30400, 'single')
