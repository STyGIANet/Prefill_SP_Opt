import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('validator', ROOT / 'scripts/validate_bench_results.py')
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)


@pytest.fixture
def artifacts(tmp_path):
    (tmp_path / 'summary.txt').write_text('Benchmark summary: all finite metrics\n')
    raw = tmp_path / 'raw'
    raw.mkdir()
    kr = tmp_path / 'kernels'
    kr.mkdir()
    (kr / 'all-manifest.json').write_text(json.dumps(dict(status='passed', mode='saturating', selected_sha256='abc')))
    events = tmp_path / 'events'
    for policy in v.POLICIES:
        for label, count, cc in [('warmup', 20, 4), ('cc1', 500, 1), ('cc4', 500, 4)]:
            data = dict(completed=count, failed=0, num_prompts=count, concurrency=cc,
                        max_new_tokens=1, request_rate=None, **{key: 1.0 for key in v.METRICS})
            (raw / f'{policy}_{label}.json').write_text(json.dumps(data))
            (raw / f'{policy}_{label}.txt').write_text(f'completed: {count} / {count}\nfailed: 0\n')
        for role in ('prefill', 'decode'):
            (raw / f'{policy}_{role}.log').write_text('server ready\n')
            d = events / policy / role
            d.mkdir(parents=True)
            for pid in (11, 12):
                records = [dict(event='configured', pid=pid, root=str(kr), header_sha256='abc'),
                           dict(event='kernel_loaded', pid=pid, library=str(kr / 'cached_ops/test.so'))]
                (d / f'{pid}.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in records))
        (raw / f'{policy}_router.log').write_text('router ready\n')
    return tmp_path, raw, events, kr


def check(a):
    root, raw, events, kr = a
    return v.validate(root, raw, events, [1, 4], 500, kr)


def test_complete(artifacts):
    assert len(check(artifacts)) == 6


def test_invalid_summary(artifacts):
    (artifacts[0] / 'summary.txt').write_text('LAPS N/A\n')
    with pytest.raises(ValueError, match='invalid metrics'):
        check(artifacts)


@pytest.mark.parametrize('key,value', [('failed', 1), ('completed', 499), ('num_prompts', 499),
    ('concurrency', 8), ('mean_ttft_ms', 'N/A'), ('p90_ttft_ms', float('nan')),
    ('request_throughput_req_s', float('inf')), ('mean_ttft_ms', None), ('completed', True)])
def test_invalid_result(artifacts, key, value):
    path = artifacts[1] / 'laps_cc4.json'
    data = json.loads(path.read_text())
    data[key] = value
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        check(artifacts)


@pytest.mark.parametrize('file', ['laps_cc4.json', 'disaggregation_cc1.txt', 'vanilla_sglang_warmup.json',
                                  'laps_prefill.log', 'laps_router.log'])
def test_missing(artifacts, file):
    (artifacts[1] / file).unlink()
    with pytest.raises(ValueError):
        check(artifacts)


def test_log_failure_despite_good_json(artifacts):
    (artifacts[1] / 'laps_cc4.txt').write_text('completed: 500 / 500\nfailed: 1\n')
    with pytest.raises(ValueError):
        check(artifacts)


@pytest.mark.parametrize('mutation', ['missing', 'wrong_root', 'wrong_hash', 'outside_cache', 'missing_load'])
def test_kernel_provenance(artifacts, mutation):
    path = artifacts[2] / 'laps/decode/12.jsonl'
    rows = [json.loads(s) for s in path.read_text().splitlines()]
    if mutation == 'missing':
        path.unlink()
    else:
        if mutation == 'wrong_root': rows[0]['root'] = '/wrong'
        if mutation == 'wrong_hash': rows[0]['header_sha256'] = 'wrong'
        if mutation == 'outside_cache': rows[1]['library'] = '/wrong/test.so'
        if mutation == 'missing_load': rows = rows[:1]
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    with pytest.raises(ValueError):
        check(artifacts)


def test_cli_failure_status(artifacts):
    root, raw, events, kr = artifacts
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/validate_bench_results.py'), str(root),
        '--kernel-events', str(events), '--kernel-root', str(kr), '--requested', '500',
        '--concurrency', '1', '4', '--upstream-status', '1'], capture_output=True, text=True)
    assert result.returncode == 1
    report = json.loads((root / 'validation.json').read_text())
    assert report['status'] == 'failed' and 'Upstream process failed' in report['error']


def test_guarded_copy(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    spec = importlib.util.spec_from_file_location('prepare', ROOT / 'scripts/prepare_bench_fixed.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = ROOT / 'vendor/LAPS/bench_laps_prefill_throughput/scripts/bench_h100_4gpu.sh'
    before = source.read_bytes()
    module.prepare(source, tmp_path)
    text = (tmp_path / 'upstream-guarded.sh').read_text()
    assert 'pkill' not in text
    assert text.count('OWNED_PIDS+=("$!")') == 3
    assert text.count('"$SAFE_LAUNCHER"') == 3
    assert 'kernel-events/${label}/prefill' in text
    assert 'kernel-events/${label}/decode' in text
    assert source.read_bytes() == before
    import hashlib
    provenance = json.loads((tmp_path / 'provenance.json').read_text())
    assert provenance['vendor_diff_sha256'] == hashlib.sha256((tmp_path / 'vendor.diff').read_bytes()).hexdigest()
    subprocess.run(['bash', '-n', str(tmp_path / 'upstream-guarded.sh')], check=True)
