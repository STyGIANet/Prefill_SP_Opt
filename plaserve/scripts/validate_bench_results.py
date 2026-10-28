"""Fail closed on incomplete upstream PD benchmark artifacts (CPU only)."""
import argparse
import json
import math
from pathlib import Path
import re

POLICIES = ('vanilla_sglang', 'disaggregation', 'laps')
METRICS = ('total_duration_s', 'total_input_tokens', 'prefill_throughput_tok_s',
           'request_throughput_req_s', 'median_ttft_ms', 'mean_ttft_ms',
           'p50_ttft_ms', 'p90_ttft_ms', 'p99_ttft_ms')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(root, logs, events, counts, requested, kernel_root, tp=2):
    summary = root / 'summary.txt'
    require(summary.is_file(), f'Missing upstream summary: {summary}')
    require(not re.search(r'\b(?:N/A|NaN|inf)\b', summary.read_text(), re.I),
            'Upstream summary contains invalid metrics')
    kernel_root = kernel_root.resolve()
    km = json.loads((kernel_root / 'all-manifest.json').read_text())
    require(km['status'] == 'passed' and km['mode'] == 'saturating', 'Kernel validation not passed')
    results = []
    for policy in POLICIES:
        for label, cc, num in [('warmup', 4, 20)] + [(f'cc{c}', c, requested) for c in counts]:
            stem = f'{policy}_{label}'
            path = root / 'raw' / f'{stem}.json'
            require(path.is_file(), f'Missing policy/concurrency result: {path}')
            data = json.loads(path.read_text())
            for key, expected in [('completed', num), ('num_prompts', num), ('failed', 0),
                                  ('concurrency', cc), ('max_new_tokens', 1)]:
                require(type(data.get(key)) is int and data[key] == expected,
                        f'{stem}: {key}={data.get(key)!r}, expected {expected}')
            for key in METRICS:
                value = data.get(key)
                require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
                        f'{stem}: invalid/missing metric {key}={value!r}')
            require(data['request_throughput_req_s'] > 0, f'{stem}: zero throughput')
            # Reject invalid additional metrics too, excluding documented nullable metadata.
            for key, value in data.items():
                if key in ('request_rate', 'concurrency'):
                    continue
                require(value is not None and not (isinstance(value, str) and value.lower() in ('n/a', 'nan', 'inf'))
                        and not (isinstance(value, float) and not math.isfinite(value)),
                        f'{stem}: invalid value for {key}')
            log = logs / f'{stem}.txt'
            require(log.is_file(), f'Missing benchmark log: {log}')
            text = log.read_text()
            require(re.findall(r'completed:\s*(\d+)\s*/\s*(\d+)', text) == [(str(num), str(num))],
                    f'{stem}: missing/inconsistent completed log count')
            require(re.findall(r'failed:\s*(\d+)', text) == ['0'], f'{stem}: missing/nonzero log failures')
            require(not re.search(r'\[FAIL\]|\[warn\]|Traceback|\b(?:N/A|NaN|inf)\b', text, re.I),
                    f'{stem}: benchmark log reports failure or invalid metrics')
            if label != 'warmup':
                results.append(dict(policy=policy, **data))
        for role in ('prefill', 'decode'):
            log = logs / f'{policy}_{role}.log'
            require(log.is_file() and log.stat().st_size > 0, f'Missing server log: {log}')
            require('LAPS_VERIFY ' not in log.read_text(), f'Diagnostic replay logging enabled: {log}')
            directory = events / policy / role
            configured, loaded = set(), set()
            for path in directory.glob('*.jsonl'):
                for line in path.read_text().splitlines():
                    event = json.loads(line)
                    if event['event'] == 'configured':
                        require(Path(event['root']).resolve() == kernel_root and
                                event['header_sha256'] == km['selected_sha256'],
                                f'Wrong corrected kernel provenance: {path}')
                        configured.add(event['pid'])
                    elif event['event'] == 'kernel_loaded':
                        require(Path(event['library']).resolve().is_relative_to(kernel_root / 'cached_ops'),
                                f'Kernel outside corrected cache: {path}')
                        loaded.add(event['pid'])
                    elif event['event'] == 'error':
                        raise ValueError(f'Kernel hook error: {event}')
            require(len(configured & loaded) >= tp,
                    f'{policy}/{role}: corrected kernel loads missing in {tp} workers')
        router = logs / f'{policy}_router.log'
        require(router.is_file() and router.stat().st_size > 0, f'Missing router log: {router}')
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('results', type=Path)
    p.add_argument('--logs', type=Path)
    p.add_argument('--kernel-events', type=Path, required=True)
    p.add_argument('--kernel-root', type=Path, required=True)
    p.add_argument('--concurrency', nargs='+', type=int, required=True)
    p.add_argument('--requested', type=int, required=True)
    p.add_argument('--upstream-status', type=int, default=0)
    a = p.parse_args()
    report = {'status': 'failed'}
    try:
        require(a.requested > 0 and a.concurrency and all(c > 0 for c in a.concurrency)
                and len(set(a.concurrency)) == len(a.concurrency), 'Invalid expected workload')
        require(a.upstream_status == 0, f'Upstream process failed: {a.upstream_status}')
        report['results'] = validate(a.results, a.logs or a.results / 'raw', a.kernel_events,
                                    a.concurrency, a.requested, a.kernel_root)
        report['status'] = 'passed'
    except (ValueError, OSError, KeyError, TypeError) as error:
        report['error'] = str(error)
    if a.results.is_dir():
        (a.results / 'validation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
