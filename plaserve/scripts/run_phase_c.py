"""Run three validated PD throughput repeats; stop between repeats on interference."""
import argparse
import csv
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import pwd
import statistics
import subprocess
import sys

from vendor_provenance import snapshot

ROOT = Path(__file__).resolve().parents[1]
POLICIES = ('vanilla_sglang', 'disaggregation', 'laps')
CCS = (1, 4, 8, 16)


def save(path, data):
    path.write_text(json.dumps(data, indent=2))


def query(args):
    r = subprocess.run(args, capture_output=True, text=True, timeout=15)
    if r.returncode:
        raise RuntimeError(f'GPU query failed: {r.stderr}')
    return r.stdout


def check_free(output, repeat):
    gpus = query(['nvidia-smi', '-i', '2,3,4,5', '--query-gpu=index,uuid,memory.used,utilization.gpu', '--format=csv'])
    processes = query(['nvidia-smi', '-i', '2,3,4,5', '--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader'])
    save(output / f'before-repeat-{repeat}.json', dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                                     gpus=gpus, processes=processes))
    if processes.strip():
        raise RuntimeError(f'GPUs 2-5 occupied before repeat {repeat}; no processes stopped: {processes}')


def interference(path, user):
    foreign, unknown, other_gpu_activity, errors = [], [], [], []
    if not path.is_file():
        return dict(foreign_users=[], unknown_owners=[], other_gpu_activity=[], errors=['Missing GPU monitoring log'])
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if r['gpus']['status'] or r['processes']['status']:
            errors.append(r)
            continue
        mapping = {}
        for row in csv.reader(io.StringIO(r['gpus']['stdout'])):
            if row and row[0].strip().isdigit():
                idx = int(row[0].strip())
                mapping[row[1].strip()] = idx
                if idx in (0, 1) and (int(row[2].strip().split()[0]) > 0 or int(row[3].strip().split()[0]) > 0):
                    other_gpu_activity.append(dict(utc=r['utc'], gpu=idx, memory=row[2].strip(), utilization=row[3].strip()))
        owners = {}
        for row in r.get('owners', {}).get('stdout', '').splitlines()[1:]:
            fields = row.split(None, 2)
            if len(fields) >= 2:
                owners[fields[0]] = dict(user=fields[1], command=fields[2] if len(fields) > 2 else '')
        for row in csv.reader(io.StringIO(r['processes']['stdout'])):
            if not row or row[0].strip() not in mapping:
                continue
            gpu = mapping[row[0].strip()]
            if gpu not in (2, 3, 4, 5):
                continue
            pid = row[1].strip()
            owner = owners.get(pid)
            if owner is None:
                unknown.append(dict(utc=r['utc'], gpu=gpu, pid=pid))
            elif owner['user'] != user:
                foreign.append(dict(utc=r['utc'], gpu=gpu, pid=pid, **owner))
    return dict(foreign_users=foreign, unknown_owners=unknown, other_gpu_activity=other_gpu_activity, errors=errors)


def aggregate(repeats, output):
    records = []
    all_results = []
    for root in repeats:
        validation = json.loads((root / 'validation.json').read_text())
        if validation['status'] != 'passed':
            raise RuntimeError(f'Cannot aggregate invalid repeat {root}')
        all_results.append({(r['policy'], r['concurrency']): r for r in validation['results']})
    def stats(values):
        return dict(mean=statistics.mean(values), std=statistics.stdev(values), min=min(values), max=max(values), values=values)
    for cc in CCS:
        for policy in POLICIES:
            record = dict(policy=policy, concurrency=cc)
            for metric in ('request_throughput_req_s', 'mean_ttft_ms', 'p90_ttft_ms'):
                record[metric] = stats([r[policy, cc][metric] for r in all_results])
            records.append(record)
    ratios = {str(cc): stats([r['laps', cc]['request_throughput_req_s'] /
                             r['vanilla_sglang', cc]['request_throughput_req_s'] for r in all_results]) for cc in CCS}
    save(output / 'aggregate.json', dict(repeats=3, standard_deviation='sample (ddof=1)', results=records,
                                       laps_vanilla_paired_ratios=ratios))
    lines = ['Mean ± sample standard deviation across three repeats.', '',
             '| Policy | Concurrency | req/s | Mean request latency (ms) | P90 request latency (ms) |',
             '|---|---:|---:|---:|---:|']
    for r in records:
        cells = [f'{r[k]["mean"]:.2f} ± {r[k]["std"]:.2f}' for k in
                 ('request_throughput_req_s', 'mean_ttft_ms', 'p90_ttft_ms')]
        lines.append(f'| {r["policy"]} | {r["concurrency"]} | ' + ' | '.join(cells) + ' |')
    lines += ['', '| Concurrency | LAPS/vanilla mean ± std | Range |', '|---:|---:|---:|']
    for cc, r in ratios.items():
        lines.append(f'| {cc} | {r["mean"]:.4f} ± {r["std"]:.4f} | {r["min"]:.4f}–{r["max"]:.4f} |')
    (output / 'summary.md').write_text('\n'.join(lines) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    output = a.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    if os.environ.get('CONDA_DEFAULT_ENV') != 'plaserve':
        raise RuntimeError('Activate plaserve')
    provenance = snapshot(ROOT / 'vendor/LAPS', output)
    provenance['scripts_sha256'] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [Path(__file__), ROOT / 'scripts/run_bench_fixed_kernels.sh', ROOT / 'scripts/prepare_bench_fixed.py',
                     ROOT / 'scripts/validate_bench_results.py', ROOT / 'scripts/monitor_bench_gpus.py']}
    save(output / 'provenance.json', provenance)
    manifest = dict(status='running', completed_repeats=[], planned_repeats=3, num_prompts=500,
                    concurrency=list(CCS), policies=list(POLICIES))
    save(output / 'manifest.json', manifest)
    repeats = []
    try:
        for repeat in range(1, 4):
            check_free(output, repeat)
            directory = output / f'repeat-{repeat}'
            env = os.environ.copy()
            env.update(NUM_PROMPTS='500', CONCURRENCY_LEVELS='1 4 8 16', BACKEND='nixl')
            command = ['bash', str(ROOT / 'scripts/run_bench_fixed_kernels.sh'), str(directory)]
            print(f'Starting repeat {repeat}: {directory}', flush=True)
            with (output / f'repeat-{repeat}-wrapper.log').open('w') as log:
                result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
            validation = json.loads((directory / 'validation.json').read_text())
            prov = json.loads((directory / 'provenance.json').read_text())
            assert prov['vendor_head'] == provenance['vendor_head']
            assert prov['vendor_diff_sha256'] == provenance['vendor_diff_sha256']
            assert hashlib.sha256((directory / 'vendor.diff').read_bytes()).hexdigest() == prov['vendor_diff_sha256']
            activity = interference(directory / 'gpu-samples.jsonl', pwd.getpwuid(os.getuid()).pw_name)
            save(directory / 'interference.json', activity)
            if result.returncode or validation['status'] != 'passed':
                raise RuntimeError(f'Repeat {repeat} failed: returncode={result.returncode}, validation={validation.get("error")}')
            if len(validation['results']) != 12:
                raise RuntimeError(f'Repeat {repeat}: missing policy/concurrency pairs')
            manifest['completed_repeats'].append(str(directory))
            save(output / 'manifest.json', manifest)
            repeats.append(directory)
            print(f'Repeat {repeat} passed all 12 policy/concurrency validations', flush=True)
            if activity['foreign_users']:
                raise RuntimeError(f'Other user appeared on GPUs 2-5 in repeat {repeat}; stopping after completed repeat. See interference.json')
            if activity['errors'] or activity['unknown_owners']:
                raise RuntimeError(f'Incomplete GPU ownership observations in repeat {repeat}; stopping for review')
        aggregate(repeats, output)
        manifest['status'] = 'passed'
    except BaseException as error:
        manifest.update(status='stopped', error=repr(error))
        print(manifest['error'], flush=True)
        raise
    finally:
        save(output / 'manifest.json', manifest)
    print((output / 'summary.md').read_text(), flush=True)


if __name__ == '__main__':
    main()
