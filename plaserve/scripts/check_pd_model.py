"""Compare single-server graph outputs with LAPS prefill/decode routing."""
import argparse
from contextlib import contextmanager, ExitStack
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from urllib.error import URLError, HTTPError

from check_blackwell_model import api, cases
from vendor_provenance import snapshot

ROOT = Path(__file__).resolve().parents[1]


def write(path, data):
    path.write_text(json.dumps(data, indent=2))


def gpu_check(output, label):
    info = subprocess.check_output(['nvidia-smi', '-i', '2,3,4,5',
        '--query-gpu=index,uuid,memory.used,utilization.gpu', '--format=csv'], text=True)
    processes = subprocess.check_output(['nvidia-smi', '-i', '2,3,4,5',
        '--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader'], text=True)
    write(output / f'{label}-gpu-check.json', dict(gpus=info, processes=processes))
    if processes.strip():
        raise RuntimeError('GPUs 2-5 occupied; inspect GPU check. No processes stopped.')
    print(f'{label}: GPUs 2-5 have no compute processes', flush=True)


def env_for(output, kernel, role, gpus, coverage=False):
    env = os.environ.copy()
    for key in ('LAPS_REPLAY_CONTROL_DIR', 'LAPS_LOGITS_DIR', 'LAPS_PD_COVERAGE_DIR', 'LAPS_DIAGNOSTIC_CASE_FILE'):
        env.pop(key, None)
    env.update(CUDA_VISIBLE_DEVICES=gpus, LAPS_FLASHINFER_KERNEL_ROOT=str(kernel),
               LAPS_KERNEL_EVENT_DIR=str(output / 'kernel-events' / role),
               PYTHONPATH=str(ROOT / 'scripts/laps_kernel_hook') + os.pathsep + env.get('PYTHONPATH', ''),
               LAPS_DIAGNOSTIC_CASE_FILE=str(output / 'active-case.json'),
               LAPS_LOGITS_DIR=str(output / f'{role}-logits'))
    env.setdefault('MAX_JOBS', '4')
    if coverage:
        env['LAPS_PD_COVERAGE_DIR'] = str(output / 'coverage' / role)
    return env


@contextmanager
def launch(output, role, command, env, port=None):
    write(output / f'{role}-command.json', command)
    with (output / f'{role}-server.log').open('w') as log:
        child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        with (output / 'owned-processes.jsonl').open('a') as record:
            record.write(json.dumps(dict(role=role, pid=child.pid, pgid=child.pid, command=command)) + '\n')
        try:
            if port is not None:
                deadline = time.monotonic() + 600
                while True:
                    if child.poll() is not None:
                        raise RuntimeError(f'{role} server exited {child.returncode}; inspect log')
                    try:
                        api(port, 'health')
                        break
                    except (URLError, TimeoutError):
                        if time.monotonic() >= deadline:
                            raise RuntimeError(f'{role} readiness timeout')
                        time.sleep(1)
                print(f'{role}: ready', flush=True)
            yield child
        finally:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            print(f'{role}: owned process group stopped', flush=True)


def server_command(port, role):
    command = [sys.executable, '-m', 'sglang.launch_server', '--model-path', 'Qwen/Qwen2.5-7B',
        '--tp-size', '2', '--dtype', 'bfloat16', '--attention-backend', 'flashinfer', '--page-size', '1',
        '--host', '127.0.0.1', '--port', str(port), '--mem-fraction-static', '0.75',
        '--random-seed', '1701', '--enable-fp32-lm-head', '--enable-piecewise-cuda-graph']
    if role != 'decode':
        command += ['--enable-batch-prefill-cuda-graph', '--enable-laps-scheduler', '--laps-length-threshold', '256']
    if role != 'single':
        command += ['--disaggregation-mode', role, '--disaggregation-transfer-backend', 'nixl',
                    '--disaggregation-bootstrap-port', '9301' if role == 'prefill' else '9300']
    return command


def generate(output, mode, port, case, count, rows, phase):
    write(output / 'active-case.json', dict(phase=phase, case=case, count=count))
    payload = dict(input_ids=rows, sampling_params=dict(temperature=0, max_new_tokens=count, ignore_eos=True), return_logprob=False)
    path = output / f'{mode}-{case}-{count}-{phase}-raw.json'
    try:
        response = api(port, 'generate', payload)
    except HTTPError as error:
        write(path, dict(request=payload, error=error.read().decode(errors='replace'), status=error.code))
        raise
    write(path, dict(request=payload, response=response))
    if not isinstance(response, list) or len(response) != len(rows):
        raise RuntimeError(f'{mode}/{case}: missing batched rows; inspect {path}')
    for item in response:
        ids = item.get('output_ids')
        if not isinstance(ids, list) or len(ids) != count or not all(type(x) is int for x in ids):
            raise RuntimeError(f'{mode}/{case}: incomplete token IDs; inspect {path}')
        finish = item.get('meta_info', {}).get('finish_reason', {})
        if isinstance(finish, dict) and finish.get('type') in ('abort', 'error'):
            raise RuntimeError(f'{mode}/{case}: request failed: {finish}')
    return response


def workload(output, mode, port, flush_ports, inputs):
    results = []
    for count in (1, 8):
        for case in inputs:
            write(output / 'active-case.json', dict(phase='flush'))
            for target in flush_ports:
                reply = api(target, 'flush_cache', {})
                with (output / 'flush-responses.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(mode=mode, case=case['name'], count=count, port=target, response=reply)) + '\n')
            prime = None
            if case['prime'] is not None:
                prime = generate(output, mode, port, case['name'], 1, [case['prime']], 'prime')
            response = generate(output, mode, port, case['name'], count, case['rows'], 'generate')
            results.append(dict(case=case['name'], count=count, prime_response=prime, response=response))
            write(output / f'{mode}-responses.json', results)
            print(f'{mode}: {case["name"]} tokens={count}', flush=True)
    return results


def kernel_evidence(output, kernel, role):
    km = json.loads((kernel / 'all-manifest.json').read_text())
    configured, loaded = set(), set()
    for path in (output / 'kernel-events' / role).glob('*.jsonl'):
        for line in path.read_text().splitlines():
            e = json.loads(line)
            if e['event'] == 'configured':
                if e['root'] != str(kernel) or e['header_sha256'] != km['selected_sha256']:
                    raise RuntimeError(f'{role}: wrong kernel provenance')
                configured.add(e['pid'])
            if e['event'] == 'kernel_loaded':
                if not Path(e['library']).is_relative_to(kernel / 'cached_ops'):
                    raise RuntimeError(f'{role}: kernel escaped corrected root')
                loaded.add(e['pid'])
    if len(configured & loaded) < 2:
        raise RuntimeError(f'{role}: missing corrected kernel load evidence in both TP workers')
    return sorted(configured & loaded)


def coverage(output, role):
    events = [json.loads(line) for path in (output / 'coverage' / role).glob('*.jsonl')
              for line in path.read_text().splitlines()]
    traces = [e for e in events if e['event'] == 'replay' and e['label'].get('phase') == 'generate']
    checks = dict(both_workers=len({e['pid'] for e in traces}) == 2,
        any_replay=bool(traces), padded_batch=any(e['actual_bs'] < e['target_bs'] for e in traces),
        padded_queries=any(n < e['width'] for e in traces for n in e['extend']),
        cached_history=any(n > q for e in traces for n, q in zip(e['kv'], e['extend'])),
        multiple_shapes=len({(e['target_bs'], e['width']) for e in traces}) > 1)
    return dict(checks=checks, replay_events=len(traces), shapes=sorted({(e['target_bs'], e['width']) for e in traces}))


def compare(single, pd):
    if len(single) != len(pd):
        raise RuntimeError('Incomplete workload responses')
    mismatches = []
    for a, b in zip(single, pd):
        if (a['case'], a['count']) != (b['case'], b['count']) or len(a['response']) != len(b['response']):
            raise RuntimeError('Misaligned responses')
        for row, (x, y) in enumerate(zip(a['response'], b['response'])):
            x, y = x['output_ids'], y['output_ids']
            if x != y:
                step = next(i for i, (u, v) in enumerate(zip(x, y)) if u != v)
                mismatches.append(dict(case=a['case'], count=a['count'], row=row,
                    first_differing_token=step, baseline_mode='single', baseline=x, pd=y))
    return mismatches


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--kernel-root', type=Path, required=True)
    a = p.parse_args()
    output, kernel = a.output.resolve(), a.kernel_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    provenance = snapshot(ROOT / 'vendor/LAPS', output)
    provenance['scripts_sha256'] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [Path(__file__), ROOT / 'scripts/laps_kernel_hook/sitecustomize.py', ROOT / 'scripts/laps_kernel_hook/pd_coverage.py']}
    write(output / 'provenance.json', provenance)
    inputs = cases()
    write(output / 'inputs.json', inputs)
    manifest = dict(status='running', baseline='single', candidate='pd', model='Qwen/Qwen2.5-7B',
                    dtype='bfloat16', fp32_lm_head=True, tp=2, backend='nixl', kernel_root=str(kernel),
                    inputs_sha256=hashlib.sha256((output / 'inputs.json').read_bytes()).hexdigest())
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM, interrupt)
    try:
        km = json.loads((kernel / 'all-manifest.json').read_text())
        if km['status'] != 'passed' or km['mode'] != 'saturating':
            raise RuntimeError('Corrected attention validation not passed')
        for port in (30400, 30300, 30301, 30302, 9300, 9301):
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', port))
        gpu_check(output, 'before-single')
        with launch(output, 'single', server_command(30400, 'single'), env_for(output, kernel, 'single', '2,3', True), 30400):
            single = workload(output, 'single', 30400, [30400], inputs)
        manifest['kernel_workers'] = {'single': kernel_evidence(output, kernel, 'single')}
        manifest['coverage'] = {'single': coverage(output, 'single')}
        write(output / 'manifest.json', manifest)
        gpu_check(output, 'before-pd')
        with ExitStack() as stack:
            stack.enter_context(launch(output, 'decode', server_command(30301, 'decode'), env_for(output, kernel, 'decode', '4,5'), 30301))
            time.sleep(5)
            stack.enter_context(launch(output, 'prefill', server_command(30300, 'prefill'), env_for(output, kernel, 'prefill', '2,3', True), 30300))
            router = [sys.executable, '-m', 'sglang_router.launch_router', '--pd-disaggregation', '--mini-lb',
                      '--prefill', 'http://127.0.0.1:30300', '9301', '--decode', 'http://127.0.0.1:30301',
                      '--host', '127.0.0.1', '--port', '30302']
            router_env = os.environ.copy()
            router_env['CUDA_VISIBLE_DEVICES'] = '2,3,4,5'
            stack.enter_context(launch(output, 'router', router, router_env, 30302))
            pd = workload(output, 'pd', 30302, [30300, 30301], inputs)
        mismatches = compare(single, pd)
        write(output / 'mismatches.json', mismatches)
        manifest.update(mismatches=len(mismatches), completed_outputs=sum(len(x['response']) for x in pd))
        for role in ('prefill', 'decode'):
            manifest['kernel_workers'][role] = kernel_evidence(output, kernel, role)
        manifest['coverage']['pd'] = coverage(output, 'prefill')
        manifest['api_cache_hits'] = {mode: sum(r.get('meta_info', {}).get('cached_tokens', 0) > 0
            for item in results if item['case'].startswith('cached-') for r in item['response'])
            for mode, results in [('single', single), ('pd', pd)]}
        if not all(all(c['checks'].values()) for c in manifest['coverage'].values()):
            raise RuntimeError('Required graph/cache coverage incomplete')
        if not all(manifest['api_cache_hits'].values()):
            raise RuntimeError('API cache-hit evidence missing')
        manifest['status'] = 'failed' if mismatches else 'passed'
        if mismatches:
            manifest['error'] = 'Exact token mismatches; FP32 investigation required'
    except BaseException as error:
        manifest.update(status='failed', error=repr(error))
        raise
    finally:
        write(output / 'manifest.json', manifest)
    print(json.dumps(manifest, indent=2), flush=True)
    return int(manifest['status'] != 'passed')


if __name__ == '__main__':
    raise SystemExit(main())
