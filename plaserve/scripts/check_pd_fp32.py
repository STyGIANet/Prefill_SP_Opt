"""Adapt PD outputs to the unchanged FP32 reference checker, preserving originals."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from vendor_provenance import snapshot


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    a.output.mkdir(parents=True, exist_ok=False)
    provenance = snapshot(root / 'vendor/LAPS', a.output)
    original = root / 'scripts/check_fp32_divergences.py'
    provenance.update(reference_script_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
                      source=str(a.source.resolve()), candidate_mapping='graphs in normalized files means PD')
    (a.output / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    normalized = a.output / 'normalized'
    normalized.mkdir()
    for name in ('inputs.json', 'single-responses.json'):
        shutil.copyfile(a.source / name, normalized / name)
    shutil.copyfile(a.source / 'pd-responses.json', normalized / 'graphs-responses.json')
    mismatches = json.loads((a.source / 'mismatches.json').read_text())
    if not mismatches:
        raise RuntimeError('No PD mismatches to reference')
    for item in mismatches:
        item['graphs'] = item.pop('pd')
    (normalized / 'mismatches.json').write_text(json.dumps(mismatches, indent=2))
    (normalized / 'manifest.json').write_text(json.dumps(dict(baseline='single', candidate='graphs',
        note='Compatibility layout: graphs means PD, not a new SGLang run')))
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '2'
    env.pop('PYTHONPATH', None)
    for key in list(env):
        if key.startswith('LAPS_'):
            env.pop(key)
    reference = a.output / 'reference'
    command = [sys.executable, str(original), '--sources', str(normalized), '--output', str(reference)]
    (a.output / 'command.json').write_text(json.dumps(command, indent=2))
    subprocess.run(command, env=env, check=True)
    (reference / 'provenance.json').write_text(json.dumps(snapshot(root / 'vendor/LAPS', reference), indent=2))
    records = json.loads((reference / 'reference.json').read_text())
    for record in records:
        record['single_token'] = record.pop('baseline_token')
        record['pd_token'] = record.pop('graph_token')
        record['single_logit'] = record.pop('baseline_logit')
        record['pd_logit'] = record.pop('graph_logit')
        record['margin_single_minus_pd'] = record.pop('margin_baseline_minus_graph')
    (a.output / 'pd-reference.json').write_text(json.dumps(records, indent=2))
    if hashlib.sha256(original.read_bytes()).hexdigest() != provenance['reference_script_sha256']:
        raise RuntimeError('Original reference checker changed')


if __name__ == '__main__':
    main()
