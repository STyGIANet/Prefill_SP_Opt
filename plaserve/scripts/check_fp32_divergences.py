"""Evaluate saved first-divergence contexts with a single-GPU FP32 reference."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(status='running', model='Qwen/Qwen2.5-7B',
                    torch=torch.__version__, transformers=transformers.__version__,
                    cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                    dtype='float32', attention='eager', tf32=False)
    try:
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '2':
            raise RuntimeError('Reference requires CUDA_VISIBLE_DEVICES=2')
        points = {}
        for root in args.sources:
            workload = {x['name']: x['rows'] for x in json.loads((root / 'inputs.json').read_text())}
            run = json.loads((root / 'manifest.json').read_text())
            responses = {}
            for mode in (run['baseline'], 'graphs'):
                responses[mode] = {(x['case'], x['count'], i): r['output_ids']
                    for x in json.loads((root / f'{mode}-responses.json').read_text())
                    for i, r in enumerate(x['response'])}
            for mismatch in json.loads((root / 'mismatches.json').read_text()):
                case, row, count, step = (mismatch[k] for k in
                    ('case', 'row', 'count', 'first_differing_token'))
                a = responses[run['baseline']][case, count, row]
                b = responses['graphs'][case, count, row]
                assert a == mismatch['baseline'] and b == mismatch['graphs']
                assert a[:step] == b[:step] and a[step] != b[step]
                tokens = workload[case][row] + a[:step]
                key = (tuple(tokens), a[step], b[step])
                point = points.setdefault(key, dict(case=case, row=row, step=step,
                    input_ids=tokens, agreed_prefix=a[:step], baseline_token=a[step],
                    graph_token=b[step], sources=[]))
                point['sources'].append(dict(output=str(root.resolve()), count=count,
                                            baseline_mode=run['baseline']))
        points = list(points.values())
        (args.output / 'inputs.json').write_text(json.dumps(points, indent=2))
        manifest['inputs_sha256'] = hashlib.sha256((args.output / 'inputs.json').read_bytes()).hexdigest()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        model = AutoModelForCausalLM.from_pretrained(manifest['model'],
            torch_dtype=torch.float32, attn_implementation='eager', local_files_only=True)
        model = model.eval().to('cuda:0')
        dtypes = sorted({str(p.dtype) for p in model.parameters()})
        assert dtypes == ['torch.float32'], dtypes
        manifest.update(parameter_dtypes=dtypes, revision=getattr(model.config, '_commit_hash', None),
                        gpu=torch.cuda.get_device_name(0), points=len(points))
        results = []
        with torch.inference_mode():
            for point in points:
                ids = torch.tensor([point['input_ids']], dtype=torch.long, device='cuda:0')
                logits = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                               use_cache=False).logits[0, -1].float().cpu()
                assert torch.isfinite(logits).all()
                a, b = point['baseline_token'], point['graph_token']
                top = torch.argsort(logits, descending=True, stable=True)[:5]
                result = dict(point, baseline_logit=logits[a].item(), graph_logit=logits[b].item(),
                    margin_baseline_minus_graph=(logits[a] - logits[b]).item(),
                    argmax=int(logits.argmax()), top_ids=top.tolist(), top_logits=logits[top].tolist())
                results.append(result)
                print(json.dumps({k: v for k, v in result.items() if k not in ('input_ids', 'sources')}), flush=True)
                (args.output / 'reference.json').write_text(json.dumps(results, indent=2))
        manifest['status'] = 'completed'
    except BaseException as error:
        manifest.update(status='failed', error=repr(error))
        raise
    finally:
        (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
