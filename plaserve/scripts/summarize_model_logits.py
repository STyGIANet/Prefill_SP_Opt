"""Compare saved prefill and decode logits without enabling API logprobs."""
import argparse
import json
from pathlib import Path


def identify_rows(metadata, workload):
    case = next(c for c in workload if c["name"] == metadata["label"]["case"])
    offset = 0
    result = []
    for size, total in zip(metadata["extend_lens"], metadata["seq_lens"]):
        tokens = metadata["input_ids"][offset:offset + size]
        offset += size
        matches = [i for i, row in enumerate(case["rows"])
                   if size > 0 and len(row) == total and row[-size:] == tokens]
        if len(matches) != 1:
            raise RuntimeError(f"Cannot uniquely match diagnostic row: {case['name']} size={size} total={total}")
        result.append((case["name"], metadata["label"]["count"], matches[0]))
    if offset != len(metadata["input_ids"]):
        raise RuntimeError("Input length differs from recorded extend lengths")
    return result


def collect(directory, workload, responses):
    import torch
    result, requests = {}, {}
    for path in sorted(directory.glob("*.pt")):
        record = torch.load(path, map_location="cpu", weights_only=True)
        if record["tp_rank"] is None:
            raise RuntimeError("Worker did not expose TP rank; cannot select rank-zero logits")
        if record["tp_rank"] != 0:
            continue
        label = record["label"]
        scope = (path.name.split("-")[0], label["case"], label["count"])
        if record.get("kind", "extend") == "extend":
            initial = identify_rows(record, workload)
            for pool, key, total in zip(record.get("req_pool_indices", []), initial, record["seq_lens"]):
                requests[scope + (pool,)] = (key, total)
            keys = [key + (0,) for key in initial]
        else:
            keys = []
            for pool, total in zip(record["req_pool_indices"], record["seq_lens"]):
                key, prompt_length = requests[scope + (pool,)]
                step = total - prompt_length
                if step <= 0:
                    raise RuntimeError(f"Invalid decode step: {key} {step}")
                keys.append(key + (step,))
            if len(keys) != len(record["input_ids"]):
                raise RuntimeError("Decode input count differs from request count")
        if len(keys) != len(record["logits"]):
            raise RuntimeError("Logit rows differ from metadata")
        for index, (key, logits) in enumerate(zip(keys, record["logits"])):
            # Overlapped execution can compute logits after the last requested token.
            # These have no corresponding returned output to compare.
            if key[3] >= len(responses[key[:3]]):
                continue
            if key[3] and record["input_ids"][index] != responses[key[:3]][key[3] - 1]:
                raise RuntimeError(f"Decode input differs from generated history: {key}")
            if key in result:
                raise RuntimeError(f"Duplicate first-prefill row {key}; inspect captured metadata")
            if not torch.isfinite(logits).all():
                raise RuntimeError(f"Nonfinite logits in {key}")
            result[key] = logits
    return result


def response_tokens(root, mode):
    items = json.loads((root / f"{mode}-responses.json").read_text())
    return {(item["case"], item["count"], row): response["output_ids"]
            for item in items for row, response in enumerate(item["response"])}


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.output
    workload = json.loads((root / "inputs.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    baseline_tokens = response_tokens(root, manifest["baseline"])
    graph_tokens = response_tokens(root, "graphs")
    baseline = collect(root / (manifest["baseline"] + "-logits"), workload, baseline_tokens)
    graphs = collect(root / "graphs-logits", workload, graph_tokens)
    if not baseline or baseline.keys() != graphs.keys():
        raise RuntimeError(f"Missing/misaligned captures: baseline={list(baseline)} graphs={list(graphs)}")
    reports = []
    for key in sorted(baseline):
        step = key[3]
        if baseline_tokens[key[:3]][:step] != graph_tokens[key[:3]][:step]:
            continue
        a, b = baseline[key], graphs[key]
        if a.shape != b.shape:
            raise RuntimeError(f"Vocabulary shape differs: {key}")
        # Stable ordering makes equal maxima use the lowest token ID, matching
        # torch.argmax. topk alone does not provide deterministic tie ordering.
        ai = torch.argsort(a, descending=True, stable=True)[:5]
        bi = torch.argsort(b, descending=True, stable=True)[:5]
        av, bv = a[ai], b[bi]
        at, bt = baseline_tokens[key[:3]][step], graph_tokens[key[:3]][step]
        delta = b - a
        report = {"case": key[0], "count": key[1], "row": key[2], "step": step,
                  "same_argmax": ai[0].item() == bi[0].item(),
                  "max_abs": delta.abs().max().item(),
                  "rms": delta.square().mean().sqrt().item(),
                  "centered_max_abs": (delta - delta.mean()).abs().max().item(),
                  "baseline_margin": (av[0] - av[1]).item(),
                  "graphs_margin": (bv[0] - bv[1]).item(),
                  "baseline_top_ids": ai.tolist(), "baseline_top_logits": av.tolist(),
                  "graphs_top_ids": bi.tolist(), "graphs_top_logits": bv.tolist(),
                  "baseline_at_graphs_winner": a[bi[0]].item(),
                  "graphs_at_baseline_winner": b[ai[0]].item(),
                  "baseline_max_tie_ids": (a == av[0]).nonzero().flatten().tolist(),
                  "graphs_max_tie_ids": (b == bv[0]).nonzero().flatten().tolist(),
                  "baseline_generated": at, "graphs_generated": bt,
                  "baseline_generated_gap": (av[0] - a[at]).item(),
                  "graphs_generated_gap": (bv[0] - b[bt]).item(),
                  "baseline_gap_for_graphs_token": (av[0] - a[bt]).item(),
                  "graphs_gap_for_baseline_token": (bv[0] - b[at]).item()}
        reports.append(report)
        print(f"{key}: argmax={ai[0].item()}/{bi[0].item()} "
              f"max_abs={report['max_abs']:.6g} rms={report['rms']:.6g} "
              f"margins={report['baseline_margin']:.6g}/{report['graphs_margin']:.6g}")
        if at != bt:
            print(f"  generated={at}/{bt}; max ties={report['baseline_max_tie_ids']}/"
                  f"{report['graphs_max_tie_ids']}; selected gaps="
                  f"{report['baseline_generated_gap']:.6g}/{report['graphs_generated_gap']:.6g}; "
                  f"cross gaps={report['baseline_gap_for_graphs_token']:.6g}/"
                  f"{report['graphs_gap_for_baseline_token']:.6g}")
    (root / "logit-comparison.json").write_text(json.dumps(reports, indent=2))
    print(f"Saved {len(reports)} row comparisons; this diagnostic does not declare correctness passed.")


if __name__ == "__main__":
    main()
