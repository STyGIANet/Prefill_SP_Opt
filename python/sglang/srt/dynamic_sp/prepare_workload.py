# Copyright 2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""Prepare real prompt arrivals for online scheduling, with optional trace export."""

import argparse
import hashlib
import json
import random
from pathlib import Path

from sglang.srt.dynamic_sp.arrivals import (
    poisson_arrival_times,
    request_rate_for_token_load,
)


def make_schedule(prompts, chunk_tokens, world_size=8, seed=0):
    if not prompts or chunk_tokens <= 0 or world_size <= 0:
        raise ValueError("Prompts, chunk size and world size must be positive")
    if any(not tokens for tokens in prompts):
        raise ValueError("Prompts cannot be empty")
    rng = random.Random(seed)
    # Alternating small and large degrees exercises both directions, including
    # degrees which do not divide eight. This is test input, not a policy.
    degrees = []
    low, high = 1, world_size
    while low <= high:
        degrees.append(low)
        if low != high:
            degrees.append(high)
        low, high = low + 1, high - 1
    plans = []
    for iteration, end in enumerate(
        range(chunk_tokens, max(map(len, prompts)) + chunk_tokens, chunk_tokens)
    ):
        steps = []
        for index, tokens in enumerate(prompts):
            if end - chunk_tokens >= len(tokens):
                continue
            ranks = list(range(world_size))
            rng.shuffle(ranks)
            degree = degrees[(iteration + index) % world_size]
            steps.append(
                {
                    "request_id": f"prompt-{index}",
                    "ranks": ranks[:degree],
                    "end": min(end, len(tokens)),
                }
            )
        plans.append({"version": len(plans), "steps": steps})
    plans[0]["admit"] = [
        {"request_id": f"prompt-{i}", "token_ids": tokens}
        for i, tokens in enumerate(prompts)
    ]
    # Verify migration without recomputation after all prefixes are resident.
    for degree in (1, world_size):
        steps = []
        for i, tokens in enumerate(prompts):
            ranks = [(i + r) % world_size for r in range(degree)]
            steps.append(
                {"request_id": f"prompt-{i}", "ranks": ranks, "end": len(tokens)}
            )
        plans.append({"version": len(plans), "steps": steps})
    plans.append(
        {"version": len(plans), "release": [f"prompt-{i}" for i in range(len(prompts))]}
    )
    return plans


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--dataset", default="emozilla/pg19")
    parser.add_argument("--dataset-config", default="default")
    parser.add_argument("--dataset-revision", default="main")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-column", default="text")
    parser.add_argument(
        "--num-prompts",
        default="all",
        help="all (default) or a positive count of eligible records",
    )
    parser.add_argument("--input-tokens", type=int, default=8192)
    parser.add_argument(
        "--length-mode",
        choices=("fixed", "natural"),
        default="fixed",
        help="fixed skips short records; natural preserves lengths up to input-tokens",
    )
    parser.add_argument("--chunk-tokens", type=int, default=1024)
    parser.add_argument(
        "--export-trace",
        action="store_true",
        help="Also write the deterministic resize test trace",
    )
    parser.add_argument(
        "--offered-load",
        type=float,
        default=8192.0,
        help="Expected offered input tokens per second (Poisson arrivals)",
    )
    parser.add_argument("--arrival-seed", type=int, default=0)
    parser.add_argument(
        "--arrival-pattern", choices=("poisson", "burst"), default="poisson"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.num_prompts == "all":
        args.num_prompts = None
    else:
        try:
            args.num_prompts = int(args.num_prompts)
        except ValueError:
            parser.error("num-prompts must be all or a positive integer")
        if args.num_prompts <= 0:
            parser.error("num-prompts must be all or a positive integer")
    if min(args.input_tokens, args.chunk_tokens) <= 0:
        parser.error("Token counts must be positive")

    try:
        request_rate_for_token_load([args.input_tokens], args.offered_load)
    except ValueError as exc:
        parser.error(str(exc))

    from datasets import load_dataset
    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoConfig, AutoTokenizer

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(
        (output / name).exists()
        for name in ("workload.json", "schedule.json", "manifest.json")
    ):
        raise FileExistsError(
            "Choose a fresh output directory to preserve previous runs"
        )
    local_model = Path(args.model)
    model_revision = None
    if local_model.is_dir():
        model_path = str(local_model.resolve())
    else:
        info = HfApi().model_info(args.model, revision=args.model_revision)
        model_revision = info.sha
        has_safetensors = any(
            s.rfilename.endswith(".safetensors") for s in info.siblings
        )
        model_path = snapshot_download(
            args.model,
            revision=model_revision,
            allow_patterns=[
                "*.json",
                "*.model",
                "*.tiktoken",
                "*.txt",
                "*.safetensors" if has_safetensors else "*.bin",
            ],
        )
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=False)
    if config.architectures != ["LlamaForCausalLM"]:
        raise ValueError("The selected model must use LlamaForCausalLM")
    if args.input_tokens > config.max_position_embeddings:
        raise ValueError("input-tokens exceeds the model context length")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    dataset_revision = (
        HfApi().dataset_info(args.dataset, revision=args.dataset_revision).sha
    )
    dataset = load_dataset(
        args.dataset,
        args.dataset_config,
        split=args.split,
        revision=dataset_revision,
        streaming=True,
    )
    prompts, source_rows = [], []
    bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    needed = args.input_tokens - len(bos)
    if needed <= 0:
        raise ValueError("input-tokens must leave room for text after BOS")
    rows_read = 0
    for row in dataset:
        rows_read += 1
        text = row[args.text_column]
        if not isinstance(text, str):
            raise TypeError("The dataset text column must contain strings")
        if not text.strip():
            continue
        tokens = tokenizer.encode(
            text, add_special_tokens=False, truncation=True, max_length=needed
        )
        if not tokens or (args.length_mode == "fixed" and len(tokens) < needed):
            continue
        prompts.append(bos + tokens)
        source_rows.append(
            {
                "row_index": rows_read - 1,
                "title": row.get("short_book_title"),
                "url": row.get("url"),
            }
        )
        if len(prompts) == args.num_prompts:
            break
    if not prompts:
        raise ValueError("Dataset yielded no eligible prompts")
    if args.num_prompts is not None and len(prompts) != args.num_prompts:
        raise ValueError(f"Dataset yielded only {len(prompts)} eligible prompts")
    mean_prompt_tokens = sum(map(len, prompts)) / len(prompts)
    request_rate = request_rate_for_token_load(map(len, prompts), args.offered_load)
    arrival_times = (
        poisson_arrival_times(len(prompts), request_rate, args.arrival_seed)
        if args.arrival_pattern == "poisson"
        else [0.0] * len(prompts)
    )
    workload = [
        {
            "request_id": f"prompt-{i}",
            "token_ids": tokens,
            "arrival_time_seconds": arrival_times[i],
        }
        for i, tokens in enumerate(prompts)
    ]
    encoded = json.dumps(workload, indent=2)
    (output / "workload.json").write_text(encoded)
    if args.export_trace:
        plans = make_schedule(prompts, args.chunk_tokens, seed=args.seed)
        (output / "schedule.json").write_text(json.dumps(plans, indent=2))
    manifest = {
        "model": args.model,
        "model_revision": model_revision,
        "model_path": model_path,
        "dataset": args.dataset,
        "dataset_revision": dataset_revision,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "text_column": args.text_column,
        "rows_read": rows_read,
        "source_rows": source_rows,
        "num_prompts": len(prompts),
        "input_tokens": args.input_tokens,
        "length_mode": args.length_mode,
        "chunk_tokens": args.chunk_tokens,
        "seed": args.seed,
        "world_size": 8,
        "arrival_pattern": args.arrival_pattern,
        "mean_prompt_tokens": mean_prompt_tokens,
        "offered_load_input_tokens_per_second": args.offered_load
        if args.arrival_pattern == "poisson"
        else None,
        "arrival_rate_prompts_per_second": request_rate
        if args.arrival_pattern == "poisson"
        else None,
        "arrival_seed": args.arrival_seed,
        "last_arrival_seconds": arrival_times[-1],
        "construction": (
            "One prefix per distinct dataset row; BOS per prompt; "
            + (
                "rows shorter than the requested length are skipped"
                if args.length_mode == "fixed"
                else "natural lengths retained up to input_tokens; empty token sequences skipped"
            )
        ),
        "workload_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
        "scheduling": "online",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    load_description = (
        f"offered_input_tokens/s={args.offered_load:g} "
        f"derived_requests/s={request_rate:g}"
        if args.arrival_pattern == "poisson"
        else "arrival_pattern=burst"
    )
    print(f"[dynamic-sp] {load_description}", flush=True)
    print(
        f"[dynamic-sp] prepared {len(prompts)} prompts; mean_tokens={mean_prompt_tokens:g} "
        f"length_mode={args.length_mode} token_limit={args.input_tokens} "
        f"from {args.dataset}/{args.split}; chunk={args.chunk_tokens}; manifest={output / 'manifest.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
