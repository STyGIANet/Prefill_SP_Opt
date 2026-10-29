"""Run one dynamic_sp experiment by launching `python -m sglang.srt.dynamic_sp`
under torchrun, with the arguments run_dsp_dataset_sweep.sh passes.

The run replays a prepared workload, <workload-root>/<dataset>/<model>/
l<load>_s<seed>/prepared, and writes results.json and run.log to
<root>/<dataset>/<model>/l<load>_s<seed>/<name>, the layout run_experiments.py
report reads. A missing workload, or one with another prompt count than the
preset, is prepared first (CPU only) under <root> from the --dataset preset:
lmsys, wildchat, pg19, pg19_natural, or longbench_v2. Arguments after `--`
go to the runner unchanged. The GPUs are shared: it stops if another user's job
holds a GPU.

--prefill-only runs every scheduler without decode, as launch_dynamic_sp_8gpu.sh
does by default: one output token per request, maxsp at SP over all --gpus,
and results under work/dataset_sweep_prefill (8 GPUs) or
work/dataset_sweep_prefill_<N>gpu. --gpus N uses GPUs 0..N-1 only.

--code runs sglang from another checkout, e.g. a git worktree of an earlier
commit. --scheduler laps runs the native LAPS baseline (laps_baseline) with one
prefill-only replica per GPU, so it has no decode GPUs and no TBT.

  python python/sglang/srt/dynamic_sp/run_dynamic_sp.py --scheduler dynamic --load 8192 \\
      --name dynamic_w1 -- --window-seconds 1
  python python/sglang/srt/dynamic_sp/run_dynamic_sp.py --scheduler tetris --dry-run
  python python/sglang/srt/dynamic_sp/run_dynamic_sp.py --scheduler dynamic --prefill-only
  python python/sglang/srt/dynamic_sp/run_dynamic_sp.py --scheduler dynamic --prefill-only \\
      --gpus 4 --dataset longbench_v2 --load 16384
  python python/sglang/srt/dynamic_sp/run_dynamic_sp.py --scheduler dynamic \\
      --name dynamic_before --code ~/Prefill_SP_Opt_before/python
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# The report and GPU-guard helpers live with the other benchmark scripts.
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "benchmark/cdsp"))
from run_experiments import CDSP, REPO, WORK, other_gpu_jobs

DATA = Path.home() / "data"
MODEL_PATHS = {
    "llama3_8b": DATA / "models/Meta-Llama-3-8B",
    "qwen3_8b": DATA / "models/Qwen3-8B",
}
# prepare_workload options per dataset. Chat sets take the first user turn of
# the test partition (every tenth row), as run_dsp_dataset_sweep.sh does;
# WildChat takes 10000 so about 50 prompts of 4K+ tokens arrive.
# Long-context records are capped; 32K leaves room under Qwen3-8B's 40K context.
PREPARE = {
    "lmsys": [
        "--dataset",
        DATA / "datasets/lmsys_chat_1m.jsonl",
        "--text-column",
        "conversations",
        "--split",
        "test",
        "--num-prompts",
        "1000",
    ],
    "wildchat": [
        "--dataset",
        DATA / "datasets/wildchat_1m.jsonl",
        "--text-column",
        "conversations",
        "--split",
        "test",
        "--num-prompts",
        "10000",
    ],
    "pg19": [
        "--dataset",
        DATA / "datasets/pg19_train0.jsonl",
        "--text-column",
        "text",
        "--split",
        "train",
        "--num-prompts",
        "100",
        "--input-tokens",
        "8192",
    ],
    # Whole books that fit Qwen3-8B's 40960-token context with one output token;
    # longer books are excluded rather than cut, keeping the natural lengths.
    "pg19_natural": [
        "--dataset",
        DATA / "datasets/pg19_train0.jsonl",
        "--text-column",
        "text",
        "--split",
        "train",
        "--num-prompts",
        "all",
        "--max-prompt-tokens",
        "40959",
    ],
    "longbench_v2": [
        "--dataset",
        DATA / "datasets/longbench_v2.jsonl",
        "--text-column",
        "context",
        "--split",
        "train",
        "--num-prompts",
        "100",
        "--input-tokens",
        "32768",
    ],
}


def _profiles(model, prefill):
    """Tetris and LoongServe profiles for the prefill workers."""
    if prefill != 8:
        return (
            WORK / f"pd_profiles/tetris_{model}_p{prefill}.json",
            WORK / f"pd_profiles/loongserve_{model}_p{prefill}.json",
        )
    return (
        WORK / f"tetris_fix_profiles/{model}/profile.json",
        WORK / f"loongserve_{model}_full_v40.json",
    )


def scheduler_args(scheduler, model, prefill):
    """--scheduler value and its options, as the sweep maps scheduler names."""
    tetris, loongserve = _profiles(model, prefill)
    if scheduler == "dp":
        return ["--scheduler", "request", "--sp-sizes", "1"]
    if scheduler == "maxsp":
        return ["--scheduler", "request", "--sp-sizes", str(prefill)]
    if scheduler == "tetris":
        return [
            "--scheduler",
            "tetris",
            "--tetris-profile",
            str(tetris),
            "--tetris-chunk-solver",
            "newton",
        ]
    if scheduler == "loongserve":
        return ["--scheduler", "loongserve", "--loongserve-profile", str(loongserve)]
    if scheduler == "dynamic":
        return ["--scheduler", "dynamic", "--dynamic-profile", str(tetris)]
    return ["--scheduler", scheduler]


def _workload_dir(args):
    run = Path(args.dataset) / args.model / f"l{args.load}_s{args.seed}"
    preset = PREPARE.get(args.dataset)
    if preset is not None and args.num_prompts is not None:
        count = preset.index("--num-prompts") + 1
        preset = [*preset[:count], str(args.num_prompts), *preset[count + 1 :]]
    for root in (args.workload_root, args.root):
        prepared = root / run / "prepared"
        if not (prepared / "workload.json").exists():
            continue
        # Skip workloads prepared with another prompt count than requested.
        manifest = json.loads((prepared / "manifest.json").read_text())
        requested = (
            None if preset is None else preset[preset.index("--num-prompts") + 1]
        )
        if requested in (None, "all") or str(manifest["num_prompts"]) == str(requested):
            return prepared
    if preset is None:
        sys.exit(f"[stop] no prepared workload for {run} and no {args.dataset} preset")
    # Under --root, so earlier workloads in --workload-root are never overwritten.
    prepared = args.root / run / "prepared"
    prepare = [
        sys.executable,
        "-m",
        "sglang.srt.dynamic_sp.prepare_workload",
        "--model",
        str(MODEL_PATHS[args.model]),
        *map(str, preset),
        "--offered-load",
        str(args.load),
        "--arrival-seed",
        str(args.seed),
        "--output-dir",
        str(prepared),
    ]
    if args.dry_run:
        print(f"[dry-run] prepare: {' '.join(prepare)}")
        return prepared
    print(f"[prepare] {prepared}", flush=True)
    subprocess.run(prepare, check=True)
    return prepared


def laps_command(args, output, model_path):
    return [
        sys.executable,
        "-m",
        "sglang.srt.dynamic_sp.laps_baseline",
        "--model-path",
        model_path,
        "--dtype",
        "bfloat16",
        "--workload",
        str(output / "workload.json"),
        "--output",
        str(output / "results.json"),
        "--chunk-size",
        "1024",
        "--max-local-tokens",
        "600000",
        "--max-running-requests",
        "32",
        "--max-batch-tokens",
        "32768",
        "--warmup-requests",
        "8",
        "--warmup-tokens",
        "8192",
        "--progress-throughput-target",
        str(args.load),
        "--laps-config",
        str(args.code / "sglang/srt/dynamic_sp/laps_native.json"),
        "--gpus",
        str(args.gpus),
        *args.runner_args,
    ]


def command(args, output, model_path):
    if args.scheduler == "laps":
        return laps_command(args, output, model_path)
    runner = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={args.gpus}",
        "-m",
        "sglang.srt.dynamic_sp",
        "--model-path",
        model_path,
        "--dtype",
        "bfloat16",
        "--attention-backend",
        "ring",
        "--execution-mode",
        "async",
        "--workload",
        str(output / "workload.json"),
        "--chunk-size",
        "1024",
        "--max-running-requests",
        "32",
        "--max-batch-tokens",
        "32768",
        "--max-local-tokens",
        "131072",
        "--kv-transport",
        "targeted",
        "--shared-attention",
        "auto",
        "--migration-overlap",
        "--output",
        str(output / "results.json"),
        "--warmup-requests",
        "8",
        "--warmup-tokens",
        "8192",
        "--progress-throughput-target",
        str(args.load),
        "--no-verify",
        "--prewarm-groups",
        "all",
        *scheduler_args(args.scheduler, args.model, args.gpus - args.decode_gpus),
    ]
    if args.decode_gpus:
        runner += [
            "--decode-gpus",
            str(args.decode_gpus),
            "--decode-tokens",
            str(args.decode_tokens),
        ]
    return runner + args.runner_args


def run(args):
    name = args.name or args.scheduler
    run_dir = args.root / args.dataset / args.model / f"l{args.load}_s{args.seed}"
    output = run_dir / name
    if (output / "results.json").exists() and not args.rerun:
        sys.exit(f"[stop] {output} already has results; pass --rerun to replace them")
    prepared = _workload_dir(args)
    if args.dry_run and not (prepared / "manifest.json").exists():
        print(" ".join(command(args, output, str(MODEL_PATHS[args.model]))))
        return
    model_path = json.loads((prepared / "manifest.json").read_text())["model_path"]
    if args.dry_run:
        print(f"[dry-run] workload {prepared}")
        print(" ".join(command(args, output, model_path)))
        return
    jobs = other_gpu_jobs()
    if jobs:
        sys.exit(f"[stop] other GPU jobs {jobs}")
    shutil.rmtree(output, ignore_errors=True)
    output.mkdir(parents=True)
    for file in ("workload.json", "manifest.json"):
        shutil.copyfile(prepared / file, output / file)
    env = {
        **os.environ,
        "PYTHONPATH": f"{args.code}:{os.environ.get('PYTHONPATH', '')}",
        "CUDA_VISIBLE_DEVICES": ",".join(map(str, range(args.gpus))),
    }
    if args.scheduler == "tetris":
        subprocess.run(
            [
                sys.executable,
                "-m",
                "sglang.srt.dynamic_sp.tetris_profile",
                "check-data",
                "--profile",
                scheduler_args("tetris", args.model, args.gpus - args.decode_gpus)[3],
                "--evaluation-manifest",
                str(output / "manifest.json"),
            ],
            env=env,
            check=True,
        )
    print(f"[run] {output}; full log: {output / 'run.log'}", flush=True)
    with (output / "run.log").open("w") as log:
        process = subprocess.Popen(
            command(args, output, model_path),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for line in process.stdout:
            log.write(line)
            if args.verbose or line.startswith("[dynamic-sp]"):
                print(line, end="", flush=True)
        returncode = process.wait()
    if returncode or not (output / "results.json").exists():
        sys.exit(f"[stop] run failed (exit {returncode}); see {output / 'run.log'}")
    subprocess.run(
        [
            sys.executable,
            str(CDSP / "summarize_pd_results.py"),
            str(args.root),
            "--datasets",
            args.dataset,
            "--models",
            args.model,
            "--loads",
            str(args.load),
            "--seed",
            str(args.seed),
            "--schedulers",
            name,
        ],
        check=False,
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--scheduler",
        required=True,
        help="dynamic, request, dp, maxsp, tetris, loongserve, laps, greedy, "
        "or module:factory",
    )
    parser.add_argument("--dataset", default="wildchat")
    parser.add_argument(
        "--num-prompts", type=int, help="Prompts to prepare (default: the preset's)"
    )
    parser.add_argument(
        "--model", choices=("qwen3_8b", "llama3_8b"), default="qwen3_8b"
    )
    parser.add_argument("--load", type=int, default=8192, help="Offered input tokens/s")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--name", help="Output directory name (default: the scheduler name)"
    )
    parser.add_argument(
        "--gpus", type=int, default=8, help="Use GPUs 0..N-1 (default: 8)"
    )
    parser.add_argument("--decode-gpus", type=int, default=4)
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument(
        "--prefill-only",
        action="store_true",
        help="No decode GPUs, as launch_dynamic_sp_8gpu.sh runs by default",
    )
    parser.add_argument(
        "--root",
        type=Path,
        help="Results root (default: work/dataset_sweep_pd_tp4, or with "
        "--prefill-only work/dataset_sweep_prefill[_<N>gpu])",
    )
    parser.add_argument(
        "--workload-root",
        type=Path,
        default=WORK / "dataset_sweep_v2",
        help="Prepared workloads; falls back to --root",
    )
    parser.add_argument(
        "--code",
        type=Path,
        default=REPO / "python",
        help="Python root of the sglang checkout to run (default: this repo)",
    )
    parser.add_argument("--rerun", action="store_true", help="Replace existing results")
    parser.add_argument("--dry-run", action="store_true", help="Print the command only")
    parser.add_argument(
        "--verbose", action="store_true", help="Echo the full runner log"
    )
    args, extra = parser.parse_known_args()
    if extra and extra[0] == "--":
        extra = extra[1:]
    elif extra:
        parser.error(f"unrecognized arguments {extra}; pass runner options after --")
    args.runner_args = extra
    if args.prefill_only:
        args.decode_gpus = 0
    if args.root is None:
        if not args.prefill_only:
            args.root = WORK / "dataset_sweep_pd_tp4"
        elif args.gpus == 8:
            args.root = WORK / "dataset_sweep_prefill"
        else:
            args.root = WORK / f"dataset_sweep_prefill_{args.gpus}gpu"
    if args.gpus - args.decode_gpus < 1:
        parser.error("--gpus must exceed --decode-gpus")
    args.code = args.code.expanduser().resolve()
    if not (args.code / "sglang/srt/dynamic_sp").is_dir():
        parser.error(f"--code {args.code} has no sglang/srt/dynamic_sp")
    run(args)


if __name__ == "__main__":
    main()
