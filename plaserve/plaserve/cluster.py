"""Launch the official artifact and preserve its provenance and outputs.

The independent scheduler in this package is used by the simulator, not silently
injected into the authors' SGLang runtime. The benchmark executes their code.
"""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys

PROFILES = {
    "h200": {"revision": "main", "script": "run_all_benchmarks.sh",
             "models": ("7b", "14b", "32b", "72b"), "backend": "mooncake"},
    "h100-4": {"revision": "main", "script": "run_all_h100_4gpu.sh",
               "models": ("7b", "14b"), "backend": "nixl"},
    # The published docs do not establish model-selection arguments for this
    # script. Run its full suite instead of inventing unsupported CLI options.
    "a100-2": {"revision": "a100-eval", "script": "run_a100_all.sh",
               "models": (), "backend": "nixl"},
}


def make_plan(profile: str, models: list[str] | None, upstream: Path) -> dict:
    spec = PROFILES[profile]
    models = list(models) if models is not None else ([] if profile == "a100-2" else ["7b"])
    if profile == "a100-2" and models:
        raise ValueError("a100-2 runs the upstream full 0.5B/3B/7B suite; omit --models")
    if any(model not in spec["models"] for model in models) or len(set(models)) != len(models):
        raise ValueError(f"{profile}: models must be unique and drawn from {spec['models']}")
    required_gpus = (4 if profile == "h100-4" else 2 if profile == "a100-2"
                     or models == ["7b"] else 8)
    directory = upstream.resolve() / "bench_laps_prefill_throughput"
    return {"profile": profile, "models": models, "required_gpus": required_gpus,
            "upstream": str(upstream.resolve()), "recommended_revision": spec["revision"],
            "cwd": str(directory), "command": ["bash", str(directory / spec["script"]), *models],
            "backend_default": spec["backend"], "python": sys.executable,
            "scope": "one allocated GPU node; upstream launches prefill/decode/router processes"}


def capture(command: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f"Command failed: {shlex.join(command)}\n{result.stderr.strip() or result.stdout.strip()}")
    return result.stdout.strip()


def inspect_runtime(upstream: Path, required_gpus: int, backend: str) -> dict:
    # Only package metadata and GPU properties; never dump the process environment
    # or authentication tokens into experiment manifests.
    program = """
import importlib.metadata, json, pathlib
import torch, sglang
packages = {}
for name in ('torch', 'sglang', 'sgl-kernel', 'transformers', 'flashinfer-python',
             'mooncake-transfer-engine', 'nixl', 'sglang-router'):
    try:
        packages[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        packages[name] = None
print(json.dumps({'packages': packages, 'cuda_runtime': torch.version.cuda,
                  'sglang_path': str(pathlib.Path(sglang.__file__).resolve()),
                  'gpus': [{'index': i, 'name': torch.cuda.get_device_name(i),
                            'vram_bytes': torch.cuda.get_device_properties(i).total_memory}
                           for i in range(torch.cuda.device_count())]}))
"""
    runtime = json.loads(capture([sys.executable, "-c", program]))
    if len(runtime["gpus"]) < required_gpus:
        raise RuntimeError(f"Need {required_gpus} visible GPUs; detected {len(runtime['gpus'])}. Run inside the allocated node.")
    source = Path(runtime["sglang_path"])
    if not source.is_relative_to(upstream.resolve() / "python"):
        raise RuntimeError("The active Python imports a different SGLang installation. Activate the PLAserve cluster environment.")
    transfer_package = "nixl" if backend == "nixl" else "mooncake-transfer-engine"
    if runtime["packages"].get(transfer_package) is None:
        raise RuntimeError(f"Missing {transfer_package}; run scripts/setup_cluster.sh for this profile")
    return runtime


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stream_benchmark(command: list[str], cwd: Path, environment: dict, log: Path) -> int:
    with log.open("w") as target:
        process = subprocess.Popen(command, cwd=cwd, env=environment, text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            for line in process.stdout:
                target.write(line)
                target.flush()
                print(line, end="", flush=True)
            return process.wait()
        except BaseException:
            # Terminate only the process group launched by this wrapper.
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=10)
            except ProcessLookupError:
                pass
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
        finally:
            process.stdout.close()


def run_cluster(args) -> None:
    from .cli import write_json
    upstream = Path(args.upstream).resolve()
    plan = make_plan(args.profile, args.models, upstream)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    script = Path(plan["command"][1])
    if not script.is_file():
        raise RuntimeError(f"Missing upstream benchmark: {script}. Fetch {plan['recommended_revision']} with scripts/fetch_upstream.sh.")
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Result directory is not empty: {output}. Choose a new --output.")
    if output.is_relative_to(upstream):
        raise ValueError("--output must be outside the upstream checkout")
    backend = os.environ.get("BACKEND", plan["backend_default"])
    if backend not in ("mooncake", "nixl"):
        raise ValueError("BACKEND must be mooncake or nixl")
    runtime = inspect_runtime(upstream, plan["required_gpus"], backend)
    revision = capture(["git", "rev-parse", "HEAD"], upstream)
    dirty = capture(["git", "status", "--porcelain", "--untracked-files=no"], upstream)
    if dirty:
        raise RuntimeError("Upstream tracked files are modified. Commit your changes or use a clean checkout so the manifest identifies the executed code.")
    environment = os.environ.copy()
    environment["PYTHON"] = sys.executable
    environment.setdefault("BACKEND", backend)
    # Prefixes are set only by the caller, allowing cluster-managed caches.
    recorded_env = {key: environment[key] for key in
                    ("CUDA_VISIBLE_DEVICES", "BACKEND", "IB_DEVICE", "NUM_PROMPTS",
                     "PYTHON", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE") if key in environment}
    dataset = upstream / "data" / "lmsys_chat_10k.jsonl"
    if not dataset.is_file():
        raise RuntimeError(f"Missing artifact dataset: {dataset}. See the selected upstream artifact instructions.")
    manifest = {"project": "PLAserve", "execution": "official_laps_sglang", "plan": plan,
                "source": {"url": "https://github.com/Jianshu-She/LAPS", "commit": revision},
                "runtime": runtime, "environment": recorded_env,
                "dataset": {"path": str(dataset), "sha256": checksum(dataset)},
                "started_utc": datetime.now(timezone.utc).isoformat(), "status": "running"}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "manifest.json", manifest)
    before = {path: path.stat().st_mtime_ns for path in upstream.rglob("summary.txt")}
    try:
        code = stream_benchmark(plan["command"], Path(plan["cwd"]), environment, output / "benchmark.log")
        manifest["exit_code"] = code
        changed = [path for path in upstream.rglob("summary.txt")
                   if before.get(path) != path.stat().st_mtime_ns]
        copied = []
        for summary in changed:
            relative = summary.parent.relative_to(upstream)
            destination = output / "artifacts" / relative
            if summary.parent.name.startswith("results"):
                shutil.copytree(summary.parent, destination)
                copied.append(str(destination.relative_to(output)))
            else:
                destination.mkdir(parents=True, exist_ok=True)
                shutil.copy2(summary, destination / summary.name)
                copied.append(str((destination / summary.name).relative_to(output)))
        manifest["artifacts"] = copied
        if code:
            raise RuntimeError(f"Upstream benchmark exited with {code}; inspect {output / 'benchmark.log'}")
        if not copied:
            raise RuntimeError("Benchmark exited successfully but produced no new summary.txt; inspect benchmark.log and upstream result paths")
        manifest["status"] = "completed"
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = str(error)
        raise
    finally:
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "manifest.json", manifest)
    print(f"PLAserve cluster run saved to {output}")
