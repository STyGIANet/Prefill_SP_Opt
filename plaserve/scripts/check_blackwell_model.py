"""Full-model eager/graph greedy-token regression check. Run on advent.

Uses two sequential, ordinary (non-disaggregated) servers on the same GPUs.
This tests full-model generation and cache reuse, not PD transport or logit
equivalence. Replay instrumentation is mandatory; logprob requests are avoided
because they disable this checkout's batch-prefill graph path.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import Request, urlopen


def api(port, route, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = Request(f"http://127.0.0.1:{port}/{route}", data=data,
                  headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=180) as response:
        body = response.read()
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return body.decode()


def cases():
    rng = random.Random(1701)
    result = []
    # Unique first tokens prevent accidental prefix hits in cold cases.
    for i, lengths in enumerate([
        [7, 15], [16, 17, 31], [32, 33, 63, 64],
        [65, 127, 128, 129, 255], [256, 256],
        [9] * 7, [20] * 8, [257, 300], [1],
    ]):
        rows = [[1000 + i * 16 + j] + [rng.randrange(100, 9000)
                for _ in range(n - 1)] for j, n in enumerate(lengths)]
        result.append({"name": f"cold-{i}", "rows": rows, "prime": None})
    for i, (history, suffixes) in enumerate([
        (32, [3, 7, 9]), (128, [15, 17, 31, 33, 63]),
        (256, [1, 16, 32, 64, 128, 129, 255]),
    ]):
        prefix = [3000 + i] + [rng.randrange(100, 9000) for _ in range(history - 1)]
        rows = [prefix + [5000 + j] + [rng.randrange(100, 9000)
                for _ in range(n - 1)] for j, n in enumerate(suffixes)]
        result.append({"name": f"cached-{i}", "rows": rows, "prime": prefix})
    return result


def replay_control_env(env, enabled, mode, output):
    # Candidate only, including when the parent shell contains a stale setting.
    env.pop("LAPS_REPLAY_CONTROL_DIR", None)
    if enabled and mode == "graphs":
        hook = Path(__file__).resolve().parent / "laps_kernel_hook"
        env["LAPS_REPLAY_CONTROL_DIR"] = str((output / "replay-control").resolve())
        env["LAPS_DIAGNOSTIC_CASE_FILE"] = str((output / "graphs-active-case.json").resolve())
        env["PYTHONPATH"] = str(hook) + os.pathsep + env.get("PYTHONPATH", "")


@contextmanager
def server(args, mode, output):
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", args.port))  # Fail before spawning if occupied.
    command = [sys.executable, "-m", "sglang.launch_server",
               "--model-path", args.model, "--tp-size", str(args.tp),
               "--attention-backend", "flashinfer", "--page-size", "1",
               "--host", "127.0.0.1", "--port", str(args.port),
               "--mem-fraction-static", "0.75", "--random-seed", "1701",
               "--enable-laps-scheduler", "--laps-length-threshold", "256"]
    if mode == "eager":
        command += ["--disable-cuda-graph"]
    elif mode == "piecewise":
        command += ["--enable-piecewise-cuda-graph"]
    elif mode == "graphs":
        command += ["--enable-piecewise-cuda-graph", "--enable-batch-prefill-cuda-graph"]
    command += args.server_arg
    (output / f"{mode}-command.json").write_text(json.dumps(command, indent=2))
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpus
    if args.dump_logits:
        if not args.kernel_root:
            raise ValueError("--dump-logits requires --kernel-root for the worker hook")
        env["LAPS_LOGITS_DIR"] = str((output / f"{mode}-logits").resolve())
        env["LAPS_DIAGNOSTIC_CASE_FILE"] = str((output / f"{mode}-active-case.json").resolve())
    if args.kernel_root:
        root = Path(args.kernel_root).resolve()
        hook = Path(__file__).resolve().parent / "laps_kernel_hook"
        if not (hook / "sitecustomize.py").is_file():
            raise RuntimeError("Missing scripts/laps_kernel_hook/sitecustomize.py")
        env["LAPS_FLASHINFER_KERNEL_ROOT"] = str(root)
        env["LAPS_KERNEL_EVENT_DIR"] = str((output / f"{mode}-kernel-events").resolve())
        env["PYTHONPATH"] = str(hook) + os.pathsep + env.get("PYTHONPATH", "")
        env.setdefault("MAX_JOBS", "4")
    replay_control_env(env, args.replay_control, mode, output)
    with (output / f"{mode}-server.log").open("w") as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                 env=env, start_new_session=True)
        try:
            end = time.monotonic() + args.startup_timeout
            while True:
                if child.poll() is not None:
                    raise RuntimeError(f"{mode} server exited {child.returncode}; inspect its log")
                try:
                    api(args.port, "health")
                    break
                except (URLError, TimeoutError):
                    if time.monotonic() >= end:
                        raise RuntimeError(f"{mode} server readiness timed out")
                    time.sleep(1)
            yield
        finally:
            # Signal only this invocation's process group, never global matches.
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            # Clean workers that outlived the server parent, within its group.
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def generate(port, rows, count):
    response = api(port, "generate", {
        "input_ids": rows,
        "sampling_params": {"temperature": 0, "max_new_tokens": count,
                            "ignore_eos": True},
        "return_logprob": False,
    })
    if not isinstance(response, list) or len(response) != len(rows):
        raise RuntimeError(f"Unexpected batched response: {str(response)[:500]}")
    for item in response:
        ids = item.get("output_ids")
        if not isinstance(ids, list) or len(ids) != count:
            raise RuntimeError("Missing or incomplete output_ids; preserve response and inspect API")
    return response


def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    workload = cases()
    if args.cases:
        selected = args.cases.split(",")
        if set(selected) - {case["name"] for case in workload}:
            raise ValueError("Unknown diagnostic case name")
        workload = [case for case in workload if case["name"] in selected]
    encoded = json.dumps(workload, indent=2)
    (output / "inputs.json").write_text(encoded)
    manifest = {"model": args.model, "gpus": args.gpus, "tp": args.tp,
                "baseline": args.baseline, "candidate": "graphs",
                "replay_control": args.replay_control,
                "inputs_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
                "status": "running", "scope": "greedy token equivalence, not logits or PD"}
    if args.kernel_root:
        root = Path(args.kernel_root).resolve()
        kernel_manifest = json.loads((root / "all-manifest.json").read_text())
        if kernel_manifest["status"] != "passed" or kernel_manifest["mode"] != "saturating":
            raise RuntimeError("Run the complete corrected attention suite before full-model validation")
        manifest["kernel_root"] = str(root)
        manifest["kernel_validation"] = kernel_manifest
    all_results = {}
    try:
        base_port = args.port
        for offset, mode in enumerate((args.baseline, "graphs")):
            args.port = base_port + offset
            results = []
            all_results[mode] = results
            with server(args, mode, output):
                for count in args.counts:
                    for case in workload:
                        label_path = output / f"{mode}-active-case.json"
                        label_path.write_text(json.dumps({"phase": "prime"}))
                        api(args.port, "flush_cache", {})
                        prime = None
                        if case["prime"] is not None:
                            prime = generate(args.port, [case["prime"]], 1)
                        label_path.write_text(json.dumps({"phase": "generate", "case": case["name"], "count": count}))
                        response = generate(args.port, case["rows"], count)
                        results.append({"case": case["name"], "count": count,
                                        "prime_response": prime, "response": response})
                        (output / f"{mode}-responses.json").write_text(json.dumps(results, indent=2))
                        print(f"{mode}: {case['name']} tokens={count}", flush=True)
            print(f"{mode} server stopped", flush=True)
            if args.kernel_root:
                loaded_pids = set()
                for path in (output / f"{mode}-kernel-events").glob("*.jsonl"):
                    for line in path.read_text().splitlines():
                        event = json.loads(line)
                        if event["event"] == "kernel_loaded":
                            loaded_pids.add(event["pid"])
                manifest.setdefault("kernel_worker_pids", {})[mode] = sorted(loaded_pids)
                if len(loaded_pids) < args.tp:
                    raise RuntimeError(f"{mode}: corrected kernel loading not verified in {args.tp} workers")
        if args.replay_control:
            from summarize_replay_control import summarize
            summary = summarize(output / "replay-control")
            (output / "replay-control-summary.json").write_text(json.dumps(summary, indent=2))
            if summary["errors"] or len(summary["worker_pids"]) < args.tp:
                directory = output / "replay-control"
                directory.mkdir(exist_ok=True)
                with (directory / "harness.jsonl").open("a") as stream:
                    stream.write(json.dumps({"event": "error", "error":
                        "Cannot verify replay control entry point/comparisons in all TP workers"}) + "\n")
                raise RuntimeError("Replay control missing worker comparisons or logged errors")
        mismatches = []
        for eager, graph in zip(all_results[args.baseline], all_results["graphs"]):
            for i, (a, b) in enumerate(zip(eager["response"], graph["response"])):
                if a["output_ids"] != b["output_ids"]:
                    first = next(j for j, (x, y) in enumerate(zip(a["output_ids"], b["output_ids"])) if x != y)
                    mismatches.append({"case": eager["case"], "count": eager["count"],
                                       "row": i, "first_differing_token": first,
                                       "baseline_mode": args.baseline,
                                       "baseline": a["output_ids"], "graphs": b["output_ids"]})
        (output / "mismatches.json").write_text(json.dumps(mismatches, indent=2))
        traces = []
        for line in (output / "graphs-server.log").read_text().splitlines():
            if "LAPS_VERIFY " in line:
                traces.append(json.loads(line.split("LAPS_VERIFY ", 1)[1]))
        (output / "replay-coverage.json").write_text(json.dumps(traces, indent=2))
        baseline_log = (output / f"{args.baseline}-server.log").read_text()
        if "LAPS_VERIFY " in baseline_log:
            raise RuntimeError("Baseline unexpectedly replayed batch-prefill graphs")
        coverage = {
            "any_replay": bool(traces),
            "padded_batch": any(t["actual_bs"] < t["target_bs"] for t in traces),
            "padded_queries": any(n < t["width"] for t in traces for n in t["extend"]),
            "cached_history": any(n > e for t in traces for n, e in zip(t["kv"], t["extend"])),
            "multiple_shapes": len({(t["target_bs"], t["width"]) for t in traces}) > 1,
        }
        cached = [item for item in all_results["graphs"] if item["case"].startswith("cached-")]
        coverage["api_cache_hit"] = any(r.get("meta_info", {}).get("cached_tokens", 0) > 0
                                         for item in cached for r in item["response"])
        manifest.update(coverage=coverage, mismatches=len(mismatches), replay_events=len(traces))
        if mismatches:
            raise RuntimeError(f"{len(mismatches)} token mismatches; inspect mismatches.json")
        if not all(coverage.values()):
            raise RuntimeError(f"Incomplete coverage: {coverage}; token agreement alone is insufficient")
        manifest["status"] = "passed"
        print("PASS full-model greedy tokens and required graph/cache coverage", flush=True)
    except BaseException as error:
        manifest.update(status="failed", error=str(error))
        raise
    finally:
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--gpus", default="2,3")
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--port", type=int, default=30400)
    parser.add_argument("--startup-timeout", type=int, default=600)
    parser.add_argument("--output", required=True)
    parser.add_argument("--kernel-root", help="Validated saturating kernel experiment directory")
    parser.add_argument("--replay-control", action="store_true",
                        help="Compare candidate batch-prefill replay with identical padded eager execution")
    parser.add_argument("--dump-logits", action="store_true", help="Save full prefill logits inside workers")
    parser.add_argument("--cases", help="Comma-separated diagnostic case names; default all")
    parser.add_argument("--counts", type=int, nargs="+", default=[1, 8], choices=[1, 8])
    parser.add_argument("--baseline", choices=["eager", "cuda", "piecewise"], default="eager",
                        help="eager disables CUDA graphs; cuda uses default decode graphs; "
                             "piecewise also enables piecewise compilation but not batch-prefill graphs")
    parser.add_argument("--server-arg", action="append", default=[])
    run(parser.parse_args())
