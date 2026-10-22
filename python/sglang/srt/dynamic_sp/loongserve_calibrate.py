# Copyright 2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""Independent synthetic SIB measurements, all SP degrees and actual batches."""

import argparse
import json
import os
import socket
import statistics
import time
from pathlib import Path
from datetime import timedelta
import torch
import torch.distributed as dist
from sglang.srt.dynamic_sp.communication import GroupCommunication, rendezvous_store
from sglang.srt.dynamic_sp.engine import DynamicSPPrefillEngine
from sglang.srt.dynamic_sp.kv_storage import NativeKVStorage
from sglang.srt.dynamic_sp.llama import load_native_llama
from sglang.srt.dynamic_sp.plan import PrefillPlan, Prompt, PromptStep


def calibration_points(lengths, batch_sizes, capacity, context, degree, grid):
    """Six fitting shapes and two independent checks, without Cartesian expansion."""
    limit = degree * capacity
    if lengths is None:
        lengths = [degree * n for n in (128, 512, 2048, 8192, 32768, 65536)]
    values = sorted({n for n in lengths if n <= min(context, limit)})
    if len(values) < 4:
        raise ValueError(
            f"SP={degree}: supply at least four feasible calibration lengths"
        )
    if grid == "full":
        shapes = {tuple([n] * b) for n in values for b in batch_sizes}
        shapes |= {
            tuple(max(1, n // (i + 1)) for i in range(b))
            for n in values
            for b in batch_sizes
        }
        held_out = {(values[-1],), (max(1, values[-1] // 2), max(1, values[-1] // 4))}
        return [
            (shape, "validation" if shape in held_out else "fit")
            for shape in sorted(shapes | held_out)
            if sum(shape) <= limit
        ]
    training = values[:-1]
    indices = {0, len(training) // 2, max(0, len(training) - 2), len(training) - 1}
    anchors = [training[i] for i in sorted(indices)]
    shapes = {(n,) for n in anchors}
    base = anchors[-2] if len(anchors) > 1 else anchors[0]
    shapes |= {(max(1, base // 2), max(1, base // 2)), (base, max(1, base // 2))}
    fit = [(shape, "fit") for shape in sorted(shapes) if sum(shape) <= limit]
    longest = values[-1]
    checks = {(longest,), (max(1, longest // 2), max(1, longest // 4))} - shapes
    return fit + [
        (shape, "validation") for shape in sorted(checks) if sum(shape) <= limit
    ]


def assign_points(points, groups):
    """Longest estimated work first, without evaluation data or fitted timings."""
    assignments = [[] for _ in range(groups)]
    loads = [0] * groups
    for index in sorted(
        range(len(points)),
        key=lambda i: sum(n * n + 8192 * n for n in points[i][0]),
        reverse=True,
    ):
        group = min(range(groups), key=lambda g: loads[g])
        assignments[group].append(index)
        loads[group] += sum(n * n + 8192 * n for n in points[index][0])
    return assignments


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32")
    )
    p.add_argument(
        "--attention-backend", default="ring", choices=("ring", "gathered", "streaming")
    )
    p.add_argument(
        "--kv-transport", default="targeted", choices=("targeted", "all_gather")
    )
    p.add_argument("--shared-attention", default="auto", choices=("auto", "sdpa"))
    p.add_argument(
        "--migration-overlap", action=argparse.BooleanOptionalAction, default=True
    )
    p.add_argument("--max-local-tokens", type=int, default=131072)
    p.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=None,
        help="Absolute prompt lengths for every degree; default scales base lengths by SP degree",
    )
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--concurrency", choices=("parallel", "serial"), default="parallel")
    p.add_argument("--grid", choices=("lean", "full"), default="lean")
    p.add_argument("--slow-point-seconds", type=float, default=1.0)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeats", type=int, default=3)
    args = p.parse_args()
    if (
        min(
            *(args.lengths or (1,)),
            *args.batch_sizes,
            args.warmup,
            args.repeats,
            args.max_local_tokens,
        )
        <= 0
    ):
        p.error("Calibration dimensions must be positive")
    if not 0 < args.slow_point_seconds < float("inf"):
        p.error("slow-point-seconds must be finite and positive")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    dist.init_process_group("nccl", timeout=timedelta(minutes=30))
    control = dist.new_group(backend="gloo", timeout=timedelta(minutes=30))
    try:
        world, global_rank = dist.get_world_size(), dist.get_rank()
        model = load_native_llama(args.model_path, dtype=args.dtype, local_rank=rank)
        by_degree = {
            degree: calibration_points(
                args.lengths,
                args.batch_sizes,
                args.max_local_tokens,
                model.context_length,
                degree,
                args.grid,
            )
            for degree in range(1, world + 1)
        }
        total_points = sum(map(len, by_degree.values()))
        completed_points = 0
        calibration_started = time.perf_counter()
        pool = NativeKVStorage(
            model,
            args.max_local_tokens,
            world,
            args.attention_backend,
            execution_tokens=min(
                args.max_local_tokens,
                max(sum(shape) for points in by_degree.values() for shape, _ in points),
            ),
        )
        hosts, names = [None] * world, [None] * world
        dist.all_gather_object(hosts, socket.gethostname(), group=control)
        dist.all_gather_object(names, torch.cuda.get_device_name(rank), group=control)
        nodes = [list(dict.fromkeys(hosts)).index(host) for host in hosts]
        metadata = {
            "model_path": os.path.realpath(args.model_path),
            "dtype": args.dtype,
            "gpu_names": names,
            "world_size": world,
            "execution_mode": "async",
            "attention_backend": args.attention_backend,
            "kv_transport": args.kv_transport,
            "shared_attention": args.shared_attention,
            "migration_overlap": args.migration_overlap,
            "kv_storage": "native_sglang_slot_pool",
            "measurement": "full_batch_to_first_tokens",
            "grid": args.grid,
            "length_policy": "explicit_absolute" if args.lengths else "sp_scaled",
            "lengths": args.lengths,
            "base_lengths": None
            if args.lengths
            else [128, 512, 2048, 8192, 32768, 65536],
            "calibration_concurrency": args.concurrency,
            "assignment": "longest_estimated_work_first",
            "slow_point_seconds": args.slow_point_seconds,
            "warmup": args.warmup,
            "repeats": args.repeats,
        }
        store = rendezvous_store(control, 1800)
        samples, skipped = [], []

        def save_samples():
            target = Path(args.output)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(
                    {
                        "metadata": metadata,
                        "nodes": nodes,
                        "samples": samples,
                        "skipped": skipped,
                    },
                    indent=2,
                )
                + "\n"
            )
            temporary.replace(target)

        for degree in range(1, world + 1):
            points = by_degree[degree]
            group_count = (
                min(len(points), world // degree)
                if args.concurrency == "parallel"
                else 1
            )
            assignments = assign_points(points, group_count)
            group_index = global_rank // degree
            active = group_index < group_count
            members = tuple(range(group_index * degree, (group_index + 1) * degree))
            local_samples = []
            comm = None
            if global_rank == 0:
                print(
                    f"[dynamic-sp] calibration SP={degree} groups={group_count} points={len(points)} completed={completed_points}/{total_points}",
                    flush=True,
                )
            if active:
                comm = GroupCommunication(store, members, model.device, 1800)
                engine = DynamicSPPrefillEngine(
                    model,
                    control_group=None,
                    communication=comm,
                    kv_pool=pool,
                    max_local_tokens=args.max_local_tokens,
                    attention_backend=args.attention_backend,
                    kv_transport=args.kv_transport,
                    shared_attention=args.shared_attention,
                    migration_overlap=args.migration_overlap,
                    logits_mode="completed",
                )
            # Only degree boundaries synchronize the entire worker set. Each
            # disjoint group advances through its own points and repetitions.
            for point_index in assignments[group_index] if active else ():
                lengths, role = points[point_index]
                label = f"SP={degree} group={group_index} point={point_index + 1}/{len(points)}"
                prompts = tuple(
                    Prompt(
                        str(i), tuple((17 * j + 1) % model.vocab_size for j in range(n))
                    )
                    for i, n in enumerate(lengths)
                )
                steps, offset = [], 0
                # Plans use group-local ranks, including groups not containing rank 0.
                ranks = tuple(range(degree))
                for prompt, n in zip(prompts, lengths):
                    shift = offset % degree
                    steps.append(
                        PromptStep(prompt.request_id, ranks[shift:] + ranks[:shift], n)
                    )
                    offset += n
                durations = []
                for repeat in range(args.warmup + args.repeats):
                    if comm.rank == 0:
                        phase = (
                            "warmup"
                            if repeat < args.warmup
                            else f"measurement {repeat - args.warmup + 1}/{args.repeats} (maximum)"
                        )
                        print(
                            f"[dynamic-sp] calibration {label} {role} lengths={lengths} {phase}",
                            flush=True,
                        )
                    torch.cuda.synchronize()
                    comm.control.barrier().wait()
                    start = time.perf_counter()
                    result = engine.execute(
                        PrefillPlan(engine.version, tuple(steps), prompts)
                        if engine.rank == 0
                        else None
                    )

                    def sample():
                        if engine.rank == 0:
                            for value in result.values():
                                value.argmax().item()

                    engine._checked("sample first tokens", sample)
                    engine._agree(engine.version, "Calibration version mismatch")
                    elapsed = time.perf_counter() - start
                    # Aggregation and cleanup are outside the measured interval.
                    elapsed = max(comm.objects(elapsed))
                    if repeat >= args.warmup:
                        durations.append(elapsed)
                    engine.execute(
                        PrefillPlan(
                            engine.version, release=tuple(p.request_id for p in prompts)
                        )
                        if engine.rank == 0
                        else None
                    )
                    if repeat >= args.warmup and elapsed >= args.slow_point_seconds:
                        break
                pool.assert_empty()
                if comm.rank == 0:
                    record = {
                        "point_index": point_index,
                        "role": role,
                        "degree": degree,
                        "lengths": lengths,
                        "seconds": statistics.median(durations),
                        "repetitions_seconds": durations,
                        "worker_ranks": members,
                        "concurrent_groups": group_count,
                    }
                    local_samples.append(record)
                    # Preserve completed points even if a later group fails.
                    shard = Path(args.output).with_suffix(
                        f".sp{degree}.group{group_index}.json"
                    )
                    shard.parent.mkdir(parents=True, exist_ok=True)
                    temporary = shard.with_suffix(".tmp")
                    temporary.write_text(json.dumps(local_samples, indent=2) + "\n")
                    temporary.replace(shard)
                    print(
                        f"[dynamic-sp] calibration {label} seconds={record['seconds']:.6f} measurements={len(durations)} elapsed={time.perf_counter() - calibration_started:.1f}s",
                        flush=True,
                    )
            if comm is not None:
                comm.close()
            gathered = [None] * world
            dist.all_gather_object(gathered, local_samples, group=control)
            degree_samples = sorted(
                (sample for records in gathered for sample in records),
                key=lambda sample: sample["point_index"],
            )
            if [sample["point_index"] for sample in degree_samples] != list(
                range(len(points))
            ):
                raise RuntimeError(
                    f"SP={degree}: incomplete or duplicate calibration samples"
                )
            samples.extend(degree_samples)
            completed_points += len(degree_samples)
            if global_rank == 0:
                save_samples()
        if global_rank == 0:
            save_samples()  # Include capacity exclusions after the final measured point.
        pool.assert_empty()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
