# Copyright 2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""Explicit GPU calibration for the Tetris prefill latency model.

Run with torchrun and the same model, precision, transport, and GPU topology as
serving. This command measures isolated chunks; it does not run automatically.
"""

import argparse
import json
import os
import socket
import statistics
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from sglang.srt.dynamic_sp.communication import GroupCommunication, rendezvous_store
from sglang.srt.dynamic_sp.engine import DynamicSPPrefillEngine
from sglang.srt.dynamic_sp.kv_storage import NativeKVStorage
from sglang.srt.dynamic_sp.llama import load_native_llama
from sglang.srt.dynamic_sp.plan import PrefillPlan, Prompt, PromptStep


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--execution-mode", choices=("async", "sync"), default="async")
    parser.add_argument(
        "--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16"
    )
    parser.add_argument("--degrees", type=int, nargs="+", default=None)
    parser.add_argument("--histories", type=int, nargs="+", default=[0, 2048, 8192])
    parser.add_argument(
        "--lengths", type=int, nargs="+", default=[256, 1024, 2048, 8192]
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-local-tokens", type=int, default=65536)
    parser.add_argument(
        "--attention-backend", choices=["ring", "gathered", "streaming"], default="ring"
    )
    parser.add_argument(
        "--kv-transport", choices=["targeted", "all_gather"], default="targeted"
    )
    parser.add_argument("--shared-attention", choices=["auto", "sdpa"], default="auto")
    parser.add_argument(
        "--migration-overlap", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if (
        args.warmup < 1
        or args.repeats < 1
        or min(args.histories) < 0
        or min(args.lengths) <= 0
    ):
        parser.error("Require warmup >= 1, repeats >= 1, histories >= 0, lengths > 0")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.set_num_threads(1)
    dist.init_process_group("nccl", timeout=timedelta(minutes=30))
    control = dist.new_group(backend="gloo", timeout=timedelta(minutes=30))
    try:
        world = dist.get_world_size()
        degrees = args.degrees or [2**i for i in range(world.bit_length())]
        if any(s < 1 or s > world for s in degrees):
            raise ValueError("Degrees must fit world size")
        model = load_native_llama(
            args.model_path, dtype=args.dtype, local_rank=local_rank
        )
        kv_pool = NativeKVStorage(
            model,
            args.max_local_tokens,
            world,
            args.attention_backend,
            execution_tokens=max(*args.lengths, *args.histories),
        )
        engine = DynamicSPPrefillEngine(
            model,
            control_group=control,
            kv_pool=kv_pool,
            max_local_tokens=args.max_local_tokens,
            attention_backend=args.attention_backend,
            kv_transport=args.kv_transport,
            shared_attention=args.shared_attention,
            migration_overlap=args.migration_overlap,
            logits_mode="completed",
        )
        hosts = engine._all_objects(socket.gethostname())
        names = engine._all_objects(torch.cuda.get_device_name(local_rank))
        unique_hosts = list(dict.fromkeys(hosts))
        samples = []
        version = 0
        active_engine = engine
        store = (
            rendezvous_store(control, 1800) if args.execution_mode == "async" else None
        )

        def execute(steps=(), admit=(), release=()):
            nonlocal version
            plan = PrefillPlan(version, steps, admit, release)
            result = (
                active_engine.execute(plan if dist.get_rank() == 0 else None)
                if active_engine is not None
                else {}
            )
            version += 1
            return result

        with torch.inference_mode():
            for degree in sorted(set(degrees)):
                ranks = tuple(range(degree))
                communication = None
                if args.execution_mode == "async":
                    version = 0
                    active_engine = None
                    if dist.get_rank() < degree:
                        communication = GroupCommunication(
                            store, ranks, model.device, 1800
                        )
                        active_engine = DynamicSPPrefillEngine(
                            model,
                            control_group=None,
                            communication=communication,
                            kv_pool=kv_pool,
                            max_local_tokens=args.max_local_tokens,
                            attention_backend=args.attention_backend,
                            kv_transport=args.kv_transport,
                            shared_attention=args.shared_attention,
                            migration_overlap=args.migration_overlap,
                            logits_mode="completed",
                        )
                for history in sorted(set(args.histories)):
                    for length in sorted(set(args.lengths)):
                        total = history + length
                        # Tetris Implementation: short-context checkpoints (e.g. 8K
                        # Meta-Llama-3-8B) skip grid points beyond the model context.
                        if total > model.context_length:
                            continue
                        if (total + degree - 1) // degree > args.max_local_tokens:
                            raise ValueError(
                                f"Calibration point ({degree}, {history}, {length}) exceeds capacity"
                            )
                        times = []
                        prompt = Prompt(
                            "calibration",
                            tuple(
                                (17 * i + 1) % model.vocab_size for i in range(total)
                            ),
                        )
                        for iteration in range(args.warmup + args.repeats):
                            if history:
                                execute(
                                    (PromptStep(prompt.request_id, ranks, history),),
                                    (prompt,),
                                )
                            dist.barrier(group=control)
                            torch.cuda.synchronize()
                            started = time.perf_counter()
                            logits = execute(
                                (PromptStep(prompt.request_id, ranks, total),),
                                () if history else (prompt,),
                            )

                            def sample(logits=logits, prompt=prompt):
                                if dist.get_rank() == 0:
                                    torch.argmax(logits[prompt.request_id]).item()

                            if args.execution_mode == "async":
                                if active_engine is not None:
                                    active_engine._checked("sample and publish", sample)
                                    active_engine._agree(
                                        version, "Calibration versions differ"
                                    )
                            else:
                                sample()
                            elapsed = time.perf_counter() - started
                            durations = engine._all_objects(elapsed)
                            # Nonparticipants do no prefill work. Their host
                            # scheduling delays are not a group service sample.
                            elapsed = max(
                                durations[:degree]
                                if args.execution_mode == "async"
                                else durations
                            )
                            if iteration >= args.warmup:
                                times.append(elapsed)
                            execute(release=(prompt.request_id,))
                        samples.append(
                            {
                                "degree": degree,
                                "history": history,
                                "length": length,
                                "seconds": statistics.median(times),
                                "repetitions_seconds": times,
                            }
                        )
                        if dist.get_rank() == 0:
                            print(
                                f"[dynamic-sp] calibration degree={degree} history={history} length={length} seconds={statistics.median(times):.6f}",
                                flush=True,
                            )

                        def save():
                            if dist.get_rank() == 0:
                                raw = {
                                    "nodes": [
                                        unique_hosts.index(host) for host in hosts
                                    ],
                                    "metadata": {
                                        "kv_storage": "native_sglang_slot_pool",
                                        "model_path": os.path.realpath(args.model_path),
                                        "dtype": args.dtype,
                                        "gpu_names": names,
                                        "world_size": world,
                                        "attention_backend": args.attention_backend,
                                        "kv_transport": args.kv_transport,
                                        "shared_attention": args.shared_attention,
                                        "migration_overlap": args.migration_overlap,
                                        "measurement": "isolated_chunk_to_first_token",
                                        "execution_mode": args.execution_mode,
                                        "warmup": args.warmup,
                                        "repeats": args.repeats,
                                    },
                                    "samples": samples,
                                }
                                target = Path(args.output)
                                target.parent.mkdir(parents=True, exist_ok=True)
                                temp = target.with_suffix(target.suffix + ".tmp")
                                temp.write_text(json.dumps(raw, indent=2) + "\n")
                                temp.replace(target)

                        engine._checked("save calibration", save)
                if communication is not None:
                    communication.close()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
