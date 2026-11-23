# Copyright 2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""GPU roofline probe of the production SP RingWorkspace and native KV pool.

Measures one attention layer with the production stream/event dependencies,
packing, indexed local attention and async local checks. No model is loaded.
Compute FLOPs and compulsory memory bytes are analytical, not hardware counters.
"""

# Timing closures are executed synchronously before advancing their loops.
# ruff: noqa: B023

import argparse
import json
import os
import random
import statistics
from datetime import timedelta
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch
import torch.distributed as dist

from .communication import GroupCommunication, rendezvous_store
from .engine import DynamicSPPrefillEngine
from .kv_storage import NativeKVStorage, write_layer
from .ring import RingWorkspace, rotate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--degrees", type=int, nargs="+", default=list(range(1, 9)))
    p.add_argument("--histories", type=int, nargs="+", default=[0, 8192, 32768, 131072])
    p.add_argument(
        "--chunks",
        type=int,
        nargs="+",
        default=[
            128,
            256,
            512,
            768,
            1024,
            1536,
            2048,
            3072,
            4096,
            6144,
            8192,
            12288,
            16384,
        ],
    )
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--warmup", type=int, default=3)
    a = p.parse_args()
    if a.repeats < 1 or a.warmup < 1 or any(d not in range(1, 9) for d in a.degrees):
        p.error("Require positive repeats and degrees from 1 through 8")
    if any(h < 0 or h % 8 for h in a.histories) or any(
        q <= 0 or q % 8 for q in a.chunks
    ):
        p.error(
            "Histories must be nonnegative and chunks positive, both multiples of eight"
        )
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    torch.manual_seed(1234 + rank)
    dist.init_process_group("nccl", timeout=timedelta(minutes=10))
    control = dist.new_group(backend="gloo", timeout=timedelta(minutes=10))
    device = torch.device("cuda", rank)
    world = dist.get_world_size()
    if world != 8:
        raise ValueError("This probe targets eight GPUs")
    output = Path(a.output)
    store = rendezvous_store(control, 600)

    def all_values(value):
        values = [None] * world
        dist.all_gather_object(values, value, group=control)
        return values

    def gpu_time(fn, repeats=None):
        for _ in range(a.warmup):
            fn()
        torch.cuda.synchronize()
        times = []
        for _ in range(repeats or a.repeats):
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start.record()
            fn()
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end) / 1000)
        return times

    try:
        # All GPUs active during hardware probes. Count both copy reads and writes.
        x = torch.randn((4096, 4096), device=device, dtype=torch.bfloat16)
        y = torch.randn_like(x)
        z = torch.empty_like(x)

        def gemm():
            for _ in range(5):
                torch.mm(x, y, out=z)

        gemm_t = gpu_time(gemm)
        compute = 5 * 2 * 4096**3 / statistics.median(gemm_t)
        x = y = z = None
        x = torch.ones(64 * 1024 * 1024, device=device, dtype=torch.float32)
        y = torch.empty_like(x)

        def copy():
            for _ in range(5):
                y.copy_(x)

        copy_t = gpu_time(copy)
        bandwidth = 5 * 2 * x.numel() * x.element_size() / statistics.median(copy_t)
        x = y = None
        raw = {
            "metadata": {
                "gpu_names": all_values(torch.cuda.get_device_name(rank)),
                "torch_version": torch.__version__,
                "dtype": "bfloat16",
                "query_heads": 32,
                "kv_heads": 8,
                "head_dim": 128,
                "scope": "Production RingWorkspace prepare+attention; native SlotKV; async local checks; no model weights",
                "flop_definition": "Useful QK and AV multiply-add FLOPs, causal pairs only; excludes softmax",
                "memory_definition": "Compulsory Q/K/V reads and output writes, not measured DRAM traffic",
                "network_definition": "Bytes sent per rank; simultaneous receive included in measured P2P service time",
                "placement": "Ranks 0..degree-1; different degrees may cross PCIe/NUMA boundaries",
                "repeats": a.repeats,
                "warmup": a.warmup,
                "execution": "production-overlap",
                "aggregation": "Median of per-repetition maximum rank durations",
                "order_seed": 2026,
            },
            "hardware": all_values(
                {"compute_flops_s": compute, "memory_bytes_s": bandwidth}
            ),
            "transport": [],
            "samples": [],
            "correctness": [],
        }
        for degree in a.degrees:
            active = rank < degree
            comm = (
                GroupCommunication(store, tuple(range(degree)), device, 600)
                if active
                else None
            )
            model = SimpleNamespace(
                device=device,
                dtype=torch.bfloat16,
                num_layers=1,
                num_heads=32,
                num_kv_heads=8,
                head_dim=128,
            )
            pool = (
                NativeKVStorage(
                    model,
                    (max(a.histories) + max(a.chunks) + degree - 1) // degree,
                    degree,
                    "ring",
                    reserve_bytes=0,
                    execution_tokens=0,
                )
                if active
                else None
            )
            e = SimpleNamespace(
                communication=comm,
                model=model,
                kv_pool=pool,
                rank=rank,
                shared_attention="auto",
                metrics={"fused_attention_calls": 0},
                _local_checks_only=True,
                _failed=False,
            )
            e._checked = MethodType(DynamicSPPrefillEngine._checked, e)

            def exchange(send, recv):
                for work in rotate(
                    e, send, recv, (rank + 1) % degree, (rank - 1) % degree
                ):
                    work.wait()

            if degree > 1:
                for size in [4096, 65536, 262144, 1048576, 4194304, 16777216, 67108864]:
                    dist.barrier(group=control)
                    times = None
                    if active:
                        send = torch.zeros(
                            size // 2, device=device, dtype=torch.bfloat16
                        )
                        recv = torch.empty_like(send)
                        times = gpu_time(lambda: exchange(send, recv))
                        del send, recv
                    timings = all_values(times)[:degree]
                    record = {
                        "degree": degree,
                        "bytes_per_rank": size,
                        "seconds": statistics.median(
                            max(t[i] for t in timings) for i in range(a.repeats)
                        ),
                        "rank_repetitions": timings,
                    }
                    raw["transport"].append(record)
            grid = [(h, q, False) for h in a.histories for q in a.chunks]
            random.Random(2026).shuffle(grid)
            for history, chunk, check in [(128, 128, True)] + grid:
                dist.barrier(group=control)
                payload = None
                if active:
                    n = len(range(rank, history + chunk, degree))
                    prefix = len(range(rank, history, degree))
                    nq = n - prefix
                    query = torch.randn(
                        nq, 32, 128, device=device, dtype=torch.bfloat16
                    )
                    kv = torch.randn(n, 2, 8, 128, device=device, dtype=torch.bfloat16)
                    pool.begin()
                    slots = pool.allocate(n)
                    write_layer(slots, 0, 0, kv[:, 0], kv[:, 1])
                    state = SimpleNamespace(
                        end=history + chunk, ranks=tuple(range(degree)), kv=slots
                    )
                    layouts = [(state, prefix, 0, nq)]
                    positions = (
                        torch.arange(nq, device=device) * degree
                        + prefix * degree
                        + rank
                    )
                    workspace = RingWorkspace(e, layouts)
                    buffers = workspace.buffers[0]

                    def attention():
                        accumulators = workspace.prepare(0, query, positions, layouts)
                        return workspace.attention(query, layouts, accumulators)

                    if check:
                        result = attention()
                        padded = torch.zeros_like(buffers[0])
                        padded[:n].copy_(kv)
                        shards = [torch.empty_like(padded) for _ in range(degree)]
                        if degree > 1:
                            comm.data.allgather([shards], [padded]).wait()
                        else:
                            shards = [padded]
                        full = torch.stack(shards, dim=1).flatten(0, 1)[
                            : history + chunk
                        ]
                        k = (
                            full[:, 0]
                            .repeat_interleave(4, dim=1)
                            .transpose(0, 1)
                            .float()
                        )
                        v = (
                            full[:, 1]
                            .repeat_interleave(4, dim=1)
                            .transpose(0, 1)
                            .float()
                        )
                        qp = (
                            torch.arange(nq, device=device) * degree
                            + prefix * degree
                            + rank
                        )
                        mask = (
                            qp[:, None]
                            >= torch.arange(history + chunk, device=device)[None, :]
                        )
                        ref = torch.nn.functional.scaled_dot_product_attention(
                            query.transpose(0, 1).float(), k, v, attn_mask=mask
                        ).transpose(0, 1)
                        error = (result.float() - ref).abs().max().item()
                        payload = {
                            "max_abs_error": error,
                            "passed": bool(
                                torch.allclose(
                                    result.float(), ref, atol=0.025, rtol=0.025
                                )
                            ),
                        }
                    else:
                        times = gpu_time(attention)

                        def transport_only():
                            for hop in range(degree - 1):
                                exchange(buffers[hop % 2], buffers[(hop + 1) % 2])

                        net_times = (
                            gpu_time(transport_only)
                            if degree > 1
                            else [0.0] * a.repeats
                        )
                        payload = {
                            "seconds": statistics.median(times),
                            "repetitions_seconds": times,
                            "transport_seconds": statistics.median(net_times),
                            "transport_repetitions_seconds": net_times,
                        }
                    workspace.drain()
                    pool.rollback()
                    del workspace, query, kv, buffers, slots, state, layouts
                values = all_values(payload)[:degree]
                if check:
                    raw["correctness"].append({"degree": degree, "ranks": values})
                    if not all(v["passed"] for v in values):
                        raise RuntimeError(
                            f"Attention reference mismatch for SP={degree}"
                        )
                    continue
                # The local cyclic query positions give slightly unequal causal FLOPs.
                flops = [
                    4
                    * 32
                    * 128
                    * sum(
                        p + 1 for p in range(r, history + chunk, degree) if p >= history
                    )
                    for r in range(degree)
                ]
                compulsory = 2 * (
                    2 * (history + chunk) * 8 * 128 + 2 * (chunk / degree) * 32 * 128
                )
                # The backend sends equally sized padded buffers, even for uneven shards.
                network = (
                    (degree - 1)
                    * ((history + chunk + degree - 1) // degree)
                    * 2
                    * 8
                    * 128
                    * 2
                )
                critical_times = [
                    max(v["repetitions_seconds"][i] for v in values)
                    for i in range(a.repeats)
                ]
                seconds = statistics.median(critical_times)
                record = {
                    "degree": degree,
                    "history": history,
                    "chunk": chunk,
                    "seconds": seconds,
                    "local_flops_mean": sum(flops) / degree,
                    "compulsory_memory_bytes": compulsory,
                    "network_send_bytes": network,
                    "critical_repetitions_seconds": critical_times,
                    "transport_seconds": statistics.median(
                        max(v["transport_repetitions_seconds"][i] for v in values)
                        for i in range(a.repeats)
                    ),
                    "rank_measurements": values,
                }
                raw["samples"].append(record)
                if rank == 0:
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_text(json.dumps(raw, indent=2) + "\n")
                    print(
                        f"[roofline] SP={degree} history={history} chunk={chunk}: {seconds * 1000:.3f} ms, transport={record['transport_seconds'] * 1000:.3f} ms",
                        flush=True,
                    )
            e.kv_pool = None
            pool = None
            if comm is not None:
                comm.close()
            dist.barrier(group=control)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
