"""Discrete-event execution with deterministic arrival/completion ordering."""

from dataclasses import asdict
import heapq
import itertools
import math
import statistics

from .controller import percentile
from .model import LatencyModel, Request, SimConfig
from .scheduler import Scheduler, batch_record


def metrics(completed: list[dict], batches: list[dict], workers: int) -> dict:
    if not completed:
        return {"requests": 0, "duration_s": 0.0, "rps": 0.0, "tokens_per_s": 0.0,
                "mean_prefill_ms": 0.0, "p90_prefill_ms": 0.0, "p99_prefill_ms": 0.0,
                "mean_queue_ms": 0.0, "slo_requests": 0, "slo_violation_rate": None,
                "graph_batch_fraction": 0.0, "padding_fraction": 0.0,
                "batches": 0, "utilization": 0.0, "by_class": {}}
    duration = max(r["finish"] for r in completed) - min(r["arrival"] for r in completed)
    latencies = [r["prefill_latency"] for r in completed]
    deadline_reqs = [r for r in completed if r["deadline"] is not None]
    graph_batches = [b for b in batches if b["shape"]]
    padded = sum(b["shape"]["depth"] * b["shape"]["length"] for b in graph_batches)
    useful = sum(sum(b["lengths"]) for b in graph_batches)
    by_class = {}
    for kind in ("short", "long"):
        subset = [r for r in completed if r["kind"] == kind]
        if subset:
            slo = [r for r in subset if r["deadline"] is not None]
            by_class[kind] = {"requests": len(subset),
                              "mean_prefill_ms": statistics.mean(r["prefill_latency"] for r in subset) * 1000,
                              "p90_prefill_ms": percentile([r["prefill_latency"] for r in subset], 0.9) * 1000,
                              "slo_violation_rate": sum(r["slo_violation"] for r in slo) / len(slo) if slo else None}
    return {"requests": len(completed), "duration_s": duration,
            "rps": len(completed) / duration,
            "tokens_per_s": sum(r["tokens"] for r in completed) / duration,
            "mean_prefill_ms": statistics.mean(latencies) * 1000,
            "p90_prefill_ms": percentile(latencies, 0.9) * 1000,
            "p99_prefill_ms": percentile(latencies, 0.99) * 1000,
            "mean_queue_ms": statistics.mean(r["queue_time"] for r in completed) * 1000,
            "slo_requests": len(deadline_reqs),
            "slo_violation_rate": sum(r["slo_violation"] for r in deadline_reqs) / len(deadline_reqs) if deadline_reqs else None,
            "graph_batch_fraction": len(graph_batches) / len(batches) if batches else 0.0,
            "padding_fraction": (padded - useful) / padded if padded else 0.0,
            "batches": len(batches),
            "utilization": sum(b["finish"] - b["start"] for b in batches) / (workers * duration),
            "by_class": by_class}


def simulate(requests: list[Request], config: SimConfig | None = None,
             model: LatencyModel | None = None, policy: str = "full") -> dict:
    config, model = config or SimConfig(), model or LatencyModel()
    scheduler = Scheduler(config, model, policy)
    # Validate the entire trace before partial execution.
    ids = set()
    for r in requests:
        if r.id in ids:
            raise ValueError(f"duplicate request id: {r.id}")
        ids.add(r.id)
        if r.tokens + r.history > config.kv_token_budget:
            raise ValueError(f"request {r.id}: context exceeds per-worker KV token budget")
    events: list[tuple] = []
    counter = itertools.count()

    def push(time, kind, payload=None):
        heapq.heappush(events, (time, next(counter), kind, payload))

    for r in sorted(requests, key=lambda r: (r.arrival, r.id)):
        push(r.arrival, "arrival", r)
    controlled = policy == "full" and config.workers > 1 and bool(requests)
    if controlled:
        push(min(r.arrival for r in requests) + config.control_period, "control")
    batches, wake_times = [], {}
    while events and len(scheduler.completed) < len(requests):
        now = events[0][0]
        control_due = False
        while events and events[0][0] == now:
            _, _, kind, payload = heapq.heappop(events)
            if kind == "arrival":
                scheduler.submit(payload)
            elif kind == "complete":
                scheduler.finish(payload, now)
            elif kind == "control":
                control_due = True
            elif kind == "wake" and wake_times.get(payload) == now:
                del wake_times[payload]
        if len(scheduler.completed) == len(requests):
            break
        if control_due:
            scheduler.rebalance(now)
            push(now + config.control_period, "control")
        for worker in scheduler.workers:
            batch, wake = scheduler.next_batch(worker.id, now)
            if batch:
                finish = now + batch.estimated_service
                batches.append(batch_record(batch, finish))
                push(finish, "complete", batch)
                wake_times.pop(worker.id, None)
            elif math.isfinite(wake) and (worker.id not in wake_times or wake < wake_times[worker.id]):
                if wake <= now:
                    raise RuntimeError("scheduler failed to advance time")
                wake_times[worker.id] = wake
                push(wake, "wake", worker.id)
    if len(scheduler.completed) != len(requests):
        raise RuntimeError("scheduler stalled with unfinished requests")
    completed = sorted(scheduler.completed, key=lambda r: r["id"])
    return {"schema_version": 1, "execution": "simulation", "policy": policy,
            "config": asdict(config), "latency_model": asdict(model),
            "metrics": metrics(completed, batches, config.workers),
            "requests": completed, "batches": batches, "migrations": scheduler.migrations}
