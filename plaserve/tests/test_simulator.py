import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from plaserve import LatencyModel, Request, SimConfig, simulate
from plaserve.controller import PressureController, Signal
from plaserve.gpu import gpu_config
from plaserve.model import GraphCatalog
from plaserve.scheduler import POLICIES, Scheduler
from plaserve.workload import generate, load_trace, save_trace


class ModelTests(unittest.TestCase):
    def test_compute_memory_crossing(self):
        model = LatencyModel()
        for history in (0, 1, 512, 1000000):
            length = model.boundary(history)
            compute = model.alpha * length * (length + 2 * history) + model.beta * length
            memory = model.gamma_write * length + model.gamma_read * history
            self.assertAlmostEqual(compute, memory, places=10)
        self.assertAlmostEqual(model.boundary(10**12), model.gamma_read / (2 * model.alpha), places=5)

    def test_graph_includes_history_and_dummy_lanes(self):
        catalog = GraphCatalog(SimConfig(kv_token_budget=250))
        shape = catalog.nearest([17, 24, 28], [1, 2, 3])
        self.assertEqual((shape.length, shape.depth, shape.kv_tokens([1, 2, 3])), (32, 4, 134))
        self.assertIsNone(catalog.nearest([128, 128], [1, 1]))

    def test_invalid_input_fails_early(self):
        for kwargs in ({"arrival": math.nan}, {"tokens": 0}, {"tokens": True},
                       {"history": -1}, {"deadline": -1}):
            valid = {"id": "r", "arrival": 0, "tokens": 1}
            valid.update(kwargs)
            with self.assertRaises(ValueError):
                Request(**valid)
        for kwargs in ({"workers": 0}, {"wait_min": 1, "wait_max": 0},
                       {"graph_depths": (2, 1)}, {"workers": 3, "min_pool": 2}):
            with self.assertRaises(ValueError):
                SimConfig(**kwargs)
        with self.assertRaises(ValueError):
            simulate([Request("x", 0, 257)], SimConfig(kv_token_budget=256))
        with self.assertRaises(ValueError):
            simulate([Request("x", 0, 1), Request("x", 1, 1)])


class SchedulingTests(unittest.TestCase):
    def test_all_policies_complete_each_request_once_and_respect_capacity(self):
        requests = generate(200, rate=4000, seed=9)
        for workers in (1, 4):
            for policy in POLICIES:
                with self.subTest(workers=workers, policy=policy):
                    config = SimConfig(workers=workers, kv_token_budget=8192)
                    report = simulate(requests, config, policy=policy)
                    self.assertEqual({r["id"] for r in report["requests"]}, {r.id for r in requests})
                    self.assertEqual(len(report["requests"]), len(requests))
                    processed = {}
                    by_worker = {}
                    for batch in report["batches"]:
                        self.assertLessEqual(batch["kv_tokens"], config.kv_token_budget)
                        self.assertGreater(batch["finish"], batch["start"])
                        by_worker.setdefault(batch["worker"], []).append(batch)
                        for rid, length in zip(batch["ids"], batch["lengths"]):
                            processed[rid] = processed.get(rid, 0) + length
                        if policy in ("disaggregation", "full"):
                            self.assertEqual(len(set(batch["classes"])), 1)
                    self.assertEqual(processed, {r.id: r.tokens for r in requests})
                    for batches in by_worker.values():
                        for previous, following in zip(batches, batches[1:]):
                            self.assertGreaterEqual(following["start"], previous["finish"])

    def test_wait_window_collects_same_length_bucket(self):
        requests = [Request("a", 0, 24), Request("b", 0.001, 32)]
        config = SimConfig(max_batch=2, graph_depths=(1, 2), wait_max=0.005)
        full = simulate(requests, config)
        self.assertEqual(full["batches"][0]["ids"], ["a", "b"])
        self.assertAlmostEqual(full["batches"][0]["start"], 0.001)
        self.assertEqual(full["batches"][0]["shape"], {"length": 32, "depth": 2})
        eager = simulate(requests, config, policy="disaggregation")
        self.assertEqual(eager["batches"][0]["start"], 0)

    def test_deadline_bypasses_minimum_wait(self):
        request = Request("urgent", 0, 8, deadline=0.0001)
        config = SimConfig(wait_min=0.001, wait_max=0.01, slack=0.0002)
        report = simulate([request], config)
        self.assertEqual(report["batches"][0]["start"], 0)
        self.assertFalse(report["requests"][0]["slo_violation"])

    def test_urgent_bucket_precedes_aged_deadline_free_bucket(self):
        scheduler = Scheduler(SimConfig(max_queue_wait=0.01), LatencyModel(), "full")
        scheduler.submit(Request("aged", 0, 128))
        scheduler.submit(Request("urgent", 0.02, 8, deadline=0.0201))
        batch, _ = scheduler.next_batch(0, 0.02)
        self.assertEqual(batch.work[0].request.id, "urgent")

    def test_graph_padding_can_fall_back_without_exceeding_memory(self):
        report = simulate([Request("tight", 0, 25, history=230)],
                          SimConfig(kv_token_budget=256))
        self.assertIsNone(report["batches"][0]["shape"])
        self.assertEqual(report["batches"][0]["kv_tokens"], 255)

    def test_awd_reduces_depth_after_underfill_and_clips_window(self):
        scheduler = Scheduler(SimConfig(max_batch=8), LatencyModel(), "full")
        scheduler.submit(Request("a", 0, 16))
        batch, wake = scheduler.next_batch(0, 0)
        self.assertIsNone(batch)
        self.assertEqual(wake, 0.005)
        batch, _ = scheduler.next_batch(0, wake)
        self.assertEqual(scheduler.workers[0].adaptive[16].depth, 1)
        scheduler.finish(batch, wake + batch.estimated_service)
        scheduler.submit(Request("b", 0.01, 16))
        batch, _ = scheduler.next_batch(0, 0.01)
        self.assertIsNotNone(batch)
        self.assertEqual(scheduler.workers[0].adaptive[16].window, scheduler.config.wait_min)

    def test_long_chunks_keep_class_and_advance_cached_history(self):
        report = simulate([Request("long", 0, 600, 100)],
                          SimConfig(long_chunk=256), policy="full")
        self.assertEqual([b["lengths"] for b in report["batches"]], [[256], [256], [88]])
        self.assertEqual([b["histories"] for b in report["batches"]], [[100], [356], [612]])
        self.assertTrue(all(b["kind"] == "long" for b in report["batches"]))
        self.assertTrue(all(b["shape"] is None for b in report["batches"]))

    def test_short_deadline_interrupts_long_between_chunks(self):
        config = SimConfig(long_chunk=256, wait_max=0.005)
        report = simulate([Request("long", 0, 2048),
                           Request("short", 0.0001, 8, deadline=0.002)], config)
        self.assertEqual(report["batches"][0]["ids"], ["long"])
        self.assertEqual(report["batches"][1]["ids"], ["short"])
        self.assertFalse(next(r for r in report["requests"] if r["id"] == "short")["slo_violation"])

    def test_offline_workload_flushes_and_reports_no_slo(self):
        report = simulate(generate(50, seed=1, slo=None), SimConfig(workers=4))
        self.assertEqual(report["metrics"]["requests"], 50)
        self.assertIsNone(report["metrics"]["slo_violation_rate"])
        self.assertTrue(all(r["deadline"] is None for r in report["requests"]))

    def test_single_busy_worker_applies_migration_only_after_completion(self):
        config = SimConfig(workers=4, cooldown=0)
        scheduler = Scheduler(config, LatencyModel(), "full")
        for i in range(3):
            scheduler.submit(Request(f"long{i}", 0, 2048))
        active = [scheduler.next_batch(i, 0)[0] for i in (2, 3)]
        for i in range(30):
            scheduler.submit(Request(f"short{i}", 0.001, 16))
        scheduler.rebalance(0.001)
        pending = next(w for w in scheduler.workers if w.pending_kind)
        self.assertEqual(pending.kind, "long")
        self.assertEqual(pending.pending_kind, "short")
        scheduler.finish(next(b for b in active if b.worker == pending.id), 0.002)
        self.assertEqual(pending.kind, "short")
        self.assertIsNone(pending.pending_kind)
        self.assertEqual(scheduler.migrations[-1]["event"], "applied")

    def test_eager_padding_budget_limits_mixed_batch(self):
        config = SimConfig(max_batch=4, kv_token_budget=1024, eager_padding=True)
        report = simulate([Request("a", 0, 512), Request("b", 0, 8), Request("c", 0, 8)],
                          config, policy="fcfs")
        self.assertEqual(len(report["batches"][0]["ids"]), 2)
        self.assertEqual(report["batches"][0]["kv_tokens"], 1024)

    def test_empty_and_deterministic(self):
        self.assertEqual(simulate([])["metrics"]["requests"], 0)
        trace = generate(100, rate=10000)
        self.assertEqual(simulate(trace), simulate(trace))


class ControllerTests(unittest.TestCase):
    def test_one_move_cooldown_and_minimum_pool(self):
        c = PressureController(SimConfig(workers=4, cooldown=1))
        signals = [Signal(0, "short", 10, 0, 0), Signal(1, "long", 0, 0, 0),
                   Signal(2, "long", 0, 0, 0), Signal(3, "long", 0, 0, 0)]
        self.assertEqual(c.decide(0, signals), (1, "short"))
        self.assertIsNone(c.decide(0.5, signals))
        self.assertEqual(c.decide(1, signals), (1, "short"))
        minimum = [Signal(0, "short", 10, 0, 0), Signal(1, "long", 0, 0, 0)]
        self.assertIsNone(c.decide(2, minimum))

    def test_equal_negative_pressure_does_not_oscillate(self):
        c = PressureController(SimConfig(workers=4))
        signals = [Signal(i, "short" if i < 2 else "long", 0, 0, 1) for i in range(4)]
        self.assertIsNone(c.decide(0, signals))


class InterfaceTests(unittest.TestCase):
    def test_jsonl_round_trip_and_seed(self):
        trace = generate(20, seed=42)
        self.assertEqual(trace, generate(20, seed=42))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            save_trace(trace, path)
            self.assertEqual(load_trace(path), trace)
            path.write_text('{"tokens": 0}\n')
            with self.assertRaisesRegex(ValueError, ":1:"):
                load_trace(path)

    def test_gpu_dry_run_needs_no_gpu_dependencies(self):
        result = subprocess.run([sys.executable, "-m", "plaserve", "gpu", "--dry-run"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        plan = json.loads(result.stdout)
        self.assertEqual(plan["config"]["workers"], 1)
        self.assertEqual(plan["model"], "Qwen/Qwen2.5-0.5B-Instruct")

    def test_gpu_rejects_cached_history_instead_of_recomputing_silently(self):
        result = subprocess.run([sys.executable, "-m", "plaserve", "gpu", "--dry-run",
                                 "--trace", "examples/mixed.jsonl"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn("history=0", result.stderr)

    def test_gpu_configuration_rejects_chunking_without_kv(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text('{"workers":1,"eager_padding":true,"long_chunk":8}')
            with self.assertRaisesRegex(ValueError, "persistent KV"):
                gpu_config(str(path))


if __name__ == "__main__":
    unittest.main()
