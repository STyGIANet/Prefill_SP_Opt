import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


control = load("replay_control", "scripts/laps_kernel_hook/replay_control.py")
summary = load("summarize_replay_control", "scripts/summarize_replay_control.py")
checker = load("check_blackwell_model", "scripts/check_blackwell_model.py")


class ReplayControlTests(unittest.TestCase):
    def test_valid_cached_and_padded_slots(self):
        control.validate_slots([0, 3, 0, 4, 0, 0], [1, 2, 3, 1, 2, 4, 1],
                               [3, 3, 1], [1, 1], 2)

    def test_live_zero_slot_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "invalid real"):
            control.validate_slots([0, 3], [0, 3], [2], [1], 2)

    def test_prefix_write_alias_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "alias"):
            control.validate_slots([0, 3, 0, 4], [1, 3, 3, 4], [2, 2], [1, 1], 2)

    def test_duplicate_suffix_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "alias"):
            control.validate_slots([0, 3, 0, 3], [1, 3, 1, 3], [2, 2], [1, 1], 2)

    def test_padding_and_wrong_suffix_rejected(self):
        for slots in ([2, 3, 0, 0], [0, 4, 0, 0], [0, 3, 0, 8]):
            with self.subTest(slots=slots), self.assertRaises(RuntimeError):
                control.validate_slots(slots, [1, 3, 1], [2, 1], [1], 2)

    def test_candidate_only_environment(self):
        for enabled, mode in [(False, "graphs"), (True, "eager"), (True, "piecewise"), (True, "graphs")]:
            env = {"PYTHONPATH": "existing", "LAPS_REPLAY_CONTROL_DIR": "stale"}
            checker.replay_control_env(env, enabled, mode, Path("/tmp/run"))
            active = enabled and mode == "graphs"
            self.assertEqual("LAPS_REPLAY_CONTROL_DIR" in env, active)
            if active:
                self.assertEqual(env["LAPS_REPLAY_CONTROL_DIR"], "/tmp/run/replay-control")
                self.assertEqual(env["LAPS_DIAGNOSTIC_CASE_FILE"], "/tmp/run/graphs-active-case.json")
                self.assertTrue(env["PYTHONPATH"].endswith(os.pathsep + "existing"))
            else:
                self.assertEqual(env["PYTHONPATH"], "existing")

    def test_cli_flag(self):
        result = subprocess.run([sys.executable, "scripts/check_blackwell_model.py", "--help"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn("--replay-control", result.stdout)

    def test_worker_hook_without_kernel_override_fails_loudly(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            package = base
            for name in ("sglang", "srt", "model_executor"):
                package = package / name
                package.mkdir()
                (package / "__init__.py").write_text("")
            (package / "piecewise_cuda_graph_runner.py").write_text("class PiecewiseCudaGraphRunner: pass\n")
            env = os.environ.copy()
            env.pop("LAPS_FLASHINFER_KERNEL_ROOT", None)
            env["LAPS_REPLAY_CONTROL_DIR"] = str(base / "events")
            env["PYTHONPATH"] = str(Path("scripts/laps_kernel_hook").resolve()) + os.pathsep + str(base)
            result = subprocess.run([sys.executable, "-c",
                                     "import sglang.srt.model_executor.piecewise_cuda_graph_runner"],
                                    env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            events = [json.loads(line) for p in (base / "events").glob("*.jsonl")
                      for line in p.read_text().splitlines()]
            self.assertEqual(events[0]["event"], "error")
            self.assertIn("Cannot locate", events[0]["error"])
            env.pop("LAPS_REPLAY_CONTROL_DIR")
            result = subprocess.run([sys.executable, "-c",
                                     "import sglang.srt.model_executor.piecewise_cuda_graph_runner; "
                                     "import sys; assert 'torch' not in sys.modules; "
                                     "assert 'replay_control' not in sys.modules"],
                                    env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_missing_entry_point_logs_error(self):
        with patch.object(control, "record") as record:
            with self.assertRaises(AttributeError):
                control.instrument(SimpleNamespace())
            self.assertEqual(record.call_args.args[0]["event"], "error")

    def test_proxy_calls_control_once(self):
        calls = []
        graph = object()
        proxy = control.ReplayProxy(graph, lambda g: calls.append(g))
        proxy.replay()
        with self.assertRaises(RuntimeError):
            proxy.replay()
        self.assertEqual(calls, [graph])

    def test_wrapper_runs_after_planning_and_restores_graph_on_failure(self):
        class Runner:
            def replay_batch_prefill(self, forward_batch, **kwargs):
                # init_forward_metadata_replay_batch_prefill_cuda_graph
                self.planned = True
                self.batch_prefill_graphs[(2, 16)].replay()
                return "output"

            def _find_batch_prefill_graph(self, bs, width):
                return (2, 16)

        module = SimpleNamespace(PiecewiseCudaGraphRunner=Runner, __name__="fixture")
        graph = object()
        runner = Runner()
        runner.batch_prefill_graphs = {(2, 16): graph}
        batch = SimpleNamespace(batch_size=2, extend_seq_lens_cpu=[8, 9])
        torch = SimpleNamespace(cuda=SimpleNamespace(is_current_stream_capturing=lambda: False))
        def compare(*args):
            self.assertTrue(runner.planned)
            self.assertIs(args[-1], graph)
        with patch.object(control, "record"), patch.dict(sys.modules, torch=torch):
            control.instrument(module)
            with patch.object(control, "compare", side_effect=compare):
                self.assertEqual(runner.replay_batch_prefill(batch), "output")
            self.assertIs(runner.batch_prefill_graphs[(2, 16)], graph)
            with patch.object(control, "compare", side_effect=RuntimeError("failed")):
                with self.assertRaisesRegex(RuntimeError, "failed"):
                    runner.replay_batch_prefill(batch)
            self.assertIs(runner.batch_prefill_graphs[(2, 16)], graph)
            torch.cuda.is_current_stream_capturing = lambda: True
            with patch.object(control, "compare") as comparison:
                with self.assertRaisesRegex(RuntimeError, "capture"):
                    runner.replay_batch_prefill(batch)
                comparison.assert_not_called()

    def events(self, directory, events):
        (directory / "1.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))

    def test_summary_weights_rms_and_preserves_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            base = {"event": "comparison", "pid": 1, "actual_bs": 2, "numel": 10,
                    "argmax_agreement": 2, "bitwise_equal": False, "max_abs": 4, "rms": 2}
            self.events(directory, [{"event": "installed", "pid": 1}, base,
                                    dict(base, numel=30, rms=0, max_abs=0, bitwise_equal=True,
                                         argmax_agreement=1), {"event": "error", "error": "failure"}])
            result = summary.summarize(directory)
            self.assertEqual(result["rms"], 1)
            self.assertEqual(result["max_abs"], 4)
            self.assertEqual(result["argmax_disagreements"], 1)
            self.assertEqual(result["bitwise_equal_replays"], 1)
            self.assertEqual(len(result["errors"]), 1)

    def test_empty_or_installed_only_is_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self.assertTrue(summary.summarize(directory)["errors"])
            self.events(directory, [{"event": "installed", "pid": 1}])
            self.assertTrue(summary.summarize(directory)["errors"])

    def test_malformed_json_fails_with_location(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "1.jsonl").write_text('{"event":')
            with self.assertRaisesRegex(ValueError, "1.jsonl:1"):
                summary.summarize(directory)
