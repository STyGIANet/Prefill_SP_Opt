from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from plaserve.cluster import make_plan, run_cluster


class ClusterTests(unittest.TestCase):
    def test_profile_commands_and_allocations(self):
        root = Path("vendor/LAPS")
        self.assertEqual(make_plan("h200", None, root)["required_gpus"], 2)
        self.assertEqual(make_plan("h200", ["32b"], root)["required_gpus"], 8)
        h100 = make_plan("h100-4", ["7b", "14b"], root)
        self.assertEqual(h100["required_gpus"], 4)
        self.assertTrue(h100["command"][1].endswith("run_all_h100_4gpu.sh"))
        a100 = make_plan("a100-2", None, root)
        self.assertEqual(len(a100["command"]), 2)
        self.assertEqual(a100["recommended_revision"], "a100-eval")
        for profile, models in (("h100-4", ["72b"]), ("h200", ["7b", "7b"]),
                                ("a100-2", ["7b"])):
            with self.assertRaises(ValueError):
                make_plan(profile, models, root)

    def test_dry_run_without_upstream_or_cuda(self):
        result = subprocess.run([sys.executable, "-m", "plaserve", "cluster", "--dry-run"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["required_gpus"], 2)

    def fixture(self, directory, exit_code=0, summary=True):
        upstream = Path(directory) / "LAPS"
        benchmark = upstream / "bench_laps_prefill_throughput"
        benchmark.mkdir(parents=True)
        (upstream / "data").mkdir()
        (upstream / "data" / "lmsys_chat_10k.jsonl").write_text('{"prompt":"fixture"}\n')
        script = benchmark / "run_all_benchmarks.sh"
        script.write_text("#!/usr/bin/env bash\nset -eu\nprintf 'fixture benchmark\\n'\n"
                          + ("mkdir -p results_fixture/raw\nprintf 'summary\\n' > results_fixture/summary.txt\n"
                             "printf '{}' > results_fixture/raw/metrics.json\n" if summary else "")
                          + f"exit {exit_code}\n")
        args = SimpleNamespace(profile="h200", models=["7b"], upstream=str(upstream),
                               output=str(Path(directory) / "results"), dry_run=False)
        return upstream, args

    def git_capture(self, command, cwd=None):
        if command == ["git", "rev-parse", "HEAD"]:
            return "abcdef1234567890"
        if command[:2] == ["git", "status"]:
            return ""
        raise AssertionError(f"Unexpected command: {command}")

    def test_launcher_records_provenance_logs_and_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            upstream, args = self.fixture(directory)
            with patch("plaserve.cluster.inspect_runtime", return_value={"gpus": [{"name": "fixture"}]}), \
                 patch("plaserve.cluster.capture", side_effect=self.git_capture), \
                 patch.dict(os.environ, {"HF_TOKEN": "do-not-record", "NUM_PROMPTS": "10"}), \
                 redirect_stdout(io.StringIO()):
                run_cluster(args)
            output = Path(args.output)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["project"], "PLAserve")
            self.assertEqual(manifest["source"]["commit"], "abcdef1234567890")
            self.assertEqual(manifest["environment"]["NUM_PROMPTS"], "10")
            self.assertNotIn("HF_TOKEN", manifest["environment"])
            self.assertNotIn("do-not-record", (output / "manifest.json").read_text())
            copied = output / "artifacts" / "bench_laps_prefill_throughput" / "results_fixture"
            self.assertTrue((copied / "raw" / "metrics.json").is_file())
            self.assertIn("fixture benchmark", (output / "benchmark.log").read_text())
            self.assertEqual(len(manifest["dataset"]["sha256"]), 64)

    def test_failed_benchmark_preserves_partial_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            upstream, args = self.fixture(directory, exit_code=3)
            with patch("plaserve.cluster.inspect_runtime", return_value={}), \
                 patch("plaserve.cluster.capture", side_effect=self.git_capture), \
                 redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "exited with 3"):
                run_cluster(args)
            manifest = json.loads((Path(args.output) / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["exit_code"], 3)
            self.assertTrue(manifest["artifacts"])

    def test_no_results_is_failure_and_nonempty_destination_is_protected(self):
        with tempfile.TemporaryDirectory() as directory:
            upstream, args = self.fixture(directory, summary=False)
            with patch("plaserve.cluster.inspect_runtime", return_value={}), \
                 patch("plaserve.cluster.capture", side_effect=self.git_capture), \
                 redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "no new summary"):
                run_cluster(args)
            with self.assertRaisesRegex(ValueError, "not empty"):
                run_cluster(args)

    def test_missing_upstream_gives_actionable_error(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(profile="h200", models=None, upstream=directory,
                                   output=str(Path(directory) / "output"), dry_run=False)
            with self.assertRaisesRegex(RuntimeError, "fetch_upstream"):
                run_cluster(args)


if __name__ == "__main__":
    unittest.main()
