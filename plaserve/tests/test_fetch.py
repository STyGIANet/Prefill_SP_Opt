import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class FetchTests(unittest.TestCase):
    def fixture(self, directory, fail=False):
        root = Path(directory)
        scripts = root / "scripts"
        scripts.mkdir()
        shutil.copy2(Path(__file__).resolve().parent.parent / "scripts" / "fetch_upstream.sh",
                     scripts / "fetch_upstream.sh")
        binaries = root / "bin"
        binaries.mkdir()
        fake_git = binaries / "git"
        fake_git.write_text("""#!/usr/bin/env python3
import os, pathlib, sys
args = sys.argv[1:]
if args[0] == 'clone':
    if os.environ.get('FETCH_FIXTURE_FAIL') == '1':
        sys.exit(9)
    pathlib.Path(args[-1], 'source.txt').write_text('fixture')
elif args[2] == 'rev-parse':
    if args[-1] == 'a100-eval^{commit}':
        sys.exit(1)
    print('1234567890abcdef')
elif args[2] != 'checkout':
    sys.exit(8)
""")
        fake_git.chmod(0o755)
        environment = os.environ.copy()
        environment["PATH"] = f"{binaries}:{environment['PATH']}"
        environment["FETCH_FIXTURE_FAIL"] = "1" if fail else "0"
        return scripts / "fetch_upstream.sh", environment

    def test_remote_branch_resolves_and_revision_is_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            script, environment = self.fixture(directory)
            result = subprocess.run(["bash", str(script), "a100-eval"], env=environment,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            vendor = Path(directory) / "vendor"
            self.assertEqual((vendor / "LAPS_REVISION.txt").read_text().strip(), "1234567890abcdef")
            self.assertTrue((vendor / "LAPS" / "source.txt").is_file())
            self.assertFalse(list(vendor.glob(".LAPS-fetch.*")))
            # An existing user checkout is preserved on a repeat invocation.
            repeated = subprocess.run(["bash", str(script), "main"], env=environment,
                                      capture_output=True, text=True)
            self.assertEqual(repeated.returncode, 1)
            self.assertTrue((vendor / "LAPS" / "source.txt").is_file())

    def test_failed_clone_cleans_partial_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            script, environment = self.fixture(directory, fail=True)
            result = subprocess.run(["bash", str(script), "main"], env=environment,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 9)
            vendor = Path(directory) / "vendor"
            self.assertFalse((vendor / "LAPS").exists())
            self.assertFalse(list(vendor.glob(".LAPS-fetch.*")))


if __name__ == "__main__":
    unittest.main()
