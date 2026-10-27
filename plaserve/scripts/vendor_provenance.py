"""Snapshot vendor HEAD plus all tracked staged/unstaged changes."""
import hashlib
import subprocess
from pathlib import Path


def snapshot(repo, output):
    repo, output = Path(repo).resolve(), Path(output)
    head = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    diff = subprocess.check_output(['git', '-C', str(repo), 'diff', '--binary', 'HEAD', '--'])
    path = output / 'vendor.diff'
    path.write_bytes(diff)
    return {'vendor_head': head, 'vendor_diff': path.name,
            'vendor_diff_sha256': hashlib.sha256(diff).hexdigest()}
