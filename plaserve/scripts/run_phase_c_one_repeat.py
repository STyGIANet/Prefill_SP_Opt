"""Usage: python scripts/run_phase_c_one_repeat.py CAMPAIGN_DIR REPEAT

Run one extra Phase C repeat into an existing campaign, using run_phase_c's checks."""
import hashlib, json, os, pwd, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_phase_c as m

output = Path(sys.argv[1]).resolve()
repeat = int(sys.argv[2])
base = json.loads((output / 'provenance.json').read_text())
m.check_free(output, repeat)
directory = output / f'repeat-{repeat}'
env = os.environ.copy()
env.update(NUM_PROMPTS='500', CONCURRENCY_LEVELS='1 4 8 16', BACKEND='nixl')
print(f'Starting repeat {repeat}: {directory}', flush=True)
with (output / f'repeat-{repeat}-wrapper.log').open('w') as log:
    result = subprocess.run(['bash', str(ROOT / 'scripts/run_bench_fixed_kernels.sh'), str(directory)],
                            env=env, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
validation = json.loads((directory / 'validation.json').read_text())
prov = json.loads((directory / 'provenance.json').read_text())
assert prov['vendor_head'] == base['vendor_head']
assert prov['vendor_diff_sha256'] == base['vendor_diff_sha256']
activity = m.interference(directory / 'gpu-samples.jsonl', pwd.getpwuid(os.getuid()).pw_name)
m.save(directory / 'interference.json', activity)
print(json.dumps(dict(returncode=result.returncode, validation=validation['status'], results=len(validation.get('results', [])),
                      foreign_users=len(activity['foreign_users']), unknown_owners=len(activity['unknown_owners']),
                      errors=len(activity['errors']))), flush=True)
