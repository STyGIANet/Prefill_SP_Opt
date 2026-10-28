"""Record read-only GPU/load observations until the owning wrapper stops us."""
import datetime
import json
from pathlib import Path
import subprocess
import sys
import time


def query(args):
    result = subprocess.run(args, capture_output=True, text=True, timeout=10)
    return dict(status=result.returncode, stdout=result.stdout, stderr=result.stderr)


with Path(sys.argv[1]).open('a', buffering=1) as stream:
    while True:
        record = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
        record['gpus'] = query(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu', '--format=csv'])
        record['processes'] = query(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory', '--format=csv'])
        pids = []
        for row in record['processes']['stdout'].splitlines()[1:]:
            columns = row.split(',')
            if len(columns) > 1 and columns[1].strip().isdigit():
                pids.append(columns[1].strip())
        if pids:
            record['owners'] = query(['ps', '-o', 'pid,user,args', '-p', ','.join(pids)])
        stream.write(json.dumps(record) + '\n')
        time.sleep(5)
