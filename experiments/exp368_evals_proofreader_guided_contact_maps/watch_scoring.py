"""Start frozen scoring tasks after their corresponding generation jobs finish.

This finite dependency watcher submits only the five predeclared validation
scoring jobs. It records submissions through launch.py, detects previous
submissions on restart, and exits once all five exist. Failed parents or scoring
jobs stop the watcher visibly for the monitoring agent to investigate.
"""

import json
import subprocess
import sys
import time
from datetime import UTC, datetime

from runtime import HERE

RESULT_ROOT = 's3://marinfold-exp368-rno2a/MarinFold/exp368/results'
DEPENDENCIES = [('exp368-pilot-direct-rno-v1', '42-regional-pilot-direct-score-000',
                 'exp368-pilot-direct-score-rno-v1')]
DEPENDENCIES += [(f'exp368-validation-pool-rno-{shard:03d}',
                  f'43-regional-validation-pool-score-{shard:03d}',
                  f'exp368-validation-score-rno-{shard:03d}') for shard in range(4)]


def main() -> None:
    """Advance known dependencies without changing scientific source snapshots."""
    client = HERE.parent.parent / 'scratch/exp368-iris-client'
    query = json.dumps({'query': {'job_id_prefix': '/bizon/exp368', 'limit': 100}})
    command = ['uv', 'run', '--project', str(client), '--package', 'marin-iris',
               'iris', '--cluster', 'marin', 'rpc', 'controller', 'list-jobs', '--json', query]
    previous = None
    while True:
        state = json.loads(subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL))
        if state.get('has_more'):
            raise ValueError('Iris status truncated; refusing an incomplete dependency view')
        jobs = {job['job_id']: job for job in state['jobs']}
        submitted = {json.loads(line)['job_id'] for line in (HERE / 'data/dispatches.jsonl').read_text().splitlines()}
        status = {}
        for parent, task, name in DEPENDENCIES:
            parent_id, child_id = '/bizon/' + parent, '/bizon/' + name
            if parent_id not in jobs:
                raise ValueError('Missing parent job: ' + parent_id)
            if jobs[parent_id]['state'] in {'JOB_STATE_FAILED', 'JOB_STATE_KILLED'}:
                raise RuntimeError('Generation needs recovery: ' + parent_id)
            if child_id in submitted:
                if child_id in jobs and jobs[child_id]['state'] in {'JOB_STATE_FAILED', 'JOB_STATE_KILLED'}:
                    raise RuntimeError('Scoring needs recovery: ' + child_id)
                status[name] = jobs.get(child_id, {}).get('state', 'SUBMITTED')
                continue
            if jobs[parent_id]['state'] != 'JOB_STATE_SUCCEEDED':
                status[name] = 'WAITING_FOR_GENERATION'
                continue
            subprocess.run([sys.executable, str(HERE / 'launch.py'), '--kind', 'regional-recover',
                '--cluster', 'cw-rno2a', '--name', name, '--', '--task', task,
                '--result-root', RESULT_ROOT], check=True)
            status[name] = 'SUBMITTED'
        snapshot = dict(observed_at=datetime.now(UTC).isoformat(), dependencies=status)
        (HERE / '_cache/scoring_dependencies.json').write_text(json.dumps(snapshot, indent=2))
        if status != previous:
            print(json.dumps(snapshot), flush=True)
            previous = status
        if all(value != 'WAITING_FOR_GENERATION' for value in status.values()):
            print('All five scoring dependencies submitted; watcher finished.', flush=True)
            return
        time.sleep(120)


if __name__ == '__main__':
    main()
