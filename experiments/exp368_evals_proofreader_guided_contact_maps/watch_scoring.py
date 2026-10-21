"""Start frozen scoring tasks after their corresponding generation jobs finish.

This finite dependency watcher submits only the five predeclared validation
scoring jobs. It records submissions through launch.py, detects previous
submissions on restart, and exits once all five exist. Failed parents or scoring
jobs stop the watcher visibly for the monitoring agent to investigate.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import UTC, datetime

from runtime import HERE, load_release

RESULT_ROOT = 's3://marinfold-exp368-rno2a/MarinFold/exp368/results'
DEPENDENCIES = [('exp368-pilot-direct-rno-v1', '42-regional-pilot-direct-score-000',
                 'exp368-pilot-direct-score-rno-v1')]
DEPENDENCIES += [(f'exp368-validation-pool-rno-{shard:03d}',
                  f'43-regional-validation-pool-score-{shard:03d}',
                  f'exp368-validation-score-rno-{shard:03d}') for shard in range(4)]


def validation_artifacts_complete() -> bool:
    """Validate all immutable generation barriers when scheduler status is stale."""
    plan = json.loads((HERE / 'validation_pool.json').read_text())
    plan_sha = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    tasks = {row['task_id']: row for row in map(json.loads, (HERE / 'data/server_tasks.jsonl').read_text().splitlines())}
    api = load_release(('storage',))
    fs, root = api.storage.filesystem(RESULT_ROOT + '/validation-pool')
    names = set(fs.find(root))
    markers = [root + f'/shard-{shard:03d}.complete.json' for shard in range(4)]
    if not all(key in names for key in markers):
        return False
    barriers = []
    for shard, key in enumerate(markers):
        value = json.loads(fs.cat(key))
        expected = dict(plan_sha256=plan_sha, kind='ordinary-generation', shards=4,
            proteins=len(plan['entries'][shard::4]),
            code_sha256=tasks[f'41-regional-validation-pool-{shard:03d}']['source']['sha256'])
        if any(value.get(field) != item for field, item in expected.items()):
            raise ValueError('Generation artifact barrier differs from its frozen task: ' + key)
        barriers.append(value)
    expected_groups = {root + f'/{entry}/pool-{size}.done.json'
                       for entry in plan['entries'] for size in plan['pool_sizes']}
    if not expected_groups <= names:
        raise ValueError('Shard barriers exist but a planned output group is missing')
    evidence = dict(verified_at=datetime.now(UTC).isoformat(), result_root=RESULT_ROOT,
                    expected_groups=len(expected_groups), barriers=barriers,
                    reason='Central peer status stale during regional controller outage')
    (HERE / 'data/validation_generation_barriers.json').write_text(json.dumps(evidence, indent=2) + '\n')
    return True


def main() -> None:
    """Advance known dependencies without changing scientific source snapshots."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifact-barriers', action='store_true',
                        help='Permit scoring after all validated generation artifacts commit')
    args = parser.parse_args()
    client = HERE.parent.parent / 'scratch/exp368-iris-client'
    query = json.dumps({'query': {'job_id_prefix': '/bizon/exp368', 'limit': 250}})
    command = ['uv', 'run', '--project', str(client), '--package', 'marin-iris',
               'iris', '--cluster', 'marin', 'rpc', 'controller', 'list-jobs', '--json', query]
    previous = None
    artifacts_ready = False
    while True:
        state = json.loads(subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL))
        if state.get('has_more'):
            raise ValueError('Iris status truncated; refusing an incomplete dependency view')
        jobs = {job['job_id']: job for job in state['jobs']}
        submitted = {json.loads(line)['job_id'] for line in (HERE / 'data/dispatches.jsonl').read_text().splitlines()}
        if args.artifact_barriers and not artifacts_ready:
            artifacts_ready = validation_artifacts_complete()
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
            if jobs[parent_id]['state'] != 'JOB_STATE_SUCCEEDED' and not (
                    artifacts_ready and parent.startswith('exp368-validation-pool-')):
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
