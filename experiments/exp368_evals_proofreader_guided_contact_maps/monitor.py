"""Read owned Iris jobs and preserve monitoring evidence across restarts."""

import argparse
import json
import subprocess
from collections import Counter
from datetime import UTC, datetime

from runtime import HERE, ROOT, load_release


def main() -> None:
    """Print compact, actionable task state; keep full logs in the saved snapshot."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--control', default=ROOT + '/server-v1')
    parser.add_argument('--server', action='store_true', help='Inspect the retired server snapshot instead of live Iris jobs')
    args = parser.parse_args()
    if not args.server:
        client = HERE.parent.parent / 'scratch/exp368-iris-client'
        query = json.dumps({'query': {'job_id_prefix': '/bizon/exp368', 'limit': 100}})
        command = ['uv', 'run', '--project', str(client), '--package', 'marin-iris',
                   'iris', '--cluster', 'marin', 'rpc', 'controller', 'list-jobs', '--json', query]
        result = subprocess.run(command, text=True, capture_output=True, check=True)
        state = json.loads(result.stdout)
        if state.get('has_more'):
            raise ValueError('Job list truncated; increase the monitor query limit')
        path = HERE / '_cache/jobs_status.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        snapshot = dict(observed_at=datetime.now(UTC).isoformat(), **state)
        path.write_text(json.dumps(snapshot, indent=2))
        print(dict(Counter(job['state'] for job in state['jobs'])))
        for job in state['jobs']:
            if job['state'] == 'JOB_STATE_RUNNING':
                print(job['job_id'], job['state'], 'preemptions=' + str(job.get('preemption_count', 0)))
            elif job['state'] == 'JOB_STATE_FAILED':
                print(job['job_id'], job['state'], job.get('error', '')[-240:])
        pending = [job for job in state['jobs'] if job['state'] == 'JOB_STATE_PENDING']
        if pending:
            print('Pending:', len(pending), sorted({job.get('pending_reason', '') for job in pending}))
        return
    api = load_release(('storage',))
    fs, root = api.storage.filesystem(args.control)
    state = json.loads(fs.cat(root + '/status.json'))
    snapshot = dict(observed_at=datetime.now(UTC).isoformat(), control=args.control, **state)
    path = HERE / '_cache/server_status.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, indent=2))
    print(f"active={len(state['active'])} finished={len(state['finished'])} "
          f"artifact_transfer_GB={state['artifact_transfer_bytes']/1e9:.3f}")
    for name, item in state['active'].items():
        tail = item.get('log_tail', '').strip().splitlines()
        progress = [line for line in tail if '[exp368]' in line]
        print(name, 'gpu=' + str(item['gpu']), (progress or tail or ['starting'])[-1][-240:])
    for name, item in state['finished'].items():
        print(name, item['state'], 'exit=' + str(item['exit_code']))


if __name__ == '__main__':
    main()
