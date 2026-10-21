"""Resume a frozen server task directly against durable, colocated storage.

The original source bundle remains the execution authority: recovery does not
change its code hash, sharding, plan, or sampling seeds. Committed trajectories
are read directly from S3, so a preempted worker cannot lose the entire cache.
"""

import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

from regional_inputs import stage_regional
from runtime import ROOT, load_release


def main() -> None:
    """Verify and execute one existing task with its original output prefix."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--control', default=ROOT + '/server-v1')
    parser.add_argument('--task', required=True)
    parser.add_argument('--regional', action='store_true')
    parser.add_argument('--result-root', help='Durable result root; defaults to the original server results')
    args = parser.parse_args()
    if '/' in args.task or '..' in args.task:
        raise ValueError('Task must be a simple identifier')
    api = load_release(('storage',))
    fs, root = api.storage.filesystem(args.control)
    task = json.loads(fs.cat(root + '/queue/' + args.task + '.json'))
    content = fs.cat(task['bundle'].removeprefix('s3://'))
    if hashlib.sha256(content).hexdigest() != task['bundle_sha256']:
        raise ValueError('Recovery source bundle hash mismatch')
    source = Path('/tmp/exp368-recovery') / task['bundle_sha256']
    source.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(content), mode='r:gz') as archive:
        archive.extractall(source, filter='data')
    kind = task['kind']
    if kind not in {'guided', 'score', 'pool'}:
        raise ValueError('Unsupported recovery task kind')
    command = (['bash', 'pool_bootstrap.sh'] if kind == 'pool' else
               [sys.executable, 'worker.py' if kind == 'guided' else 'score_pool.py'])
    results = args.result_root or args.control + '/results'
    command += ['--plan', task['plan'], '--out', results + '/' + task['output'],
                '--shard', str(task['shard']), '--shards', str(task['shards'])]
    if kind == 'score':
        command += ['--input', results + '/' + task['input']]
    if task.get('smoke'):
        command += ['--smoke']
    env = dict(os.environ)
    env.pop('EXP368_SHARED_MODELS', None)
    env.pop('EXP368_TARGETS', None)
    if args.regional:
        env = stage_regional(api)
    print('[exp368-recovery] frozen task ' + args.task, flush=True)
    subprocess.run(command, cwd=source, env=env, check=True)


if __name__ == '__main__':
    main()
