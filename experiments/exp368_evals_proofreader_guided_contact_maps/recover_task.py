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

from runtime import ROOT, load_release


def main() -> None:
    """Verify and execute one existing task with its original output prefix."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--control', default=ROOT + '/server-v1')
    parser.add_argument('--task', required=True)
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
    if kind not in {'guided', 'score'}:
        raise ValueError('This recovery image supports guided/scoring tasks')
    command = [sys.executable, 'worker.py' if kind == 'guided' else 'score_pool.py',
        '--plan', task['plan'], '--out', args.control + '/results/' + task['output'],
        '--shard', str(task['shard']), '--shards', str(task['shards'])]
    if kind == 'score':
        command += ['--input', args.control + '/results/' + task['input']]
    if task.get('smoke'):
        command += ['--smoke']
    env = dict(os.environ)
    env.pop('EXP368_SHARED_MODELS', None)
    env.pop('EXP368_TARGETS', None)
    print('[exp368-recovery] frozen task ' + args.task, flush=True)
    subprocess.run(command, cwd=source, env=env, check=True)


if __name__ == '__main__':
    main()
