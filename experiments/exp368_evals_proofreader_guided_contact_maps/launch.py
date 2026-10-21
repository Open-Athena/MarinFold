"""Submit independent batch H100 jobs and retain exact recovery commands."""

import argparse
import hashlib
import json
import shlex
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from runtime import load_release

HERE = Path(__file__).resolve().parent
IMAGE = 'pytorch/pytorch@sha256:b574d4ccf6d8856a5d87dcadc667aa4f95dc18d337ef3a28d02b7b01897d7081'


def main() -> None:
    """Bundle the pinned source and write durable job ownership metadata."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--kind', choices=['guided', 'pool', 'score', 'serve', 'recover', 'mirror',
                                         'regional-recover', 'regional-pool'], default='guided')
    parser.add_argument('--cluster', default='cw-us-east-02a')
    parser.add_argument('--dry-run', action='store_true')
    args, remaining = parser.parse_known_args()
    if remaining and remaining[0] == '--':
        remaining = remaining[1:]
    load_release(('records', 'storage', 'model'))
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.iterdir()
             if p.suffix in {'.py', '.sh', '.toml', '.lock', '.json'} and p.name != 'code_manifest.json'}
    source = dict(files=files, sha256=hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=HERE, text=True).strip())
    (HERE / 'code_manifest.json').write_text(json.dumps(source, indent=2))
    client = HERE.parent.parent / 'scratch/exp368-iris-client'
    server = args.kind == 'serve'
    mirror = args.kind == 'mirror'
    regional = args.kind in {'regional-recover', 'regional-pool'}
    if regional and args.cluster != 'cw-rno2a':
        raise ValueError('Regional recovery must use the verified RNO2A mirror')
    if args.cluster != 'cw-us-east-02a' and not ((mirror or regional) and args.cluster == 'cw-rno2a'):
        raise ValueError('All new jobs must be colocated with storage; preemption can repeat model staging')
    entrypoint = (['uv', 'run', '--no-project', '--with', 's3fs==2026.2.0', '--with',
                   'huggingface-hub==1.7.1', 'python', 'prepare_regional_mirror.py'] if mirror else
                  ['bash', 'pool_bootstrap.sh' if args.kind in {'pool', 'regional-pool'} else 'bootstrap.sh',
                   *(['--score'] if args.kind == 'score' else ['--serve'] if server else
                     ['--recover', '--regional'] if regional else ['--recover'] if args.kind == 'recover' else [])])
    command = ['uv', 'run', '--project', str(client), '--package', 'marin-iris',
        'iris', '--cluster', 'marin', 'job', 'run', '--target-cluster', args.cluster,
        '--priority', 'batch', '--user', 'bizon', '--job-name', args.name, '--no-wait',
        '--enable-extra-resources', '--cpu', '32' if server else '2' if mirror else '8',
        '--memory', '512GB' if server else '8GB' if mirror else '64GB', '--disk', '16GB' if mirror else '256GB',
        '--no-sync', '--timeout', '259200', '--max-retries', '0' if server else '2', '--task-image',
        'ghcr.io/astral-sh/uv:python3.12-bookworm-slim' if mirror else
        'vllm/vllm-openai:v0.9.2' if args.kind in {'pool', 'serve', 'regional-pool'} else IMAGE,
        '--exclude', r'(^|/)(_cache|\.venv|wandb|data|plots|__pycache__)/',
        '--include', 'release_api/*.py',
        '--include', 'code_manifest.json',
        '-e', 'EXP368_JOB_ID', '/bizon/' + args.name,
        *([] if mirror else ['--gpu', 'H100x8' if server else 'H100x1']),
        '--', *entrypoint, *remaining]
    exact = shlex.join(command)
    print(exact, flush=True)
    if args.dry_run:
        return
    result = subprocess.run(command, cwd=HERE, text=True, capture_output=True)
    print(result.stdout + result.stderr, flush=True)
    if result.returncode or '/bizon/' + args.name not in result.stdout + result.stderr:
        raise RuntimeError('Iris did not confirm the expected job; inspect before resubmitting')
    client_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=client, text=True).strip()
    state = dict(job_id='/bizon/' + args.name, cluster=args.cluster, client_commit=client_commit,
        submitted_at=datetime.now(UTC).isoformat(), resubmit_command=exact,
        restart_count=0, monitoring_owner='exp368-root-agent', source=source)
    path = HERE / 'data/dispatches.jsonl'
    path.parent.mkdir(exist_ok=True)
    with path.open('a') as stream:
        stream.write(json.dumps(state) + '\n')
    scratch = HERE.parent.parent / 'scratch/20261021-exp368_monitoring_state.json'
    previous = json.loads(scratch.read_text()) if scratch.exists() else {'jobs': {}}
    previous['jobs'][state['job_id']] = state
    previous['monitoring_owner'] = state['monitoring_owner']
    scratch.parent.mkdir(exist_ok=True)
    scratch.write_text(json.dumps(previous, indent=2))


if __name__ == '__main__':
    main()
