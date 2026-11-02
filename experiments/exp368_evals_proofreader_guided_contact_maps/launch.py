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
                                         'regional-recover', 'regional-pool', 'runtime-smoke', 'backend-check'], default='guided')
    parser.add_argument('--cluster', default='cw-us-east-02a')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--failure-retries', type=int)
    args, remaining = parser.parse_known_args()
    if args.failure_retries is not None and args.failure_retries < 0:
        raise ValueError('Failure retries must be nonnegative')
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
    runtime_smoke = args.kind == 'runtime-smoke'
    backend_check = args.kind == 'backend-check'
    gb200 = args.cluster == 'cw-us-east-08a'
    regional = args.kind in {'regional-recover', 'regional-pool'}
    if regional and args.cluster not in {'cw-rno2a', 'cw-us-east-08a'}:
        raise ValueError('Regional recovery requires a configured colocated mirror')
    if (runtime_smoke or backend_check) and args.cluster != 'cw-us-east-08a':
        raise ValueError('The ARM/GB200 runtime smoke must use cw-us-east-08a')
    if args.cluster != 'cw-us-east-02a' and not (((mirror or regional) and args.cluster in {'cw-rno2a', 'cw-us-east-08a'})
                                               or ((runtime_smoke or backend_check) and args.cluster == 'cw-us-east-08a')):
        raise ValueError('All new jobs must be colocated with storage; preemption can repeat model staging')
    region = 'US-EAST-08A' if gb200 else 'RNO2A'
    pool = args.kind in {'pool', 'regional-pool'}
    failure_retries = args.failure_retries if args.failure_retries is not None else (
        0 if server or runtime_smoke or backend_check else 2)
    entrypoint = (['bash', 'bootstrap_gb200.sh', '--runtime-smoke'] if runtime_smoke else
                 ['bash', 'bootstrap_gb200.sh', '--backend-check'] if backend_check else
                 ['uv', 'run', '--no-project', '--with', 's3fs==2026.2.0', '--with',
                   'huggingface-hub==1.7.1', 'python', 'prepare_regional_mirror.py', '--region', region] if mirror else
                  ['bash', 'pool_bootstrap.sh' if pool else 'bootstrap_gb200.sh' if gb200 else 'bootstrap.sh',
                   *(['--score'] if args.kind == 'score' else ['--serve'] if server else
                     ['--recover', '--regional', '--region', region] if regional else
                     ['--recover'] if args.kind == 'recover' else [])])
    if mirror:
        task_image = 'ghcr.io/astral-sh/uv@sha256:e5b65587bce7de595f299855d7385fe7fca39b8a74baa261ba1b7147afa78e58'
    elif gb200 and pool:
        task_image = 'vllm/vllm-openai@sha256:bc219c29e691e8b21d6da64b6322bda869f1cb7ab584463bdac02d01d3d6c7f9'
    elif gb200:
        task_image = 'ghcr.io/astral-sh/uv@sha256:a2e8b64aed3382a20c03fc6f1e03fe95fb312425e071ad27d7936a0f02c3e497'
    else:
        task_image = 'vllm/vllm-openai:v0.9.2' if pool or server else IMAGE
    command = ['uv', 'run', '--project', str(client), '--package', 'marin-iris',
        'iris', '--cluster', 'marin', 'job', 'run', '--target-cluster', args.cluster,
        '--priority', 'batch', '--user', 'bizon', '--job-name', args.name, '--no-wait',
        '--enable-extra-resources', '--cpu', '32' if server else '2' if mirror else '8',
        '--memory', '512GB' if server else '8GB' if mirror else '64GB', '--disk', '16GB' if mirror else '256GB',
        '--no-sync', '--timeout', '1800' if runtime_smoke or backend_check else '259200',
        '--max-retries', str(failure_retries), '--task-image', task_image,
        '--exclude', r'(^|/)(_cache|\.venv|wandb|data|plots|__pycache__)/',
        '--include', 'release_api/*.py',
        '--include', 'code_manifest.json',
        '-e', 'EXP368_JOB_ID', '/bizon/' + args.name,
        *([] if mirror else ['--gpu', 'GB200x1' if gb200 else 'H100x8' if server else 'H100x1']),
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
