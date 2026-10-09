"""Submit and record CoreWeave root jobs through the central Iris controller."""

import argparse
import hashlib
import json
import netrc
import os
import shlex
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from huggingface_hub import get_token

HERE = Path(__file__).resolve().parent
TRAIN_IMAGE = 'pytorch/pytorch@sha256:b574d4ccf6d8856a5d87dcadc667aa4f95dc18d337ef3a28d02b7b01897d7081'
GEN_IMAGE = 'vllm/vllm-openai:v0.9.2'


def main() -> None:
    """Launch exactly the requested bounded job and persist recovery metadata."""
    parser = argparse.ArgumentParser()
    parser.add_argument('kind', choices=['generate', 'train', 'evaluate', 'collect', 'publish', 'verify-public'])
    parser.add_argument('--name', required=True)
    parser.add_argument('--gpus', type=int, choices=[0, 1, 8], default=1)
    parser.add_argument('--cluster', default='cw-us-east-02a')
    parser.add_argument('--dry-run', action='store_true')
    args, remaining = parser.parse_known_args()
    if remaining and remaining[0] == '--':
        remaining = remaining[1:]
    if args.gpus == 0 and args.kind not in {'collect','publish'}:
        raise ValueError('Predictor and training jobs require a GPU')
    bootstrap = 'generate_bootstrap.sh' if args.kind == 'generate' else 'train_bootstrap.sh'
    command = ['uv', 'run', '--project', '/home/bizon/git/marin', '--package', 'marin-iris',
        'iris', '--cluster', 'marin', 'job', 'run', '--target-cluster', args.cluster,
        '--priority', 'batch', '--user', 'bizon', '--job-name', args.name, '--no-wait',
        '--enable-extra-resources', '--cpu', str(max(8, args.gpus*4)),
        '--memory', '512GB' if args.gpus == 8 else '64GB', '--disk', '256GB', '--no-sync',
        '--timeout', str(3*86400), '--max-retries', '2', '--task-image',
        GEN_IMAGE if args.kind == 'generate' else TRAIN_IMAGE,
        '--exclude', r'(^|/)(_cache|\.venv|wandb|data|plots|__pycache__)/']
    command += ['-e','EXP356_JOB_ID','/bizon/'+args.name]
    if args.gpus:
        command += ['--gpu', f'H100x{args.gpus}']
    secret = None
    secret_name = None
    if args.kind != 'generate':
        command += ['-e', 'PROOFREADER_GPUS', str(args.gpus)]
    if args.kind == 'train':
        secret = os.getenv('WANDB_API_KEY')
        if not secret:
            auth = netrc.netrc().authenticators('api.wandb.ai')
            if auth is None:
                raise ValueError('Missing W&B credentials')
            secret = auth[2]
        secret_name = 'WANDB_API_KEY'
    if args.kind == 'publish':
        secret = get_token()
        if not secret:
            raise ValueError('Missing Hugging Face publication credentials')
        secret_name = 'HF_TOKEN'
    if secret_name:
        command += ['-e',secret_name,'<redacted>']
    command += ['--', 'bash', bootstrap]
    if args.kind in {'evaluate', 'collect', 'publish', 'verify-public'}:
        command += ['--'+args.kind]
    command += remaining
    redacted = shlex.join(command)
    print(redacted, flush=True)
    if args.dry_run:
        return
    if secret:
        command[command.index('<redacted>')] = secret
    sources = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(HERE.iterdir())
        if p.is_file() and p.suffix in {'.py','.sh','.toml','.lock'}}
    source_manifest = dict(files=sources,
        sha256=hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest(),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip())
    (HERE/'code_manifest.json').write_text(json.dumps(source_manifest,indent=2))
    submitted = datetime.now(UTC).isoformat()
    result = subprocess.run(command, cwd=HERE, text=True, capture_output=True)
    output = result.stdout + result.stderr
    if secret:
        output = output.replace(secret, '<redacted>')
    print(output, flush=True)
    if result.returncode:
        raise RuntimeError(f'Iris submission failed: {result.returncode}')
    job_id = '/bizon/' + args.name
    if job_id not in output:
        raise ValueError('Inspect Iris: submission did not confirm the expected job ID')
    state = dict(job_id=job_id, kind=args.kind, cluster=args.cluster, submitted_at=submitted,
                 resubmit_command=redacted, restart_count=0, monitoring_owner='exp356-root-agent',
                 source=source_manifest,launcher_arguments=sys.argv[1:])
    path = HERE / '_cache/jobs'
    path.mkdir(parents=True, exist_ok=True)
    (path / f'{args.name}.json').write_text(json.dumps(state, indent=2))
    data = HERE / 'data'
    data.mkdir(exist_ok=True)
    with (data / 'dispatches.jsonl').open('a') as handle:
        handle.write(json.dumps(state) + '\n')


if __name__ == '__main__':
    main()
