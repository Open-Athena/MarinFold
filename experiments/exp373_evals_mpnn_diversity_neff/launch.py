"""Submit independent batch jobs beside the two existing CoreWeave exports."""

import argparse
import configparser
import hashlib
import io
import json
import os
import shlex
import subprocess
import zipfile
from pathlib import Path

import iris
from fray.iris_backend import FrayIrisClient
from fray.types import Entrypoint, JobRequest, ResourceConfig, create_environment
from iris.cli.connect import open_iris_client

HERE = Path(__file__).resolve().parent


def main() -> None:
    """Package the exact sources and frozen inputs before submitting GPU work."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--models', nargs='+', default=['mpnn','default'])
    parser.add_argument('--shards', type=int, default=6)
    parser.add_argument('--only-shards', nargs='+', type=int)
    parser.add_argument('--attempt', type=int, default=1)
    args = parser.parse_args()
    assert len(args.models)*args.shards <= 12
    plan = json.loads((HERE/'data/plan.json').read_text())
    archive = io.BytesIO()
    sources = [HERE/name for name in ['worker.py','contacts.py','data/plan.json','inputs/targets.json']]
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for path in sources:
            bundle.write(path, path.relative_to(HERE))
    data = archive.getvalue()
    digest = hashlib.sha256(data).hexdigest()
    local = HERE/'scratch'/f'{digest}.zip'
    local.parent.mkdir(exist_ok=True)
    local.write_bytes(data)
    remote = plan['artifact_root']+'/source/'+local.name
    url = remote.replace('hf://buckets/','https://huggingface.co/buckets/').replace('/MarinFold/', '/MarinFold/resolve/',1)
    print('BUNDLE', digest, len(data), 'bytes', flush=True)
    if not args.submit:
        return
    subprocess.run(['hf','buckets','cp',str(local),remote],check=True)
    credentials = configparser.ConfigParser()
    credentials.read(Path.home()/'.aws/credentials')
    cw = credentials['cw']
    environment = {
        'FSSPEC_S3':json.dumps({'key':cw['aws_access_key_id'], 'secret':cw['aws_secret_access_key'],
                      'endpoint_url':'http://cwlota.com','config_kwargs':{'s3':{'addressing_style':'virtual'}}}),
        'HF_TOKEN':os.environ.get('HF_TOKEN') or (Path.home()/'.cache/huggingface/token').read_text().strip(),
        'HF_HUB_DISABLE_PROGRESS_BARS':'1', 'TOKENIZERS_PARALLELISM':'false',
    }
    image = 'vllm/vllm-openai:v0.9.2'
    directory = '/tmp/exp373-'+digest[:12]
    download = ('import hashlib,io,urllib.request,zipfile; '
                f'd=urllib.request.urlopen({url!r},timeout=120).read(); '
                f'assert hashlib.sha256(d).hexdigest()=={digest!r}; '
                f'zipfile.ZipFile(io.BytesIO(d)).extractall({directory!r})')
    package = 'marinfold @ git+https://github.com/Open-Athena/MarinFold.git@'+plan['marinfold_revision']+'#subdirectory=marinfold'
    record_path = HERE/'data/launches.json'
    records = json.loads(record_path.read_text()) if record_path.exists() else []
    config = Path(iris.__file__).parent/'config/cw-us-east-02a.yaml'
    with open_iris_client(config_file=config, workspace=None) as connection:
        client = FrayIrisClient.from_iris_client(connection)
        for model in args.models:
            for shard in ([0] if args.smoke else (args.only_shards or range(args.shards))):
                shell = '\n'.join([
                    'set -euo pipefail',
                    'python3 -c '+shlex.quote(download),
                    'cd '+shlex.quote(directory),
                    'python3 -m pip install --quiet "fsspec==2026.1.0" "s3fs==2026.1.0" "aiobotocore==2.26.0" "huggingface_hub==1.18.0"',
                    'python3 -m pip install --quiet --no-deps '+shlex.quote(package),
                    'export VLLM_PORT=$(python3 -c \'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()\')',
                    f'exec python3 worker.py --model {model} --shard {shard} --shards {args.shards}'+(' --smoke' if args.smoke else ''),
                ])
                name=f'exp373-{model}-'+('smoke' if args.smoke else f's{shard:02d}')+f'-a{args.attempt:02d}'
                job = client.submit(JobRequest(name=name,
                    entrypoint=Entrypoint.from_binary('bash',['-lc',shell]),
                    resources=ResourceConfig.with_gpu('H100',count=1,cpu=8,ram='64Gi',disk='40Gi',image=image),
                    environment=create_environment(docker_image=image,env_vars=environment,setup_scripts=[]),
                    priority=3,max_retries_failure=0,max_retries_preemption=3))
                records.append({'job_id':job.job_id,'model':model,'shard':shard,'shards':args.shards,
                                'smoke':args.smoke,'source_sha256':digest,'source_uri':remote,
                                'cluster':'cw-us-east-02a','attempt':args.attempt})
                record_path.write_text(json.dumps(records,indent=2)+'\n')
                print('SUBMITTED',job.job_id,flush=True)


if __name__ == '__main__':
    main()
