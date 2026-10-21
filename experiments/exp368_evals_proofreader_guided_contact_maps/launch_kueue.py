"""Submit owned batch jobs through the existing Kueue queue during Iris outages.

This uses the cluster's normal admission, quota and priority controls. It does
not modify any shared scheduler/controller object. A small immutable ConfigMap
holds the bootstrap source; model reads remain in RNO2A. Storage credentials
from the user's existing CW profile are sent directly to an owned Kubernetes
Secret and are never included in source bundles, command arguments or ledgers.
"""

import argparse
import base64
import hashlib
import io
import json
import re
import subprocess
import tarfile
from datetime import UTC, datetime

import botocore.session

from launch import IMAGE
from runtime import HERE, load_release

KUBECTL = ['kubectl', '--kubeconfig', '/home/bizon/.kube/coreweave-iris-rno2a',
           '--context', 'marin-rn02a_RNO2A', '-n', 'iris']
LABELS = {'marinfold-experiment': 'exp368', 'marinfold-owner': 'bizon'}
BOOTSTRAP = HERE / 'data/native_bootstrap.json'


def create(resource: dict) -> dict:
    """Create one owned object without updating existing cluster resources."""
    result = subprocess.run([*KUBECTL, 'create', '-f', '-', '-o', 'json'],
        input=json.dumps(resource), capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def prepare() -> None:
    """Create a source snapshot and a private storage Secret once."""
    if BOOTSTRAP.exists():
        raise FileExistsError('Native bootstrap already prepared; reuse its recorded objects')
    load_release()
    files = {path.name: path.read_bytes() for path in HERE.iterdir()
        if path.is_file() and path.suffix in {'.py', '.sh', '.toml', '.lock', '.json'}
        and path.name != 'code_manifest.json'}
    files.update({str(path.relative_to(HERE)): path.read_bytes() for path in (HERE / 'release_api').glob('*.py')})
    hashes = {name: hashlib.sha256(content).hexdigest() for name, content in files.items()}
    source = dict(files=hashes, sha256=hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=HERE, text=True).strip())
    files['code_manifest.json'] = json.dumps(source, indent=2).encode()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w:gz') as archive:
        for name, content in sorted(files.items()):
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    payload = buffer.getvalue()
    if len(payload) > 700_000:
        raise ValueError('Bootstrap source exceeds the bounded ConfigMap payload')
    digest = hashlib.sha256(payload).hexdigest()
    config_name = 'exp368-native-source-' + digest[:12]
    secret_name = 'exp368-native-storage-' + digest[:12]
    config = create(dict(apiVersion='v1', kind='ConfigMap',
        metadata=dict(name=config_name, labels=LABELS), immutable=True,
        binaryData={'source.tar.gz': base64.b64encode(payload).decode()}))
    credentials = botocore.session.Session(profile='cw').get_credentials().get_frozen_credentials()
    storage = dict(key=credentials.access_key, secret=credentials.secret_key,
        endpoint_url='https://cwobject.com', config_kwargs={'s3': {'addressing_style': 'virtual'}})
    if credentials.token:
        storage['token'] = credentials.token
    secret = create(dict(apiVersion='v1', kind='Secret', type='Opaque',
        metadata=dict(name=secret_name, labels=LABELS), immutable=True,
        stringData={'FSSPEC_S3': json.dumps(storage)}))
    record = dict(config_map=config_name, config_map_uid=config['metadata']['uid'],
        secret=secret_name, secret_uid=secret['metadata']['uid'], source=source,
        bundle_sha256=digest, bundle_bytes=len(payload), created_at=datetime.now(UTC).isoformat())
    BOOTSTRAP.write_text(json.dumps(record, indent=2) + '\n')
    (HERE / '_cache' / (config_name + '.tar.gz')).write_bytes(payload)
    print(json.dumps({key: record[key] for key in ['config_map', 'secret', 'bundle_bytes']}))


def main() -> None:
    """Prepare private bootstrap objects or submit one resumable frozen task."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--name')
    parser.add_argument('--control')
    parser.add_argument('--task')
    parser.add_argument('--result-root')
    parser.add_argument('--pool', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare()
        return
    if not all([args.name, args.control, args.task, args.result_root]):
        raise ValueError('Submission requires name, control, task and result root')
    if not re.fullmatch(r'exp368-native-[a-z0-9-]+', args.name) or len(args.name) > 63:
        raise ValueError('Job name must use the owned exp368-native prefix')
    if not args.result_root.startswith('s3://marinfold-exp368-rno2a/MarinFold/exp368/'):
        raise ValueError('Native RNO workers must write to the verified regional bucket')
    bootstrap = json.loads(BOOTSTRAP.read_text())
    script = '''set -euo pipefail
python -m pip install --quiet --no-deps uv==0.10.7
python - <<'PY'
import hashlib,tarfile
from pathlib import Path
path=Path('/bundle/source.tar.gz')
assert hashlib.sha256(path.read_bytes()).hexdigest() == 'BUNDLE_DIGEST'
with tarfile.open(path) as archive:
    archive.extractall('/app', filter='data')
PY
cd /app
exec bash BOOTSTRAP_SCRIPT --recover --regional --region RNO2A "$@"
'''.replace('BUNDLE_DIGEST', bootstrap['bundle_sha256']).replace(
        'BOOTSTRAP_SCRIPT', 'pool_bootstrap.sh' if args.pool else 'bootstrap.sh')
    env = [dict(name='FSSPEC_S3', valueFrom=dict(secretKeyRef=dict(name=bootstrap['secret'], key='FSSPEC_S3'))),
           dict(name='EXP368_JOB_ID', value='kueue://cw-rno2a/iris/' + args.name),
           dict(name='UV_LINK_MODE', value='copy')]
    resources = {'cpu': '8', 'memory': '64Gi', 'ephemeral-storage': '256Gi', 'nvidia.com/gpu': '1'}
    task = dict(name='task', image='vllm/vllm-openai:v0.9.2' if args.pool else IMAGE,
        command=['bash', '-c', script, 'exp368-bootstrap'],
        args=['--control', args.control, '--task', args.task, '--result-root', args.result_root],
        env=env, resources=dict(requests=resources, limits=resources),
        volumeMounts=[dict(name='source', mountPath='/bundle', readOnly=True),
                      dict(name='work', mountPath='/app'), dict(name='shm', mountPath='/dev/shm')])
    pod = dict(restartPolicy='Never', automountServiceAccountToken=False,
        priorityClassName='iris-batch',
        nodeSelector={'iris-cw-rno2a-managed': 'true', 'iris.kueue': 'true'},
        tolerations=[dict(key='nvidia.com/gpu', operator='Exists', effect='NoSchedule'),
                     dict(key='qos.coreweave.cloud/interruptable', operator='Exists', effect='NoExecute')],
        containers=[task], volumes=[dict(name='source', configMap=dict(name=bootstrap['config_map'])),
            dict(name='work', emptyDir={}), dict(name='shm', emptyDir=dict(medium='Memory', sizeLimit='8Gi'))])
    manifest = dict(apiVersion='batch/v1', kind='Job', metadata=dict(name=args.name,
        labels={**LABELS, 'kueue.x-k8s.io/queue-name': 'cw-rno2a-lq',
                'kueue.x-k8s.io/priority-class': 'iris-accelerator-batch'}),
        spec=dict(suspend=True, backoffLimit=0, activeDeadlineSeconds=259200, ttlSecondsAfterFinished=604800,
            template=dict(metadata=dict(labels=LABELS,
                annotations={'kueue.x-k8s.io/podset-preferred-topology': 'kubernetes.io/hostname'}), spec=pod)))
    # Server validation precedes creation; the Job remains suspended until the
    # existing Kueue controller admits it using the same batch queue as Iris.
    subprocess.run([*KUBECTL, 'create', '--dry-run=server', '-f', '-', '-o', 'name'],
                   input=json.dumps(manifest), text=True, check=True)
    created = create(manifest)
    record = dict(job_id='kueue://cw-rno2a/iris/' + args.name, uid=created['metadata']['uid'],
        submitted_at=datetime.now(UTC).isoformat(), task=args.task, control=args.control,
        result_root=args.result_root, source=bootstrap['source'], manifest=manifest,
        monitoring_owner='exp368-root-agent')
    with (HERE / 'data/native_dispatches.jsonl').open('a') as stream:
        stream.write(json.dumps(record) + '\n')
    print(json.dumps({key: record[key] for key in ['job_id', 'uid', 'task', 'result_root']}))


if __name__ == '__main__':
    main()
