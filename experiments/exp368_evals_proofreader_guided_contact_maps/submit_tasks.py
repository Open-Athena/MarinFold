"""Submit immutable phase snapshots to the shared eight-GPU experiment worker."""

import argparse
import hashlib
import io
import json
import subprocess
import tarfile
import tempfile

from runtime import HERE, ROOT, load_release


def main() -> None:
    """Publish one source snapshot and explicit task descriptors, never shell code."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--control', default=ROOT + '/server-v1')
    parser.add_argument('--kind', choices=['guided', 'pool', 'score', 'length'], required=True)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--input')
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    api = load_release()
    files = {path.name: path.read_bytes() for path in HERE.iterdir()
        if path.is_file() and path.suffix in {'.py', '.sh', '.toml', '.lock', '.json'}
        and path.name != 'code_manifest.json'}
    files.update({str(path.relative_to(HERE)): path.read_bytes() for path in (HERE / 'release_api').glob('*.py')})
    hashes = {name: hashlib.sha256(content).hexdigest() for name, content in files.items()}
    manifest = dict(files=hashes, sha256=hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=HERE, text=True).strip())
    files['code_manifest.json'] = json.dumps(manifest, indent=2).encode()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w:gz') as archive:
        for name, content in sorted(files.items()):
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    content = buffer.getvalue()
    sha = hashlib.sha256(content).hexdigest()
    fs, root = api.storage.filesystem(args.control)
    bundle = args.control + '/bundles/' + sha + '.tar.gz'
    print(f'Uploading immutable source bundle: {len(content)} bytes', flush=True)
    with tempfile.NamedTemporaryFile(suffix='.tar.gz') as temporary:
        temporary.write(content)
        temporary.flush()
        fs.put_file(temporary.name, root + '/bundles/' + sha + '.tar.gz', timeout=60)
    records = []
    for shard in range(args.shards):
        task_id = f'{args.tag}-{shard:03d}'
        task = dict(kind=args.kind, plan=args.plan, output=args.output, input=args.input,
            shard=shard, shards=args.shards, smoke=args.smoke, bundle=bundle, bundle_sha256=sha)
        key = root + '/queue/' + task_id + '.json'
        if fs.exists(key):
            raise ValueError('Task already submitted; use a fresh retry tag: ' + task_id)
        api.storage.write_json(task, args.control + '/queue/' + task_id + '.json')
        records.append(dict(task_id=task_id, control=args.control, source=manifest, task=task))
    with (HERE / 'data/server_tasks.jsonl').open('a') as stream:
        for record in records:
            stream.write(json.dumps(record) + '\n')
    print(json.dumps(dict(tasks=len(records), source_sha256=manifest['sha256'], bundle=bundle)))


if __name__ == '__main__':
    main()
