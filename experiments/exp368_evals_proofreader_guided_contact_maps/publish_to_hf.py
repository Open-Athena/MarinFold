"""Publish reproducible small experiment artifacts to the public MarinFold bucket."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from huggingface_hub import HfFileSystem

from runtime import load_release

HERE = Path(__file__).resolve().parent
DESTINATION = 'hf://buckets/open-athena/MarinFold/data/exp368/reports'


def frozen_sources() -> dict[str, Path]:
    """Cache and verify the small immutable bundles used by executed phases."""
    records = [json.loads(line) for line in (HERE / 'data/server_tasks.jsonl').read_text().splitlines()]
    bundles = {record['task']['bundle_sha256']: record['task']['bundle'] for record in records}
    api = load_release(('storage',))
    result = {}
    total = 0
    for sha, uri in sorted(bundles.items()):
        if '/MarinFold/exp368/' not in uri or not uri.endswith('/bundles/' + sha + '.tar.gz'):
            raise ValueError('Unexpected immutable source bundle: ' + uri)
        destination = HERE / '_cache/source_bundles' / (sha + '.tar.gz')
        fs, key = api.storage.filesystem(uri)
        size = fs.info(key)['size']
        total += size
        if size > 10_000_000 or total > 100_000_000:
            raise ValueError('Frozen-source download exceeds its small-artifact ceiling')
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            partial = destination.with_suffix('.partial')
            fs.get_file(key, str(partial))
            partial.replace(destination)
        if hashlib.sha256(destination.read_bytes()).hexdigest() != sha:
            raise ValueError('Frozen source hash mismatch: ' + uri)
        result['source_bundles/' + destination.name] = destination
    native = HERE / 'data/native_bootstrap.json'
    if native.exists():
        bootstrap = json.loads(native.read_text())
        path = HERE / '_cache' / (bootstrap['config_map'] + '.tar.gz')
        if hashlib.sha256(path.read_bytes()).hexdigest() != bootstrap['bundle_sha256']:
            raise ValueError('Native bootstrap source hash mismatch')
        result['source_bundles/' + bootstrap['bundle_sha256'] + '.tar.gz'] = path
    return result


def main() -> None:
    """Rebuild the PDF, upload the explicit artifact set and verify public reads."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    subprocess.run([sys.executable, str(HERE / 'build_summary.py')], check=True)
    files = [path for path in HERE.iterdir() if path.is_file() and path.suffix in
             {'.py', '.md', '.json', '.sh', '.toml', '.lock'} and path.name != 'code_manifest.json']
    files += [path for directory in ['data', 'plots', 'release_api']
              for path in (HERE / directory).rglob('*') if path.is_file() and '__pycache__' not in path.parts]
    artifacts = {str(path.relative_to(HERE)): path for path in files}
    artifacts.update(frozen_sources())
    total = sum(path.stat().st_size for path in artifacts.values())
    if total > 1_000_000_000:
        raise ValueError('Small-artifact publication exceeds its 1 GB ceiling')
    manifest = {}
    for name, path in sorted(artifacts.items()):
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        manifest[name] = dict(size=path.stat().st_size, sha256=digest)
    result = dict(destination=DESTINATION, total_bytes=total, files=manifest,
                  git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=HERE, text=True).strip(),
                  working_tree_dirty=bool(subprocess.check_output(
                      ['git', 'status', '--porcelain', '--', str(HERE)], cwd=HERE, text=True).strip()),
                  published_at=datetime.now(UTC).isoformat())
    if args.dry_run:
        print(json.dumps(result, indent=2))
        return
    with tempfile.TemporaryDirectory(prefix='exp368-publication-') as temporary:
        root = Path(temporary)
        for name, path in artifacts.items():
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        (root / 'publication.json').write_text(json.dumps(result, indent=2))
        subprocess.run(['hf', 'buckets', 'sync', str(root), DESTINATION, '--quiet'], check=True)
    anonymous = HfFileSystem(token=False)
    public = DESTINATION.removeprefix('hf://')
    actual = json.loads(anonymous.read_text(public + '/publication.json'))
    if actual != result:
        raise ValueError('Public publication manifest differs from the uploaded artifacts')
    for name, item in manifest.items():
        if anonymous.info(public + '/' + name)['size'] != item['size']:
            raise ValueError('Public artifact has the wrong size: ' + name)
    for name in ['README.md', 'plots/summary.pdf']:
        if hashlib.sha256(anonymous.read_bytes(public + '/' + name)).hexdigest() != manifest[name]['sha256']:
            raise ValueError('Public report hash differs from its manifest: ' + name)
    (HERE / '_cache/publication.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(destination=DESTINATION, files=len(manifest), bytes=total), indent=2))


if __name__ == '__main__':
    main()
