"""Load the published, hash-pinned proofreader API as a model dependency.

Only the small inference source files come from the public release. Large model
files and experimental inputs are read from the co-located CoreWeave store.
"""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

from huggingface_hub import HfFileSystem

HERE = Path(__file__).resolve().parent
RELEASE = 'buckets/open-athena/MarinFold/checkpoints/exp356-exp277-frozen-contact-pdb50k-v2/step-18000'
READER = 's3://marin-us-east-02a/MarinFold/exp356/checkpoints/exp356-exp277-frozen-contact-pdb50k-v2/step-18000'
GENERATOR = 's3://marin-us-east-02a/MarinFold/exp304/model/exp277-step-266344'
TARGETS = 's3://marin-us-east-02a/MarinFold/exp356/data/v1/targets.parquet'
ROOT = 's3://marin-us-east-02a/MarinFold/exp368'


def digest_file(path: Path) -> str:
    """Hash a file without loading large weights into memory."""
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def import_file(name: str, path: Path) -> ModuleType:
    """Import the explicitly pinned standalone model API under a unique name."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_release(names: tuple[str, ...] = ('model', 'records', 'storage')) -> SimpleNamespace:
    """Fetch and verify the release's small API files, never its weight files."""
    manifest = json.loads((HERE / 'release_manifest.json').read_text())
    directory = HERE / 'release_api'
    directory.mkdir(parents=True, exist_ok=True)
    fs = HfFileSystem(token=False)
    modules = {}
    for name in names:
        filename = name + '.py'
        destination = directory / filename
        expected = manifest['files'][filename]['sha256']
        if not destination.exists() or digest_file(destination) != expected:
            destination.write_bytes(fs.read_bytes(RELEASE + '/' + filename))
        if digest_file(destination) != expected:
            raise ValueError(f'Published API hash changed: {filename}')
        modules[name] = import_file('exp368_release_' + name, destination)
    return SimpleNamespace(**modules)


def stage_reader(api: SimpleNamespace, destination: Path) -> None:
    """Stage and hash-check the released reader against its public manifest."""
    api.storage.stage_directory(READER, destination, include_training_state=False)
    manifest = json.loads((HERE / 'release_manifest.json').read_text())
    for name in ['model.safetensors', 'heads.pt', 'config.json', 'proofreader.json',
                 'tokenizer.json', 'tokenizer_config.json']:
        if digest_file(destination / name) != manifest['files'][name]['sha256']:
            raise ValueError(f'Checkpoint differs from release: {name}')


def verify_source() -> dict:
    """Reject a bundle that differs from the source manifest saved at launch."""
    manifest = json.loads((HERE / 'code_manifest.json').read_text())
    for name, expected in manifest['files'].items():
        if digest_file(HERE / name) != expected:
            raise ValueError(f'Bundled source changed after manifest creation: {name}')
    return manifest


def pin_run(api: SimpleNamespace, plan: dict, out: str, shard: int, **extra) -> dict:
    """Pin code, plan and inputs before reading any partial resume outputs."""
    source = verify_source()
    pin = dict(plan_sha256=hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest(),
        code_sha256=source['sha256'], generator=GENERATOR, reader=READER, **extra)
    fs, root = api.storage.filesystem(out)
    key = root + f'/shard-{shard:03d}.pin.json'
    if fs.exists(key) and json.loads(fs.cat(key)) != pin:
        raise ValueError('Code/plan changed on resume; use a fresh output prefix')
    api.storage.write_json(pin, out + f'/shard-{shard:03d}.pin.json')
    return pin
