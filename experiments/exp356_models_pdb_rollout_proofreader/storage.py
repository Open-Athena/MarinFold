"""Durable, co-located storage and provenance for the proofreader experiment."""

import hashlib
import csv
import io
import json
import os
import shutil
from pathlib import Path

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = 's3://marin-us-east-02a/MarinFold/exp356'
GENERATOR = 's3://marin-us-east-02a/MarinFold/exp304/model/exp277-step-266344'


def filesystem(uri: str):
    """Use Iris-injected S3 settings in pods and the named CW profile locally."""
    options = {}
    if uri.startswith('s3://') and not os.environ.get('FSSPEC_S3'):
        options = dict(profile='cw', endpoint_url='https://cwobject.com',
                       config_kwargs={'s3': {'addressing_style': 'virtual'}})
    endpoint = os.environ.get('EXP356_S3_ENDPOINT')
    if uri.startswith('s3://') and endpoint:
        if endpoint != 'https://cwobject.com':
            raise ValueError('The storage override must address the same CoreWeave object store')
        # Long-lived clients stalled with both LOTA and the regional origin,
        # while fresh clients could write immediately. Avoid reusing connections;
        # credentials still come from Iris and storage stays in the same region.
        options.update(endpoint_url=endpoint,config_kwargs={'connect_timeout':5,'read_timeout':30,
            'retries':{'max_attempts':2},'s3':{'addressing_style':'virtual'},
            'connector_args':{'force_close':True,'keepalive_timeout':None}})
    return fsspec.core.url_to_fs(uri, **options)


def read_rows(uri: str) -> list[dict]:
    """Read a parquet through the configured filesystem."""
    fs, path = filesystem(uri)
    with fs.open(path, 'rb') as handle:
        return pq.read_table(handle).to_pylist()


def write_rows(rows: list[dict], uri: str) -> None:
    """Publish a complete parquet object after serialization succeeds."""
    buffer = pa.BufferOutputStream()
    pq.write_table(pa.Table.from_pylist(rows), buffer, compression='zstd')
    fs, path = filesystem(uri)
    fs.pipe_file(path, buffer.getvalue().to_pybytes())


def write_json(value: dict, uri: str) -> None:
    """Write a JSON object, used last as the commit marker for grouped artifacts."""
    fs, path = filesystem(uri)
    # Bound the whole operation as well as socket reads, so a stuck client fails
    # visibly and the job can recover from its last committed optimizer cursor.
    fs.pipe_file(path, json.dumps(value, indent=2).encode(), timeout=60)


def write_csv(rows: list[dict], uri: str) -> None:
    """Persist measured predictor timings in the repository's portable CSV format."""
    if not rows:
        raise ValueError('Cannot write an empty timing ledger')
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    fs, path = filesystem(uri)
    fs.pipe_file(path, buffer.getvalue().encode())


def stage_directory(source: str, destination: Path, *, include_training_state: bool = True) -> None:
    """Stage a flat checkpoint directory with size-checked resumable files."""
    fs, root = filesystem(source)
    files = [f for f in fs.ls(root, detail=True) if f['type'] == 'file']
    if not include_training_state:
        files = [f for f in files if Path(f['name']).name != 'training_state.pt']
    if not files:
        raise FileNotFoundError(source)
    marker = destination / '.source.json'
    if destination.exists() and (not marker.exists() or json.loads(marker.read_text()) != source):
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for item in files:
        target = destination / Path(item['name']).name
        if target.exists() and target.stat().st_size == item['size']:
            continue
        temporary = target.with_suffix(target.suffix + '.partial')
        fs.get_file(item['name'], str(temporary))
        if temporary.stat().st_size != item['size']:
            raise ValueError(f'Staging size mismatch: {target.name}')
        temporary.replace(target)
    marker.write_text(json.dumps(source))


def upload_directory(source: Path, destination: str) -> dict:
    """Upload flat checkpoint files and return their sizes and SHA256 hashes."""
    fs, root = filesystem(destination)
    manifest = {}
    for path in sorted(source.iterdir()):
        if not path.is_file() or path.name.startswith('.'):
            continue
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
        fs.put_file(str(path), root + '/' + path.name, timeout=300)
        manifest[path.name] = dict(size=path.stat().st_size, sha256=digest.hexdigest())
    return manifest
