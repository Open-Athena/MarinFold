"""Arrow-backed rollout loading and deterministic distributed epoch ordering."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from records import make_example, prefix_length
from storage import filesystem


def stage_rollouts(source: str, destination: Path) -> dict:
    """Download a committed, audited dataset once per training node."""
    fs, root = filesystem(source)
    manifest = json.loads(fs.cat(root + '/_SUCCESS.json'))
    destination.mkdir(parents=True, exist_ok=True)
    for item in manifest['files']:
        target = destination / item['name']
        if target.exists() and target.stat().st_size == item['size']:
            continue
        temporary = target.with_suffix('.partial')
        fs.get_file(root + '/' + item['name'], str(temporary))
        if temporary.stat().st_size != item['size']:
            raise ValueError(f'Dataset staging failed: {target}')
        temporary.replace(target)
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest


def read_table(directory: Path, split: str) -> pa.Table:
    """Load one split and explicitly remove unparseable/empty trajectories."""
    manifest = json.loads((directory / 'manifest.json').read_text())
    tables = []
    for item in manifest['files']:
        table = pq.read_table(directory / item['name'])
        table = table.filter(pc.equal(table['split'], split))
        table = table.filter(pc.and_(pc.equal(table['malformed'], 0), pc.greater(pc.list_value_length(table['contact_ends']), 0)))
        if len(table):
            tables.append(table)
    if not tables:
        raise ValueError(f'No retained examples in {split}')
    return pa.concat_tables(tables)


def example(table: pa.Table, index: int, assessment_id: int, epoch: int, k: int | None = None):
    """Materialize a single prefix without unpacking the corpus into Python lists."""
    row = table.slice(index, 1).to_pylist()[0]
    size = len(row['contact_ends'])
    count = prefix_length(size, row['identity'], epoch) if k is None else min(k, size)
    return make_example(row, count, assessment_id)


def epoch_order(size: int, epoch: int, world_size: int, accumulation: int) -> np.ndarray:
    """Shuffle all examples, padding only the final global accumulation group."""
    rng = np.random.default_rng(356 + epoch)
    order = rng.permutation(size)
    multiple = world_size * accumulation
    padding = (-size) % multiple
    if padding:
        order = np.concatenate([order, np.resize(order, padding)])
    return order.reshape(-1, accumulation, world_size)


def fingerprint(manifest: dict) -> str:
    """Identify the exact immutable training objects and their audit."""
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
