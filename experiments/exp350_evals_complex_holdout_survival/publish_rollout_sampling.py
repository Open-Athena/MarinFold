"""Publish exp350 individual rollouts and their derived sampling diagnostics."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import fsspec

HERE = Path(__file__).resolve().parent
PREFIX = 'hf://buckets/open-athena/MarinFold/data/evals/exp350_foldbench_pair_holdout/contact_eval_v1/sampling_v1'


def main() -> None:
    """Verify manifest hashes and upload the raw and derived directories."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--derived', type=Path, default=HERE / 'data/sampling_v1')
    parser.add_argument('--source-s3', help='Optionally download raw files within CoreWeave first')
    args = parser.parse_args()
    manifest = json.loads((args.derived / 'manifest.json').read_text())
    if args.source_s3:
        filesystem, root = fsspec.core.url_to_fs(args.source_s3)
        entries = filesystem.find(root, detail=True)
        size = sum(entry['size'] for entry in entries.values() if entry['type'] == 'file')
        if size > 1_000_000_000:
            raise ValueError(f'Unexpectedly large sampling export: {size} bytes')
        for name, entry in entries.items():
            if entry['type'] != 'file':
                continue
            destination = args.raw / name.removeprefix(root + '/')
            destination.parent.mkdir(parents=True, exist_ok=True)
            filesystem.get_file(name, str(destination))
    for directory, hashes in ((args.raw, manifest['raw_files']),
                              (args.derived, manifest['derived'])):
        for name, expected in hashes.items():
            path = directory / name
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f'Artifact hash mismatch: {path}')
    for local, suffix in ((args.raw, 'raw'), (args.derived, 'derived')):
        subprocess.run(['uvx', '--from', 'huggingface-hub>=2.1,<3', 'hf',
                        'buckets', 'sync', str(local), f'{PREFIX}/{suffix}'], check=True)


if __name__ == '__main__':
    main()
