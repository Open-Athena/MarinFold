"""Publish verified conditional-null outputs to the public MarinFold bucket."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    """Verify all saved hashes before uploading result tables and null draws."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--derived', type=Path, default=HERE / 'data/sampling_controls_v1')
    parser.add_argument('--raw', type=Path, default=Path('/tmp/exp350-sampling-controls-v1'))
    args = parser.parse_args()
    manifest = json.loads((args.derived / 'manifest.json').read_text())
    for directory, hashes in ((args.derived, manifest['derived_sha256']),
                              (args.raw, manifest['raw_sha256'])):
        for name, expected in hashes.items():
            if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f'Artifact hash mismatch: {directory / name}')
    for directory, suffix in ((args.derived, 'derived'), (args.raw, 'raw')):
        subprocess.run(['uvx', '--from', 'huggingface-hub>=2.1,<3', 'hf', 'buckets',
                        'sync', str(directory), f"{manifest['public_prefix']}/{suffix}"], check=True)


if __name__ == '__main__':
    main()
