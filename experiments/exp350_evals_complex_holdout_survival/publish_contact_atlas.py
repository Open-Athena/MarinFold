"""Publish the contact-map PDF, page previews and its reproducible CSV inputs."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from build_contact_atlas import DATA, HERE, PAGES, PDF

PUBLIC = ('hf://buckets/open-athena/MarinFold/data/evals/exp350_foldbench_pair_holdout/'
          'contact_eval_v1/contact_atlas_v1')


def main() -> None:
    """Create a hash-checked report bundle and publish it to the shared bucket."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--staging', type=Path, default=Path('/tmp/exp350-contact-atlas-public'))
    args = parser.parse_args()
    provenance = json.loads((DATA / 'manifest.json').read_text())
    for name, expected in provenance['csv_sha256'].items():
        if hashlib.sha256((DATA / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Plot-data hash mismatch: {name}')
    args.staging.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PDF, args.staging / PDF.name)
    shutil.copytree(DATA, args.staging / 'data', dirs_exist_ok=True)
    shutil.copytree(PAGES, args.staging / 'pages', dirs_exist_ok=True)
    manifest = {'public_prefix': PUBLIC, 'pdf': PDF.name,
                'source_script_sha256': hashlib.sha256((HERE / 'build_contact_atlas.py').read_bytes()).hexdigest(),
                'files': {str(p.relative_to(args.staging)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(args.staging.rglob('*')) if p.is_file() and p != args.staging / 'bundle_manifest.json'}}
    (args.staging / 'bundle_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    subprocess.run(['uvx', '--from', 'huggingface-hub>=2.1,<3', 'hf', 'buckets',
                    'sync', str(args.staging), PUBLIC], check=True)
    print(f'Published {PDF.name} to {PUBLIC}', flush=True)


if __name__ == '__main__':
    main()
