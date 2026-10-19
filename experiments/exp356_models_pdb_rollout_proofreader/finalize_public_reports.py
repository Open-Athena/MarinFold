"""Publish post-release verification evidence without changing verified model bytes."""

import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import HfFileSystem

from publish_to_hf import file_manifest, sync
from storage import ROOT, upload_directory, write_json

HERE = Path(__file__).resolve().parent


def main() -> None:
    """Commit an updated report bundle after the anonymous inference check passes."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--reports-local', type=Path, required=True)
    args = parser.parse_args()
    release = json.loads((args.reports_local/'release.json').read_text())
    verification = json.loads((HERE/'data/public_verification.json').read_text())
    if verification['checkpoint'] != release['model'] or not all(verification[key] for key in
            ['anonymous_download','all_file_hashes_verified','offline_inference']):
        raise ValueError('Anonymous verification does not establish this release')
    local = HERE/'_cache/final-reports'
    shutil.copytree(args.reports_local, local, dirs_exist_ok=True)
    marker = local/'_SUCCESS.json'
    marker.unlink(missing_ok=True)
    for path in sorted((HERE/'data').glob('public_verification*')):
        shutil.copyfile(path, local/path.name)
    shutil.copyfile(HERE/'plots/summary.pdf', local/'summary.pdf')
    files = file_manifest(local)
    total = sum(item['size'] for item in files.values())
    if total > 10_000_000_000:
        raise ValueError('Final reports exceed the explicit-approval transfer limit')
    run, step = release['model'].rstrip('/').split('/')[-2:]
    destination = f'{ROOT}/reports/release/{run}/{step}-verified'
    uploaded = upload_directory(local, destination)
    if uploaded != files:
        raise ValueError('Final report files changed during upload')
    manifest = dict(release, files=files, public_verification=verification)
    write_json(manifest, destination+'/_SUCCESS.json')
    marker.write_text(json.dumps(manifest, indent=2))
    sync(local, release['reports'])
    fs = HfFileSystem(token=False)
    public = release['reports'].removeprefix('hf://')
    if json.loads(fs.cat(public+'/_SUCCESS.json')) != manifest:
        raise ValueError('Anonymous report manifest differs from the committed bundle')
    if json.loads(fs.cat(public+'/public_verification.json')) != verification:
        raise ValueError('Anonymous verification record differs from the measured result')
    for name, item in files.items():
        if fs.info(public+'/'+name)['size'] != item['size']:
            raise ValueError(f'Public report size differs: {name}')
    result = dict(reports=release['reports'], source=destination, bytes=total,
        verified_model=release['model'], anonymous_reports_verified=True)
    (HERE/'data/release_finalization.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
