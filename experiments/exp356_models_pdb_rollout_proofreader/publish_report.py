"""Publish the written report and its small reproducibility bundle."""

import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from huggingface_hub import HfFileSystem


HERE = Path(__file__).resolve().parent
DESTINATION = 'hf://buckets/open-athena/MarinFold/data/exp356/written-report'
INPUTS = [
    'structure_manifest.csv.gz',
    'validation_step2000_metrics.json',
    'frozen_validation_step2000_metrics.json',
    'exp356-exp277-frozen-contact-pdb50k-v2_learning.csv',
    'release_selection.json',
    'final_test_metrics.json',
    'final_test_calibration.csv',
]


def main() -> None:
    """Upload only report artifacts, then anonymously verify every file hash."""
    subprocess.run(['uv', 'run', '--project', str(HERE), '--no-sync',
                    'python', str(HERE / 'plot_report.py')], check=True)
    sources = [HERE / name for name in ['report.md', 'plot_report.py', 'build_summary.py',
                                        'publish_report.py', 'pyproject.toml', 'uv.lock']]
    sources += [HERE / 'data' / name for name in INPUTS]
    sources += sorted((HERE / 'plots').glob('report_*.png*'))
    total = sum(path.stat().st_size for path in sources)
    if total > 100_000_000:
        raise ValueError(f'The written-report bundle is unexpectedly large: {total} bytes')
    with tempfile.TemporaryDirectory(prefix='exp356-written-report-') as temporary:
        staging = Path(temporary)
        manifest = {}
        for source in sources:
            relative = source.relative_to(HERE)
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            manifest[str(relative)] = {
                'bytes': target.stat().st_size,
                'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
            }
        (staging / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        subprocess.run(['hf', 'buckets', 'sync', str(staging), DESTINATION, '--quiet'], check=True)
        anonymous = HfFileSystem(token=False)
        root = DESTINATION.removeprefix('hf://')
        for relative, expected in manifest.items():
            with anonymous.open(f'{root}/{relative}', 'rb') as handle:
                payload = handle.read()
            if len(payload) != expected['bytes'] or hashlib.sha256(payload).hexdigest() != expected['sha256']:
                raise ValueError(f'Public report verification failed: {relative}')
        with anonymous.open(f'{root}/manifest.json') as handle:
            if json.load(handle) != manifest:
                raise ValueError('Public report manifest differs from the local bundle')
        print(json.dumps({'destination': DESTINATION, 'verified_files': len(manifest) + 1,
                          'artifact_bytes': total}, indent=2))


if __name__ == '__main__':
    main()
