"""Consolidate contemporaneous training-validation timings for publication."""

import argparse
import io
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from storage import ROOT, filesystem

HERE = Path(__file__).resolve().parent


def main() -> None:
    """Preserve every recorded validation invocation, including resumed evaluations."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    fs, root = filesystem(f'{ROOT}/runs/{args.run}/validation-timings')
    files = sorted(fs.glob(root+'/**/*.csv'))
    if not files:
        raise FileNotFoundError(root)
    with ThreadPoolExecutor(max_workers=8) as pool:
        blobs = list(pool.map(fs.cat_file, files))
    frames = []
    for source, blob in zip(files, blobs, strict=True):
        frame = pd.read_csv(io.BytesIO(blob))
        frame['measurement_file'] = source.removeprefix(root+'/')
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    destination = HERE/'data'/f'{args.run}_validation_timings.csv.gz'
    result.to_csv(destination,index=False)
    print(f'{len(result)} measured inputs from {len(files)} files: {destination}')


if __name__ == '__main__':
    main()
