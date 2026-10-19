"""Salvage a fully written local checkpoint through the co-located S3 origin."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import fsspec


def main() -> None:
    """Upload and commit model/optimizer files without changing recovery pointers."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--local', type=Path, required=True)
    parser.add_argument('--destination', required=True)
    args = parser.parse_args()
    metadata = json.loads((args.local/'proofreader.json').read_text())
    if not (args.local/'training_state.pt').exists():
        raise FileNotFoundError('Recovery requires the optimizer and exact training cursor')
    fs, root = fsspec.core.url_to_fs(args.destination, endpoint_url='https://cwobject.com',
        config_kwargs={'connect_timeout':5,'read_timeout':30,'retries':{'max_attempts':2},
                       's3':{'addressing_style':'virtual'}})
    files = {}
    for path in sorted(args.local.iterdir()):
        if not path.is_file() or path.name.startswith('.') or path.name=='manifest.json':
            continue
        started = time.monotonic()
        with path.open('rb') as handle:
            digest = hashlib.file_digest(handle,'sha256').hexdigest()
        fs.put_file(str(path),root+'/'+path.name)
        size = path.stat().st_size
        if fs.info(root+'/'+path.name)['size'] != size:
            raise ValueError(f'Recovery upload size mismatch: {path.name}')
        files[path.name] = dict(size=size,sha256=digest)
        print(json.dumps(dict(file=path.name,bytes=size,seconds=time.monotonic()-started)),flush=True)
    manifest = dict(metadata,files=files,recovery_transport='same-region CoreWeave S3 origin')
    fs.pipe_file(root+'/manifest.json',json.dumps(manifest,indent=2).encode())
    print(json.dumps(dict(recovered=args.destination,step=metadata['step'])),flush=True)


if __name__ == '__main__':
    main()
