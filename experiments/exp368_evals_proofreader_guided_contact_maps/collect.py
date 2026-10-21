"""Collect bounded experiment outputs without copying any model checkpoints."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from runtime import load_release


def main() -> None:
    """Mirror small completed prediction/timing artifacts for local analysis."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--remote', required=True)
    parser.add_argument('--local', type=Path, required=True)
    args = parser.parse_args()
    if '/MarinFold/exp368/' not in args.remote or 'checkpoint' in args.remote:
        raise ValueError('Collector only accepts exp368 result prefixes')
    api = load_release(('storage',))
    fs, root = api.storage.filesystem(args.remote)
    objects = fs.find(root, detail=True)
    entries = [(name, item['size']) for name, item in objects.items() if item['type'] == 'file']
    if sum(size for _, size in entries) > 5 * 10**9:
        raise ValueError('Result collection exceeds its 5 GB transfer ceiling; consolidate in-region first')

    def download(item: tuple[str, int]) -> None:
        name, size = item
        relative = name.removeprefix(root.rstrip('/') + '/')
        destination = args.local / relative
        if destination.exists() and destination.stat().st_size == size:
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(destination.suffix + '.partial')
        fs.get_file(name, str(partial))
        if partial.stat().st_size != size:
            raise ValueError('Size mismatch: ' + relative)
        partial.replace(destination)

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(download, entries))
    print(json.dumps(dict(files=len(entries), bytes=sum(size for _, size in entries), local=str(args.local))))


if __name__ == '__main__':
    main()
