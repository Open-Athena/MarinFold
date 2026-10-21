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
    entries = [(name, item['size'], item.get('ETag')) for name, item in objects.items() if item['type'] == 'file']
    if sum(size for _, size, _ in entries) > 5 * 10**9:
        raise ValueError('Result collection exceeds its 5 GB transfer ceiling; consolidate in-region first')
    ledger_path = args.local / '_collection.json'
    previous = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}

    def download(item: tuple[str, int, str | None]) -> tuple[str, dict]:
        name, size, etag = item
        relative = name.removeprefix(root.rstrip('/') + '/')
        destination = args.local / relative
        metadata = dict(size=size, etag=etag)
        if etag and previous.get(relative) == metadata and destination.exists() and destination.stat().st_size == size:
            return relative, metadata
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(destination.suffix + '.partial')
        fs.get_file(name, str(partial))
        if partial.stat().st_size != size:
            raise ValueError('Size mismatch: ' + relative)
        partial.replace(destination)
        return relative, metadata

    with ThreadPoolExecutor(max_workers=8) as executor:
        current = dict(executor.map(download, entries))
    ledger_path.write_text(json.dumps(current, indent=2))
    print(json.dumps(dict(files=len(entries), bytes=sum(size for _, size, _ in entries), local=str(args.local))))


if __name__ == '__main__':
    main()
