"""Copy a stopped, small result prefix with a byte-for-byte verification ledger."""

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from runtime import load_release


def main() -> None:
    """Verify a collected snapshot, publish it regionally, and read back every file."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--destination', required=True)
    parser.add_argument('--local', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    for uri in [args.source, args.destination]:
        if '/MarinFold/exp368/' not in uri or 'checkpoint' in uri:
            raise ValueError('Migration is limited to exp368 results')
    if args.source == args.destination:
        raise ValueError('Source and destination must differ')
    api = load_release(('storage',))
    source_fs, source_root = api.storage.filesystem(args.source)
    destination_fs, destination_root = api.storage.filesystem(args.destination)
    snapshot = json.loads((args.local / '_collection.json').read_text())

    def listing(fs, root: str) -> dict:
        return {name.removeprefix(root.rstrip('/') + '/'): dict(size=item['size'], etag=item.get('ETag'))
                for name, item in fs.find(root, detail=True).items() if item['type'] == 'file'}

    if listing(source_fs, source_root) != snapshot:
        raise ValueError('Source changed since collection; stop writers and recollect')
    total_bytes = sum(item['size'] for item in snapshot.values())
    if total_bytes > 1_000_000_000:
        raise ValueError('Migration exceeds the 1 GB result-copy ceiling')
    existing = listing(destination_fs, destination_root)
    if existing.keys() - snapshot.keys():
        raise ValueError('Destination contains unrelated files')

    def transfer(relative: str) -> tuple[str, dict]:
        path = args.local / relative
        value = path.read_bytes()
        if len(value) != snapshot[relative]['size']:
            raise ValueError('Local size mismatch: ' + relative)
        # These tiny, unencrypted, single-PUT objects use a content-MD5 ETag.
        # Reject other ETag forms rather than inferring source-byte integrity.
        etag = (snapshot[relative]['etag'] or '').strip('"')
        if len(etag) != 32 or hashlib.md5(value, usedforsecurity=False).hexdigest() != etag:
            raise ValueError('Local content differs from the source ETag: ' + relative)
        sha256 = hashlib.sha256(value).hexdigest()
        target = destination_root + '/' + relative
        if relative not in existing:
            destination_fs.put_file(str(path), target, timeout=60)
        if hashlib.sha256(destination_fs.cat(target)).hexdigest() != sha256:
            raise ValueError('Destination conflicts with snapshot: ' + relative)
        return relative, dict(snapshot[relative], sha256=sha256)

    with ThreadPoolExecutor(max_workers=16) as executor:
        verified = dict(executor.map(transfer, sorted(snapshot)))
    if listing(source_fs, source_root) != snapshot:
        raise ValueError('Source changed during migration')
    manifest = dict(source=args.source, destination=args.destination,
                    verified_at=datetime.now(timezone.utc).isoformat(),
                    total_bytes=total_bytes, files=verified)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(files=len(verified), bytes=total_bytes, manifest=str(args.manifest))), flush=True)


if __name__ == '__main__':
    main()
