# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0
"""Stage the bucket shards exp230 reads, and write the shard manifest.

**Run this with an interpreter that has ``huggingface_hub>=1.5``** — the bucket
API (``list_bucket_tree`` / ``download_bucket_files``) does not exist below
that, and ``snapshot_download`` cannot see buckets at all.  Everything
downstream imports ``marinfold``, whose transformers pins
``huggingface_hub<1``, so the two halves deliberately do not share a venv:

    /home/bizon/anaconda3/bin/python stage.py --work /data/exp230_multi --arm pdb
    uv run --with 'huggingface_hub>=1.5' python stage.py --work ... --arm afdb --n-shards 40

The manifest records the **corpus-wide sorted** listing for every arm, not just
the shards downloaded, so a partial draw still addresses the right row of the
published parquet (see ``corpus_sources.list_shards``).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from corpus_sources import ARMS, BUCKET, MANIFEST_NAME


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, default=Path("/data/exp230_multi"))
    ap.add_argument("--arm", action="append", choices=sorted(ARMS), default=None,
                    help="repeatable; default all three")
    ap.add_argument("--n-shards", type=int, default=None,
                    help="download only the first N shards of the seeded shuffle "
                         "(the manifest still lists every shard)")
    ap.add_argument("--seed", type=int, default=230)
    ap.add_argument("--list-only", action="store_true")
    a = ap.parse_args()

    from huggingface_hub import HfApi

    api = HfApi()
    cache = a.work / "cache"
    cache.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    tree = [p.path for p in api.list_bucket_tree(BUCKET, token=False)]
    print(f"[bucket] listed {len(tree):,} paths in {time.time() - t0:.0f}s", flush=True)

    manifest_file = cache / MANIFEST_NAME
    manifest = json.loads(manifest_file.read_text()) if manifest_file.exists() else {}

    arms = a.arm or sorted(ARMS)
    for arm in arms:
        spec = ARMS[arm]
        paths = sorted(p for p in tree
                       if p.startswith(spec.prefix + "/") and p.endswith(".parquet"))
        if not paths:
            raise SystemExit(f"no shards under {spec.prefix}/")
        manifest[arm] = paths
        manifest_file.write_text(json.dumps(manifest, indent=1))
        print(f"[{arm}] {len(paths):,} shards listed", flush=True)

        if a.list_only:
            continue

        # Restrict the draw to the eligible range (AFDB: round-0 shards only)
        # while the manifest keeps the corpus-wide listing, so shard ordinals
        # stay corpus-wide addresses.
        eligible = paths[spec.first_shard:]
        if spec.first_shard:
            print(f"[{arm}] eligible: {len(eligible):,} shards from ordinal "
                  f"{spec.first_shard} — {spec.first_shard_note}", flush=True)
        want = eligible
        if a.n_shards is not None and a.n_shards < len(eligible):
            # Shuffle a SORTED list with a fixed seed: reproducible, unlike
            # #163's selector, which shuffled an unstably-ordered glob.
            import random

            rng = random.Random(a.seed)
            want = sorted(rng.sample(eligible, a.n_shards))
        todo = [(p, cache / p.replace("/", "__")) for p in want
                if not (cache / p.replace("/", "__")).exists()]
        if not todo:
            print(f"[{arm}] all {len(want)} wanted shards already staged", flush=True)
            continue
        t1 = time.time()
        print(f"[{arm}] downloading {len(todo)} shard(s)", flush=True)
        api.download_bucket_files(BUCKET, todo, token=False)
        got = sum(dst.stat().st_size for _, dst in todo)
        dt = time.time() - t1
        print(f"[{arm}] {got / 1e6:.0f} MB in {dt:.0f}s ({got / 1e6 / max(dt, 1):.1f} MB/s)",
              flush=True)

    print(f"[manifest] -> {manifest_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
