# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Cloud-side crops-corpus mirror: HF bucket -> GCS, run ON an iris pod.

The workstation's uplink to GCS is ~2.5 MB/s (asymmetric link), so a 55 GB
workstation->GCS upload takes ~6 h. A pod in the marin cluster reads the HF
bucket and writes to gs://marin-us-east5 both at cloud speeds, finishing in
minutes. This is the fast path (``mirror_crops_corpus.py`` is the slow local one).

The pod reads with the ``hf`` CLI (``hf buckets cp``, huggingface_hub >= 1.5 +
hf_xet — pip-installed into a scratch prefix at launch so it never perturbs the
locked training env) and writes with gcsfs using the pod's service account (which
has marin-us-east5 write, same as marin's checkpoint writes). Resumable: skips
shards already present in GCS with a matching size.

Env:
* ``HF_BIN``   path to an ``hf`` (>=1.5) binary (default: ``hf`` on PATH)
* ``MIRROR_SPLITS`` comma list (default ``train,val,test``)
* ``MIRROR_WORKERS`` thread count (default 16 — cloud-side, no uplink cap)
"""
from __future__ import annotations

import os
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import gcsfs

BUCKET_ROOT = (
    "hf://buckets/open-athena/MarinFold/data/document_structures/contacts_and_crops_v1"
)
DST = (
    "gs://marin-us-east5/protein-structure/MarinFold/exp132_contacts_and_crops_v1/documents"
)
SHARDS = {"train": 2067, "val": 22, "test": 22}
HF_BIN = os.environ.get("HF_BIN", "hf")

fs = gcsfs.GCSFileSystem(skip_instance_cache=True)


def existing_sizes(split: str) -> dict[str, int]:
    try:
        return {os.path.basename(i["name"]): i["size"] for i in fs.ls(f"{DST}/{split}", detail=True)}
    except FileNotFoundError:
        return {}


def one(split: str, fn: str, tmp: str) -> int:
    src = f"{BUCKET_ROOT}/{split}/{fn}"
    local = os.path.join(tmp, fn)
    subprocess.run([HF_BIN, "buckets", "cp", src, local], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    sz = os.path.getsize(local)
    with open(local, "rb") as a, fs.open(f"{DST}/{split}/{fn}", "wb") as b:
        while True:
            chunk = a.read(16 << 20)
            if not chunk:
                break
            b.write(chunk)
    os.remove(local)
    return sz


def main() -> int:
    splits = [s.strip() for s in os.environ.get("MIRROR_SPLITS", "train,val,test").split(",") if s.strip()]
    workers = int(os.environ.get("MIRROR_WORKERS", "16"))
    tmp = tempfile.mkdtemp(prefix="mirror_")
    t0 = time.time()
    grand_done = grand_total = 0
    for split in splits:
        n = SHARDS[split]
        have = existing_sizes(split)
        # We don't know each shard's size a priori (bucket ls is another call);
        # re-copy only those missing from GCS entirely (size check on re-runs is
        # cheap since present ones are skipped by name).
        names = [f"contacts_and_crops_v1-{i:05d}-of-{n:05d}.parquet" for i in range(n)]
        todo = [fn for fn in names if fn not in have]
        grand_total += len(todo)
        print(f"[{split}] {n} shards, {len(names)-len(todo)} present, {len(todo)} to copy", flush=True)
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(one, split, fn, tmp): fn for fn in todo}
            for fut in as_completed(futs):
                fn = futs[fut]
                try:
                    fut.result()
                    done += 1
                    grand_done += 1
                    if done % 100 == 0 or done == len(todo):
                        el = time.time() - t0
                        print(f"[{split}] {done}/{len(todo)} ({el:.0f}s, {grand_done/el:.1f}/s)", flush=True)
                except Exception as e:  # noqa: BLE001
                    err = getattr(e, "stderr", b"")
                    print(f"[{split}] FAIL {fn}: {type(e).__name__} {err[:200]!r}", flush=True)
        print(f"[{split}] done {done}/{len(todo)}", flush=True)
    # verify
    miss = 0
    for split in splits:
        have = existing_sizes(split)
        n = SHARDS[split]
        for i in range(n):
            fn = f"contacts_and_crops_v1-{i:05d}-of-{n:05d}.parquet"
            if fn not in have:
                miss += 1
    print(f"MIRROR DONE {grand_done} copied in {time.time()-t0:.0f}s; missing={miss}", flush=True)
    return 1 if miss else 0


if __name__ == "__main__":
    raise SystemExit(main())
