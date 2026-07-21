# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Cloud-side crops-corpus mirror: HF bucket -> GCS, run ON an iris pod.

The workstation's uplink to GCS is ~2.5 MB/s (asymmetric link), so a 55 GB
workstation->GCS upload takes ~6 h. A pod in the marin cluster reads the HF
bucket and writes to gs://marin-us-east5 both at cloud speeds, finishing in
minutes. This is the fast path (``mirror_crops_corpus.py`` is the slow local one).

The pod reads with ``huggingface_hub.download_bucket_files`` (>= 1.5 + hf_xet,
pip-installed at launch — anon, no token) and writes with gcsfs using the pod's
service account (which has marin-us-east5 write, same as marin's checkpoint
writes). Resumable: skips shards already present in GCS.

Env:
* ``MIRROR_SPLITS``  comma list (default ``train,val,test``)
* ``MIRROR_WORKERS`` thread count (default 16 — cloud-side, no uplink cap)
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


_PKGS = ["huggingface_hub>=1.5,<2", "hf_xet", "gcsfs", "google-auth"]


def _have_bucket_api() -> bool:
    try:
        from huggingface_hub import download_bucket_files  # noqa: F401
        import gcsfs  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def _ensure_deps() -> None:
    """Install the bucket API (hub>=1.5) + gcsfs into THIS interpreter.

    The synced project env pins hub 0.36 (transitive), which lacks the bucket
    Python API. We upgrade in-place using the same interpreter iris launched, so
    there's no venv/PATH ambiguity (the reason a ``bash -lc`` wrapper failed).
    """
    if _have_bucket_api():
        return
    attempts = (
        [sys.executable, "-m", "pip", "install", "-q", "--upgrade", *_PKGS],
        ["uv", "pip", "install", "-q", "--upgrade", "--python", sys.executable, *_PKGS],
    )
    for cmd in attempts:
        try:
            subprocess.run(cmd, check=True)
            if _have_bucket_api():
                return
        except Exception as e:  # noqa: BLE001
            print(f"[ensure_deps] {cmd[0]} failed: {e}", flush=True)
    subprocess.run([sys.executable, "-m", "ensurepip", "--upgrade"], check=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", *_PKGS], check=True)


_ensure_deps()

import gcsfs  # noqa: E402
import huggingface_hub  # noqa: E402
from huggingface_hub import download_bucket_files  # noqa: E402

print(f"[mirror] python={sys.executable} hub={huggingface_hub.__version__} "
      f"gcsfs={gcsfs.__version__}", flush=True)

BUCKET_ID = "open-athena/MarinFold"
BUCKET_PREFIX = "data/document_structures/contacts_and_crops_v1"
DST = (
    "gs://marin-us-east5/protein-structure/MarinFold/exp132_contacts_and_crops_v1/documents"
)
SHARDS = {"train": 2067, "val": 22, "test": 22}

fs = gcsfs.GCSFileSystem(skip_instance_cache=True)


def existing(split: str) -> set[str]:
    try:
        return {os.path.basename(i["name"]) for i in fs.ls(f"{DST}/{split}", detail=True)}
    except FileNotFoundError:
        return set()


def one(split: str, fn: str, tmp: str) -> int:
    src = f"{BUCKET_PREFIX}/{split}/{fn}"
    local = os.path.join(tmp, fn)
    last = None
    for attempt in range(4):
        try:
            download_bucket_files(BUCKET_ID, files=[(src, local)], token=False,
                                  raise_on_missing_files=True)
            sz = os.path.getsize(local)
            with open(local, "rb") as a, fs.open(f"{DST}/{split}/{fn}", "wb") as b:
                while True:
                    chunk = a.read(16 << 20)
                    if not chunk:
                        break
                    b.write(chunk)
            if fs.info(f"{DST}/{split}/{fn}")["size"] != sz:
                raise RuntimeError("size mismatch after write")
            os.remove(local)
            return sz
        except Exception as e:  # noqa: BLE001
            last = e
            if os.path.exists(local):
                try:
                    os.remove(local)
                except OSError:
                    pass
            time.sleep(2 * (attempt + 1))
    raise last


def main() -> int:
    splits = [s.strip() for s in os.environ.get("MIRROR_SPLITS", "train,val,test").split(",") if s.strip()]
    workers = int(os.environ.get("MIRROR_WORKERS", "16"))
    tmp = tempfile.mkdtemp(prefix="mirror_")
    t0 = time.time()
    grand = 0
    for split in splits:
        n = SHARDS[split]
        have = existing(split)
        names = [f"contacts_and_crops_v1-{i:05d}-of-{n:05d}.parquet" for i in range(n)]
        todo = [fn for fn in names if fn not in have]
        print(f"[{split}] {n} shards, {len(names)-len(todo)} present, {len(todo)} to copy", flush=True)
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(one, split, fn, tmp): fn for fn in todo}
            for fut in as_completed(futs):
                fn = futs[fut]
                try:
                    fut.result()
                    done += 1
                    grand += 1
                    if done % 100 == 0 or done == len(todo):
                        el = time.time() - t0
                        print(f"[{split}] {done}/{len(todo)} ({el:.0f}s, {grand/el:.1f}/s)", flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f"[{split}] FAIL {fn}: {type(e).__name__} {str(e)[:200]}", flush=True)
        print(f"[{split}] done {done}/{len(todo)}", flush=True)
    miss = 0
    for split in splits:
        have = existing(split)
        for i in range(SHARDS[split]):
            if f"contacts_and_crops_v1-{i:05d}-of-{SHARDS[split]:05d}.parquet" not in have:
                miss += 1
    print(f"MIRROR DONE {grand} copied in {time.time()-t0:.0f}s; missing={miss}", flush=True)
    return 1 if miss else 0


if __name__ == "__main__":
    raise SystemExit(main())
