# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Cloud-side corpus mirror: HF bucket -> GCS, run ON an iris pod.

The workstation's uplink to GCS is ~2.5 MB/s (asymmetric link), so a >50 GB
workstation->GCS upload takes many hours. A pod in the marin cluster reads the HF
bucket and writes to gs://marin-* both at cloud speeds, finishing in minutes. This
is the fast path (``mirror_crops_corpus.py`` is the slow local one).

The pod reads with ``huggingface_hub.download_bucket_files`` (>= 1.5 + hf_xet,
supplied via ``uv run --with`` -- anon, but pass HF_TOKEN for a higher rate limit)
and writes with gcsfs using the pod's service account (which has marin-* write, the
same SA that writes marin checkpoints). Resumable: skips objects already present in
GCS with a matching size.

Parameterized by env (defaults mirror the contacts-and-crops-v1 corpus for
backwards-compat):
* ``MIRROR_SRC_BUCKET``   HF bucket id                 (default ``open-athena/MarinFold``)
* ``MIRROR_SRC_PREFIX``   path prefix within the bucket to mirror (recursive)
* ``MIRROR_DST``          gs:// destination dir; the SRC_PREFIX-relative path is preserved under it
* ``MIRROR_WORKERS``      thread count (default 8 -- HF 429-rate-limits higher)
"""
from __future__ import annotations

import os
import posixpath
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import gcsfs
import huggingface_hub
from huggingface_hub import download_bucket_files, list_bucket_tree

# hub>=1.5 (bucket Python API) + hf_xet + gcsfs come from `uv run --with ...` at
# launch, so the locked training env is never perturbed (see README launch cmd).
print(f"[mirror] python hub={huggingface_hub.__version__} gcsfs={gcsfs.__version__}", flush=True)

SRC_BUCKET = os.environ.get("MIRROR_SRC_BUCKET", "open-athena/MarinFold")
SRC_PREFIX = os.environ.get(
    "MIRROR_SRC_PREFIX", "data/document_structures/contacts_and_crops_v1"
).rstrip("/")
DST = os.environ["MIRROR_DST"].rstrip("/")
WORKERS = int(os.environ.get("MIRROR_WORKERS", "8"))
_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or False

fs = gcsfs.GCSFileSystem(skip_instance_cache=True)


def _rel(path: str) -> str:
    return path[len(SRC_PREFIX):].lstrip("/")


def existing() -> dict[str, int]:
    try:
        return {i["name"].split(f"{DST.split('://',1)[1]}/", 1)[-1]: i["size"]
                for i in fs.find(DST, detail=True).values()}
    except Exception:  # noqa: BLE001
        return {}


def one(src_path: str, tmp: str) -> int:
    rel = _rel(src_path)
    local = os.path.join(tmp, rel.replace("/", "_"))
    dst = f"{DST}/{rel}"
    last = None
    for attempt in range(8):
        try:
            download_bucket_files(SRC_BUCKET, files=[(src_path, local)], token=_TOKEN,
                                  raise_on_missing_files=True)
            sz = os.path.getsize(local)
            with open(local, "rb") as a, fs.open(dst, "wb") as b:
                while True:
                    chunk = a.read(16 << 20)
                    if not chunk:
                        break
                    b.write(chunk)
            if fs.info(dst)["size"] != sz:
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
            time.sleep(min(60, 3 * (2 ** attempt)) + (hash(rel) % 5))
    raise last


def main() -> int:
    print(f"[mirror] {SRC_BUCKET}:{SRC_PREFIX} -> {DST} (workers={WORKERS})", flush=True)
    items = list(list_bucket_tree(SRC_BUCKET, prefix=SRC_PREFIX, recursive=True, token=_TOKEN))
    srcs = [getattr(i, "path") for i in items
            if type(i).__name__ == "BucketFile" and getattr(i, "path", "").endswith(".parquet")]
    have = existing()
    todo = [s for s in srcs if _rel(s) not in have]
    print(f"[mirror] {len(srcs)} parquet; {len(srcs) - len(todo)} present; {len(todo)} to copy", flush=True)
    tmp = tempfile.mkdtemp(prefix="mirror_")
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(one, s, tmp): s for s in todo}
        for fut in as_completed(futs):
            s = futs[fut]
            try:
                fut.result()
                done += 1
                if done % 100 == 0 or done == len(todo):
                    el = time.time() - t0
                    print(f"[mirror] {done}/{len(todo)} ({el:.0f}s, {done/el:.1f}/s)", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[mirror] FAIL {_rel(s)}: {type(e).__name__} {str(e)[:200]}", flush=True)
    have2 = existing()
    miss = [s for s in srcs if _rel(s) not in have2]
    print(f"MIRROR DONE {done} copied in {time.time()-t0:.0f}s; missing={len(miss)}", flush=True)
    return 1 if miss else 0


if __name__ == "__main__":
    raise SystemExit(main())
