# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Mirror the contacts-and-crops-v1 corpus (exp132/#132) to region-local GCS.

The corpus is published only to the open-athena HF **bucket**
(``data/document_structures/contacts_and_crops_v1/{train,val,test}/``). HF
buckets are NOT levanter/fsspec-addressable — the tokenize step on the iris/TPU
worker resolves ``open-athena/MarinFold`` as a dataset repo and 404s (the
recurring exp53/85/108/120 gotcha). So we mirror the parquet shards byte-for-byte
to GCS next to the TPU, where the training tokenize step reads them.

The shards already carry a single ``document`` text column (plus crop-stat
columns), so no re-writing is needed — a straight copy preserves the published
bytes. Destination mirrors exp53's layout (data named by the datagen experiment,
separate from the exp137 training-run prefix):

    gs://marin-us-east5/protein-structure/MarinFold/
        exp132_contacts_and_crops_v1/documents/{train,val,test}/

Two sources:

* ``--source bucket`` (default, reproducible): download each shard from the HF
  bucket with the ``hf buckets`` CLI, then upload to GCS.
* ``--source local --local-dir <dir>``: copy directly from a local corpus dir
  (e.g. exp132's generation scratch ``/data/exp132_contacts_and_crops_v1_scratch/
  documents``), skipping the HF round-trip. This is what the exp137 launch used
  (the scratch shards are byte-identical to the published bucket — same
  deterministic generation, verified by shard size).

    uv run python mirror_crops_corpus.py --source local \\
        --local-dir /data/exp132_contacts_and_crops_v1_scratch/documents

Needs an ``hf`` CLI with ``buckets`` support on PATH for ``--source bucket``
(system hf, huggingface_hub >= 1.5). ``gcloud storage`` (or gsutil) for uploads.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import tempfile

BUCKET_ROOT = (
    "hf://buckets/open-athena/MarinFold/data/document_structures/contacts_and_crops_v1"
)
DEST_ROOT = (
    "gs://marin-us-east5/protein-structure/MarinFold/exp132_contacts_and_crops_v1/documents"
)
SPLITS = ("train", "val", "test")
SHARDS = {"train": 2067, "val": 22, "test": 22}


def _find_hf() -> str:
    import shutil

    for d in os.environ.get("PATH", "").split(os.pathsep):
        p = os.path.join(d, "hf")
        if os.path.exists(p) and ".venv" not in p and "/venv/" not in p:
            if subprocess.run([p, "buckets", "--help"], capture_output=True).returncode == 0:
                return p
    w = shutil.which("hf")
    if w and subprocess.run([w, "buckets", "--help"], capture_output=True).returncode == 0:
        return w
    raise RuntimeError("no `hf` with `buckets` support on PATH (need huggingface_hub >= 1.5)")


def _gcs_cp(src: str, dst: str) -> None:
    subprocess.run(["gcloud", "storage", "cp", src, dst], check=True)


def mirror_local(local_dir: str, splits: tuple[str, ...]) -> None:
    for split in splits:
        src = f"{local_dir.rstrip('/')}/{split}/*.parquet"
        dst = f"{DEST_ROOT}/{split}/"
        print(f"[mirror] local {src} -> {dst}", flush=True)
        _gcs_cp(src, dst)
        print(f"[mirror] {split} done", flush=True)


def mirror_bucket(splits: tuple[str, ...]) -> None:
    hf = _find_hf()
    tmp = tempfile.mkdtemp(prefix="exp137_crops_")
    for split in splits:
        n = SHARDS[split]
        for si in range(n):
            fn = f"contacts_and_crops_v1-{si:05d}-of-{n:05d}.parquet"
            local = os.path.join(tmp, fn)
            subprocess.run([hf, "buckets", "cp", f"{BUCKET_ROOT}/{split}/{fn}", local], check=True)
            _gcs_cp(local, f"{DEST_ROOT}/{split}/{fn}")
            os.remove(local)
            if si % 100 == 0 or si == n - 1:
                print(f"[mirror] {split} {si + 1}/{n}", flush=True)
    print("[mirror] bucket mirror done", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=("bucket", "local"), default="bucket")
    ap.add_argument("--local-dir", default=None, help="local corpus dir (for --source local)")
    ap.add_argument("--splits", default="train,val,test")
    a = ap.parse_args()
    splits = tuple(s.strip() for s in a.splits.split(",") if s.strip())

    if a.source == "local":
        if not a.local_dir:
            raise SystemExit("--source local requires --local-dir")
        mirror_local(a.local_dir, splits)
    else:
        mirror_bucket(splits)
    print(f"[mirror] corpus mirrored to {DEST_ROOT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
