# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage ESM-Atlas backbones — one CoreWeave shard, HF source to CoreWeave S3.

The ESM-Atlas arm of #266. Same destination format as the AFDB arm
(`backbone.encode_backbone`), but a different source shape, and it needs no
GCP at all:

* AFDB rows carry a `gcs_uri` into a requester-pays bucket, which only GCP Iris
  workers can read — hence the AFDB arm's separate GCP staging job.
* ESM-Atlas structures live as **inline `cif_content`** in the public HF bucket
  `open-athena/esm-atlas-esmfold2-distill`, readable anonymously from anywhere.
  So a CoreWeave pod reads HF directly and writes staged backbones to CoreWeave
  object storage in one hop.

**Shard-aligned join.** The decontaminated corpus
(`contacts_v1_esm_atlas_decontam/train/shard-NNNNN-of-03338.parquet`) and the
source (`structures/parts/part_NNNNN.parquet`) are both 3,338 files and index
aligned — verified: corpus shard *i* is a subset of source part *i* on the
shards checked. That is what makes this tractable: each worker holds one
shard's ~20 k kept ids, not a 65 M-key set (which would be several GB per
worker).

Note the exp139 GCS mirror of these structures no longer exists — only its
`documents/` survived — so HF is the source of truth.
"""

from __future__ import annotations

import argparse
import sys
import time

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq

CORPUS = ("hf://buckets/open-athena/MarinFold/data/document_structures/"
          "contacts_v1_esm_atlas_decontam/train")
SOURCE = "hf://buckets/open-athena/esm-atlas-esmfold2-distill/structures/parts"
NUM_SHARDS = 3338

# Provenance carried from the decontaminated corpus. The ESM-Atlas corpus has
# no `struct_cluster_id` or `round` (both are AFDB-only), so the AFDB arm's
# column list does not apply verbatim.
CARRY = {"seq_len": "seq_len", "global_plddt": "parent_global_plddt",
         "seq_cluster_id": "seq_cluster_id", "split": "split",
         "contacts_emitted": "native_contacts_emitted", "sha1": "native_sha1",
         "cluster_size": "cluster_size", "ptm": "ptm"}


def _log(msg: str) -> None:
    print(f"[exp266-esm-stage] {msg}", file=sys.stderr, flush=True)


def stage_shard(index: int, out_prefix: str) -> int:
    """Corpus shard + source part -> one staged backbone parquet. Resumable."""
    from backbone import encode_backbone, prepare_structure
    from stage_rows import _structure_from_cif

    out_uri = f"{out_prefix.rstrip('/')}/backbones-{index:05d}-of-{NUM_SHARDS:05d}.parquet"
    out_fs, out_path = fsspec.core.url_to_fs(out_uri)
    if out_fs.exists(out_path):
        _log(f"skip {index} (output exists)")
        return 0

    t0 = time.perf_counter()
    with fsspec.open(f"{CORPUS}/shard-{index:05d}-of-{NUM_SHARDS:05d}.parquet", "rb") as h:
        keep = pq.read_table(h, columns=["entry_id", *CARRY]).to_pylist()
    kept = {r["entry_id"]: r for r in keep}

    with fsspec.open(f"{SOURCE}/part_{index:05d}.parquet", "rb") as h:
        src = pq.read_table(h, columns=["entry_id", "cif_content"]).to_pylist()
    _log(f"{index}: {len(kept):,} kept of {len(src):,} in part "
         f"({time.perf_counter()-t0:.0f}s to read)")

    rows, filtered = [], 0
    for s in src:
        meta = kept.get(s["entry_id"])
        if meta is None:
            continue                       # dropped by #225's decontamination
        structure = prepare_structure(
            _structure_from_cif(s["cif_content"], entry_id=s["entry_id"])
        )
        try:
            staged = encode_backbone(structure)
        except ValueError as exc:
            msg = str(exc)
            # The same two designed-in filters as the AFDB arm. Anything else --
            # a missing mainchain atom, an inexact coordinate, non-contiguous
            # numbering -- is a real surprise about the input and must surface.
            if "non-canonical residues" in msg or "expected 1 chain" in msg:
                filtered += 1
                continue
            raise
        staged["entry_id"] = s["entry_id"]
        for col, name in CARRY.items():
            if col in meta:
                staged[name] = meta[col]
        rows.append(staged)

    if not rows:
        raise ValueError(f"shard {index}: no rows staged from {len(src)} structures")
    with fsspec.open(out_uri, "wb") as h:
        pq.write_table(pa.Table.from_pylist(rows), h, compression="zstd")
    _log(f"{index}: staged {len(rows):,} ({filtered} filtered) in "
         f"{time.perf_counter()-t0:.0f}s -> {out_uri}")
    return len(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--shard", required=True, metavar="I/N",
                    help="This task's slice of the 3,338 shards, interleaved.")
    ap.add_argument("--max-shards", type=int, default=None, help="Smoke cap.")
    args = ap.parse_args()

    i, n = (int(x) for x in args.shard.split("/"))
    # Interleave: the corpus is not length-sorted here, but interleaving still
    # spreads any per-region slowness evenly across tasks.
    mine = list(range(i, NUM_SHARDS, n))
    if args.max_shards is not None:
        mine = mine[: args.max_shards]
    _log(f"shard {i}/{n}: {len(mine)} of {NUM_SHARDS} corpus shards")

    total, started = 0, time.perf_counter()
    for k, idx in enumerate(mine, 1):
        total += stage_shard(idx, args.out_prefix)
        rate = k / (time.perf_counter() - started) * 3600
        _log(f"progress {k}/{len(mine)} shards ({rate:.1f}/h, {total:,} backbones)")
    _log(f"done: {total:,} backbones in {(time.perf_counter()-started)/60:.0f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
