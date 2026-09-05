# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""ESM-Atlas arm of #266 — source cif to redesigned documents in ONE pass.

No staging stage, unlike the AFDB arm, and no coordinate encoding at all.

The AFDB arm stages backbones because AFDB's bucket is requester-pays and only
GCP Iris workers can read it; the staged rows carry coordinates as int32
milli-angstroms, which is exact because AFDB mmCIF is written with 3 decimals.

Neither holds here:

* ESM-Atlas structures are inline `cif_content` in a **public** HF bucket, so a
  CoreWeave pod reads them directly — there is nothing to stage around.
* ESM-Atlas cifs carry up to **11 decimals** (~77 % of values are float16-exact,
  the rest need float64), so milli-angstrom rounding is lossy, and measurably
  so: on 200 structures it changed **114 of 200 documents** — same contact
  counts, different contact sets at the margin. Exactly the sensitivity the
  pyconfind backend difference showed.

So the structure never leaves float64: parsed once, used for ProteinMPNN
coordinates and for pyconfind, and discarded. That also drops ~1.2 TB of
staged artifact that would have been needed to store float64 coordinates.

**Relabelling an all-atom structure is fine.** `relabel_sequence` renames
residues and leaves the original side chains in place, which now disagree with
the new names — and that does not matter, because confind ignores input side
chains and rebuilds rotamers from the residue names (the very first thing this
experiment verified). No stripping needed.

The corpus and source are both 3,338 files and index-aligned (corpus shard *i*
is a subset of source part *i*, verified), so each task joins shard-by-shard
with a ~20 k-key set rather than a 65 M-key one.
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq

CORPUS = ("hf://buckets/open-athena/MarinFold/data/document_structures/"
          "contacts_v1_esm_atlas_decontam/train")
SOURCE = "hf://buckets/open-athena/esm-atlas-esmfold2-distill/structures/parts"
NUM_SHARDS = 3338

# ESM-Atlas provenance. It has no struct_cluster_id or round -- both AFDB-only.
CARRY = {"seq_len": "parent_seq_len", "global_plddt": "global_plddt",
         "seq_cluster_id": "seq_cluster_id", "split": "split",
         "contacts_emitted": "native_contacts_emitted", "sha1": "native_sha1",
         "cluster_size": "cluster_size", "ptm": "ptm"}


def _log(msg: str) -> None:
    print(f"[exp266-esm] {msg}", file=sys.stderr, flush=True)


def _documents_for_one(payload):
    """Process-pool entry point: (cif text, entry_id, meta, designs) -> rows."""
    import generate_rows
    from backbone import prepare_structure, relabel_sequence
    from stage_rows import _structure_from_cif
    from marinfold.document_structures.contacts_v1 import generate_document

    cif, entry_id, meta, designs = payload
    structure = prepare_structure(_structure_from_cif(cif, entry_id=entry_id))
    rotamers = generate_rows._load_rotamer_library()

    out = []
    for d in designs:
        result = generate_document(
            relabel_sequence(structure, d.sequence),
            entry_id=f"{entry_id}#{d.design_index}",
            rotamer_library=rotamers,
        )
        if result is None:
            continue
        row = result.metadata_row()
        row["entry_id"] = entry_id
        row["design_index"] = d.design_index
        row["mpnn_temperature"] = d.mpnn_temperature
        row["mpnn_score"] = d.mpnn_score
        row["identity_to_native"] = d.identity_to_native
        row.update(meta)
        out.append(row)
    return out


def process_shard(index: int, args, pool) -> int:
    from backbone import backbone_coords, prepare_structure, residue_sequence
    from redesign import BackboneEntry, batch_by_exact_length, design_batch
    from stage_rows import _structure_from_cif

    out_uri = f"{args.out_prefix.rstrip('/')}/documents-{index:05d}-of-{NUM_SHARDS:05d}.parquet"
    out_fs, out_path = fsspec.core.url_to_fs(out_uri)
    if out_fs.exists(out_path):
        _log(f"skip {index} (output exists)")
        return 0

    t0 = time.perf_counter()
    with fsspec.open(f"{CORPUS}/shard-{index:05d}-of-{NUM_SHARDS:05d}.parquet", "rb") as h:
        kept = {r["entry_id"]: r for r in
                pq.read_table(h, columns=["entry_id", *CARRY]).to_pylist()}
    with fsspec.open(f"{SOURCE}/part_{index:05d}.parquet", "rb") as h:
        src = pq.read_table(h, columns=["entry_id", "cif_content"]).to_pylist()
    _log(f"{index}: {len(kept):,} kept of {len(src):,} ({time.perf_counter()-t0:.0f}s read)")

    entries, payload_meta, filtered = [], {}, 0
    for s in src:
        meta = kept.get(s["entry_id"])
        if meta is None:
            continue                        # dropped by #225's decontamination
        try:
            structure = prepare_structure(
                _structure_from_cif(s["cif_content"], entry_id=s["entry_id"]))
            if len(structure) == 0 or sum(1 for _ in structure[0]) != 1:
                filtered += 1
                continue                    # monomers only, as contacts-v1 requires
            seq = residue_sequence(structure)
            if "X" in seq:
                filtered += 1
                continue                    # ProteinMPNN has no token for these
            _chains, coords = backbone_coords(structure)
        except ValueError:
            filtered += 1
            continue
        entries.append(BackboneEntry(s["entry_id"], seq, coords))
        payload_meta[s["entry_id"]] = (
            s["cif_content"], {n: meta[c] for c, n in CARRY.items() if c in meta})

    t_gpu = time.perf_counter()
    designs_by: dict[str, list] = {}
    for batch in batch_by_exact_length(
        entries, max_batch=args.max_batch, max_batch_residues=args.max_batch_residues,
        designs_per_backbone=len(args.temperatures),
    ):
        for d in design_batch(batch, device=args.device,
                              temperatures=tuple(args.temperatures)):
            designs_by.setdefault(d.entry_id, []).append(d)
    gpu_s = time.perf_counter() - t_gpu

    t_cpu = time.perf_counter()
    payloads = [(payload_meta[e][0], e, payload_meta[e][1], ds)
                for e, ds in designs_by.items()]
    rows = []
    for recs in pool.map(_documents_for_one, payloads, chunksize=4):
        rows.extend(recs)
    cpu_s = time.perf_counter() - t_cpu

    if not rows:
        raise ValueError(f"shard {index}: no documents from {len(entries)} backbones")
    with fsspec.open(out_uri, "wb") as h:
        pq.write_table(pa.Table.from_pylist(rows), h, compression="zstd")
    _log(f"{index}: {len(rows):,} docs from {len(entries):,} backbones "
         f"({filtered} filtered) in {time.perf_counter()-t0:.0f}s "
         f"(gpu {gpu_s:.0f}s cpu {cpu_s:.0f}s)")
    return len(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--shard", required=True, metavar="I/N")
    ap.add_argument("--temperatures", type=float, nargs="+", default=[0.1, 0.2],
                    help="Two by default for this arm: exp266's AFDB run showed "
                         "the 8-slot ladder spans almost nothing (identity "
                         "0.373->0.345, density flat, T=0.5 refolds worse), and "
                         "ESM-Atlas's value is backbone diversity, not more "
                         "sequences per backbone.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-batch", type=int, default=256)
    ap.add_argument("--max-batch-residues", type=int, default=100_000)
    ap.add_argument("--max-shards", type=int, default=None, help="Smoke cap.")
    ap.add_argument("--cpu-workers", type=int, default=14)
    args = ap.parse_args()

    from redesign import load_model
    import generate_rows

    i, n = (int(x) for x in args.shard.split("/"))
    mine = list(range(i, NUM_SHARDS, n))
    if args.max_shards is not None:
        mine = mine[: args.max_shards]
    _log(f"shard {i}/{n}: {len(mine)} of {NUM_SHARDS} corpus shards, "
         f"{len(args.temperatures)} designs each")

    load_model(args.device)
    # Warm the rotamer library before forking, or the pool children race on the
    # download and one parses a half-written library (the AFDB arm's bug).
    if generate_rows._load_rotamer_library() is None:
        raise RuntimeError("rotamer library failed to load in the parent")
    _log(f"model + rotamers ready; {args.cpu_workers} document processes")

    total, started = 0, time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.cpu_workers) as pool:
        for k, idx in enumerate(mine, 1):
            total += process_shard(idx, args, pool)
            _log(f"progress {k}/{len(mine)} shards, {total:,} documents")
    _log(f"done: {total:,} documents in {(time.perf_counter()-started)/60:.0f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
