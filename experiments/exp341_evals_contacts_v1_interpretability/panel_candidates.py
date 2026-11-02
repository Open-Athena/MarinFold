# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""A3/A4 panel, stage 1: removed and kept AFDB candidates with ground truth.

The held-out panel (plan section 4.7) is built from AFDB structures that #225
removed from the training corpus (seen by exp199, never by exp232/exp277),
plus kept structures as A4's exposure control. Each needs a sequence, ground
truth contacts, length and pLDDT. All of it is read from published contacts-v1
documents rather than recomputed from mmCIF with pyconfind:

- A3/A4 score R-precision against binary ground truth (degree >= 0.001,
  separation >= 6). A document with ``truncated = False`` emits exactly that
  contact set, strongest first, so its contacts *are* the ground truth.
  Truncated documents are excluded, which drops some long proteins; the
  stage reports how many.
- Removed rows are located through the #225 drop list's ``(shard, row)``
  pointers into the unfiltered AFDB corpus
  (``data/document_structures/contacts_v1/train``); each pointer's
  ``entry_id`` is checked against the drop list.
- Kept candidates are sampled from the same shard indices of the
  decontaminated corpus (``data/document_structures/contacts_v1_decontam``),
  which is the AFDB arm exp232 trained on.

A seeded random subset of shards is read (column projection over HTTP range
requests on the public HF bucket). Stage 2 counts each candidate's surviving
training relatives with MMseqs2; the panel is then stratified on that count.

Writes ``<output-dir>/candidates.parquet`` with one row per candidate:
``set`` (removed/kept), ``entry_id``, ``shard``, ``sequence``, ``seq_len``,
``global_plddt``, ``struct_cluster_id``, ``contacts`` (0-based ``[i, j]``
pairs, i < j), and for removed rows the drop list's nearest-reference
identity and coverage.

    uv run python panel_candidates.py --droplist droplist_final.parquet \
        --shards 300 --output-dir /path/to/panel
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from hf_bucket import read_columns
from marinfold.document_structures.contacts_v1.read import live_contacts, sequence_from_document
from marinfold.document_structures.contacts_v1.vocab import NUM_POSITION_INDICES

N_SHARDS = 2067
UNFILTERED = "data/document_structures/contacts_v1/train/contacts_v1-{:05d}-of-02067.parquet"
DECONTAM = "data/document_structures/contacts_v1_decontam/train/contacts_v1-{:05d}-of-02067.parquet"
COLUMNS = ["entry_id", "document", "seq_len", "n_term_index", "truncated", "global_plddt", "struct_cluster_id"]
SEED = 341


def read_shard(template: str, shard: int) -> pd.DataFrame:
    return read_columns(template.format(shard), COLUMNS, block_size=1 << 22).to_pandas()


def decode(row) -> dict:
    """Sequence and 0-based sequence-index contact pairs from one document."""
    sequence = sequence_from_document(row.document, row.seq_len, row.n_term_index)
    cut = row.document.index("<begin_statements>")
    pairs = []
    for a, b in live_contacts(row.document[cut:]):
        i = (a - row.n_term_index) % NUM_POSITION_INDICES
        j = (b - row.n_term_index) % NUM_POSITION_INDICES
        pairs.append(sorted((i, j)))
    return {
        "entry_id": row.entry_id,
        "sequence": sequence,
        "seq_len": int(row.seq_len),
        "global_plddt": float(row.global_plddt),
        "struct_cluster_id": row.struct_cluster_id,
        "contacts": sorted(pairs),
    }


def shard_candidates(shard: int, removed: pd.DataFrame, kept_per_shard: int) -> tuple[list[dict], dict]:
    unfiltered = read_shard(UNFILTERED, shard)
    pointers = removed[removed["shard"] == shard]
    picked = unfiltered.iloc[pointers["row"].to_numpy()]
    mismatched = (picked["entry_id"].to_numpy() != pointers["entry_id"].to_numpy()).sum()
    if mismatched:
        raise ValueError(f"shard {shard}: {mismatched} drop-list pointers do not match their row's entry_id")
    picked = picked.assign(identity=pointers["identity"].to_numpy(), coverage=pointers["coverage"].to_numpy())

    rows = []
    for row in picked[~picked["truncated"]].itertuples(index=False):
        rows.append({"set": "removed", "shard": shard, **decode(row),
                     "identity": float(row.identity), "coverage": float(row.coverage)})

    decontam = read_shard(DECONTAM, shard)
    complete = decontam[~decontam["truncated"]]
    rng = np.random.default_rng([SEED, shard])
    sample = complete.iloc[rng.choice(len(complete), size=min(kept_per_shard, len(complete)), replace=False)]
    for row in sample.itertuples(index=False):
        rows.append({"set": "kept", "shard": shard, **decode(row), "identity": np.nan, "coverage": np.nan})

    counts = {
        "removed_pointers": len(pointers),
        "removed_truncated": int(picked["truncated"].sum()),
        "kept_documents": len(decontam),
        "kept_truncated": int(decontam["truncated"].sum()),
    }
    return rows, counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--droplist", type=Path, required=True, help="#225 droplist_final.parquet")
    parser.add_argument("--shards", type=int, default=300, help="number of AFDB shards to read (of 2,067)")
    parser.add_argument("--kept-per-shard", type=int, default=40)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    droplist = pd.read_parquet(args.droplist)
    removed = droplist[droplist["arm"] == "afdb"]
    shards = sorted(np.random.default_rng(SEED).choice(N_SHARDS, size=args.shards, replace=False).tolist())

    rows: list[dict] = []
    totals: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for done, (shard_rows, counts) in enumerate(
            pool.map(lambda s: shard_candidates(s, removed, args.kept_per_shard), shards), start=1
        ):
            rows += shard_rows
            for key, value in counts.items():
                totals[key] = totals.get(key, 0) + value
            if done % 25 == 0 or done == len(shards):
                print(f"{done}/{len(shards)} shards, {len(rows):,} candidates", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_parquet(args.output_dir / "candidates.parquet", index=False)
    summary = {
        "shards": shards,
        **totals,
        "candidates_by_set": frame["set"].value_counts().to_dict(),
    }
    (args.output_dir / "candidates_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "shards"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
