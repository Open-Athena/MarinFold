# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Select the round-0 proteins of the contacts-v1 **val** split and emit both:

1. ``targets_val_r0.parquet`` — one row per protein (entry_id, L, sequence, n_gt,
   gt_contacts), the input the exp100 constrained-decoding pipeline consumes to
   produce the **regenerated validation set** (feed it to exp100's
   ``gen_prompts.py`` + ``gen_constrained_worker_vllm.py``, exactly like the
   train run — see this dir's README "Regenerated validation set").
2. ``orig_r0_val/*.parquet`` — the ORIGINAL round-0 val documents (``document``
   column), tokenized as the ``contacts-v1-val-orig`` validation component (the
   apples-to-apples partner of the regenerated val: same proteins, one
   realization each, differ only in content).

This mirrors exp100's ``select_round0_all.py`` (train split) with a small,
self-contained doc parser (no marinfold dependency) using the same GT definition:
seq-separation >= 6, positions in N-terminus-relative sequence order.

    uv run python select_round0_val.py

Val is small (9,558 round-0 proteins of the 41,954-row val split), so this runs
locally in seconds.
"""
from __future__ import annotations

import argparse
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import gcsfs
import pyarrow as pa
import pyarrow.parquet as pq

VAL_DIR = ("marin-us-east5/protein-structure/MarinFold/exp53_contacts_v1_5x/"
           "documents/val")
N_VAL_SHARDS = 22
PREFIX = ("gs://marin-us-east5/protein-structure/MarinFold/"
          "exp120_regen_vs_reepoch_contacts_v1/data")

NUM_POS = 2000   # contacts-v1 position-token wrap
MIN_SEP = 6      # contacts-v1 min_seq_separation
BEGIN = "<begin_statements>"
CONTACT_RE = re.compile(r"<contact>\s+<p(\d+)>\s+<p(\d+)>")
NTERM_RE = re.compile(r"<n-term>\s+<p(\d+)>")
RES_RE = re.compile(r"<p(\d+)>\s+<([A-Z]{3})>")

# Standard 3-letter -> 1-letter amino-acid map (matches contacts-v1 residues;
# unknown/non-standard fall back to "X"). Equivalent to exp100 parse_doc's
# THREE_TO_ONE derived from marinfold's _ONE_LETTER_TO_THREE.
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}


def shard_path(i: int) -> str:
    return f"{VAL_DIR}/contacts_v1-{i:05d}-of-{N_VAL_SHARDS:05d}.parquet"


def parse_doc(doc: str):
    """(L, one-letter sequence, sorted GT pairs in seq-index space) or None.

    Mirrors exp100 select_targets.parse_doc.
    """
    cut = doc.index(BEGIN) + len(BEGIN)
    prefix, struct = doc[:cut], doc[cut:]
    m = NTERM_RE.search(prefix)
    if not m:
        return None
    nterm = int(m.group(1))
    pos_in_seq = sorted({int(p) for p in re.findall(r"<p(\d+)>", prefix)},
                        key=lambda p: (p - nterm) % NUM_POS)
    seqidx = {p: (p - nterm) % NUM_POS for p in pos_in_seq}
    res_of_pos = {int(p): aa for p, aa in RES_RE.findall(prefix)}
    if not all(p in res_of_pos for p in pos_in_seq):
        return None
    seq = "".join(THREE_TO_ONE.get(res_of_pos[p], "X") for p in pos_in_seq)
    gt = set()
    for a, b in CONTACT_RE.findall(struct):
        ia, ib = seqidx.get(int(a)), seqidx.get(int(b))
        if ia is None or ib is None or ia == ib or abs(ia - ib) < MIN_SEP:
            continue
        gt.add((min(ia, ib), max(ia, ib)))
    return len(pos_in_seq), seq, sorted(gt)


def process_shard(fs, si: int) -> tuple[list[dict], list[dict]]:
    with fs.open(shard_path(si), "rb") as fh:
        tbl = pq.read_table(fh, columns=["entry_id", "document", "round"]).to_pylist()
    targets, docs = [], []
    for r in tbl:
        if r["round"] != 0:
            continue
        parsed = parse_doc(r["document"])
        if parsed is None:
            continue
        L, seq, gt = parsed
        targets.append(dict(entry_id=r["entry_id"], L=L, sequence=seq, n_gt=len(gt),
                            gt_contacts=[[i, j] for (i, j) in gt]))
        docs.append(dict(entry_id=r["entry_id"], document=r["document"]))
    return targets, docs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets-out", default="data/targets_val_r0.parquet")
    ap.add_argument("--targets-gcs-out",
                    default=f"{PREFIX}/targets_val_r0.parquet")
    ap.add_argument("--docs-dest", default=f"{PREFIX}/orig_r0_val")
    ap.add_argument("--docs-shards", type=int, default=4)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()

    fs = gcsfs.GCSFileSystem()
    all_targets: list[dict] = []
    all_docs: list[dict] = []
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(process_shard, fs, si): si for si in range(N_VAL_SHARDS)}
        for fut in as_completed(futs):
            t, d = fut.result()
            all_targets.extend(t); all_docs.extend(d)

    all_targets.sort(key=lambda c: c["entry_id"])
    all_docs.sort(key=lambda c: c["entry_id"])
    n0 = sum(1 for t in all_targets if t["n_gt"] == 0)
    Ls = [t["L"] for t in all_targets]
    print(f"round-0 val proteins: {len(all_targets)}  (0-contact: {n0})  "
          f"L min/mean/max: {min(Ls)}/{sum(Ls)//len(Ls)}/{max(Ls)}", flush=True)

    # (1) targets parquet (local + GCS) for the exp100 regen pipeline
    import os
    os.makedirs(os.path.dirname(a.targets_out) or ".", exist_ok=True)
    tgt_tbl = pa.Table.from_pylist(all_targets)
    pq.write_table(tgt_tbl, a.targets_out)
    with fs.open(a.targets_gcs_out, "wb") as fh:
        pq.write_table(tgt_tbl, fh)
    print(f"wrote targets: {a.targets_out} + {a.targets_gcs_out}", flush=True)

    # (2) original round-0 val docs corpus (the contacts-v1-val-orig component)
    dest = a.docs_dest.rstrip("/")
    per = (len(all_docs) + a.docs_shards - 1) // a.docs_shards
    for si in range(a.docs_shards):
        lo, hi = si * per, min((si + 1) * per, len(all_docs))
        if lo >= hi:
            break
        with fs.open(f"{dest}/orig_r0_val-{si:05d}-of-{a.docs_shards:05d}.parquet", "wb") as fh:
            pq.write_table(pa.Table.from_pylist(all_docs[lo:hi]), fh)
    print(f"wrote {len(all_docs)} original round-0 val docs to {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
