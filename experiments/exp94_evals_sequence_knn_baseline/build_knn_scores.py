# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Step 3 — copy neighbor contacts through alignments into [L, L] scores (Pass 2).

For each eval protein we take its top-k mmseqs hits (ranked by bitscore), map every
training contact through the local alignment onto the query's residues, and vote.
Two accumulators per protein:

* ``count[i, j]`` — integer votes (each neighbor contributes at most once per pair).
* ``W[i, j]``     — the same votes weighted by the neighbor's bitscore.

The saved ``score`` is ``tiebreak_matrix(count, W)`` (exp82): votes are primary, the
bitscore-weighted term lives in ``[0, 0.5)`` and only reorders pairs tied on votes —
which rescues long-range AUC at zero top-K cost (plain integer votes tie ~2/3 of
pairs at 0). The raw ``count`` is saved alongside under key ``count`` so the metric
step can report plain-vs-tiebreak.

Per (k, self-mode) one directory ``_scratch/scores/k{K}_{self,noself}/``; ``noself``
drops hits with ``fident==1 & qcov==1 & tcov==1`` (an eval protein verbatim in the
train set). The matrix is ``[L, L]`` in 0-based input-seq coordinates, matching the
GT universe; a protein with no usable hit yields an all-zero matrix (correct null).

    uv run python build_knn_scores.py --scratch _scratch --gt _scratch/gt_universe.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from knn_lib import MIN_SEP, target_to_query_map, tiebreak_matrix

KS = (1, 5, 10, 25, 50)
M8_COLS = ["query", "target", "fident", "alnlen", "qcov", "tcov", "evalue", "bits",
           "qstart", "qend", "tstart", "tend", "qaln", "taln"]


def load_lengths(gt_path: Path) -> dict[str, int]:
    """`{dataset}__{stem}` -> L from the GT universe."""
    out: dict[str, int] = {}
    for line in gt_path.open():
        r = json.loads(line)
        out[f"{r['dataset']}__{r['stem']}"] = int(r["L"])
    return out


def load_hits(m8: Path) -> dict[str, list[dict]]:
    """Parse aln.m8 into per-query hit lists sorted by bitscore (desc)."""
    by_query: dict[str, list[dict]] = defaultdict(list)
    with m8.open() as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            rec = dict(zip(M8_COLS, f))
            by_query[rec["query"]].append({
                "target": rec["target"], "fident": float(rec["fident"]),
                "qcov": float(rec["qcov"]), "tcov": float(rec["tcov"]),
                "bits": float(rec["bits"]), "qstart": int(rec["qstart"]),
                "tstart": int(rec["tstart"]), "qaln": rec["qaln"], "taln": rec["taln"],
            })
    for hits in by_query.values():
        hits.sort(key=lambda h: h["bits"], reverse=True)
    return by_query


def is_self(h: dict) -> bool:
    return h["fident"] >= 1.0 and h["qcov"] >= 1.0 and h["tcov"] >= 1.0


def load_contacts(store_dir: Path, needed: set[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load 0-based contacts for the needed doc_ids, reading only the shards they touch."""
    by_shard: dict[str, set[str]] = defaultdict(set)
    for doc_id in needed:
        by_shard[doc_id.split("_", 1)[0]].add(doc_id)
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for shard, ids in by_shard.items():
        path = store_dir / f"{shard}.parquet"
        if not path.exists():
            continue
        tbl = pq.read_table(path, columns=["doc_id", "i", "j"])
        d = tbl.column("doc_id").to_pylist()
        icol = tbl.column("i").to_pylist()
        jcol = tbl.column("j").to_pylist()
        for doc_id, ii, jj in zip(d, icol, jcol):
            if doc_id in ids:
                out[doc_id] = (np.asarray(ii, np.int64), np.asarray(jj, np.int64))
    return out


def vote(hits: list[dict], contacts: dict, L: int) -> tuple[np.ndarray, np.ndarray]:
    """Accumulate integer + bitscore-weighted votes from a protein's chosen hits."""
    count = np.zeros((L, L), np.float32)
    W = np.zeros((L, L), np.float32)
    for h in hits:
        cij = contacts.get(h["target"])
        if cij is None:
            continue
        t2q = target_to_query_map(h["qaln"], h["taln"], h["qstart"], h["tstart"])
        if not t2q:
            continue
        bits = h["bits"]
        seen: set[tuple[int, int]] = set()
        ti_arr, tj_arr = cij
        for ti, tj in zip(ti_arr.tolist(), tj_arr.tolist()):
            qi = t2q.get(ti)
            qj = t2q.get(tj)
            if qi is None or qj is None:
                continue
            lo, hi = (qi, qj) if qi < qj else (qj, qi)
            if hi - lo < MIN_SEP:          # re-apply separation in QUERY coords
                continue
            if (lo, hi) in seen:           # dedup within this neighbor
                continue
            seen.add((lo, hi))
            count[lo, hi] += 1.0
            count[hi, lo] += 1.0
            W[lo, hi] += bits
            W[hi, lo] += bits
    return count, W


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--ks", type=int, nargs="+", default=list(KS))
    args = ap.parse_args()

    lengths = load_lengths(args.gt)
    by_query = load_hits(args.scratch / "aln.m8")
    print(f"[score] {len(by_query)}/{len(lengths)} eval proteins have >=1 hit", flush=True)

    # Targets we'll ever need: top-max(k) of each query under both self-modes.
    kmax = max(args.ks)
    needed: set[str] = set()
    for hits in by_query.values():
        needed.update(h["target"] for h in hits[:kmax])
        noself = [h for h in hits if not is_self(h)]
        needed.update(h["target"] for h in noself[:kmax])
    print(f"[score] loading contacts for {len(needed):,} distinct training neighbors ...", flush=True)
    contacts = load_contacts(args.scratch / "contacts_store", needed)
    print(f"[score] loaded {len(contacts):,} neighbor contact sets", flush=True)

    modes = {"self": lambda hs: hs, "noself": lambda hs: [h for h in hs if not is_self(h)]}
    out_root = args.scratch / "scores"
    summary_rows: list[dict] = []
    for key, L in lengths.items():
        hits_all = by_query.get(key, [])
        best = hits_all[0] if hits_all else None
        summary_rows.append({
            "query": key, "n_hits": len(hits_all),
            "best_bits": best["bits"] if best else 0.0,
            "best_fident": best["fident"] if best else 0.0,
            "best_qcov": best["qcov"] if best else 0.0,
            "has_self": any(is_self(h) for h in hits_all),
        })
        for mode_name, pick in modes.items():
            ranked = pick(hits_all)
            for k in args.ks:
                out_dir = out_root / f"k{k}_{mode_name}"
                out_dir.mkdir(parents=True, exist_ok=True)
                count, W = vote(ranked[:k], contacts, L)
                score = tiebreak_matrix(count, W).astype(np.float16)
                np.savez_compressed(out_dir / f"{key}.npz",
                                    score=score, count=count.astype(np.float16))

    summ = args.scratch / "knn_hit_summary.csv"
    with summ.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    n_self = sum(r["has_self"] for r in summary_rows)
    n_hit = sum(r["n_hits"] > 0 for r in summary_rows)
    print(f"[score] wrote scores for {len(args.ks)} k x 2 modes; "
          f"{n_hit}/{len(lengths)} with a hit, {n_self} with a verbatim self-hit", flush=True)
    print(f"[score] hit summary -> {summ}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
