# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Build rollout targets from the exp89 *eval* set (not the exp98 train targets).

The short-document-bias probe needs eval proteins with (a) a one-letter sequence
(to rebuild fresh contacts-v1 realizations) and (b) the canonical ground-truth
contact set. exp89's ``gt_universe.jsonl`` has the GT contacts (pyconfind, in
input-seq coords) + the resolved-residue universe; the exp74/exp78 manifests have
the ``input_seq``. We join them and emit a ``targets.parquet`` with exp98/exp102's
schema so ``gen_prompts.py`` and the rollout worker are drop-in.

Ground-truth contact definition — identical to exp89 ``compute_metrics.true_matrix``
+ ``resolved_pairs``: a pair ``(i, j)`` is a GT contact iff ``degree >= 0.001``,
``j - i >= 6`` (MIN_SEP), ``i < j < L``, **and both endpoints are resolved** in the
GT structure (the eval metric only ever scores resolved pairs). We also record
``n_gt_all`` (same but without the resolved restriction) for reference.

Selection is length-stratified (equal-count L bins, one seeded pick per bin) so the
probe spans short->long proteins, where any length-dependent bias will show.

    uv run python select_eval_targets.py --n 12 --min-gt 8 --out data/eval/targets.parquet
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random

import pyarrow as pa
import pyarrow.parquet as pq

MIN_DEG, MIN_SEP = 0.001, 6

DEFAULT_GT = ("/home/bizon/git/MarinFold-exp120/experiments/"
              "exp89_evals_contacts_v1_model_on_eval_set/data/gt_universe.jsonl")
DEFAULT_MANIFEST_DIR = ("/home/bizon/git/MarinFold-exp78/experiments/"
                        "exp78_evals_esmfold_contacts/data")
MANIFESTS = ("eval_manifest_foldbench.csv", "eval_manifest_exp65.csv")


def load_sequences(manifest_dir: str) -> dict[tuple[str, str], str]:
    seqs: dict[tuple[str, str], str] = {}
    for fn in MANIFESTS:
        with open(os.path.join(manifest_dir, fn)) as fh:
            for row in csv.DictReader(fh):
                seqs[(row["dataset"], row["stem"])] = row["input_seq"]
    return seqs


def gt_contacts(rec: dict) -> tuple[list[list[int]], int]:
    """(resolved-restricted canonical GT contact list [[i,j],...], n_gt_all).

    n_gt_all counts contacts that pass degree/sep but ignores the resolved mask.
    """
    L = rec["L"]
    resolved = set(rec["resolved"])
    canon: list[list[int]] = []
    n_all = 0
    for i, j, d in rec["contacts"]:
        i, j = int(i), int(j)
        if i > j:
            i, j = j, i
        if d >= MIN_DEG and (j - i) >= MIN_SEP and 0 <= i < j < L:
            n_all += 1
            if i in resolved and j in resolved:
                canon.append([i, j])
    return canon, n_all


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default=DEFAULT_GT)
    ap.add_argument("--manifest-dir", default=DEFAULT_MANIFEST_DIR)
    ap.add_argument("--n", type=int, default=12, help="number of targets to keep")
    ap.add_argument("--min-gt", type=int, default=8, help="minimum resolved GT contacts")
    ap.add_argument("--max-len", type=int, default=None, help="cap L (None = no cap)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="data/eval/targets.parquet")
    a = ap.parse_args()

    seqs = load_sequences(a.manifest_dir)
    recs = [json.loads(l) for l in open(a.gt)]
    print(f"gt_universe: {len(recs)} proteins; manifest seqs: {len(seqs)}", flush=True)

    pool = []
    for rec in recs:
        key = (rec["dataset"], rec["stem"])
        seq = seqs.get(key)
        if seq is None or len(seq) != rec["L"]:
            continue
        canon, n_all = gt_contacts(rec)
        n_gt = len(canon)
        if n_gt < a.min_gt:
            continue
        if a.max_len is not None and rec["L"] > a.max_len:
            continue
        pool.append(dict(
            entry_id=f"{rec['dataset']}__{rec['stem']}",
            dataset=rec["dataset"], stem=rec["stem"],
            L=int(rec["L"]), n_resolved=int(rec["n_resolved"]),
            sequence=seq, n_gt=int(n_gt), n_gt_all=int(n_all),
            gt_contacts=canon,
        ))
    print(f"eligible (n_gt>={a.min_gt}"
          + (f", L<={a.max_len}" if a.max_len else "") + f"): {len(pool)}", flush=True)
    if a.n >= len(pool):
        raise SystemExit(f"--n {a.n} >= eligible {len(pool)}")

    # length-stratified: sort by L, equal-count bins over rank, one seeded pick/bin.
    pool.sort(key=lambda r: r["L"])
    rng = random.Random(a.seed)
    picks = []
    n = len(pool)
    for b in range(a.n):
        lo = (b * n) // a.n
        hi = ((b + 1) * n) // a.n
        if hi > lo:
            picks.append(pool[rng.randrange(lo, hi)])
    picks.sort(key=lambda r: r["L"])

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    pq.write_table(pa.Table.from_pylist(picks), a.out)
    print(f"\nselected {len(picks)} targets -> {a.out}")
    print(f"{'entry_id':<26}{'L':>5}{'n_res':>7}{'n_gt':>6}{'n_gt_all':>9}{'gt/L':>7}")
    for r in picks:
        print(f"{r['entry_id']:<26}{r['L']:>5}{r['n_resolved']:>7}{r['n_gt']:>6}"
              f"{r['n_gt_all']:>9}{r['n_gt']/r['L']:>7.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
