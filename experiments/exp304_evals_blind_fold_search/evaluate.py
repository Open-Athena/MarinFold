#!/usr/bin/env python
"""Reveal exp301 references only after blind candidate IDs are sealed."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from search_policy import canonical

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
SOURCE = HERE.parent / "exp301_evals_fold_switching_proteins" / "data"


def score_candidate(row: dict, target: dict, primary: bool, split: str,
                    method: str, rank: int) -> dict:
    """Score a sealed map on resolved positions and fold-specific contacts."""
    common = {int(position) for position in target["common_positions"]}
    complete = canonical(row["contacts"]) | canonical(row["given"])
    predicted = {pair for pair in complete if pair[0] in common and pair[1] in common}
    fold1 = set(canonical(target["contacts_fold1"]))
    fold2 = set(canonical(target["contacts_fold2"]))
    a, b, shared = fold1 - fold2, fold2 - fold1, fold1 & fold2
    lo, hi = int(target["fs_lo"]), int(target["fs_hi"])
    touches_region = lambda pair: lo <= pair[0] < hi or lo <= pair[1] < hi
    a_fs = {pair for pair in a if touches_region(pair)}
    b_fs = {pair for pair in b if touches_region(pair)}
    pred_fs = {pair for pair in predicted if touches_region(pair)}
    recall_a = len(predicted & a) / len(a) if a else np.nan
    recall_b = len(predicted & b) / len(b) if b else np.nan
    recall_a_fs = len(pred_fs & a_fs) / len(a_fs) if a_fs else np.nan
    recall_b_fs = len(pred_fs & b_fs) / len(b_fs) if b_fs else np.nan
    return {
        "pair_id": target["pair_id"], "method": method, "rank": rank,
        "candidate_id": row["candidate_id"], "source_arm": row["arm"],
        "split": split, "primary": primary, "tier": target["tier"],
        "seq_class": target["seq_class"], "L": target["L"],
        "n_pred": len(predicted), "n_pred_fs": len(pred_fs),
        "finished": bool(row["finished"]),
        "recall_a": recall_a, "recall_b": recall_b, "phi": recall_a - recall_b,
        "recall_a_fs": recall_a_fs, "recall_b_fs": recall_b_fs,
        "phi_fs": recall_a_fs - recall_b_fs,
        "recall_shared": len(predicted & shared) / len(shared) if shared else np.nan,
        "precision_union": len(predicted & (fold1 | fold2)) / len(predicted) if predicted else np.nan,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=HERE / "_cache" / "raw")
    args = parser.parse_args()
    seal_path = DATA / "sealed_shortlists.csv"
    expected = (DATA / "sealed_shortlists.sha256").read_text().split()[0]
    actual = hashlib.sha256(seal_path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError("sealed shortlist hash mismatch")
    shortlist = pd.read_csv(seal_path)
    cohort = pd.read_csv(DATA / "cohort.csv").set_index("pair_id")
    truth = {
        row["pair_id"]: row
        for row in pd.read_parquet(SOURCE / "eval_targets.parquet").to_dict("records")
        if row["role"] == "foldswitch"
    }
    selected: dict[tuple[str, str], list[tuple[str, int]]] = {}
    for row in shortlist.itertuples():
        selected.setdefault((row.pair_id, row.candidate_id), []).append((row.method, int(row.rank)))
    rows = []
    all_rows = []
    for file in sorted(args.raw.glob("shard-*.parquet")):
        for record in pd.read_parquet(file).to_dict("records"):
            pid = record["pair_id"]
            result = score_candidate(record, truth[pid], bool(cohort.loc[pid, "primary"]),
                                     str(cohort.loc[pid, "split"]), record["arm"], 0)
            all_rows.append(result)
            for method, rank in selected.get((pid, record["candidate_id"]), []):
                rows.append({**result, "method": method, "rank": rank})
    scored = pd.DataFrame(rows).sort_values(["pair_id", "method", "rank"])
    if len(scored) != len(shortlist):
        raise ValueError(f"scored {len(scored)} of {len(shortlist)} sealed candidates")
    scored.to_csv(DATA / "scored_shortlists.csv", index=False)
    cache = HERE / "_cache"
    cache.mkdir(exist_ok=True)
    pd.DataFrame(all_rows).to_parquet(cache / "scored_all_candidates.parquet", index=False)
    print(f"scored {len(scored)} sealed maps across {scored.pair_id.nunique()} proteins")


if __name__ == "__main__":
    main()
