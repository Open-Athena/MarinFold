#!/usr/bin/env python
"""Seal then score 100/200/500 independent-rollout budget shortlists."""

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from analyze import score_group
from evaluate import score_candidate
from search_policy import canonical, diverse_indices

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RAW = HERE / "_cache" / "iid500"
SOURCE = HERE.parent / "exp301_evals_fold_switching_proteins" / "data"
BUDGETS = (100, 200, 500)


def raw_files() -> list[Path]:
    """Return a complete one-file-per-target budget cohort."""
    files = sorted(RAW.glob("shard-*.parquet"))
    expected = set(pd.read_parquet(DATA / "search_targets.parquet").pair_id)
    observed = set()
    for file in files:
        frame = pd.read_parquet(file, columns=["pair_id"])
        if frame.pair_id.nunique() != 1:
            raise ValueError(file)
        observed.add(frame.pair_id.iloc[0])
    if len(files) != len(expected) or observed != expected:
        raise ValueError(f"iid500 cohort incomplete: files={len(files)}, targets={len(observed)}")
    return files


def seal(files: list[Path]) -> None:
    """Choose all budget shortlists using only model outputs."""
    rows = []
    for file in files:
        frame = pd.read_parquet(file).sort_values("rollout")
        if len(frame) != 500 or frame.arm.nunique() != 1 or frame.arm.iloc[0] != "root":
            raise ValueError(f"{file}: expected 500 independent maps")
        for budget in BUDGETS:
            pool = frame.iloc[:budget]
            eligible = pool[pool.finished].reset_index(drop=True)
            maps = [canonical(pairs) for pairs in eligible.contacts]
            for rank, index in enumerate(diverse_indices(maps, 16), 1):
                row = eligible.iloc[index]
                rows.append({"pair_id": row.pair_id, "budget": budget,
                             "rank": rank, "candidate_id": row.candidate_id})
    path = DATA / "sealed_budget_shortlists.csv"
    pd.DataFrame(rows).sort_values(["pair_id", "budget", "rank"]).to_csv(path, index=False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (DATA / "sealed_budget_shortlists.sha256").write_text(
        f"{digest}  sealed_budget_shortlists.csv\n")
    print(f"sealed {len(rows)} budget candidates; SHA256 {digest}")


def score(files: list[Path]) -> None:
    """Reveal references after confirming the fixed shortlist hash."""
    path = DATA / "sealed_budget_shortlists.csv"
    expected = (DATA / "sealed_budget_shortlists.sha256").read_text().split()[0]
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError("budget shortlist hash mismatch")
    shortlist = pd.read_csv(path)
    cohort = pd.read_csv(DATA / "cohort.csv").set_index("pair_id")
    prior = pd.read_csv(SOURCE / "fold_preference.csv").set_index("pair_id")
    truth = pd.read_parquet(SOURCE / "eval_targets.parquet").set_index("pair_id")
    selected = {(row.pair_id, int(row.budget), row.candidate_id): int(row.rank)
                for row in shortlist.itertuples()}
    output = []
    for file in files:
        frame = pd.read_parquet(file).sort_values("rollout")
        pid = str(frame.pair_id.iloc[0])
        target = truth.loc[pid].to_dict()
        target["pair_id"] = pid
        scored = [score_candidate(record, target, bool(cohort.loc[pid, "primary"]),
                                  str(cohort.loc[pid, "split"]), "iid", 0)
                  for record in frame.to_dict("records")]
        dominant = "fold1" if float(prior.loc[pid, "phi"]) >= 0 else "fold2"
        for budget in BUDGETS:
            pool = pd.DataFrame(scored[:budget])
            ranks = {cid: rank for (p, b, cid), rank in selected.items() if p == pid and b == budget}
            blind = pool[pool.candidate_id.isin(ranks)].copy()
            blind["rank"] = blind.candidate_id.map(ranks)
            if len(blind) != len(ranks):
                raise ValueError(f"{pid}: budget {budget} selected IDs absent")
            blind_result = score_group(blind, dominant)
            oracle_result = score_group(pool, dominant)
            output.append({"pair_id": pid, "budget": budget,
                           "primary": bool(cohort.loc[pid, "primary"]),
                           "split": str(cohort.loc[pid, "split"]),
                           "dominant": dominant,
                           **blind_result,
                           "oracle_minor_enrichment": oracle_result["minority_enrichment"],
                           "oracle_minor_recall": oracle_result["minority_recall"],
                           "oracle_dual_hit": oracle_result["dual_contact_hit"],
                           "generated_tokens": int(frame.iloc[:budget].n_tokens.sum())})
    per = pd.DataFrame(output).sort_values(["pair_id", "budget"])
    per.to_csv(DATA / "budget_per_protein.csv", index=False)
    test = per[per.primary & (per.split == "test")]
    summary = test.groupby("budget").agg(
        n_test=("pair_id", "size"), mean_minor_enrichment=("minority_enrichment", "mean"),
        mean_minor_recall=("minority_recall", "mean"),
        blind_dual_hits=("dual_contact_hit", "sum"),
        oracle_dual_hits=("oracle_dual_hit", "sum"),
        mean_tokens=("generated_tokens", "mean"),
    )
    summary.to_csv(DATA / "budget_summary.csv")
    print(summary.to_string())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["seal", "score"])
    args = parser.parse_args()
    files = raw_files()
    if args.phase == "seal":
        seal(files)
    else:
        score(files)


if __name__ == "__main__":
    main()
