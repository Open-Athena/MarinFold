#!/usr/bin/env python
"""Seal beam shortlists, then score both reference contact modes."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
EXP304 = EXPERIMENTS / "exp304_evals_blind_fold_search"
EXP301 = EXPERIMENTS / "exp301_evals_fold_switching_proteins"
sys.path.insert(0, str(EXP304))
from analyze import MIN_CONTACTS_FS, MIN_ENRICHMENT, MIN_RECALL, paired_interval, score_group  # noqa: E402
from evaluate import score_candidate  # noqa: E402
from search_policy import canonical, diverse_indices  # noqa: E402


def raw_files(mode: str, allow_partial: bool) -> list[Path]:
    """Return one raw rollout table per non-capped fold-switch target."""
    root = HERE / "_cache" / mode / "foldswitch"
    files = sorted(path for path in root.glob("*.parquet")
                   if not path.name.endswith(".timing.parquet"))
    if len(files) != 67 and not allow_partial:
        raise ValueError(f"expected 67 fold-switch result files, got {len(files)}")
    if not files:
        raise FileNotFoundError(root)
    return files


def seal(files: list[Path], mode: str, n_rollouts: int) -> None:
    """Select 16 diverse finished maps per protein without opening references."""
    rows = []
    for path in files:
        frame = pd.read_parquet(path).sort_values("rollout")
        if len(frame) != n_rollouts or frame.rollout.tolist() != list(range(n_rollouts)):
            raise ValueError(f"{path.stem}: incomplete rollout set")
        eligible = frame[frame.finished].reset_index(drop=True)
        maps = [canonical(pairs) for pairs in eligible.contacts]
        for rank, index in enumerate(diverse_indices(maps, 16), 1):
            rows.append({"pair_id": path.stem, "mode": mode,
                         "rollout": int(eligible.iloc[index].rollout), "rank": rank})
    output = HERE / "data" / f"sealed_{mode}.csv"
    pd.DataFrame(rows).sort_values(["pair_id", "rank"]).to_csv(output, index=False)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(".sha256").write_text(f"{digest}  {output.name}\n")
    print(f"sealed {len(rows)} maps from {len(files)} proteins; SHA256 {digest}")


def fold_hits(group: pd.DataFrame, recall_cutoff: float) -> tuple[bool, bool]:
    """Apply exp304's region-specific contact criterion to one candidate pool."""
    valid = group.finished & (group.n_pred_fs >= MIN_CONTACTS_FS)
    fold1 = (valid & (group.recall_a_fs >= recall_cutoff)
             & (group.phi_fs >= MIN_ENRICHMENT)).any()
    fold2 = (valid & (group.recall_b_fs >= recall_cutoff)
             & (-group.phi_fs >= MIN_ENRICHMENT)).any()
    return bool(fold1), bool(fold2)


def score(files: list[Path], mode: str, n_rollouts: int) -> None:
    """Reveal references only after verifying the frozen shortlist hash."""
    shortlist_path = HERE / "data" / f"sealed_{mode}.csv"
    expected = shortlist_path.with_suffix(".sha256").read_text().split()[0]
    if hashlib.sha256(shortlist_path.read_bytes()).hexdigest() != expected:
        raise ValueError("sealed shortlist hash mismatch")
    shortlist = pd.read_csv(shortlist_path)
    ranks = {(row.pair_id, int(row.rollout)): int(row.rank)
             for row in shortlist.itertuples()}
    truth = pd.read_parquet(EXP301 / "data" / "eval_targets.parquet").set_index("pair_id")
    cohort = pd.read_csv(EXP304 / "data" / "cohort.csv").set_index("pair_id")
    prior = pd.read_csv(EXP301 / "data" / "fold_preference.csv").set_index("pair_id")
    with (EXP301 / "data" / "foldswitch_universe.jsonl").open() as source:
        mismatch = {row["pair_id"]: int(row["n_seq_mismatch"])
                    for row in (json.loads(line) for line in source)}
    iid = pd.read_csv(EXP304 / "data" / "iid_mode_coverage_per_protein.csv")
    iid = iid[iid.budget == n_rollouts].set_index("pair_id")
    iid_blind = pd.read_csv(EXP304 / "data" / "budget_per_protein.csv")
    iid_blind = iid_blind[iid_blind.budget == n_rollouts].set_index("pair_id")
    iid_timing = pd.read_csv(EXP304 / "data" / "timings.csv")
    iid_timing = iid_timing[iid_timing["mode"] == "root"].set_index("pair_id")
    rows = []
    for path in files:
        frame = pd.read_parquet(path).sort_values("rollout")
        pid = path.stem
        target = truth.loc[pid].to_dict()
        target["pair_id"] = pid
        scored = pd.DataFrame([
            score_candidate({**record, "candidate_id": f"{mode}:{record['rollout']}",
                             "arm": mode, "given": []},
                            target, bool(cohort.loc[pid, "primary"]),
                            str(cohort.loc[pid, "split"]), mode, 0)
            for record in frame.to_dict("records")
        ])
        selected = scored[scored.candidate_id.map(
            lambda candidate: (pid, int(candidate.split(":")[-1])) in ranks
        )].copy()
        selected["rank"] = selected.candidate_id.map(
            lambda candidate: ranks[(pid, int(candidate.split(":")[-1]))]
        )
        if len(selected) != sum(pair_id == pid for pair_id, _ in ranks):
            raise ValueError(f"{pid}: missing sealed candidate")
        dominant = "fold1" if float(prior.loc[pid, "phi"]) >= 0 else "fold2"
        pool_result = score_group(scored, dominant)
        blind_result = score_group(selected, dominant)
        fold1, fold2 = fold_hits(scored, MIN_RECALL)
        strict1, strict2 = fold_hits(scored, 0.50)
        blind1, blind2 = fold_hits(selected, MIN_RECALL)
        beam_seconds = float(pd.read_parquet(path.with_name(f"{pid}.timing.parquet"))
                             .elapsed_seconds.iloc[0])
        iid_seconds = float(iid_timing.loc[pid, "elapsed_seconds"])
        iid_enrichment = float(iid_blind.loc[pid, "minority_enrichment"])
        rows.append({
            "pair_id": pid, "mode": mode, "L": int(cohort.loc[pid, "L"]),
            "split": cohort.loc[pid, "split"], "primary": bool(cohort.loc[pid, "primary"]),
            "strict_exact": mismatch[pid] == 0, "n_finished": int(frame.finished.sum()),
            "mean_contacts": float(frame.n_contacts.mean()),
            "n_unique_maps": len({canonical(pairs) for pairs in frame.contacts}),
            "beam_seconds": beam_seconds, "iid_seconds": iid_seconds,
            "time_ratio": beam_seconds / iid_seconds,
            "fold1_pool": fold1, "fold2_pool": fold2, "dual_pool": fold1 and fold2,
            "dual_pool_strict": strict1 and strict2,
            "fold1_blind": blind1, "fold2_blind": blind2,
            "dual_blind": blind1 and blind2,
            "minority_enrichment_pool": pool_result["minority_enrichment"],
            "minority_enrichment_blind": blind_result["minority_enrichment"],
            "iid_minority_enrichment_blind": iid_enrichment,
            "paired_enrichment_delta": blind_result["minority_enrichment"] - iid_enrichment,
            "iid_fold1_pool": bool(iid.loc[pid, "fold1_hit"]),
            "iid_fold2_pool": bool(iid.loc[pid, "fold2_hit"]),
            "iid_dual_pool": bool(iid.loc[pid, "dual_hit"]),
            "iid_dual_blind": bool(iid_blind.loc[pid, "dual_contact_hit"]),
        })
    per = pd.DataFrame(rows).sort_values("pair_id")
    out = HERE / "data"
    per.to_csv(out / f"foldswitch_{mode}.csv", index=False)
    cohorts = {
        "primary_test": per.primary & (per.split == "test"),
        "primary_test_exact": per.primary & (per.split == "test") & per.strict_exact,
        "primary_dev": per.primary & (per.split == "dev"),
        "all_non_capped": pd.Series(True, index=per.index),
    }
    summary = []
    for name, selected in cohorts.items():
        group = per[selected]
        if group.empty:
            continue
        lo, hi = paired_interval(group.paired_enrichment_delta.to_numpy())
        pool_lo, pool_hi = paired_interval(
            group.dual_pool.astype(int).to_numpy() - group.iid_dual_pool.astype(int).to_numpy()
        )
        blind_lo, blind_hi = paired_interval(
            group.dual_blind.astype(int).to_numpy() - group.iid_dual_blind.astype(int).to_numpy()
        )
        summary.append({
            "cohort": name, "mode": mode, "n": len(group),
            "fold1_pool": int(group.fold1_pool.sum()),
            "fold2_pool": int(group.fold2_pool.sum()),
            "dual_pool": int(group.dual_pool.sum()),
            "dual_pool_strict": int(group.dual_pool_strict.sum()),
            "dual_blind": int(group.dual_blind.sum()),
            "iid_dual_pool": int(group.iid_dual_pool.sum()),
            "iid_dual_blind": int(group.iid_dual_blind.sum()),
            "paired_dual_pool_delta": float((group.dual_pool.astype(int)
                                            - group.iid_dual_pool.astype(int)).mean()),
            "paired_dual_pool_lo": pool_lo, "paired_dual_pool_hi": pool_hi,
            "paired_dual_blind_delta": float((group.dual_blind.astype(int)
                                             - group.iid_dual_blind.astype(int)).mean()),
            "paired_dual_blind_lo": blind_lo, "paired_dual_blind_hi": blind_hi,
            "mean_enrichment_blind": float(group.minority_enrichment_blind.mean()),
            "paired_enrichment_delta": float(group.paired_enrichment_delta.mean()),
            "delta_lo": lo, "delta_hi": hi,
            "mean_time_ratio": float(group.time_ratio.mean()),
            "total_beam_seconds": float(group.beam_seconds.sum()),
            "total_iid_seconds": float(group.iid_seconds.sum()),
            "total_time_ratio": float(group.beam_seconds.sum() / group.iid_seconds.sum()),
            "mean_finished": float(group.n_finished.mean()),
        })
    report = pd.DataFrame(summary)
    report.to_csv(out / f"foldswitch_summary_{mode}.csv", index=False)
    print(report.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["seal", "score"])
    parser.add_argument("--mode", required=True)
    parser.add_argument("--n-rollouts", type=int, default=100)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    files = raw_files(args.mode, args.allow_partial)
    if args.phase == "seal":
        seal(files, args.mode, args.n_rollouts)
    else:
        score(files, args.mode, args.n_rollouts)


if __name__ == "__main__":
    main()
