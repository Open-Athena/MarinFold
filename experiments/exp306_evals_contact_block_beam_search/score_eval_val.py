#!/usr/bin/env python
"""Score beam rollout votes on eval-val with exp89's unchanged metric functions."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
sys.path.insert(0, str(EXPERIMENTS / "exp89_evals_contacts_v1_model_on_eval_set"))
from compute_metrics import metric_rows, resolved_pairs, true_matrix  # noqa: E402

BASELINE = (EXPERIMENTS / "exp277_models_single_mpnn_pilot" / "data"
            / "eval_rollout_v2" / "contact_precision_all.csv")
BASELINE_TIMING = (EXPERIMENTS / "exp277_models_single_mpnn_pilot" / "data"
                   / "eval_rollout_v2" / "timings.csv")
TRUTH = (EXPERIMENTS / "exp245_evals_foldbench_held_out_monomers" / "data"
         / "gt_universe_scored.jsonl")


def score_votes(frame: pd.DataFrame, length: int) -> np.ndarray:
    """Accumulate occurrence frequency over finished, complete contact maps."""
    votes = np.zeros((length, length), dtype=np.float32)
    for row in frame.itertuples():
        if not row.finished:
            continue
        for i, j in row.contacts:
            i, j = int(i), int(j)
            if not (0 <= i < j < length and j - i >= 6):
                raise ValueError(f"invalid saved contact ({i}, {j}) for L={length}")
            votes[i, j] += 1
            votes[j, i] += 1
    return votes


def paired_interval(values: np.ndarray, seed: int = 306) -> tuple[float, float]:
    """Protein-level percentile bootstrap interval for the paired mean."""
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(10_000, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--n-rollouts", type=int, default=100)
    args = parser.parse_args()
    targets = pd.read_csv(HERE / "data" / "targets.csv")
    val_stems = set(targets.loc[targets.cohort == "eval-val", "stem"])
    if len(val_stems) != 97:
        raise ValueError("eval-val target set changed")
    truth = {}
    with TRUTH.open() as source:
        for line in source:
            record = json.loads(line)
            if record["dataset"] == "foldbench_monomer" and record["stem"] in val_stems:
                truth[record["stem"]] = record
    if len(truth) != 97:
        raise ValueError(f"only {len(truth)} eval-val truth records")
    root = HERE / "_cache" / args.mode / "eval-val"
    files = sorted(path for path in root.glob("*.parquet") if not path.name.endswith(".timing.parquet"))
    if len(files) != 97 and not args.allow_partial:
        raise ValueError(f"expected 97 eval-val result files, found {len(files)}")
    baseline = pd.read_csv(BASELINE)
    baseline = baseline[(baseline.dataset == "foldbench_monomer")
                        & (baseline.cut == "R") & baseline["range"].isin(["all", "long"])]
    baseline = baseline.set_index(["stem", "range"])
    baseline_timing = pd.read_csv(BASELINE_TIMING)
    baseline_timing = baseline_timing[baseline_timing.dataset == "foldbench_monomer"]
    baseline_timing = baseline_timing.set_index("stem")
    rows = []
    for path in files:
        frame = pd.read_parquet(path)
        if frame.stem.nunique() != 1 or frame.stem.iloc[0] != path.stem:
            raise ValueError(f"result file identity mismatch: {path}")
        stem = path.stem
        record = truth[stem]
        if len(frame) != args.n_rollouts or frame.rollout.tolist() != list(range(args.n_rollouts)):
            raise ValueError(f"{stem}: missing or duplicated rollouts")
        length = int(record["L"])
        if not (frame.L == length).all():
            raise ValueError(f"{stem}: length mismatch")
        votes = score_votes(frame, length)
        timing = pd.read_parquet(path.with_name(f"{stem}.timing.parquet")).iloc[0]
        beam_seconds = float(timing.elapsed_seconds)
        iid_seconds = float(baseline_timing.loc[stem, "elapsed_seconds"])
        resolved = np.asarray(record["resolved"], dtype=np.int64)
        indices = resolved_pairs(resolved)
        metric = pd.DataFrame(metric_rows(votes, true_matrix(length, record["contacts"]),
                                          *indices, length, with_precision=True))
        for region in ("all", "long"):
            r_precision = float(metric[(metric["range"] == region)
                                       & (metric.cut == "R")].precision.iloc[0])
            original = float(baseline.loc[(stem, region), "precision"])
            rows.append({
                "stem": stem, "L": length, "mode": args.mode, "range": region,
                "beam_r_precision": r_precision, "iid_r_precision": original,
                "paired_delta": r_precision - original,
                "beam_seconds": beam_seconds, "iid_seconds": iid_seconds,
                "time_ratio": beam_seconds / iid_seconds,
                "n_finished": int(frame.finished.sum()),
                "mean_contacts": float(frame.n_contacts.mean()),
                "mean_tokens": float(frame.n_tokens.mean()),
                "malformed": int(frame.malformed.sum()),
            })
    per = pd.DataFrame(rows).sort_values(["stem", "range"])
    out = HERE / "data"
    per.to_csv(out / f"eval_val_{args.mode}.csv", index=False)
    summary = []
    for region, group in per.groupby("range"):
        differences = group.paired_delta.to_numpy()
        lo, hi = paired_interval(differences)
        summary.append({
            "mode": args.mode, "range": region, "n": len(group),
            "beam_r_precision": group.beam_r_precision.mean(),
            "iid_r_precision": group.iid_r_precision.mean(),
            "paired_delta": differences.mean(), "delta_lo": lo, "delta_hi": hi,
            "mean_finished": group.n_finished.mean(),
            "mean_time_ratio": group.time_ratio.mean(),
            "total_beam_seconds": group.beam_seconds.sum(),
            "total_iid_seconds": group.iid_seconds.sum(),
            "total_time_ratio": group.beam_seconds.sum() / group.iid_seconds.sum(),
        })
    report = pd.DataFrame(summary)
    report.to_csv(out / f"eval_val_summary_{args.mode}.csv", index=False)
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
