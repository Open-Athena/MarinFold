# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Step 7 -- the scoreboard: three checkpoints and five baselines, three sets.

Everything upstream produced per-protein rows in #89's schema. This joins them
into one table and answers the four questions the experiment was filed on:

1. **What does each predictor score on each set?** R-precision and AUC, all- and
   long-range, per eval set.
2. **Does eval-val over-report?** The eval-val -> eval-test delta per predictor.
   For the decontaminated checkpoints both sets are held out, so a gap there is
   a property of the protein sample, not of leakage. For the #199 cooldown,
   trained on corpora that were never filtered against FoldBench, eval-val is
   partly seen and eval-test is not, so its delta minus theirs is the
   contamination estimate.
3. **How do the checkpoints compare to the baselines?** Paired deltas with
   bootstrap confidence intervals, computed per set, on the proteins both
   predictors scored.
4. **Does the viral split change any of it?** Every headline repeated on the two
   strata.

The bootstrap is over proteins, paired: for a difference between two predictors
on the same set, resample the protein index and recompute the mean difference.
That respects the pairing, which a two-sample interval would not.

    uv run python analyze.py
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import upstream as U

DATA = U.DATA
PER_PROTEIN = DATA / "per_protein.csv.gz"
HEADLINE = DATA / "headline.csv"
DELTAS = DATA / "paired_deltas.csv"
VAL_TEST = DATA / "val_vs_test.csv"
SUMMARY = DATA / "analysis_summary.json"

#: Display names, and the order every table and figure uses.
CHECKPOINTS = {
    "marinfold-exp232-decontam-m2-p06-step145199": "#232 m2-p06 (decontaminated)",
    "marinfold-exp232-decontam-m1-p02-step145199": "#232 m1-p02 (decontaminated)",
    "marinfold-exp199-cw-p06-cool-step290400": "#199 cooldown (contaminated)",
}
BASELINES = {
    "protenix-v2_single_seq": "Protenix-v2 single-seq",
    "esmfold": "ESMFold",
    "esmfold2": "ESMFold2",
    "protenix-v2_msa": "Protenix-v2 + MSA",
}
#: The KNN null is run twice, over the two corpora the two model families
#: trained on, so it is not reused from #213 and is not part of `BASELINES`.
KNN = {
    "seq-knn-k10": "seq-KNN (unfiltered corpus)",
    "seq-knn-k10-decontam": "seq-KNN (decontaminated corpus)",
}
#: #213's column names for the baselines whose scores are reused verbatim.
EXP213_COLUMNS = {
    "ESMFold": "ESMFold",
    "ESMFold2": "ESMFold2",
    "Protenix-v2 single-seq": "Protenix-v2 single-seq",
    "Protenix-v2 + MSA": "Protenix-v2 + MSA",
}
ORDER = list(CHECKPOINTS.values()) + list(BASELINES.values()) + list(KNN.values())
SETS = ("eval-val", "eval-test", "eval-denovo")
RANGES = ("all", "long")
CUTS = ("R", "AUC")

EXP213_WIDE = (U.EXPERIMENTS / "exp213_evals_train_sequence_overlap_audit"
               / "data" / "per_protein_wide.csv.gz")
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 245


def load_marinfold(path: Path) -> pd.DataFrame:
    """Per-protein rows for the three checkpoints, from the CoreWeave run."""
    frame = pd.read_csv(path)
    frame = frame[frame["range"].isin(RANGES) & frame["cut"].isin(CUTS)]
    unknown = set(frame.model) - set(CHECKPOINTS)
    if unknown:
        raise AssertionError(f"unexpected models in {path}: {sorted(unknown)}")
    frame["predictor"] = frame.model.map(CHECKPOINTS)
    return frame[["stem", "range", "cut", "predictor", "precision"]]


def load_new_baselines(path: Path) -> pd.DataFrame:
    """Per-protein rows for the baselines this experiment ran itself."""
    frame = pd.read_csv(path)
    frame = frame[(frame.predictor == "structure")
                  & frame["range"].isin(RANGES) & frame["cut"].isin(CUTS)]
    frame = frame[frame.model.isin(BASELINES)]
    frame = frame.assign(predictor=frame.model.map(BASELINES))
    return frame[["stem", "range", "cut", "predictor", "precision"]]


def load_reused_baselines(stems: set[str]) -> pd.DataFrame:
    """Published per-protein baseline rows for the units that can reuse them."""
    wide = pd.read_csv(EXP213_WIDE)
    wide = wide[wide.stem.isin(stems) & wide["range"].isin(RANGES)
                & wide["cut"].isin(CUTS)]
    rows = []
    for name, column in EXP213_COLUMNS.items():
        part = wide[["stem", "range", "cut", column]].rename(
            columns={column: "precision"})
        part["predictor"] = name
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


def load_knn(path: Path | None) -> pd.DataFrame:
    """Per-protein rows for both sequence-KNN nulls, over every scored unit."""
    if path is None or not path.exists():
        return pd.DataFrame(columns=["stem", "range", "cut", "predictor", "precision"])
    frame = pd.read_csv(path)
    frame = frame[frame["range"].isin(RANGES) & frame["cut"].isin(CUTS)
                  & frame.model.isin(KNN)]
    frame = frame.assign(predictor=frame.model.map(KNN))
    return frame[["stem", "range", "cut", "predictor", "precision"]]


def bootstrap_delta(left: np.ndarray, right: np.ndarray, *,
                    draws: int = BOOTSTRAP_DRAWS) -> tuple[float, float, float]:
    """Paired bootstrap of ``mean(left - right)``: point estimate and 95 % CI."""
    difference = left - right
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    index = generator.integers(0, len(difference), size=(draws, len(difference)))
    means = difference[index].mean(axis=1)
    return (float(difference.mean()), float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)))


def headline(per_protein: pd.DataFrame, sets: pd.DataFrame) -> pd.DataFrame:
    """Mean metric per (predictor, set, stratum, range, cut), with counts."""
    joined = per_protein.merge(sets[["stem", "eval_set", "is_viral"]], on="stem")
    rows = []
    for stratum, frame in (("all", joined),
                           ("viral", joined[joined.is_viral == 1]),
                           ("non-viral", joined[joined.is_viral == 0])):
        grouped = frame.groupby(["eval_set", "predictor", "range", "cut"])
        summary = grouped.precision.agg(["mean", "count"]).reset_index()
        summary["stratum"] = stratum
        rows.append(summary)
    out = pd.concat(rows, ignore_index=True).rename(columns={"mean": "value", "count": "n"})
    return out[["eval_set", "stratum", "predictor", "range", "cut", "value", "n"]]


def paired_deltas(per_protein: pd.DataFrame, sets: pd.DataFrame) -> pd.DataFrame:
    """Every checkpoint against every baseline, paired over proteins."""
    joined = per_protein.merge(sets[["stem", "eval_set", "is_viral"]], on="stem")
    rows = []
    for eval_set in SETS:
        for metric_range in RANGES:
            for cut in CUTS:
                slice_ = joined[(joined.eval_set == eval_set)
                                & (joined["range"] == metric_range)
                                & (joined["cut"] == cut)]
                table = slice_.pivot_table(index="stem", columns="predictor",
                                           values="precision")
                for checkpoint in CHECKPOINTS.values():
                    if checkpoint not in table:
                        continue
                    for baseline in list(BASELINES.values()) + list(KNN.values()):
                        if baseline not in table:
                            continue
                        pair = table[[checkpoint, baseline]].dropna()
                        if pair.empty:
                            continue
                        delta, low, high = bootstrap_delta(
                            pair[checkpoint].to_numpy(), pair[baseline].to_numpy())
                        rows.append({
                            "eval_set": eval_set, "range": metric_range, "cut": cut,
                            "checkpoint": checkpoint, "baseline": baseline,
                            "n": int(len(pair)), "delta": delta,
                            "ci_low": low, "ci_high": high,
                            "significant": bool(low > 0 or high < 0),
                        })
    return pd.DataFrame(rows)


def val_vs_test(per_protein: pd.DataFrame, sets: pd.DataFrame) -> pd.DataFrame:
    """The eval-val minus eval-test gap for each predictor.

    Unpaired -- the two sets are different proteins -- so the interval is a
    two-sample bootstrap over each set independently.
    """
    joined = per_protein.merge(sets[["stem", "eval_set"]], on="stem")
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    for metric_range in RANGES:
        for cut in CUTS:
            for predictor in ORDER:
                slice_ = joined[(joined.predictor == predictor)
                                & (joined["range"] == metric_range)
                                & (joined["cut"] == cut)]
                val = slice_.loc[slice_.eval_set == "eval-val", "precision"].to_numpy()
                test = slice_.loc[slice_.eval_set == "eval-test", "precision"].to_numpy()
                if not len(val) or not len(test):
                    continue
                draws = np.array([
                    generator.choice(val, len(val)).mean()
                    - generator.choice(test, len(test)).mean()
                    for _ in range(2_000)
                ])
                rows.append({
                    "range": metric_range, "cut": cut, "predictor": predictor,
                    "n_val": len(val), "n_test": len(test),
                    "val": float(val.mean()), "test": float(test.mean()),
                    "gap": float(val.mean() - test.mean()),
                    "ci_low": float(np.percentile(draws, 2.5)),
                    "ci_high": float(np.percentile(draws, 97.5)),
                })
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--marinfold", type=Path,
                        default=DATA / "coreweave_results" / "marinfold_precision.csv")
    parser.add_argument("--new-baselines", type=Path,
                        default=DATA / "baseline_precision_new.csv.gz")
    parser.add_argument("--knn", type=Path, default=DATA / "knn_precision_new.csv.gz")
    args = parser.parse_args()

    sets = pd.read_csv(DATA / "eval_sets.csv")
    sets = sets[sets.scorable == 1]
    reuse = pd.read_csv(DATA / "baseline_reuse.csv")
    reused_stems = set(reuse.loc[reuse.source == "published", "stem"])

    frames = [
        load_marinfold(args.marinfold),
        load_new_baselines(args.new_baselines),
        load_reused_baselines(reused_stems),
        load_knn(args.knn),
    ]
    per_protein = pd.concat(frames, ignore_index=True)
    per_protein = per_protein[per_protein.stem.isin(set(sets.stem))]
    per_protein.to_csv(PER_PROTEIN, index=False)

    coverage = (per_protein[(per_protein["range"] == "all") & (per_protein["cut"] == "R")]
                .groupby("predictor").stem.nunique().to_dict())
    incomplete = {name: n for name, n in coverage.items() if n != len(sets)}

    head = headline(per_protein, sets)
    head.to_csv(HEADLINE, index=False)
    deltas = paired_deltas(per_protein, sets)
    deltas.to_csv(DELTAS, index=False)
    gaps = val_vs_test(per_protein, sets)
    gaps.to_csv(VAL_TEST, index=False)

    def cell(eval_set: str, predictor: str) -> float | None:
        row = head[(head.eval_set == eval_set) & (head.predictor == predictor)
                   & (head.stratum == "all") & (head["range"] == "all")
                   & (head["cut"] == "R")]
        return None if row.empty else round(float(row.value.iloc[0]), 4)

    summary = {
        "units": int(len(sets)),
        "predictor_coverage": coverage,
        "incomplete_predictors": incomplete,
        "r_all": {s: {p: cell(s, p) for p in ORDER} for s in SETS},
        "val_minus_test": {
            row.predictor: {"gap": round(row.gap, 4),
                            "ci": [round(row.ci_low, 4), round(row.ci_high, 4)]}
            for row in gaps[(gaps["range"] == "all") & (gaps["cut"] == "R")].itertuples()
        },
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
