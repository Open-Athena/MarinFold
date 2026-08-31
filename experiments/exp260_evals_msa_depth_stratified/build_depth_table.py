# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Step 3 — join scores to MSA depth and cut the tiered tables.

Three inputs, one row per (protein, predictor):

* this run's per-protein R-precision, published to the HF bucket by the
  CoreWeave driver (``marinfold_precision.csv``);
* the baselines, which already exist for every one of these proteins — #245's
  ``per_protein.csv.gz`` for the FoldBench half and #89's
  ``contact_precision_all.csv`` for the CAMEO-hard / CASP-FM half. Nothing is
  re-run;
* the ColabFold depths from :mod:`msa_depth_modal`.

The comparison the tiers exist for is MarinFold, which never sees an MSA,
against Protenix-v2 ``+MSA``, which does, over the same proteins binned by how
much MSA there was to see. Protenix-v2 single-seq is the control that separates
"MarinFold is good here" from "this protein is easy".

Means come with percentile bootstrap intervals over proteins, because the
shallow bins are small: whether a tier can carry a claim is a property of the
table, not a footnote to it.

    uv run python build_depth_table.py
"""

import argparse
import json

import numpy as np
import pandas as pd
import upstream as U

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 0

#: The checkpoint this experiment scores, as the metric script labels it.
MARINFOLD_MODEL = "marinfold-exp232-decontam-train-m2-p06-step363000"
MARINFOLD_LABEL = "MarinFold #232 m2-p06 (step 363k)"

#: Baseline identities in #245's published per-protein table (FoldBench half).
FOLDBENCH_BASELINES = {
    "Protenix-v2 + MSA": "Protenix-v2 + MSA",
    "Protenix-v2 single-seq": "Protenix-v2 single-seq",
    "ESMFold2": "ESMFold2",
    "seq-KNN (decontaminated corpus)": "seq-KNN (decontaminated corpus)",
}

#: The same baselines in #89's unified table (CAMEO-hard / CASP-FM half), keyed
#: by (model, mode). ``predictor == "structure"`` is the contact set read off
#: the predicted structure, which is what #245 published as well.
LEGACY_BASELINES = {
    ("protenix-v2", "msa"): "Protenix-v2 + MSA",
    ("protenix-v2", "single_seq"): "Protenix-v2 single-seq",
    ("esmfold2", "single_seq"): "ESMFold2",
}

#: Reported metrics: R-precision is the headline, AUC is the ranking-quality
#: check that does not depend on the top-L cut.
METRICS = (("all", "R"), ("long", "R"), ("all", "AUC"))


#: Three eval-denovo designs have no a3m on the Modal volume (see
#: :mod:`msa_depth_modal`). They keep their scores and sit outside every bin,
#: so "all" stays the sum of the tiers.
UNMEASURED = "unmeasured"


def depth_tier(depth: float) -> str:
    """Return the requested depth bin for a raw ColabFold sequence count."""

    if pd.isna(depth):
        return UNMEASURED
    for name, low, high in U.DEPTH_TIERS:
        if depth >= low and (high is None or depth < high):
            return name
    raise ValueError(f"depth {depth} falls outside the tier definition")


def bootstrap_ci(values: np.ndarray) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean of ``values``."""

    if len(values) < 2:
        return (float("nan"), float("nan"))
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    draws = generator.choice(values, size=(BOOTSTRAP_RESAMPLES, len(values)))
    means = draws.mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def load_marinfold(results_root: str) -> pd.DataFrame:
    """Per-protein rows for the checkpoint under test."""

    frame = pd.read_csv(f"{results_root}/results/marinfold_precision.csv")
    frame = frame[frame.model == MARINFOLD_MODEL]
    if frame.empty:
        raise ValueError(f"no rows for {MARINFOLD_MODEL} in {results_root}")
    frame = frame[["dataset", "stem", "range", "cut", "precision"]].copy()
    frame["predictor"] = MARINFOLD_LABEL
    return frame


def load_foldbench_baselines() -> pd.DataFrame:
    """#245's published baselines over the FoldBench monomers."""

    frame = pd.read_csv(U.FOLDBENCH_PER_PROTEIN_URL)
    frame = frame[frame.predictor.isin(FOLDBENCH_BASELINES)].copy()
    frame["predictor"] = frame.predictor.map(FOLDBENCH_BASELINES)
    frame["dataset"] = "foldbench_monomer"
    return frame[["dataset", "stem", "range", "cut", "predictor", "precision"]]


def load_legacy_baselines() -> pd.DataFrame:
    """#89's baselines over the CAMEO-hard and CASP-FM targets."""

    path = (
        U.EXPERIMENTS
        / "exp89_evals_contacts_v1_model_on_eval_set"
        / "data/contact_precision_all.csv"
    )
    frame = pd.read_csv(path)
    frame = frame[
        frame.dataset.isin(U.NONFOLDBENCH_NATURAL_DATASETS)
        & (frame.predictor == "structure")
    ].copy()
    frame["label"] = [
        LEGACY_BASELINES.get((model, mode))
        for model, mode in zip(frame.model, frame["mode"], strict=True)
    ]
    frame = frame[frame.label.notna()].rename(columns={"label": "baseline"})
    frame = frame.drop(columns=["predictor"]).rename(columns={"baseline": "predictor"})
    return frame[["dataset", "stem", "range", "cut", "predictor", "precision"]]


def assemble(results_root: str, depths_path: str) -> pd.DataFrame:
    """One row per (protein, predictor, metric), with depth and subset joined."""

    universe = pd.read_csv(U.DATA / "universe.csv")
    depths = pd.read_csv(depths_path)
    if depths.duplicated(["stem", "msa_volume"]).any():
        raise ValueError("depth measurements are not unique per (stem, volume)")
    # Cross-volume duplicates are measured for the consistency check and dropped
    # here: each protein is scored under exactly one volume.
    depths = depths.merge(
        universe[["stem", "msa_volume"]], on=["stem", "msa_volume"], how="inner"
    )
    if len(depths) != len(universe):
        raise ValueError(
            f"depth rows {len(depths)} do not cover the universe {len(universe)}"
        )

    scores = pd.concat(
        [
            load_marinfold(results_root),
            load_foldbench_baselines(),
            load_legacy_baselines(),
        ],
        ignore_index=True,
    )
    keep = pd.MultiIndex.from_tuples(METRICS)
    scores = scores[pd.MultiIndex.from_frame(scores[["range", "cut"]]).isin(keep)]

    frame = universe.merge(
        depths[["stem", "msa_volume", "n_seqs", "n_eff_0.8", "n_eff_0.62", "query_len"]],
        on=["stem", "msa_volume"],
        how="left",
    ).merge(scores, on=["dataset", "stem"], how="inner")
    frame = frame.rename(columns={"n_seqs": "msa_depth", "n_eff_0.8": "msa_neff"})
    frame["depth_tier"] = [depth_tier(depth) for depth in frame.msa_depth]
    frame["neff_tier"] = [depth_tier(neff) for neff in frame.msa_neff]
    return frame


def tier_table(frame: pd.DataFrame, *, tier_column: str) -> pd.DataFrame:
    """Mean and bootstrap interval per (subset, tier, predictor, metric)."""

    frame = frame[frame[tier_column] != UNMEASURED]
    natural = frame[frame.subset != "foldbench_designed"].copy()
    populations = {
        "all_natural": natural,
        "foldbench_natural": natural[natural.subset == "foldbench_natural"],
        "nonfoldbench_natural": natural[natural.subset == "nonfoldbench_natural"],
        "foldbench_designed": frame[frame.subset == "foldbench_designed"],
    }
    tiers = [name for name, _, _ in U.DEPTH_TIERS]
    rows = []
    for population, subframe in populations.items():
        for tier in [*tiers, "all"]:
            selected = subframe if tier == "all" else subframe[subframe[tier_column] == tier]
            for (predictor, range_name, cut), group in selected.groupby(
                ["predictor", "range", "cut"]
            ):
                values = group.precision.dropna().to_numpy()
                if not len(values):
                    continue
                low, high = bootstrap_ci(values)
                rows.append(
                    {
                        "population": population,
                        "tier_axis": tier_column,
                        "tier": tier,
                        "predictor": predictor,
                        "range": range_name,
                        "cut": cut,
                        "n": len(values),
                        "mean": float(values.mean()),
                        "ci_low": low,
                        "ci_high": high,
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["population", "tier", "cut", "range", "predictor"], ignore_index=True
    )


def paired_deltas(frame: pd.DataFrame, *, tier_column: str) -> pd.DataFrame:
    """Per-protein MarinFold-minus-baseline differences, by tier.

    Two independent means over five proteins say very little; the same five
    proteins scored by both predictors say considerably more, because the
    protein-to-protein variance that dominates a small bin cancels. Every
    shallow-tier claim in the README rests on this table, not on the difference
    of the two columns in the tier table.
    """

    frame = frame[frame[tier_column] != UNMEASURED]
    natural = frame[frame.subset != "foldbench_designed"]
    populations = {
        "all_natural": natural,
        "foldbench_natural": natural[natural.subset == "foldbench_natural"],
        "nonfoldbench_natural": natural[natural.subset == "nonfoldbench_natural"],
    }
    tiers = [name for name, _, _ in U.DEPTH_TIERS]
    rows = []
    for population, subframe in populations.items():
        for tier in [*tiers, "all"]:
            selected = (
                subframe if tier == "all" else subframe[subframe[tier_column] == tier]
            )
            for (range_name, cut), metric_frame in selected.groupby(["range", "cut"]):
                wide = metric_frame.pivot_table(
                    index=["dataset", "stem"], columns="predictor", values="precision"
                )
                if MARINFOLD_LABEL not in wide.columns:
                    continue
                for baseline in wide.columns:
                    if baseline == MARINFOLD_LABEL:
                        continue
                    paired = wide[[MARINFOLD_LABEL, baseline]].dropna()
                    if paired.empty:
                        continue
                    difference = (
                        paired[MARINFOLD_LABEL] - paired[baseline]
                    ).to_numpy()
                    low, high = bootstrap_ci(difference)
                    rows.append(
                        {
                            "population": population,
                            "tier_axis": tier_column,
                            "tier": tier,
                            "baseline": baseline,
                            "range": range_name,
                            "cut": cut,
                            "n": len(difference),
                            "mean_delta": float(difference.mean()),
                            "ci_low": low,
                            "ci_high": high,
                            "marinfold_wins": int((difference > 0).sum()),
                        }
                    )
    return pd.DataFrame(rows).sort_values(
        ["population", "tier", "cut", "range", "baseline"], ignore_index=True
    )


def tier_counts(frame: pd.DataFrame) -> pd.DataFrame:
    """How many proteins land in each bin, on both depth axes."""

    proteins = frame.drop_duplicates(["dataset", "stem"])
    rows = []
    for axis in ("depth_tier", "neff_tier"):
        counts = (
            proteins.groupby(["subset", axis]).size().rename("n_proteins").reset_index()
        )
        counts = counts.rename(columns={axis: "tier"})
        counts.insert(0, "tier_axis", axis)
        rows.append(counts)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        default=U.RESULTS_URL,
        help="Root of the published (or local) evaluation results.",
    )
    parser.add_argument("--depths", default=str(U.DATA / "msa_depth.csv"))
    args = parser.parse_args()

    frame = assemble(args.results, args.depths)
    frame.to_csv(U.DATA / "per_protein_depth.csv", index=False)
    depth_tiers = tier_table(frame, tier_column="depth_tier")
    neff_tiers = tier_table(frame, tier_column="neff_tier")
    pd.concat([depth_tiers, neff_tiers], ignore_index=True).to_csv(
        U.DATA / "depth_tiers.csv", index=False
    )
    counts = tier_counts(frame)
    counts.to_csv(U.DATA / "tier_counts.csv", index=False)
    pd.concat(
        [
            paired_deltas(frame, tier_column="depth_tier"),
            paired_deltas(frame, tier_column="neff_tier"),
        ],
        ignore_index=True,
    ).to_csv(U.DATA / "paired_deltas.csv", index=False)

    headline = depth_tiers[
        (depth_tiers.predictor == MARINFOLD_LABEL)
        & (depth_tiers["range"] == "all")
        & (depth_tiers["cut"] == "R")
    ]
    print(headline.to_string(index=False))
    print(
        json.dumps(
            {
                "proteins": int(frame.drop_duplicates(["dataset", "stem"]).shape[0]),
                "predictors": sorted(frame.predictor.unique()),
                "depth_tier_counts": counts[counts.tier_axis == "depth_tier"]
                .groupby("tier")
                .n_proteins.sum()
                .to_dict(),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
