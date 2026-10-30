"""Paired protein bootstrap contrasts, frozen Neff tiers and sensitivity checks."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_stats import bootstrap_mean, bootstrap_slope

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
REPO = HERE.parents[1]
TIERS = ["<10", "10–<100", "100–<1000", "≥1000"]


def tier(values: pd.Series) -> pd.Series:
    """Use the prespecified boundaries, preserving missingness explicitly."""
    return (
        pd.cut(values, [0, 10, 100, 1000, np.inf], right=False, labels=TIERS)
        .astype("object")
        .fillna("missing")
    )


def estimate(values: np.ndarray) -> dict:
    """Return a named mean and paired-protein percentile interval."""
    mean, low, high, n = bootstrap_mean(values)
    return dict(mean=mean, ci_low=low, ci_high=high, n=n)


def paired(frame: pd.DataFrame, value: str) -> pd.DataFrame:
    """Join both models on protein identity and preserve missing score pairs."""
    return frame.pivot(index="stem", columns="model", values=value).dropna(
        subset=["default", "mpnn"]
    )


def main() -> None:
    """Write summary tables only from complete committed per-protein tables."""
    completeness = json.loads((DATA / "completeness.json").read_text())
    assert completeness["sample_count"] == 278400 and not completeness["missing"]
    metadata = pd.read_csv(DATA / "targets.csv")
    similarity = pd.read_csv(
        REPO
        / "experiments/exp245_evals_foldbench_held_out_monomers/data/residual_identity.csv"
    )
    similarity = similarity[similarity.eval_set.isin(["eval-val", "eval-denovo"])]
    metadata = metadata.merge(
        similarity.drop(columns=["eval_set", "seq_len"]),
        on="stem",
        validate="one_to_one",
    )
    metadata["neff_tier"] = tier(metadata.msa_neff)
    metadata["depth_tier"] = tier(metadata.msa_depth)
    metadata.to_csv(DATA / "metadata.csv", index=False)
    endpoints = []
    for filename, value, filter_block in [
        ("diversity", "jaccard", True),
        ("diversity", "entropy_per_candidate", True),
        ("diversity", "mean_contacts", True),
        ("count_controls", "jaccard", False),
    ]:
        frame = pd.read_csv(DATA / (filename + ".csv"))
        if filter_block:
            frame = frame[frame.block.eq("pooled")]
        for key, group in frame.groupby(["eval_set", "temperature", "mode", "range"]):
            pair = paired(group, value)
            endpoints.append(
                dict(
                    source=filename,
                    endpoint=value,
                    eval_set=key[0],
                    temperature=key[1],
                    mode=key[2],
                    range=key[3],
                    **estimate((pair.mpnn - pair.default).to_numpy()),
                )
            )
    pd.DataFrame(endpoints).to_csv(DATA / "diversity_contrasts.csv", index=False)
    metrics = pd.read_csv(DATA / "metrics.csv")
    scores = metrics[
        metrics.cut.eq("R") & metrics["range"].isin(["all", "long"])
    ].merge(
        metadata[
            [
                "stem",
                "msa_neff",
                "msa_depth",
                "neff_tier",
                "depth_tier",
                "is_viral",
                "best_surviving_identity_cov50",
                "best_surviving_identity_cov40",
            ]
        ],
        on="stem",
        validate="many_to_one",
    )
    aggregate = []
    for keys, group in scores.groupby(["eval_set", "temperature", "mode", "range"]):
        base = dict(
            zip(["eval_set", "temperature", "mode", "range"], keys, strict=True)
        )
        for model in ["default", "mpnn"]:
            aggregate.append(
                {
                    **base,
                    "model": model,
                    **estimate(
                        group.loc[group.model.eq(model), "precision"].to_numpy()
                    ),
                }
            )
        pair = paired(group, "precision")
        aggregate.append(
            {
                **base,
                "model": "mpnn-default",
                **estimate((pair.mpnn - pair.default).to_numpy()),
            }
        )
    pd.DataFrame(aggregate).to_csv(DATA / "accuracy_summary.csv", index=False)
    tiers, slopes, drops, subgroups = [], [], [], []
    natural = scores[scores.eval_set.eq("eval-val") & scores.temperature.eq(1)]
    for keys, group in natural.groupby(["mode", "range"]):
        base = dict(mode=keys[0], range=keys[1])
        pair = paired(group, "precision").join(
            metadata.set_index("stem"), validate="one_to_one"
        )
        pair["delta"] = pair.mpnn - pair.default
        for depth, tier_col in [("msa_neff", "neff_tier"), ("msa_depth", "depth_tier")]:
            for label in TIERS + ["missing"]:
                selected = pair[pair[tier_col].eq(label)]
                for model, col in [
                    ("default", "default"),
                    ("mpnn", "mpnn"),
                    ("mpnn-default", "delta"),
                ]:
                    tiers.append(
                        {
                            **base,
                            "depth_variable": depth,
                            "tier": label,
                            "model": model,
                            **estimate(selected[col].to_numpy()),
                        }
                    )
            lo = pair.loc[pair[tier_col].eq(TIERS[0]), "delta"].to_numpy()
            hi = pair.loc[pair[tier_col].eq(TIERS[-1]), "delta"].to_numpy()
            result = dict(
                mean=np.nan,
                ci_low=np.nan,
                ci_high=np.nan,
                n_low=len(lo),
                n_high=len(hi),
            )
            if len(lo) >= 2 and len(hi) >= 2:
                rng = np.random.default_rng(373)
                draws = hi[rng.integers(len(hi), size=(10000, len(hi)))].mean(
                    axis=1
                ) - lo[rng.integers(len(lo), size=(10000, len(lo)))].mean(axis=1)
                result.update(
                    mean=hi.mean() - lo.mean(),
                    ci_low=np.quantile(draws, 0.025),
                    ci_high=np.quantile(draws, 0.975),
                )
            drops.append({**base, "depth_variable": depth, **result})
            for sensitivity in ["unadjusted", "length_viral_adjusted", "nonviral"]:
                selected = (
                    pair if sensitivity != "nonviral" else pair[pair.is_viral.eq(0)]
                )
                cov = (
                    None
                    if sensitivity != "length_viral_adjusted"
                    else np.column_stack([np.log10(selected.L), selected.is_viral])
                )
                mean, low, high, n = bootstrap_slope(
                    selected.delta.to_numpy(), np.log10(selected[depth].to_numpy()), cov
                )
                slopes.append(
                    {
                        **base,
                        "depth_variable": depth,
                        "sensitivity": sensitivity,
                        "outcome": "paired_accuracy_difference",
                        "mean": mean,
                        "ci_low": low,
                        "ci_high": high,
                        "n": n,
                    }
                )
            for model in ["default", "mpnn"]:
                mean, low, high, n = bootstrap_slope(
                    pair[model].to_numpy(), np.log10(pair[depth].to_numpy())
                )
                slopes.append(
                    {
                        **base,
                        "depth_variable": depth,
                        "sensitivity": "unadjusted",
                        "outcome": model,
                        "mean": mean,
                        "ci_low": low,
                        "ci_high": high,
                        "n": n,
                    }
                )
        for name, selected in [
            ("viral", pair[pair.is_viral.eq(1)]),
            ("nonviral", pair[pair.is_viral.eq(0)]),
            ("raw_depth_lt10", pair[pair.msa_depth.lt(10)]),
            ("neff_lt10", pair[pair.msa_neff.lt(10)]),
            (
                "native_cov40_identity_lt0.4",
                pair[pair.best_surviving_identity_cov40.lt(0.4)],
            ),
            (
                "native_cov40_identity_ge0.4",
                pair[pair.best_surviving_identity_cov40.ge(0.4)],
            ),
        ]:
            for model in ["default", "mpnn", "delta"]:
                subgroups.append(
                    {
                        **base,
                        "subset": name,
                        "model": model,
                        **estimate(selected[model].to_numpy()),
                    }
                )
        for column in [
            "best_surviving_identity_cov50",
            "best_surviving_identity_cov40",
        ]:
            mean, low, high, n = bootstrap_slope(
                pair.delta.to_numpy(), pair[column].to_numpy()
            )
            slopes.append(
                {
                    **base,
                    "depth_variable": column,
                    "sensitivity": "exploratory",
                    "outcome": "paired_accuracy_difference",
                    "mean": mean,
                    "ci_low": low,
                    "ci_high": high,
                    "n": n,
                }
            )
    for name, table in [
        ("tier_accuracy", tiers),
        ("slopes", slopes),
        ("drop_contrasts", drops),
        ("subgroups", subgroups),
    ]:
        pd.DataFrame(table).to_csv(DATA / (name + ".csv"), index=False)
    curves = pd.read_csv(DATA / "ensemble_curves.csv")
    curve_summary = []
    for key, group in curves.groupby(["eval_set", "temperature", "mode", "range", "k"]):
        base = dict(
            zip(["eval_set", "temperature", "mode", "range", "k"], key, strict=True)
        )
        for variable in [
            "r_precision",
            "true_coverage",
            "true_discovered",
            "false_discovered",
            "discovery_precision",
        ]:
            pair = paired(group, variable)
            for model in ["default", "mpnn"]:
                curve_summary.append(
                    {
                        **base,
                        "endpoint": variable,
                        "model": model,
                        **estimate(pair[model].to_numpy()),
                    }
                )
            curve_summary.append(
                {
                    **base,
                    "endpoint": variable,
                    "model": "mpnn-default",
                    **estimate((pair.mpnn - pair.default).to_numpy()),
                }
            )
    pd.DataFrame(curve_summary).to_csv(DATA / "curve_summary.csv", index=False)
    diagnostic = pd.read_csv(DATA / "rollout_diagnostics.csv.gz")
    per_protein = diagnostic.groupby(
        ["eval_set", "model", "stem", "temperature", "mode"], as_index=False
    ).mean(numeric_only=True)
    per_protein.to_csv(DATA / "single_rollout_summary.csv", index=False)
    samples = diagnostic.groupby(
        ["eval_set", "model", "temperature", "mode"], as_index=False
    ).agg(
        samples=("stem", "size"),
        unfinished=("unfinished", "sum"),
        extended=("extended", "sum"),
        malformed=("malformed_statements", "sum"),
        invalid_positions=("invalid_positions", "sum"),
        retractions=("retractions", "sum"),
        duplicates=("duplicate_contacts", "sum"),
        empty=("contacts", lambda x: int(x.eq(0).sum())),
    )
    samples.to_csv(DATA / "diagnostic_counts.csv", index=False)
    # Existing KNN scores cover native AFDB, not the full native+redesign corpus.
    baselines = pd.read_csv(
        REPO
        / "experiments/exp245_evals_foldbench_held_out_monomers/data/per_protein.csv.gz"
    )
    baselines = baselines[
        baselines.stem.isin(metadata.loc[metadata.eval_set.eq("eval-val"), "stem"])
        & baselines.predictor.str.contains("KNN")
        & baselines.cut.eq("R")
    ]
    baselines.to_csv(DATA / "native_knn_context.csv", index=False)
    print(
        pd.DataFrame(aggregate)
        .query(
            "eval_set=='eval-val' and temperature==1 and mode=='randomized' and range=='all'"
        )
        .to_string(index=False)
    )
    print(pd.DataFrame(drops).to_string(index=False))


if __name__ == "__main__":
    main()
