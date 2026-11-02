"""Paired four-model summaries, oracle curves and Neff sensitivity analyses."""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_stats import bootstrap_slope
from extend_analysis import ENDPOINTS, summarize_matrix
from summarize import TIERS, estimate

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = DATA / "two_epoch"
MODELS = ["default", "mpnn", "mpnn2-lr5e-4", "mpnn2-lr1e-3"]
PAIRS = [(b, a) for a, b in combinations(MODELS, 2)]


def paired_values(frame: pd.DataFrame, endpoint: str) -> dict[str, np.ndarray]:
    """Pair on protein and require all four models rather than dropping failures."""
    pair = frame.pivot(index="stem", columns="model", values=endpoint).sort_index()
    assert set(pair.columns) == set(MODELS) and pair.notna().all().all()
    return {
        **{m: pair[m].to_numpy() for m in MODELS},
        **{f"{a} minus {b}": (pair[a] - pair[b]).to_numpy() for a, b in PAIRS},
    }


def aggregate(
    frame: pd.DataFrame, keys: list[str], endpoints: list[str]
) -> pd.DataFrame:
    """Compute equal-protein means and paired differences with common resamples."""
    rows = []
    for values, group in frame.groupby(keys, dropna=False):
        values = values if isinstance(values, tuple) else (values,)
        base = dict(zip(keys, values, strict=True))
        for endpoint in endpoints:
            for model, vector in paired_values(group, endpoint).items():
                rows.append(
                    {**base, "model": model, "endpoint": endpoint, **estimate(vector)}
                )
    return pd.DataFrame(rows)


def summarize_oracle() -> None:
    """Use every integer k, resampling proteins jointly along each trajectory."""
    singles = pd.read_csv(OUT / "rollout_f1_1000.csv.gz")
    rows, blocks, stability = [], [], []
    for (cohort, region, model), part in singles.groupby(
        ["eval_set", "range", "model"]
    ):
        values = part.pivot(index="stem", columns="rollout", values="f1").sort_index()
        assert values.shape == (97 if cohort == "eval-val" else 19, 1000)
        base = dict(eval_set=cohort, range=region, model=model)
        matrix = values.to_numpy()
        for endpoint, scores in [
            ("oracle_f1", np.maximum.accumulate(matrix, axis=1)),
            ("running_mean_f1", np.cumsum(matrix, axis=1) / np.arange(1, 1001)),
        ]:
            mean, lo, hi = summarize_matrix(scores)
            rows.extend(
                {
                    **base,
                    "endpoint": endpoint,
                    "k": k + 1,
                    "mean": mean[k],
                    "ci_low": lo[k],
                    "ci_high": hi[k],
                    "n": len(values),
                }
                for k in range(1000)
            )
        rows.append(
            {
                **base,
                "endpoint": "mean_f1",
                "k": 1000,
                **estimate(values.mean(axis=1).to_numpy()),
            }
        )
        for block in range(10):
            blocks.append(
                {
                    **base,
                    "block": block,
                    **estimate(matrix[:, block * 100 : (block + 1) * 100].mean(axis=1)),
                }
            )
        stability.append(
            {**base, **estimate(matrix.mean(axis=1) - matrix[:, :300].mean(axis=1))}
        )
    pd.DataFrame(rows).to_csv(OUT / "oracle_summary.csv", index=False)
    pd.DataFrame(blocks).to_csv(OUT / "f1_seed_blocks.csv", index=False)
    pd.DataFrame(stability).to_csv(OUT / "mean_f1_1000_minus_300.csv", index=False)


def exploratory(div: pd.DataFrame, curves: pd.DataFrame) -> None:
    """Repeat observational diversity and ensemble-gain associations without causal claims."""
    meta = pd.read_csv(DATA / "metadata.csv").set_index("stem")
    rows = []
    selected = div[
        div.eval_set.eq("eval-val")
        & div.temperature.eq(1)
        & div["mode"].eq("fixed")
        & div.block.eq("pooled")
    ]
    for region in ["all", "long"]:
        pair = (
            selected[selected["range"].eq(region)]
            .pivot(index="stem", columns="model", values="jaccard")
            .sort_index()
            .join(meta)
        )
        outcomes = {m: pair[m] for m in MODELS}
        outcomes.update({f"{a} minus {b}": pair[a] - pair[b] for a, b in PAIRS})
        for variable in ["msa_neff", "L", "best_surviving_identity_cov40"]:
            predictor = (
                np.log10(pair[variable].to_numpy())
                if variable in ["msa_neff", "L"]
                else pair[variable].to_numpy()
            )
            for model, values in outcomes.items():
                result = bootstrap_slope(values.to_numpy(), predictor)
                rows.append(
                    dict(
                        range=region,
                        endpoint="diversity",
                        model=model,
                        predictor=variable,
                        **dict(
                            zip(["mean", "ci_low", "ci_high", "n"], result, strict=True)
                        ),
                    )
                )
        c = curves[curves.eval_set.eq("eval-val") & curves["range"].eq(region)]
        for end in [300, 1000]:
            gains = c.pivot(index=["stem", "model"], columns="k", values="r_precision")
            gains = (
                (gains[end] - gains[1])
                .rename("gain")
                .reset_index()
                .pivot(index="stem", columns="model", values="gain")
                .sort_index()
                .join(meta)
            )
            for a, b in PAIRS:
                result = bootstrap_slope(
                    (gains[a] - gains[b]).to_numpy(),
                    np.log10(gains.msa_neff.to_numpy()),
                )
                rows.append(
                    dict(
                        range=region,
                        endpoint=f"ensemble_gain_1_to_{end}",
                        model=f"{a} minus {b}",
                        predictor="msa_neff",
                        **dict(
                            zip(["mean", "ci_low", "ci_high", "n"], result, strict=True)
                        ),
                    )
                )
    pd.DataFrame(rows).to_csv(OUT / "exploratory_associations.csv", index=False)


def neff_summary(metrics: pd.DataFrame) -> None:
    """Recompute tiers and interactions, preserving raw-depth and viral caveats."""
    meta = pd.read_csv(DATA / "metadata.csv").set_index("stem")
    selected = metrics[
        metrics.eval_set.eq("eval-val")
        & metrics.temperature.eq(1)
        & metrics.cut.eq("R")
        & metrics["range"].isin(["all", "long"])
    ]
    tiers, slopes, drops, subgroups = [], [], [], []
    for (mode, region), part in selected.groupby(["mode", "range"]):
        base = dict(mode=mode, range=region)
        pair = (
            part.pivot(index="stem", columns="model", values="precision")
            .sort_index()
            .join(meta, validate="one_to_one")
        )
        outcomes = {m: pair[m] for m in MODELS}
        outcomes.update({f"{a} minus {b}": pair[a] - pair[b] for a, b in PAIRS})
        for depth, tier_col in [("msa_neff", "neff_tier"), ("msa_depth", "depth_tier")]:
            for label in TIERS + ["missing"]:
                for model, outcome in outcomes.items():
                    tiers.append(
                        {
                            **base,
                            "depth_variable": depth,
                            "tier": label,
                            "model": model,
                            **estimate(outcome[pair[tier_col].eq(label)].to_numpy()),
                        }
                    )
            for model, outcome in outcomes.items():
                if " minus " not in model:
                    continue
                lo = outcome[pair[tier_col].eq(TIERS[0])].to_numpy()
                hi = outcome[pair[tier_col].eq(TIERS[-1])].to_numpy()
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
                drops.append(
                    {**base, "model": model, "depth_variable": depth, **result}
                )
                for sensitivity in ["unadjusted", "length_viral_adjusted", "nonviral"]:
                    keep = (
                        pair.is_viral.eq(0)
                        if sensitivity == "nonviral"
                        else np.ones(len(pair), bool)
                    )
                    p = pair.loc[keep]
                    cov = (
                        np.column_stack([np.log10(p.L), p.is_viral])
                        if sensitivity == "length_viral_adjusted"
                        else None
                    )
                    result = bootstrap_slope(
                        outcome.loc[keep].to_numpy(), np.log10(p[depth].to_numpy()), cov
                    )
                    slopes.append(
                        {
                            **base,
                            "model": model,
                            "depth_variable": depth,
                            "sensitivity": sensitivity,
                            **dict(
                                zip(
                                    ["mean", "ci_low", "ci_high", "n"],
                                    result,
                                    strict=True,
                                )
                            ),
                        }
                    )
        for subset, keep in [
            ("viral", pair.is_viral.eq(1)),
            ("nonviral", pair.is_viral.eq(0)),
            ("raw_depth_lt10", pair.msa_depth.lt(10)),
            ("neff_lt10", pair.msa_neff.lt(10)),
            ("native_cov40_identity_lt0.4", pair.best_surviving_identity_cov40.lt(0.4)),
            ("native_cov40_identity_ge0.4", pair.best_surviving_identity_cov40.ge(0.4)),
        ]:
            for model, outcome in outcomes.items():
                subgroups.append(
                    {
                        **base,
                        "subset": subset,
                        "model": model,
                        **estimate(outcome[keep].to_numpy()),
                    }
                )
        for column in [
            "best_surviving_identity_cov50",
            "best_surviving_identity_cov40",
        ]:
            for model, outcome in outcomes.items():
                if " minus " not in model:
                    continue
                result = bootstrap_slope(outcome.to_numpy(), pair[column].to_numpy())
                slopes.append(
                    {
                        **base,
                        "model": model,
                        "depth_variable": column,
                        "sensitivity": "exploratory",
                        **dict(
                            zip(["mean", "ci_low", "ci_high", "n"], result, strict=True)
                        ),
                    }
                )
    for name, rows in [
        ("tier_accuracy", tiers),
        ("slopes", slopes),
        ("drop_contrasts", drops),
        ("subgroups", subgroups),
    ]:
        pd.DataFrame(rows).to_csv(OUT / (name + ".csv"), index=False)


def main() -> None:
    """Save all plotted values and paired statistical comparisons as source CSVs."""
    keys = ["eval_set", "temperature", "mode", "range"]
    div = pd.read_csv(OUT / "diversity.csv")
    aggregate(
        div[div.block.eq("pooled") & div["range"].isin(["all", "long"])],
        keys,
        ["jaccard", "entropy_per_candidate", "mean_contacts"],
    ).to_csv(OUT / "diversity_summary.csv", index=False)
    aggregate(
        div[div.block.eq("pooled") & div["range"].eq("all-valid")],
        keys,
        ["jaccard", "mean_contacts"],
    ).to_csv(OUT / "unrestricted_diversity_summary.csv", index=False)
    aggregate(
        div[div.block.ne("pooled")], keys + ["block"], ["jaccard", "r_precision"]
    ).to_csv(OUT / "seed_blocks.csv", index=False)
    controls = pd.read_csv(OUT / "count_controls.csv")
    rows = []
    for key, part in controls.groupby(keys + ["model"]):
        rows.append(
            {
                **dict(zip(keys + ["model"], key, strict=True)),
                **estimate(part.sort_values("stem").jaccard_difference.to_numpy()),
            }
        )
    pd.DataFrame(rows).to_csv(OUT / "count_control_summary.csv", index=False)
    metrics = pd.read_csv(OUT / "metrics.csv")
    accuracy = aggregate(
        metrics[metrics.cut.eq("R") & metrics["range"].isin(["all", "long"])],
        keys,
        ["precision"],
    )
    accuracy.to_csv(OUT / "accuracy_summary.csv", index=False)
    reference = pd.read_csv(DATA / "two_epoch_reference_accuracy.csv")
    reference["model"] = "mpnn2-" + reference.trial
    comparison = accuracy[
        accuracy.temperature.eq(1)
        & accuracy["mode"].eq("randomized")
        & accuracy.model.isin(MODELS[2:])
    ].merge(
        reference,
        on=["model", "eval_set", "range"],
        validate="one_to_one",
        suffixes=("", "_exp364"),
    )
    assert len(comparison) == 8 and comparison.n.eq(comparison.n_exp364).all()
    comparison["difference"] = comparison["mean"] - comparison.r_precision
    comparison.to_csv(OUT / "reference_accuracy_check.csv", index=False)
    print(
        comparison[
            ["model", "eval_set", "range", "mean", "r_precision", "difference"]
        ].to_string(index=False),
        flush=True,
    )
    singles = pd.read_csv(OUT / "single_rollout_summary.csv")
    aggregate(
        singles,
        keys[:-1],
        [
            "precision",
            "recall",
            "long_precision",
            "long_recall",
            "contacts",
            "contact_density",
        ],
    ).to_csv(OUT / "single_rollout_summary_aggregate.csv", index=False)
    curves = pd.read_csv(OUT / "ensemble_curves_1000.csv")
    aggregate(curves, ["eval_set", "range", "k"], ENDPOINTS).to_csv(
        OUT / "curve_summary.csv", index=False
    )
    gain_rows = []
    for (cohort, region, model), part in curves.groupby(["eval_set", "range", "model"]):
        for endpoint in ENDPOINTS[:-1]:
            pair = part.pivot(index="stem", columns="k", values=endpoint).sort_index()
            gain_rows.append(
                dict(
                    eval_set=cohort,
                    range=region,
                    model=model,
                    endpoint=endpoint,
                    **estimate((pair[1000] - pair[300]).to_numpy()),
                )
            )
    pd.DataFrame(gain_rows).to_csv(OUT / "extension_gains.csv", index=False)
    summarize_oracle()
    print("Oracle and accuracy summaries complete", flush=True)
    neff_summary(metrics)
    exploratory(div, curves)
    shuffle = pd.concat(
        [
            pd.read_csv(DATA / "shuffle_controls.csv"),
            pd.read_csv(DATA / "shuffle_controls_two_epoch.csv"),
        ],
        ignore_index=True,
    )
    shuffle = shuffle.groupby(["stem", "model", "eval_set"], as_index=False).mean(
        numeric_only=True
    )
    shuffle["jaccard_sd_excess"] = shuffle.observed_jaccard_sd - shuffle.null_jaccard_sd
    shuffle["overlap_sd_excess"] = shuffle.observed_overlap_sd - shuffle.null_overlap_sd
    shuffle["mixing_difference"] = (
        shuffle.null_jaccard_sd - shuffle.half_chain_jaccard_sd
    )
    shuffle.to_csv(OUT / "shuffle_per_protein.csv", index=False)
    aggregate(
        shuffle,
        ["eval_set"],
        ["jaccard_sd_excess", "overlap_sd_excess", "mixing_difference"],
    ).to_csv(OUT / "shuffle_summary.csv", index=False)
    print("Four-model summaries complete", flush=True)


if __name__ == "__main__":
    main()
