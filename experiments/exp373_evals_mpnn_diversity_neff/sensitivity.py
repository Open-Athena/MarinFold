"""Prespecified seed/covariate checks and labeled exploratory associations."""

from pathlib import Path

import numpy as np
import pandas as pd

from analysis_stats import bootstrap_mean, bootstrap_slope

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def named(result: tuple[float, float, float, int]) -> dict:
    """Label an estimate and interval."""
    return dict(zip(["mean", "ci_low", "ci_high", "n"], result, strict=True))


def main() -> None:
    """Compute sensitivities from protein-level results, never treating samples as proteins."""
    meta = pd.read_csv(DATA / "metadata.csv").set_index("stem")
    div = pd.read_csv(DATA / "diversity.csv")
    curves = pd.read_csv(DATA / "ensemble_curves.csv")
    rows, blocks = [], []
    selected = div[
        div.eval_set.eq("eval-val") & div.temperature.eq(1) & div["mode"].eq("fixed")
    ]
    for (region, block), group in selected.groupby(["range", "block"]):
        pair = group.pivot(index="stem", columns="model", values="jaccard")
        blocks.append(
            {
                "range": region,
                "block": block,
                "endpoint": "jaccard_difference",
                **named(bootstrap_mean((pair.mpnn - pair.default).to_numpy())),
            }
        )
        if block != "pooled":
            accuracy = group.pivot(
                index="stem", columns="model", values="r_precision"
            ).dropna()
            blocks.append(
                {
                    "range": region,
                    "block": block,
                    "endpoint": "r_precision_difference",
                    **named(
                        bootstrap_mean((accuracy.mpnn - accuracy.default).to_numpy())
                    ),
                }
            )
    pd.DataFrame(blocks).to_csv(DATA / "seed_blocks.csv", index=False)
    for region in ["all", "long"]:
        group = selected[selected["range"].eq(region) & selected.block.eq("pooled")]
        pair = group.pivot(index="stem", columns="model", values="jaccard").join(meta)
        for variable, transform in [
            ("msa_neff", np.log10),
            ("L", np.log10),
            ("best_surviving_identity_cov40", lambda x: x),
        ]:
            rows.append(
                {
                    "range": region,
                    "outcome": "paired_diversity_difference",
                    "predictor": variable,
                    **named(
                        bootstrap_slope(
                            (pair.mpnn - pair.default).to_numpy(),
                            transform(pair[variable].to_numpy()),
                        )
                    ),
                }
            )
        for model in ["default", "mpnn"]:
            rows.append(
                {
                    "range": region,
                    "outcome": model + "_diversity",
                    "predictor": "msa_neff",
                    **named(
                        bootstrap_slope(
                            pair[model].to_numpy(), np.log10(pair.msa_neff.to_numpy())
                        )
                    ),
                }
            )
        gains = curves[
            curves.eval_set.eq("eval-val")
            & curves.temperature.eq(1)
            & curves["mode"].eq("fixed")
            & curves["range"].eq(region)
            & curves.k.isin([1, 300])
        ]
        gains = gains.pivot(index=["stem", "model"], columns="k", values="r_precision")
        gains["gain"] = gains[300] - gains[1]
        gains = (
            gains.reset_index()
            .pivot(index="stem", columns="model", values="gain")
            .join(meta)
        )
        for model in ["default", "mpnn"]:
            rows.append(
                {
                    "range": region,
                    "outcome": model + "_ensemble_gain_1_to_300",
                    "predictor": "msa_neff",
                    **named(
                        bootstrap_slope(
                            gains[model].to_numpy(), np.log10(gains.msa_neff.to_numpy())
                        )
                    ),
                }
            )
        rows.append(
            {
                "range": region,
                "outcome": "paired_ensemble_gain_difference",
                "predictor": "msa_neff",
                **named(
                    bootstrap_slope(
                        (gains.mpnn - gains.default).to_numpy(),
                        np.log10(gains.msa_neff.to_numpy()),
                    )
                ),
            }
        )
    pd.DataFrame(rows).to_csv(DATA / "exploratory_associations.csv", index=False)
    shuffle = pd.read_csv(DATA / "shuffle_controls.csv")
    shuffle = shuffle.groupby(["eval_set", "model", "stem"], as_index=False).mean(
        numeric_only=True
    )
    shuffle["jaccard_sd_excess"] = shuffle.observed_jaccard_sd - shuffle.null_jaccard_sd
    shuffle["overlap_sd_excess"] = shuffle.observed_overlap_sd - shuffle.null_overlap_sd
    shuffle["mixing_difference"] = (
        shuffle.null_jaccard_sd - shuffle.half_chain_jaccard_sd
    )
    summaries = []
    for subset, group in shuffle.groupby("eval_set"):
        for variable in ["jaccard_sd_excess", "overlap_sd_excess", "mixing_difference"]:
            pair = group.pivot(index="stem", columns="model", values=variable)
            for model in ["default", "mpnn"]:
                summaries.append(
                    {
                        "eval_set": subset,
                        "model": model,
                        "endpoint": variable,
                        **named(bootstrap_mean(pair[model].to_numpy())),
                    }
                )
            summaries.append(
                {
                    "eval_set": subset,
                    "model": "mpnn-default",
                    "endpoint": variable,
                    **named(bootstrap_mean((pair.mpnn - pair.default).to_numpy())),
                }
            )
    pd.DataFrame(summaries).to_csv(DATA / "shuffle_summary.csv", index=False)


if __name__ == "__main__":
    main()
