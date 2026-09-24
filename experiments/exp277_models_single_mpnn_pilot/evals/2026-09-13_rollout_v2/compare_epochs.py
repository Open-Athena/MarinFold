# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare the exp277 first and second epochs scored in one rollout-v2 run.

Both checkpoints are scored in the same driver job (`--suite exp277-epochs`), so
every protein is evaluated twice under one pinned worker, one recipe and one
cluster. That lets the comparison be *paired per protein* rather than a
difference of subset means, which matters here because the effect is small
relative to rollout sampling noise (#204 spans 0.0023 R-precision across four
evaluations of one unchanged checkpoint).

Reads the published results of a run id and writes the committed tables.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

EPOCH1_LABEL = "exp277_full_epoch_m2_p06_step266344"
EPOCH2_LABEL = "exp277_full_epoch2_from213072_step479417"
BOOTSTRAP_SEED = 277
BOOTSTRAP_DRAWS = 10_000
LOW_MSA_SET = "experiments/exp260_evals_msa_depth_stratified/data/low_msa_depth_set.csv"


def model_id(label: str) -> str:
    return f"marinfold-{label.replace('_', '-')}"


def paired_delta(frame: pd.DataFrame, rng: np.random.Generator) -> dict:
    """Bootstrap the mean per-protein epoch2 - epoch1 difference."""
    wide = frame.pivot_table(
        index=["dataset", "stem"], columns="epoch", values="precision"
    ).dropna()
    if wide.empty:
        return {"n": 0}
    delta = (wide["epoch2"] - wide["epoch1"]).to_numpy()
    draws = delta[rng.integers(0, len(delta), size=(BOOTSTRAP_DRAWS, len(delta)))]
    means = draws.mean(axis=1)
    return {
        "n": len(delta),
        "epoch1": float(wide["epoch1"].mean()),
        "epoch2": float(wide["epoch2"].mean()),
        "delta": float(delta.mean()),
        "ci_low": float(np.percentile(means, 2.5)),
        "ci_high": float(np.percentile(means, 97.5)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    precision = pd.read_csv(args.results / "marinfold_precision.csv")
    subsets = pd.read_csv(args.results / "evaluation_subsets.csv")
    timings = pd.read_csv(args.results / "timings.csv")
    precision["epoch"] = precision.model.map(
        {model_id(EPOCH1_LABEL): "epoch1", model_id(EPOCH2_LABEL): "epoch2"}
    )
    if precision.epoch.isna().any():
        raise ValueError("unexpected model identities in the precision table")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    # A rollout that reaches the token cap is excluded from voting, so a unit
    # with capped rollouts is scored on fewer votes. Report the headline over
    # every unit and, beside it, the same delta with those units dropped.
    capped = set(
        map(
            tuple,
            timings.loc[timings.unfinished_rollouts > 0, ["dataset", "stem"]]
            .drop_duplicates()
            .to_numpy(),
        )
    )
    scored = precision[precision.cut == "R"].merge(subsets, on=["dataset", "stem"])
    rows = []
    for subset in ("legacy_554", "eval-val", "eval-denovo"):
        for contact_range in ("all", "long"):
            chunk = scored[
                (scored.subset == subset) & (scored["range"] == contact_range)
            ]
            record = {"subset": subset, "range": contact_range}
            record.update(paired_delta(chunk, rng))
            uncapped = chunk[~chunk.set_index(["dataset", "stem"]).index.isin(capped)]
            dropped = paired_delta(uncapped, rng)
            record["n_uncapped"] = dropped["n"]
            record["delta_uncapped"] = dropped.get("delta")
            rows.append(record)
    headline = pd.DataFrame(rows)
    headline.to_csv(args.out / "epoch_comparison.csv", index=False)

    # Required cuts: the low-MSA-depth regime a single-sequence model exists for,
    # and the viral split. Both are reported with explicit denominators because
    # eval-test stays unread, which leaves parts of the frozen set uncovered.
    low = pd.read_csv(args.repo_root / LOW_MSA_SET)
    low["designed"] = low.designed.astype(str).str.lower() == "true"
    all_range = scored[(scored["range"] == "all")]
    cuts = []
    definitions = {
        "low_msa_natural": low[~low.designed],
        "low_msa_foldbench_only": low[
            (~low.designed) & (low.dataset == "foldbench_monomer")
        ],
        "low_msa_designed": low[low.designed],
    }
    for name, members in definitions.items():
        keys = set(map(tuple, members[["dataset", "stem"]].to_numpy()))
        chunk = all_range[all_range.set_index(["dataset", "stem"]).index.isin(keys)]
        record = {"cut": name, "frozen_members": len(members)}
        record.update(paired_delta(chunk, rng))
        cuts.append(record)
    foldbench = all_range[all_range.subset.isin(("eval-val", "eval-denovo"))]
    for name, mask in (
        ("viral", foldbench.is_viral == True),  # noqa: E712 - pandas mask
        ("non_viral", foldbench.is_viral == False),  # noqa: E712
    ):
        record = {"cut": name, "frozen_members": None}
        record.update(paired_delta(foldbench[mask], rng))
        cuts.append(record)
    pd.DataFrame(cuts).to_csv(args.out / "epoch_comparison_cuts.csv", index=False)

    capped_counts = (
        timings.loc[timings.unfinished_rollouts > 0]
        .groupby(["model_nickname", "dataset", "stem"], as_index=False)
        .unfinished_rollouts.sum()
    )
    capped_counts.to_csv(args.out / "capped_rollouts.csv", index=False)
    print(headline.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
