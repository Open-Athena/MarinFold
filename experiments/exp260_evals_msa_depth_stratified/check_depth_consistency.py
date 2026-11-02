# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Two checks on the depth measurements before anything is concluded from them.

**Against #247.** That experiment already counted sequences in the same
``protenix-foldbench-msa`` a3m files, with different code (a streaming
``>``-line count rather than the parse-and-align path in #74's
``msa_depth.py``). The two should agree exactly for every FoldBench natural
protein; a disagreement means one of the counts is wrong.

**Across the two volumes.** Eleven stems live in both ``protenix-foldbench-msa``
and ``protenix-exp74-msa``, searched about three weeks apart against a database
that grows. Their spread is the only measurement we have of how much the gap
between the two ColabFold runs moves a depth, and therefore of how comparable
the FoldBench and non-FoldBench halves of the tier table really are.

    uv run python check_depth_consistency.py
"""

import json

import pandas as pd
import upstream as U

EXP247_FEATURES = (
    U.EXPERIMENTS
    / "exp247_evals_protein_property_analysis"
    / "data/protein_features.csv"
)


def against_exp247(depths: pd.DataFrame) -> dict:
    """Compare this run's raw depths to #247's independent count."""

    published = pd.read_csv(EXP247_FEATURES)[["stem", "msa_depth"]]
    merged = published.merge(
        depths[depths.msa_volume == "foldbench"][["stem", "n_seqs"]],
        on="stem",
        how="inner",
    )
    difference = (merged.n_seqs - merged.msa_depth).abs()
    return {
        "proteins_compared": int(len(merged)),
        "exact_matches": int((difference == 0).sum()),
        "largest_absolute_difference": int(difference.max()),
        "disagreeing_stems": merged.loc[difference > 0, "stem"].tolist()[:20],
    }


def across_volumes(depths: pd.DataFrame) -> dict:
    """Compare the two ColabFold runs on the stems both volumes hold."""

    wide = depths.pivot_table(
        index="stem", columns="msa_volume", values=["n_seqs", "n_eff_0.8"]
    ).dropna()
    if wide.empty:
        return {"stems_compared": 0}
    depth_ratio = wide[("n_seqs", "foldbench")] / wide[("n_seqs", "exp74")]
    neff_ratio = wide[("n_eff_0.8", "foldbench")] / wide[("n_eff_0.8", "exp74")]
    same_tier = [
        _tier(foldbench) == _tier(exp74)
        for foldbench, exp74 in zip(
            wide[("n_seqs", "foldbench")], wide[("n_seqs", "exp74")], strict=True
        )
    ]
    return {
        "stems_compared": int(len(wide)),
        "depth_ratio_median": float(depth_ratio.median()),
        "depth_ratio_min": float(depth_ratio.min()),
        "depth_ratio_max": float(depth_ratio.max()),
        "neff_ratio_median": float(neff_ratio.median()),
        "same_depth_tier": int(sum(same_tier)),
        "per_stem": [
            {
                "stem": stem,
                "foldbench_n_seqs": int(wide.loc[stem, ("n_seqs", "foldbench")]),
                "exp74_n_seqs": int(wide.loc[stem, ("n_seqs", "exp74")]),
                "foldbench_n_eff": float(wide.loc[stem, ("n_eff_0.8", "foldbench")]),
                "exp74_n_eff": float(wide.loc[stem, ("n_eff_0.8", "exp74")]),
            }
            for stem in wide.index
        ],
    }


def _tier(depth: float) -> str:
    for name, low, high in U.DEPTH_TIERS:
        if depth >= low and (high is None or depth < high):
            return name
    raise ValueError(depth)


def main() -> None:
    depths = pd.read_csv(U.DATA / "msa_depth.csv")
    report = {
        "vs_exp247": against_exp247(depths),
        "across_volumes": across_volumes(depths),
    }
    (U.DATA / "depth_consistency.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
