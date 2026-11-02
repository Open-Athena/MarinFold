# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Does MarinFold add signal on top of a real PLM, or is it strictly worse?

The control #218 flagged as "the most likely route to a useful result if the
headline is mediocre" — and the headline was mediocre (0.2964 against ESM-2
650M's 0.4152). A model can sit below another and still be worth having, if it
fails on *different* assays. Per-assay Spearman correlation with ESM-2 is 0.696,
which is loose enough to make the question live but cannot answer it: an
ensemble has to be built variant by variant.

So this reads ProteinGym's per-variant score archive (1.9 GB, one CSV per assay
carrying every published baseline's score for every mutant), joins it to
MarinFold's cached conditionals on ``mutant``, and combines the two **by
within-assay rank** — the two score scales are unrelated, and rank-averaging is
the standard scale-free combiner.

The reported ensemble is the **equal-weight** one. The weight sweep is printed
as sensitivity, not used to pick the headline: choosing the best weight on the
benchmark and then reporting that number is selection on the test set.

Usage::

    uv run python ensemble.py --archive /data/exp218_proteingym/zero_shot_substitutions_scores.zip
"""

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import proteingym
import score as scoring
from analyze import PRIMARY_RULE

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

# The partner model. ESM-2 650M is the strongest single-sequence baseline on the
# leaderboard and the natural "would you rather just use a PLM?" comparison.
PARTNER_COLUMN = "ESM2_650M"
PARTNER_LABEL = "ESM2 (650M)"

# Weights on MarinFold's rank. 0 is pure ESM-2, 1 is pure MarinFold; 0.5 is the
# pre-registered equal-weight combination.
WEIGHTS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
EQUAL_WEIGHT = 0.5


def partner_scores(archive: Path, dms_id: str) -> pd.DataFrame:
    """``mutant`` → partner-model score for one assay, from the archive."""
    with zipfile.ZipFile(archive) as zf:
        name = next(n for n in zf.namelist() if n.endswith(f"{dms_id}.csv"))
        with zf.open(name) as handle:
            frame = pd.read_csv(io.BytesIO(handle.read()))
    if PARTNER_COLUMN not in frame.columns:
        raise ValueError(f"{dms_id}: archive has no {PARTNER_COLUMN} column.")
    return frame[["mutant", PARTNER_COLUMN]]


def combine(marinfold: np.ndarray, partner: np.ndarray, weight: float) -> np.ndarray:
    """Weighted average of within-assay ranks.

    Ranks rather than raw scores because the two models' scores are on unrelated
    scales — MarinFold's is a sum of log-ratios whose spread grows with the
    number of mutated sites, ESM-2's is its own thing. Averaging raw scores
    would silently weight by whichever has the larger variance.
    """
    left = pd.Series(marinfold).rank(pct=True).values
    right = pd.Series(partner).rank(pct=True).values
    return weight * left + (1.0 - weight) * right


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("/data/exp218_proteingym/zero_shot_substitutions_scores.zip"),
    )
    parser.add_argument(
        "--conditionals", type=Path, default=DATA / "conditionals"
    )
    args = parser.parse_args()
    if not args.archive.exists():
        raise SystemExit(
            f"{args.archive} not found. Fetch it from "
            f"https://marks.hms.harvard.edu/proteingym/ProteinGym_{proteingym.VERSION}"
            f"/zero_shot_substitutions_scores.zip (1.9 GB) — put it on /data, not /."
        )

    rule = scoring.ScoringRule(
        orderings=PRIMARY_RULE[0], min_context_fraction=PRIMARY_RULE[1]
    )
    reference = proteingym.reference()
    scorable = reference[reference.scorable]
    rows = []
    for _, meta in scorable.iterrows():
        path = args.conditionals / f"{meta.DMS_id}.npz"
        if not path.exists():
            continue
        assay = proteingym.load_assay(meta)
        conditionals = scoring.load_conditionals(path, rule.orderings)
        mine, _ = scoring.score_assay(conditionals, assay, rule)

        merged = assay.variants.assign(marinfold=mine).merge(
            partner_scores(args.archive, meta.DMS_id), on="mutant", how="inner"
        )
        merged = merged[
            np.isfinite(merged.marinfold) & np.isfinite(merged[PARTNER_COLUMN])
        ]
        if len(merged) < 2:
            raise ValueError(f"{meta.DMS_id}: {len(merged)} jointly scored variants.")

        row = {"DMS_id": meta.DMS_id, "n_variants": int(len(merged))}
        for weight in WEIGHTS:
            row[f"w{weight:g}"] = proteingym.assay_spearman(
                combine(merged.marinfold.values, merged[PARTNER_COLUMN].values, weight),
                merged.DMS_score.values,
            )
        rows.append(row)
        print(
            f"  {meta.DMS_id:<45s} n={row['n_variants']:6d} "
            f"mf={row['w1']:+.3f} esm={row['w0']:+.3f} ens={row[f'w{EQUAL_WEIGHT:g}']:+.3f}"
        )

    per_assay = pd.DataFrame(rows)
    per_assay.to_csv(DATA / "ensemble_spearman_dms_level.csv", index=False)

    curve = {}
    for weight in WEIGHTS:
        column = f"w{weight:g}"
        aggregated = proteingym.aggregate(
            per_assay[["DMS_id", column]].rename(columns={column: "spearman"}),
            reference,
        )
        curve[column] = aggregated["average_spearman"]

    marinfold_only = curve["w1"]
    partner_only = curve["w0"]
    equal = curve[f"w{EQUAL_WEIGHT:g}"]
    best_weight = max(curve, key=curve.get)
    summary = {
        "n_assays": int(len(per_assay)),
        "marinfold_only": marinfold_only,
        "partner_only": partner_only,
        "partner_label": PARTNER_LABEL,
        "equal_weight_ensemble": equal,
        "lift_over_partner": equal - partner_only,
        "weight_curve": curve,
        "best_weight_upper_bound": {
            "weight": best_weight,
            "average_spearman": curve[best_weight],
        },
    }
    (DATA / "ensemble_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nassays: {summary['n_assays']}")
    print(f"MarinFold alone           {marinfold_only:.4f}")
    print(f"{PARTNER_LABEL} alone         {partner_only:.4f}")
    print(f"equal-weight ensemble     {equal:.4f}   (lift {equal - partner_only:+.4f})")
    print("\nweight on MarinFold's rank:")
    for weight in WEIGHTS:
        marker = "  <- pre-registered" if weight == EQUAL_WEIGHT else ""
        print(f"  {weight:.1f}: {curve[f'w{weight:g}']:.4f}{marker}")
    print(
        f"\nbest weight {best_weight} -> {curve[best_weight]:.4f} "
        f"(test-set selection; reported as an upper bound)"
    )


if __name__ == "__main__":
    main()
