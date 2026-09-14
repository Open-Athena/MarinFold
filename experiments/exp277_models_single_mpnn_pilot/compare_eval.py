# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare exp277 contact R-precision with the exp232 decontaminated winner."""

import csv
from pathlib import Path


HERE = Path(__file__).resolve().parent
EXP277_RESULTS = HERE / "data" / "eval_rollout_v2" / "subset_aggregate_metrics.csv"
EXP232_RESULTS = (
    HERE.parent
    / "exp232_sweep_cv1_decontam"
    / "evals"
    / "2026-08-24_rollout_v2"
    / "data"
    / "coreweave_results"
    / "subset_aggregate_metrics.csv"
)
OUTPUT = HERE / "data" / "eval_rollout_v2" / "exp232_comparison.csv"
MODEL_277 = "marinfold-exp277-full-epoch-m2-p06-step266344"
MODEL_232 = "marinfold-exp232-decontam-train-m2-p06-step363000"
SUBSETS = ("legacy_554", "eval-val", "eval-denovo")
RANGES = ("all", "long")
TIE_THRESHOLD = 0.005


def read_r_precision(path: Path, model: str) -> dict[tuple[str, str], float]:
    """Read the headline R-precision values for one model."""

    with path.open(newline="") as source:
        rows = csv.DictReader(source)
        return {
            (row["subset"], row["range"]): float(row["precision"])
            for row in rows
            if row["model"] == model
            and row["cut"] == "R"
            and row["subset"] in SUBSETS
            and row["range"] in RANGES
        }


def main() -> None:
    exp277 = read_r_precision(EXP277_RESULTS, MODEL_277)
    exp232 = read_r_precision(EXP232_RESULTS, MODEL_232)
    expected = {(subset, distance_range) for subset in SUBSETS for distance_range in RANGES}
    if exp277.keys() != expected or exp232.keys() != expected:
        raise ValueError("comparison inputs do not contain every expected headline metric")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="") as destination:
        writer = csv.DictWriter(
            destination,
            fieldnames=(
                "subset",
                "range",
                "exp277_r_precision",
                "exp232_r_precision",
                "delta_exp277_minus_exp232",
                "absolute_delta_below_0_005",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        for subset in SUBSETS:
            for distance_range in RANGES:
                key = (subset, distance_range)
                delta = exp277[key] - exp232[key]
                writer.writerow(
                    {
                        "subset": subset,
                        "range": distance_range,
                        "exp277_r_precision": f"{exp277[key]:.12f}",
                        "exp232_r_precision": f"{exp232[key]:.12f}",
                        "delta_exp277_minus_exp232": f"{delta:+.12f}",
                        "absolute_delta_below_0_005": abs(delta) < TIE_THRESHOLD,
                    }
                )
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
