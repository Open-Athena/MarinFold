# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare exp343's contact R-precision with exp277, paired per protein.

The baseline is **not re-scored**. exp277's per-protein results were produced by
the same worker bytes
(`sha256 dd2f76dd5d34d1549e0be1197d9053d6e3aa0ef909f062ff5355d65f31cd571c`) on the
same 670 units under the same exp82 rollout recipe, so its committed CSV is the
comparison input. Drawing fresh rollouts for exp277 would only add a second,
slightly different exp277 number from a new sampling draw and invite confusion
about which one is "the" baseline. exp277 compared against exp232 this way, and
exp169 against exp232 before that.

The intervals are protein bootstraps: they describe variation across evaluation
proteins, and say nothing about training-seed or repeated-rollout variation. With
one seed per arm that limit is the dominant caveat, not a footnote.

    uv run python compare_eval.py
"""

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP343_RESULTS = HERE / "data" / "eval_rollout_v2"
EXP277_RESULTS = (
    HERE.parent / "exp277_models_single_mpnn_pilot" / "data" / "eval_rollout_v2"
)
#: The eval's model string is derived from the checkpoint label, which follows
#: AGENTS.md's `<wandb-run-name>-step-<N>` form -- so match on the experiment
#: rather than a prefix that breaks every time the label convention changes.
MODEL_343_TOKEN = "exp343"
MODEL_277 = "marinfold-exp277-full-epoch-m2-p06-step266344"
SUBSETS = ("legacy_554", "eval-val", "eval-denovo")
RANGES = ("all", "long")
#: Predeclared in the issue. An absolute eval-val delta below this is a tie, and
#: the hypothesis is that eval-val ties.
TIE_THRESHOLD = 0.005
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 343


def read_headline(path: Path, model: str | None) -> dict[tuple[str, str], float]:
    """Headline R-precision per (subset, range) for one model."""
    with path.open(newline="") as source:
        rows = [
            row
            for row in csv.DictReader(source)
            if row["cut"] == "R"
            and row["subset"] in SUBSETS
            and row["range"] in RANGES
            and (model is None or row["model"] == model)
        ]
    if model is None:
        models = {row["model"] for row in rows}
        matching = {name for name in models if MODEL_343_TOKEN in name}
        if len(matching) != 1:
            raise ValueError(f"expected one exp343 model in {path}, found {models}")
        model = matching.pop()
        rows = [row for row in rows if row["model"] == model]
    result = {(row["subset"], row["range"]): float(row["precision"]) for row in rows}
    expected = {(subset, rng) for subset in SUBSETS for rng in RANGES}
    if set(result) != expected:
        raise ValueError(f"{path} is missing {sorted(expected - set(result))}")
    return result


def read_exp343_per_protein(path: Path) -> dict[tuple[str, str, str], float]:
    """exp343 R-precision keyed by `(dataset, stem, range)`.

    `contact_precision_all.csv` carries no `subset` column -- the subset a unit
    belongs to lives in exp277's committed paired file -- so the join key is the
    unit itself. Each `(dataset, stem)` belongs to exactly one subset across all
    670 units, so this is unambiguous.

    A handful of rows have an empty `precision`: a unit with no long-range true
    contacts has no long-range R-precision. Those are skipped here and drop out
    of the pairing rather than being read as zero.
    """
    values: dict[tuple[str, str, str], float] = {}
    with path.open(newline="") as source:
        for row in csv.DictReader(source):
            if row["cut"] != "R" or not row["precision"]:
                continue
            values[(row["dataset"], row["stem"], row["range"])] = float(row["precision"])
    return values


def read_exp277_per_protein(path: Path):
    """exp277 R-precision and its subset, keyed by `(dataset, stem, range)`."""
    values: dict[tuple[str, str, str], tuple[str, float]] = {}
    with path.open(newline="") as source:
        for row in csv.DictReader(source):
            if not row["precision_exp277"]:
                continue
            values[(row["dataset"], row["stem"], row["range"])] = (
                row["subset"],
                float(row["precision_exp277"]),
            )
    return values


def bootstrap(deltas: list[float], seed: int) -> tuple[float, float]:
    """Percentile 95% interval over resampled per-protein differences."""
    if not deltas:
        raise ValueError("no paired differences to resample")
    rng = random.Random(seed)
    count = len(deltas)
    means = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        total = 0.0
        for _ in range(count):
            total += deltas[rng.randrange(count)]
        means.append(total / count)
    means.sort()
    low = means[int(0.025 * (BOOTSTRAP_RESAMPLES - 1))]
    high = means[int(0.975 * (BOOTSTRAP_RESAMPLES - 1))]
    return low, high


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=EXP343_RESULTS)
    arguments = parser.parse_args()
    results = arguments.results
    exp343 = read_headline(results / "subset_aggregate_metrics.csv", None)
    exp277 = read_headline(EXP277_RESULTS / "subset_aggregate_metrics.csv", MODEL_277)

    per_343 = read_exp343_per_protein(results / "contact_precision_all.csv")
    per_277 = read_exp277_per_protein(EXP277_RESULTS / "paired_r_precision.csv")

    rows = []
    for subset in SUBSETS:
        for distance_range in RANGES:
            key = (subset, distance_range)
            delta = exp343[key] - exp277[key]
            shared = sorted(
                k
                for k, (unit_subset, _) in per_277.items()
                if unit_subset == subset and k[2] == distance_range and k in per_343
            )
            deltas = [per_343[k] - per_277[k][1] for k in shared]
            low, high = bootstrap(deltas, BOOTSTRAP_SEED)
            paired_mean = sum(deltas) / len(deltas)
            rows.append(
                {
                    "subset": subset,
                    "range": distance_range,
                    "exp343_r_precision": f"{exp343[key]:.12f}",
                    "exp277_r_precision": f"{exp277[key]:.12f}",
                    "delta_exp343_minus_exp277": f"{delta:+.12f}",
                    "n_paired": len(deltas),
                    "paired_mean_delta": f"{paired_mean:+.12f}",
                    "ci_low": f"{low:+.12f}",
                    "ci_high": f"{high:+.12f}",
                    "absolute_delta_below_0_005": abs(delta) < TIE_THRESHOLD,
                    "interval_covers_zero": low <= 0.0 <= high,
                }
            )
    results.mkdir(parents=True, exist_ok=True)
    output = results / "exp277_comparison.csv"
    with output.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    provenance = {
        "exp343_results": str(results),
        "exp277_results": str(EXP277_RESULTS),
        "baseline_rescored": False,
        "worker_sha256": "dd2f76dd5d34d1549e0be1197d9053d6e3aa0ef909f062ff5355d65f31cd571c",
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "tie_threshold": TIE_THRESHOLD,
    }
    (results / "exp277_comparison.provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    for row in rows:
        print(
            f"{row['subset']:>12s} {row['range']:>4s}  exp343 {row['exp343_r_precision'][:7]}"
            f"  exp277 {row['exp277_r_precision'][:7]}"
            f"  delta {row['delta_exp343_minus_exp277'][:8]}"
            f"  [{row['ci_low'][:8]}, {row['ci_high'][:8]}]  n={row['n_paired']}"
            f"  tie={row['absolute_delta_below_0_005']}"
        )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
