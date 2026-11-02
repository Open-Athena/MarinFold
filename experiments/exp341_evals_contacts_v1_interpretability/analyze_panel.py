# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""A3 and A4 from the panel readouts (``score_panel.py``).

Every quantity is computed per matched pair (one removed structure and its
kept control from the same relative-count bin) and bootstrapped over pairs
within each bin (``N_BOOT`` resamples, seeded). Pairs where either structure
has no true contact in the band (R-precision NaN) are dropped for that band.

- **A3** (``data/relative_dose_response.csv``): per bin, exp232's mean
  R-precision on removed and kept structures and the removed-minus-kept gap.
  Also a piecewise-linear fit of exp232's removed R-precision against
  log(1 + relative count) with one breakpoint (grid over observed values),
  the plan's saturation test, with the breakpoint's bootstrap CI over
  removed structures; and an OLS of the same scores on log(1 + relatives),
  log length and pLDDT, because relative count falls steeply with length
  and rises with pLDDT across the bins.
- **A4** (``data/contamination_did.csv``): per bin, the difference in
  differences (exp199 - exp232 on removed) - (exp199 - exp232 on kept), and
  each of its two terms.
- **Removed-vs-kept gap checks** (``data/removed_gap_checks.json``). exp232
  scores removed structures above their matched kept controls. Two candidate
  explanations are tested: removed structures being easier because they
  resemble eval proteins (exp232's removed R-precision against the drop
  list's identity to the nearest eval reference, raw by quartile and adjusted
  for length, pLDDT and relative count), and contact density or long-range
  share differing within pairs (the per-pair gap regressed on both
  differences; the intercept is the gap left over).

    uv run python analyze_panel.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent / "data"
EXP232 = "contacts-v1-exp232-m2-p06-train-1.5B"
EXP199 = "contacts-v1-exp199-cooldown-1.5B"
BINS = ["0", "1-3", "4-15", "16-63", "64+"]
METRICS = ["r_precision", "r_precision_long"]
N_BOOT = 10_000
SEED = 341


def load() -> pd.DataFrame:
    """One row per pair: each model's score on the removed and kept member."""
    frames = []
    for tag, nickname in (("232", EXP232), ("199", EXP199)):
        scores = pd.read_csv(DATA / "panel_scores" / nickname / "scores.csv", dtype={"bin": str})
        wide = scores.pivot(index=["bin", "pair"], columns="set", values=METRICS)
        wide.columns = [f"{metric}_{tag}_{subset}" for metric, subset in wide.columns]
        frames.append(wide)
    pairs = pd.concat(frames, axis=1).reset_index()
    panel = pd.read_parquet(DATA / "panel.parquet", columns=["set", "pair", "seq_len", "global_plddt", "n_relatives"])
    removed = panel[panel["set"] == "removed"].drop(columns="set").set_index("pair")
    return pairs.join(removed, on="pair")


def boot_mean(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    draws = rng.choice(values, size=(N_BOOT, len(values)), replace=True).mean(axis=1)
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(values.mean()), float(low), float(high)


def piecewise_breakpoint(x: np.ndarray, y: np.ndarray, candidates: np.ndarray) -> float:
    """Breakpoint minimizing squared error of a continuous two-segment linear fit."""
    best, best_sse = candidates[0], np.inf
    for knot in candidates:
        design = np.column_stack([np.ones_like(x), x, np.maximum(x - knot, 0)])
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        sse = float(((design @ coef - y) ** 2).sum())
        if sse < best_sse:
            best, best_sse = knot, sse
    return float(best)


def gap_checks() -> dict:
    panel = pd.read_parquet(DATA / "panel.parquet")
    scores = pd.read_csv(DATA / "panel_scores" / EXP232 / "scores.csv", dtype={"bin": str})
    frame = panel.merge(scores[["stem", "r_precision"]], left_on="entry_id", right_on="stem")
    frame["density"] = frame["contacts"].map(len) / frame["seq_len"]
    frame["long_share"] = [np.mean([j - i >= 24 for i, j in c]) for c in frame["contacts"]]

    removed = frame[frame["set"] == "removed"].dropna(subset=["r_precision"])
    quartile = pd.qcut(removed["identity"], 4)
    by_quartile = removed.groupby(quartile, observed=True)["r_precision"].agg(["size", "mean"])
    design = np.column_stack([
        np.ones(len(removed)), removed["identity"], np.log(removed["seq_len"]),
        removed["global_plddt"], np.log1p(removed["n_relatives"]),
    ])
    identity_coef = np.linalg.lstsq(design, removed["r_precision"].to_numpy(), rcond=None)[0][1]

    wide = frame.pivot(index="pair", columns="set", values=["density", "long_share", "r_precision"]).dropna()
    gap = (wide["r_precision"]["removed"] - wide["r_precision"]["kept"]).to_numpy()
    d_density = (wide["density"]["removed"] - wide["density"]["kept"]).to_numpy()
    d_long = (wide["long_share"]["removed"] - wide["long_share"]["kept"]).to_numpy()
    coef = np.linalg.lstsq(np.column_stack([np.ones(len(gap)), d_density, d_long]), gap, rcond=None)[0]
    return {
        "identity_quartiles": [
            {"identity": str(interval), "n": int(row["size"]), "exp232_removed_r_precision": float(row["mean"])}
            for interval, row in by_quartile.iterrows()
        ],
        "identity_coef_adjusted": float(identity_coef),
        "identity_spearman": float(removed[["identity", "r_precision"]].corr("spearman").iloc[0, 1]),
        "pairs": len(gap),
        "mean_gap": float(gap.mean()),
        "mean_density_difference": float(d_density.mean()),
        "mean_long_share_difference": float(d_long.mean()),
        "gap_after_density_and_long_share": float(coef[0]),
    }


def main() -> int:
    pairs = load()
    rng = np.random.default_rng(SEED)
    dose_rows, did_rows = [], []
    for metric in METRICS:
        for label in BINS:
            group = pairs[pairs["bin"] == label]
            cols = [f"{metric}_{tag}_{subset}" for tag in ("232", "199") for subset in ("removed", "kept")]
            group = group.dropna(subset=cols)
            r232, k232 = group[f"{metric}_232_removed"].to_numpy(), group[f"{metric}_232_kept"].to_numpy()
            r199, k199 = group[f"{metric}_199_removed"].to_numpy(), group[f"{metric}_199_kept"].to_numpy()
            row = {"metric": metric, "bin": label, "n_pairs": len(group)}
            for name, values in (("removed", r232), ("kept", k232), ("gap", r232 - k232)):
                mean, low, high = boot_mean(values, rng)
                row |= {f"exp232_{name}": mean, f"exp232_{name}_lo": low, f"exp232_{name}_hi": high}
            dose_rows.append(row)
            row = {"metric": metric, "bin": label, "n_pairs": len(group)}
            for name, values in (("removed_delta", r199 - r232), ("kept_delta", k199 - k232),
                                 ("did", (r199 - r232) - (k199 - k232))):
                mean, low, high = boot_mean(values, rng)
                row |= {name: mean, f"{name}_lo": low, f"{name}_hi": high}
            did_rows.append(row)

    dose = pd.DataFrame(dose_rows)
    did = pd.DataFrame(did_rows)
    dose.to_csv(DATA / "relative_dose_response.csv", index=False, float_format="%.4f")
    did.to_csv(DATA / "contamination_did.csv", index=False, float_format="%.4f")

    # A3 saturation fit and covariate-adjusted slope on removed structures.
    fits = {}
    for metric in METRICS:
        removed = pairs.dropna(subset=[f"{metric}_232_removed"])
        x = np.log1p(removed["n_relatives"].to_numpy(dtype=float))
        y = removed[f"{metric}_232_removed"].to_numpy()
        candidates = np.unique(np.quantile(x, np.linspace(0.05, 0.95, 37)))
        knot = piecewise_breakpoint(x, y, candidates)
        boot = []
        for _ in range(1_000):
            idx = rng.integers(0, len(x), len(x))
            boot.append(piecewise_breakpoint(x[idx], y[idx], candidates))
        design = np.column_stack([
            np.ones_like(x), x, np.log(removed["seq_len"].to_numpy(dtype=float)), removed["global_plddt"].to_numpy(),
        ])
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        slopes = []
        for _ in range(1_000):
            idx = rng.integers(0, len(x), len(x))
            slopes.append(np.linalg.lstsq(design[idx], y[idx], rcond=None)[0][1])
        fits[metric] = {
            "n": len(x),
            "breakpoint_relatives": float(np.expm1(knot)),
            "breakpoint_ci_relatives": [float(np.expm1(v)) for v in np.percentile(boot, [2.5, 97.5])],
            "adjusted_slope_per_log1p_relative": float(coef[1]),
            "adjusted_slope_ci": [float(v) for v in np.percentile(slopes, [2.5, 97.5])],
            "adjusted_coef_log_length": float(coef[2]),
            "adjusted_coef_plddt": float(coef[3]),
        }
    (DATA / "relative_dose_fit.json").write_text(json.dumps(fits, indent=2) + "\n")
    checks = gap_checks()
    (DATA / "removed_gap_checks.json").write_text(json.dumps(checks, indent=2) + "\n")

    pd.set_option("display.width", 200)
    print(dose.round(3).to_string(index=False))
    print(did.round(3).to_string(index=False))
    print(json.dumps(fits, indent=2))
    print(json.dumps(checks, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
