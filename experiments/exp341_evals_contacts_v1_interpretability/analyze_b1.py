# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""B1 layer ablation: R-precision lost when one attention or MLP output is mean-ablated.

Reads ``b1_layer_ablation.py``'s ``scores/*.csv`` and ``timings/*.csv``
from one or more ``--input-dir`` (runs on different protein sets, or shards
of one), concatenates them into ``data/<prefix>_scores.csv`` and
``data/<prefix>_timings.csv``, and writes ``data/<prefix>_deltas.csv``.

Per condition, separation band (all, short, medium, long) and split (all
proteins, each eval set, and each length tercile or, with
``--length-bins``, each fixed length bin) the quantity is

    delta = R-precision(ablated) - R-precision(clean)

on the seed-averaged P(contact) matrix, the production readout, averaged
over the proteins scored under both the condition and clean (runs on
different sets may cover different conditions) and with true contacts in
the band. Ablated and clean see the same proteins and document seeds,
so the comparison is paired.

Two CIs (``N_BOOT``, seeded):

- ``lo``/``hi``: bootstrap over proteins on the seed-averaged delta. This is
  the sampling error from which proteins are in the eval set.
- ``lo_seed``/``hi_seed``: two-level bootstrap, proteins and then the
  4 document seeds within each protein, on per-seed deltas. It adds the
  error from which document realizations were drawn. Its centre is the mean
  per-seed delta, which differs from ``delta`` because R-precision of an
  averaged matrix is not the average of per-seed R-precisions.

No threshold is applied; the deltas are the raw effect sizes.

    uv run python analyze_b1.py --input-dir out/b1
    uv run python analyze_b1.py --input-dir out/b1 out/b1_extra out/b1_panel_* --prefix b1_ext \
        --length-bins 0 150 300 500 1000
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent / "data"
BANDS = ["all", "short", "medium", "long"]
N_BOOT = 10_000
SEED = 341


def load(input_dirs: list[Path], sub: str) -> pd.DataFrame:
    frames = [pd.read_csv(path, dtype={"seed": str})
              for input_dir in input_dirs for path in sorted((input_dir / sub).glob("*.csv"))]
    return pd.concat(frames, ignore_index=True)


def splits(proteins: pd.DataFrame, length_bins: list[int] | None = None) -> dict[str, np.ndarray]:
    """Boolean masks over ``proteins`` (one row per stem) for each split.

    Length splits are terciles of ``proteins`` or, given ``length_bins``
    edges, the bins ``(edge_k, edge_k+1]``.
    """
    out = {"all": np.ones(len(proteins), dtype=bool)}
    for eval_set in sorted(proteins.eval_set.unique()):
        out[eval_set] = (proteins.eval_set == eval_set).to_numpy()
    if length_bins is not None:
        for low, high in zip(length_bins[:-1], length_bins[1:], strict=False):
            out[f"length_{low + 1}-{high}"] = ((proteins.n_residues > low) & (proteins.n_residues <= high)).to_numpy()
        return out
    terciles = pd.qcut(proteins.n_residues, 3, labels=False)
    edges = proteins.groupby(terciles).n_residues.agg(["min", "max"])
    for k, (low, high) in edges.iterrows():
        out[f"length_T{k + 1}_{low}-{high}"] = (terciles == k).to_numpy()
    return out


def bootstrap(per_protein: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    """95% CI of the mean over proteins (axis 0)."""
    idx = rng.integers(0, len(per_protein), size=(N_BOOT, len(per_protein)))
    means = per_protein[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def bootstrap_seeds(per_seed: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    """95% CI of the mean, resampling proteins (axis 0) then seeds (axis 1) within each."""
    n, s = per_seed.shape
    proteins = rng.integers(0, n, size=(N_BOOT, n))
    seeds = rng.integers(0, s, size=(N_BOOT, n, s))
    means = per_seed[proteins[:, :, None], seeds].mean(axis=(1, 2))
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--prefix", default="b1", help="output names data/<prefix>_{scores,timings,deltas}.csv")
    parser.add_argument("--length-bins", type=int, nargs="+", default=None,
                        help="fixed length-bin edges instead of terciles")
    args = parser.parse_args()

    scores = load(args.input_dir, "scores")
    if scores.duplicated(["condition", "stem", "seed"]).any():
        raise ValueError("a condition, protein and seed appear in more than one input")
    scores.to_csv(DATA / f"{args.prefix}_scores.csv", index=False)
    load(args.input_dir, "timings").to_csv(DATA / f"{args.prefix}_timings.csv", index=False)

    mean = scores[scores.seed == "mean"]
    per_seed = scores[scores.seed != "mean"]
    seeds = sorted(per_seed.seed.unique(), key=int)

    def matrix(frame: pd.DataFrame, condition: str, band: str, stems: np.ndarray) -> np.ndarray:
        rows = frame[frame.condition == condition]
        if frame is mean:
            return rows.set_index("stem").loc[stems, f"rp_{band}"].to_numpy()
        return rows.pivot(index="stem", columns="seed", values=f"rp_{band}").loc[stems, seeds].to_numpy()

    rng = np.random.default_rng(SEED)
    rows = []
    for condition in sorted(c for c in mean.condition.unique() if c != "clean"):
        layer, site = int(condition[1:3]), condition.split("_")[1]
        scored = set(mean.stem[mean.condition == condition]) & set(mean.stem[mean.condition == "clean"])
        proteins = (mean[(mean.condition == "clean") & mean.stem.isin(scored)][["stem", "eval_set", "n_residues"]]
                    .sort_values("stem").reset_index(drop=True))
        stems = proteins.stem.to_numpy()
        masks = splits(proteins, args.length_bins)
        for band in BANDS:
            delta = matrix(mean, condition, band, stems) - matrix(mean, "clean", band, stems)
            delta_seed = matrix(per_seed, condition, band, stems) - matrix(per_seed, "clean", band, stems)
            clean = matrix(mean, "clean", band, stems)
            finite = np.isfinite(delta) & np.isfinite(delta_seed).all(axis=1)
            for split, split_mask in masks.items():
                mask = split_mask & finite
                if not mask.any():
                    continue
                lo, hi = bootstrap(delta[mask], rng)
                lo_seed, hi_seed = bootstrap_seeds(delta_seed[mask], rng)
                rows.append({
                    "condition": condition, "layer": layer, "site": site, "band": band, "split": split,
                    "n_proteins": int(mask.sum()), "clean_rp": clean[mask].mean(), "delta": delta[mask].mean(),
                    "lo": lo, "hi": hi, "delta_seed": delta_seed[mask].mean(), "lo_seed": lo_seed, "hi_seed": hi_seed,
                })
    out = pd.DataFrame(rows)
    out.to_csv(DATA / f"{args.prefix}_deltas.csv", index=False, float_format="%.5g")

    table = out[(out.split == "all")].pivot_table(index=["layer"], columns=["site", "band"], values="delta")
    with pd.option_context("display.width", 200, "display.float_format", "{:+.3f}".format):
        print(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
