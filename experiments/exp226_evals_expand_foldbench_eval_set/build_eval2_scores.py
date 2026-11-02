# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Step 8 — the eval2 scoreboard: six predictors over all 307 proteins.

eval2's 284 older proteins were already scored by every predictor; #226 scored
the 23 new ones. This joins the two halves into one per-protein table and
aggregates it.

The 284 half comes from **exp213's `per_protein_wide.csv.gz` verbatim** rather
than being recomputed, for the same reason #226 reused exp199's existing score
matrices: those numbers are the published reference measurement, and re-deriving
them would introduce sampling variance without adding information. The 23 half
is scored through the same metric implementation — exp82's `build_rollout_rows`
and exp78's `contact_eval`, both of which carry exp89's `compute_metrics`
functions verbatim — so the halves are commensurable.

Every aggregate is reported four ways, because eval2 pooled is 75 % de novo
design and that number answers a different question than the one #226 asked:

* **eval2** (307) and **eval2-natural** (78)
* the same at the stricter **<30 %** identity cut (275 / 61)

    uv run python build_eval2_scores.py
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
EXP213_WIDE = (HERE.parent / "exp213_evals_train_sequence_overlap_audit"
               / "data" / "per_protein_wide.csv.gz")

#: exp213's display names — the 23 half is relabelled onto these so the two
#: concatenate into one column space.
PREDICTORS = {
    "esmfold": "ESMFold",
    "esmfold2": "ESMFold2",
    "marinfold-exp199": "MarinFold #199 (1.5B, seq only)",
    "seq-knn-k10": "seq-KNN k=10 (null)",
    "protenix-v2|single_seq": "Protenix-v2 single-seq",
    "protenix-v2|msa": "Protenix-v2 + MSA",
}
MARINFOLD = "MarinFold #199 (1.5B, seq only)"
ORDER = [MARINFOLD, "Protenix-v2 single-seq", "ESMFold", "ESMFold2",
         "Protenix-v2 + MSA", "seq-KNN k=10 (null)"]

#: exp213's wide table keeps only these; the same two ranges and two cuts are
#: what the headline reports.
RANGES = ("all", "long")
CUTS = ("R", "AUC")

#: exp74/exp78 emit two contact readouts per structure predictor: ``structure``
#: (pyconfind on the predicted structure) and ``distogram`` (thresholded
#: distance bins). **exp213's published numbers are the structure ones** —
#: verified: its Protenix single-seq 0.603 / +MSA 0.812 reproduce exactly from
#: ``predictor == "structure"``, while ``distogram`` gives 0.380 / 0.465. Taking
#: the wrong one would silently halve the baselines.
STRUCTURE_READOUT = "structure"


def long_to_wide(frame: pd.DataFrame) -> pd.DataFrame:
    """Long-form metric rows -> exp213's wide (dataset, stem, range, cut) shape.

    Handles both row shapes in play: exp82's rollout rows carry ``model`` and a
    constant ``predictor='lm'``; exp74/exp78's carry ``mode`` + ``predictor``,
    and exp74's Protenix table has no ``model`` column at all.
    """
    frame = frame[frame["range"].isin(RANGES) & frame["cut"].isin(CUTS)].copy()
    if "model" not in frame:
        frame["model"] = "protenix-v2"
    frame["model"] = frame["model"].fillna("protenix-v2")
    # Structure predictors: keep the structure readout only. The LM rows have
    # predictor='lm' and are passed through untouched.
    is_structure_predictor = frame["predictor"].isin([STRUCTURE_READOUT, "distogram"])
    frame = frame[~is_structure_predictor | (frame["predictor"] == STRUCTURE_READOUT)]
    # Protenix is the only predictor with two modes, so its label needs the mode.
    key = frame["model"].where(frame["model"] != "protenix-v2",
                               frame["model"] + "|" + frame["mode"].astype(str))
    frame["label"] = key.map(PREDICTORS)
    unknown = sorted(set(key[frame["label"].isna()]))
    if unknown:
        raise SystemExit(f"unmapped predictor labels: {unknown}")
    return frame.pivot_table(index=["dataset", "stem", "range", "cut"],
                             columns="label", values="precision").reset_index()


def load_new_half(paths: dict[str, Path]) -> pd.DataFrame:
    frames = []
    for name, path in paths.items():
        if not path.exists():
            raise SystemExit(f"missing {name} rows at {path}")
        frames.append(pd.read_csv(path))
    return long_to_wide(pd.concat(frames, ignore_index=True))


#: exp213's published full-554 R-precision (all ranges), from its README's
#: headline table. The 284 half of eval2 is these rows verbatim, so if the full
#: table does not reproduce these the file being read is not the one #213
#: published and nothing downstream is comparable.
EXP213_PUBLISHED_R_ALL = {
    MARINFOLD: 0.611,
    "Protenix-v2 single-seq": 0.603,
    "ESMFold": 0.755,
    "ESMFold2": 0.786,
    "Protenix-v2 + MSA": 0.812,
}


def check_exp213_parity(old: pd.DataFrame, tol: float = 0.001) -> None:
    """The inherited half must reproduce #213's published headline."""
    cell = old[(old["range"] == "all") & (old["cut"] == "R")]
    off = {}
    for predictor, expected in EXP213_PUBLISHED_R_ALL.items():
        got = float(cell[predictor].mean())
        if abs(got - expected) > tol:
            off[predictor] = (round(got, 4), expected)
    if off:
        raise SystemExit(f"exp213's table does not reproduce its published "
                         f"R(all): {off}")
    print(f"[parity] exp213's 554 rows reproduce its published headline "
          f"({len(cell)} units, {len(EXP213_PUBLISHED_R_ALL)} predictors)", flush=True)


#: Bootstrap resamples for the paired MarinFold-minus-baseline intervals.
#: Paired because the subsets are small and every predictor sees the same
#: proteins, so the per-protein difference is far less noisy than the two means.
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 0


def paired_deltas(wide: pd.DataFrame, subsets: dict[str, pd.Series],
                  rng_name: str = "all", cut: str = "R") -> pd.DataFrame:
    """MarinFold minus each baseline, per subset, with a paired bootstrap CI."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    for label, mask in subsets.items():
        cell = wide[mask & (wide["range"] == rng_name) & (wide["cut"] == cut)]
        for predictor in ORDER:
            if predictor == MARINFOLD or predictor not in cell:
                continue
            paired = cell[[MARINFOLD, predictor]].dropna()
            if paired.empty:
                continue
            diff = (paired[MARINFOLD] - paired[predictor]).to_numpy()
            idx = rng.integers(0, len(diff), size=(N_BOOTSTRAP, len(diff)))
            means = diff[idx].mean(axis=1)
            lo, hi = np.percentile(means, [2.5, 97.5])
            rows.append({
                "subset": label, "range": rng_name, "cut": cut,
                "baseline": predictor, "n": len(diff),
                "marinfold": round(float(paired[MARINFOLD].mean()), 4),
                "baseline_mean": round(float(paired[predictor].mean()), 4),
                "delta": round(float(diff.mean()), 4),
                "ci_lo": round(float(lo), 4), "ci_hi": round(float(hi), 4),
                "significant": bool(lo > 0 or hi < 0),
            })
    return pd.DataFrame(rows)


def aggregate(wide: pd.DataFrame, subsets: dict[str, pd.Series]) -> pd.DataFrame:
    """Mean of each predictor over each subset, per (range, cut)."""
    rows = []
    for label, mask in subsets.items():
        subset = wide[mask]
        for rng in RANGES:
            for cut in CUTS:
                cell = subset[(subset["range"] == rng) & (subset["cut"] == cut)]
                entry = {"subset": label,
                         "n": len(cell[["dataset", "stem"]].drop_duplicates()),
                         "range": rng, "cut": cut}
                for predictor in ORDER:
                    if predictor in cell:
                        entry[predictor] = round(float(cell[predictor].mean()), 4)
                rows.append(entry)
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp213-wide", type=Path, default=EXP213_WIDE)
    ap.add_argument("--eval2", type=Path, default=DATA / "eval2_manifest.csv")
    ap.add_argument("--exp199-rows", type=Path,
                    default=Path("/data/exp226_gt/exp199_rows_new23.csv.gz"))
    ap.add_argument("--knn-rows", type=Path,
                    default=Path("/data/exp226_gt/knn_rows_new23.csv.gz"))
    ap.add_argument("--esm-rows", type=Path,
                    default=Path("/data/exp226_gt/esm_scores/contact_precision.csv"))
    ap.add_argument("--protenix-rows", type=Path,
                    default=Path("/data/exp226_gt/protenix_scores/contact_precision.csv"))
    ap.add_argument("--out-per-protein", type=Path, default=DATA / "eval2_per_protein.csv.gz")
    ap.add_argument("--out-headline", type=Path, default=DATA / "eval2_headline.csv")
    ap.add_argument("--out-deltas", type=Path, default=DATA / "eval2_paired_deltas.csv")
    args = ap.parse_args()

    old = pd.read_csv(args.exp213_wide)
    check_exp213_parity(old)
    new = load_new_half({
        "exp199": args.exp199_rows, "seq-knn": args.knn_rows,
        "esmfold/esmfold2": args.esm_rows, "protenix": args.protenix_rows,
    })
    missing = [p for p in ORDER if p not in new.columns]
    if missing:
        raise SystemExit(f"the 23 new proteins are missing predictors: {missing}")

    keep = ["dataset", "stem", "range", "cut"] + ORDER
    wide = pd.concat([old[keep], new[keep]], ignore_index=True)

    eval2 = pd.read_csv(args.eval2)
    wide = wide.merge(
        eval2[["dataset", "stem", "best_identity", "passes_30", "designed_any",
               "has_ground_truth", "length"]],
        on=["dataset", "stem"], how="inner")
    # Count (dataset, stem) units, never stems alone: `7ur7_A` and `8ah9_A` each
    # appear in two datasets with *different sequences*, which is why exp213 and
    # exp94 key on the pair. eval2 is 307 units over 305 unique stems.
    n_units = len(wide[wide["range"] == "all"][["dataset", "stem"]].drop_duplicates())
    if n_units != len(eval2):
        missing = (set(map(tuple, eval2[["dataset", "stem"]].values))
                   - set(map(tuple, wide[["dataset", "stem"]].values)))
        raise SystemExit(f"joined {n_units} of eval2's {len(eval2)} units; "
                         f"unscored: {sorted(missing)[:5]}")

    natural = wide["designed_any"] == 0
    strict = wide["passes_30"] == 1
    subsets = {
        "eval2 (<40% id)": pd.Series(True, index=wide.index),
        "eval2 natural": natural,
        "eval2 (<30% id)": strict,
        "eval2 natural (<30%)": natural & strict,
        "the 23 net-new": wide["dataset"] == "foldbench_rest",
        "the 284 pre-existing": wide["dataset"] != "foldbench_rest",
    }
    headline = aggregate(wide, subsets)

    deltas = paired_deltas(wide, {k: v for k, v in subsets.items()})

    args.out_per_protein.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(args.out_per_protein, index=False)
    headline.to_csv(args.out_headline, index=False)
    deltas.to_csv(args.out_deltas, index=False)
    print(f"[eval2] {n_units} proteins x {len(ORDER)} predictors "
          f"-> {args.out_per_protein}", flush=True)

    for _, row in headline[(headline["cut"] == "R")].iterrows():
        cells = "  ".join(f"{p.split(' (')[0][:22]}={row[p]:.4f}"
                          for p in ORDER if p in row and pd.notna(row[p]))
        print(f"[R {row['range']:>4}] {row['subset']:<22} n={row['n']:<4} {cells}",
              flush=True)
    print(flush=True)
    for _, row in deltas.iterrows():
        mark = "*" if row["significant"] else " "
        print(f"[delta {mark}] {row['subset']:<22} vs {row['baseline']:<32} "
              f"{row['delta']:+.4f} [{row['ci_lo']:+.4f}, {row['ci_hi']:+.4f}] n={row['n']}",
              flush=True)
    print(f"[eval2] headline -> {args.out_headline}", flush=True)
    print(f"[eval2] paired deltas -> {args.out_deltas}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
