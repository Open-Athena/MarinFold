# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Step 3 — the homology-free re-eval: stratify every predictor by train-set identity.

No new inference. Every number here already exists per protein, scored over
#89's single frozen candidate universe by #89's ``compute_metrics``; this
re-aggregates the same rows over the strata :mod:`search_overlap` assigned:

* **MarinFold #199** — ``data/exp199_rollout_rows.csv.gz``, produced by exp82's
  ``score_rollout_worker.py``, the reference scorer per #209/#212. Do **not**
  substitute #199's own published pipeline: it reads 0.023 low on this
  checkpoint, ~10x the 0.0023 replicate span.
* **Protenix-v2 (single-seq / MSA), ESMFold, ESMFold2** — exp89's
  ``contact_precision_all.csv``.
* **seq-KNN** — exp94's ``knn_precision_all.csv``, the copy-the-nearest-training-
  neighbour null. It is the *positive control* for this whole experiment: a
  predictor that works only by homology transfer must collapse on the
  homology-free subset, and if it doesn't, the stratification is wrong.

exp89's table also carries older MarinFold rows scored with the **pairwise**
recipe. Those are deliberately excluded — the same weights read ~0.086 higher
under rollout than under pairwise (#180's trap 1), so pooling recipes would
swamp every effect measured here.

Three outputs under ``data/``: per-stratum means (``strata_metrics.csv``),
paired MarinFold-minus-baseline differences with bootstrap CIs
(``paired_deltas.csv``), and the pre-registered headline subset
(``headline.csv``).

    uv run python stratify_and_compare.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from overlap_lib import STRATUM_NO_HIT, STRATUM_ORDER, STRATUM_REMOTE

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
EXP89 = REPO / "experiments/exp89_evals_contacts_v1_model_on_eval_set/data"
EXP94 = REPO / "experiments/exp94_evals_sequence_knn_baseline/data"

MARINFOLD = "MarinFold #199 (1.5B, seq only)"

#: ``(model, mode, predictor)`` in the source table -> the label used here.
#: Order is the display order; MarinFold first, then the comparators.
#:
#: The third key is not decoration: exp89's table carries Protenix twice, once
#: as ``structure`` (contacts read off the predicted structure — the canonical
#: baseline, all 554) and once as ``distogram`` (its distogram head, only 443).
#: Keying on ``(model, mode)`` alone silently pools them and drops
#: Protenix-single-seq from 0.603 to 0.380.
BASELINES = {
    ("protenix-v2", "single_seq", "structure"): "Protenix-v2 single-seq",
    ("esmfold", "single_seq", "structure"): "ESMFold",
    ("esmfold2", "single_seq", "structure"): "ESMFold2",
    ("protenix-v2", "msa", "structure"): "Protenix-v2 + MSA",
}

#: Sanity floor: the published all-range R-precision of each comparator on the
#: full 554 (exp180 ``data/structure_baselines.csv``, itself computed from
#: exp89's per-protein table). :func:`check_baselines` asserts the numbers this
#: script reconstructs still match, so a future schema change to the source
#: table can't quietly re-pool the rows again.
PUBLISHED_R_ALL = {
    "Protenix-v2 single-seq": 0.6031578401726864,
    "ESMFold": 0.7553286610904633,
    "ESMFold2": 0.7862849202602139,
    "Protenix-v2 + MSA": 0.8118214040198143,
    MARINFOLD: 0.611032,  # #212's replication of exp199 under exp82's worker
}
KNN_MODEL = "seq-knn-k10"
KNN_LABEL = "seq-KNN k=10 (null)"

PREDICTOR_ORDER = [MARINFOLD, *BASELINES.values(), KNN_LABEL]

#: What gets aggregated. R-precision is the headline; AUC is carried because it
#: is prevalence-independent and the strata differ in protein length.
CUTS = [("all", "R"), ("long", "R"), ("all", "AUC"), ("long", "AUC")]

N_BOOTSTRAP = 10_000
SEED = 0


def load_predictors(marinfold_rows: Path, baselines_csv: Path,
                    knn_csv: Path | None) -> pd.DataFrame:
    """One tidy frame: dataset, stem, predictor, range, cut, precision."""
    frames = []

    mf = pd.read_csv(marinfold_rows)
    mf = mf[["dataset", "stem", "range", "cut", "precision"]].copy()
    mf["predictor"] = MARINFOLD
    frames.append(mf)

    base = pd.read_csv(baselines_csv)
    keys = list(zip(base["model"], base["mode"], base["predictor"]))
    base = base[[k in BASELINES for k in keys]].copy()
    base["predictor"] = [BASELINES[k] for k in zip(base["model"], base["mode"],
                                                   base["predictor"])]
    frames.append(base[["dataset", "stem", "range", "cut", "precision", "predictor"]])

    if knn_csv is not None and knn_csv.exists():
        knn = pd.read_csv(knn_csv)
        knn = knn[knn["model"] == KNN_MODEL].copy()
        knn["predictor"] = KNN_LABEL
        frames.append(knn[["dataset", "stem", "range", "cut", "precision", "predictor"]])

    tidy = pd.concat(frames, ignore_index=True)
    tidy = tidy[[(r, c) in CUTS for r, c in zip(tidy["range"], tidy["cut"])]]
    return tidy


def check_baselines(wide: pd.DataFrame, tolerance: float = 5e-4) -> None:
    """Assert the full-554 means still match the published numbers.

    Cheap insurance against the failure mode that actually happened while
    building this: pooling two different Protenix predictor kinds under one
    label, which moved its R-precision by 0.22 without any error.
    """
    full = wide[(wide["range"] == "all") & (wide["cut"] == "R")]
    problems = []
    for predictor, expected in PUBLISHED_R_ALL.items():
        if predictor not in full.columns:
            continue
        got = float(full[predictor].mean())
        if abs(got - expected) > tolerance:
            problems.append(f"  {predictor}: got {got:.6f}, published {expected:.6f}")
    if problems:
        raise SystemExit(
            "full-554 R-precision does not match the published baselines:\n"
            + "\n".join(problems)
            + "\nThe source tables or the predictor selection changed; fix before "
              "trusting any stratified number."
        )
    print(f"[check] all {len(PUBLISHED_R_ALL)} full-554 means match published "
          f"values within {tolerance}")


def wide_table(tidy: pd.DataFrame, identity: pd.DataFrame) -> pd.DataFrame:
    """One row per (dataset, stem, range, cut); one column per predictor + strata."""
    wide = tidy.pivot_table(
        index=["dataset", "stem", "range", "cut"],
        columns="predictor", values="precision", aggfunc="first",
    ).reset_index()
    wide.columns.name = None
    merged = wide.merge(
        identity[["dataset", "stem", "stratum", "designed", "query_len", "n_hits",
                  "n_hits_significant", "best_identity_covered", "fold_verdict",
                  "afdb_n_hits_significant", "esm_atlas_n_hits_significant"]],
        on=["dataset", "stem"], how="left", validate="many_to_one",
    )
    missing = merged["stratum"].isna().sum()
    if missing:
        raise SystemExit(f"{missing} metric rows have no identity-table match")
    return merged


def paired_bootstrap(a: np.ndarray, b: np.ndarray, *, n: int = N_BOOTSTRAP,
                     seed: int = SEED) -> tuple[float, float, float]:
    """Mean of ``a - b`` and its percentile CI, resampling *proteins* together.

    Paired because both predictors scored the same proteins: the between-
    protein variance (which is large — proteins differ hugely in difficulty)
    cancels, and the subsets here are small enough that it would otherwise
    dominate. Pairs with a NaN on either side are dropped first.
    """
    ok = np.isfinite(a) & np.isfinite(b)
    diff = a[ok] - b[ok]
    if diff.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(n, diff.size))
    means = diff[idx].mean(axis=1)
    return float(diff.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks, ties shared — the ranking Spearman's rho is defined on."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    ranks[order] = np.arange(1, values.size + 1, dtype=float)
    sorted_values = values[order]
    start = 0
    for stop in range(1, values.size + 1):
        if stop == values.size or sorted_values[stop] != sorted_values[start]:
            if stop - start > 1:
                ranks[order[start:stop]] = ranks[order[start:stop]].mean()
            start = stop
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    rx, ry = _rankdata(x[ok]), _rankdata(y[ok])
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def identity_slopes(wide: pd.DataFrame, *, n: int = N_BOOTSTRAP,
                    seed: int = SEED) -> pd.DataFrame:
    """Spearman rho between best training identity and accuracy, per predictor.

    The binned view spends most of its statistical power on the thinly
    populated tails. This uses every protein that has a covered training hit
    and asks the same question continuously: *does accuracy track proximity to
    the training set?* — which is exp94's "the robust, confound-resistant
    signal is the slope".

    seq-KNN is the calibration: it copies its nearest neighbour, so its rho
    must be strongly positive. A predictor whose rho is ~0 is not retrieving.
    Reported pooled and on natural proteins only, since designed proteins
    cluster at the low-identity end for reasons unrelated to our training set.
    """
    rng = np.random.default_rng(seed)
    predictors = [p for p in PREDICTOR_ORDER if p in wide.columns]
    rows = []
    for split, subset in (("all", wide),
                          ("natural", wide[wide["designed"] == 0]),
                          ("designed", wide[wide["designed"] == 1])):
        for (range_, cut), group in subset.groupby(["range", "cut"]):
            # Only proteins with a covered hit have a defined identity.
            group = group[group["best_identity_covered"].notna()]
            identity = group["best_identity_covered"].to_numpy(dtype=float)
            if identity.size < 10:
                continue
            for predictor in predictors:
                values = group[predictor].to_numpy(dtype=float)
                rho = spearman(identity, values)
                idx = rng.integers(0, identity.size, size=(n, identity.size))
                boot = np.array([spearman(identity[i], values[i]) for i in idx[:1000]])
                boot = boot[np.isfinite(boot)]
                rows.append({
                    "split": split, "range": range_, "cut": cut,
                    "predictor": predictor, "n": int(identity.size), "spearman_rho": rho,
                    "ci_lo": float(np.percentile(boot, 2.5)) if boot.size else float("nan"),
                    "ci_hi": float(np.percentile(boot, 97.5)) if boot.size else float("nan"),
                })
    return pd.DataFrame(rows)


#: Columns the two aggregators always emit, so an empty subset (e.g. a cut that
#: no protein satisfies) still merges and still reports n=0 rather than
#: vanishing from the table.
MEAN_COLUMNS = ["predictor", "n", "n_valid", "mean", "sem"]
DELTA_COLUMNS = ["comparator", "n_pairs", "delta_marinfold_minus_comparator",
                 "ci_lo", "ci_hi", "significant"]


def interaction_test(wide: pd.DataFrame, homology_free: pd.Series, *,
                     label: str, n: int = N_BOOTSTRAP,
                     seed: int = SEED) -> pd.DataFrame:
    """Does removing training homologs change MarinFold's standing? (difference of differences)

    Reading two subsets' CIs side by side is not a test — the subsets contain
    different proteins, and an apparent shift can be nothing but a change in
    which proteins are averaged. This is the actual claim:

        d_i    = MarinFold_i - baseline_i          (paired, per protein)
        effect = mean(d | homology-free) - mean(d | has a homolog)

    ``d_i`` is a within-protein difference, so protein difficulty cancels
    inside each group; the two group means are then compared across
    independent protein sets by a bootstrap that resamples each group
    separately. A negative effect whose CI clears zero means MarinFold loses
    ground specifically on the proteins with no training relative — the
    leakage signature. An effect indistinguishable from zero means its
    standing does not depend on training proximity.
    """
    rng = np.random.default_rng(seed)
    comparators = [p for p in PREDICTOR_ORDER
                   if p != MARINFOLD and p in wide.columns]
    rows = []
    for split, split_mask in (("all", pd.Series(True, index=wide.index)),
                              ("natural", wide["designed"] == 0),
                              ("designed", wide["designed"] == 1)):
        for (range_, cut), group in wide[split_mask].groupby(["range", "cut"]):
            free = homology_free.reindex(group.index).fillna(False).to_numpy()
            for predictor in comparators:
                diff = (group[MARINFOLD].to_numpy(dtype=float)
                        - group[predictor].to_numpy(dtype=float))
                ok = np.isfinite(diff)
                a, b = diff[ok & free], diff[ok & ~free]
                if a.size < 3 or b.size < 3:
                    continue
                effect = float(a.mean() - b.mean())
                boot = (a[rng.integers(0, a.size, size=(n, a.size))].mean(axis=1)
                        - b[rng.integers(0, b.size, size=(n, b.size))].mean(axis=1))
                lo, hi = np.percentile(boot, [2.5, 97.5])
                rows.append({
                    "subset_definition": label, "split": split,
                    "range": range_, "cut": cut, "comparator": predictor,
                    "n_homology_free": int(a.size), "n_with_homolog": int(b.size),
                    "delta_homology_free": float(a.mean()),
                    "delta_with_homolog": float(b.mean()),
                    "effect": effect, "ci_lo": float(lo), "ci_hi": float(hi),
                    "significant": bool(lo > 0 or hi < 0),
                })
    return pd.DataFrame(rows)


def stratum_means(wide: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    predictors = [p for p in PREDICTOR_ORDER if p in wide.columns]
    if wide.empty:
        return pd.DataFrame(columns=[*group_cols, *MEAN_COLUMNS])
    for keys, group in wide.groupby(group_cols, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        for predictor in predictors:
            values = group[predictor].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            rows.append({
                **dict(zip(group_cols, keys)),
                "predictor": predictor,
                "n": len(group),
                "n_valid": finite.size,
                "mean": finite.mean() if finite.size else float("nan"),
                "sem": (finite.std(ddof=1) / np.sqrt(finite.size))
                       if finite.size > 1 else float("nan"),
            })
    return pd.DataFrame(rows)


def deltas_vs_marinfold(wide: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    comparators = [p for p in PREDICTOR_ORDER
                   if p != MARINFOLD and p in wide.columns]
    if wide.empty:
        return pd.DataFrame(columns=[*group_cols, *DELTA_COLUMNS])
    for keys, group in wide.groupby(group_cols, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        mf = group[MARINFOLD].to_numpy(dtype=float)
        for predictor in comparators:
            other = group[predictor].to_numpy(dtype=float)
            mean, lo, hi = paired_bootstrap(mf, other)
            rows.append({
                **dict(zip(group_cols, keys)),
                "comparator": predictor,
                "n_pairs": int((np.isfinite(mf) & np.isfinite(other)).sum()),
                "delta_marinfold_minus_comparator": mean,
                "ci_lo": lo, "ci_hi": hi,
                "significant": bool(np.isfinite(lo) and (lo > 0 or hi < 0)),
            })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--identity", type=Path, default=HERE / "data/eval_train_identity.csv")
    ap.add_argument("--marinfold", type=Path,
                    default=HERE / "data/exp199_rollout_rows.csv.gz")
    ap.add_argument("--baselines", type=Path, default=EXP89 / "contact_precision_all.csv")
    ap.add_argument("--knn", type=Path, default=EXP94 / "knn_precision_all.csv")
    ap.add_argument("--out-dir", type=Path, default=HERE / "data")
    args = ap.parse_args()

    identity = pd.read_csv(args.identity)
    tidy = load_predictors(args.marinfold, args.baselines, args.knn)
    wide = wide_table(tidy, identity)
    check_baselines(wide)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    stratum_dtype = pd.CategoricalDtype(STRATUM_ORDER, ordered=True)
    wide["stratum"] = wide["stratum"].astype(stratum_dtype)
    wide.to_csv(args.out_dir / "per_protein_wide.csv.gz", index=False)

    # (1) per-stratum means, pooled and split designed/natural.
    pooled = stratum_means(wide, ["range", "cut", "stratum"])
    pooled["split"] = "all"
    by_design = stratum_means(wide, ["range", "cut", "stratum", "designed"])
    by_design["split"] = np.where(by_design.pop("designed") == 1, "designed", "natural")
    overall = stratum_means(wide.assign(stratum="ALL"), ["range", "cut", "stratum"])
    overall["split"] = "all"
    metrics = pd.concat([overall, pooled, by_design], ignore_index=True)
    metrics.to_csv(args.out_dir / "strata_metrics.csv", index=False)

    # (2) paired MarinFold-minus-comparator differences.
    deltas = pd.concat([
        deltas_vs_marinfold(wide.assign(stratum="ALL"), ["range", "cut", "stratum"]),
        deltas_vs_marinfold(wide, ["range", "cut", "stratum"]),
    ], ignore_index=True)
    deltas.to_csv(args.out_dir / "paired_deltas.csv", index=False)

    # (3) the pre-registered headline: no detectable training homolog.
    #     `no_homolog_and_novel_fold` is the strictest cut available: sequence
    #     novelty *and* Foldseek novelty (TM < 0.5 to any AFDB train
    #     representative, #41/#65). Sequence novelty alone does not imply the
    #     fold is new — #94 found most of its no-hit proteins were still
    #     same_fold — and for contact prediction the fold is the channel that
    #     matters. Caveat: no Foldseek DB exists for the ESM-Atlas arm, so this
    #     cut removes AFDB fold redundancy only.
    #     `no_hit_at_all` is the *strict* companion to `no_homolog`, and the
    #     two bracket the one judgement call in this experiment. `no_homolog`
    #     applies the conventional E <= 1e-3 significance line; `no_hit_at_all`
    #     demands that MMseqs2 reported no alignment whatsoever, even the
    #     noise-level ones it emits up to E = 10. Against the AFDB arm alone
    #     that is 269 vs 120 proteins — a big enough gap that the headline must
    #     be shown to survive both, rather than resting on one threshold.
    #     (exp94 counted *any* alignment as a hit, which is why its AFDB-only
    #     "no homolog" bin was 139 rather than 269.)
    headline_mask = wide["stratum"] == STRATUM_NO_HIT
    relaxed_mask = wide["stratum"].isin([STRATUM_NO_HIT, STRATUM_REMOTE])
    subsets = {"no_homolog": headline_mask,
               "no_hit_at_all": wide["n_hits"] == 0,
               "no_or_remote_homolog": relaxed_mask,
               "no_homolog_and_novel_fold": headline_mask
               & (wide["fold_verdict"] == "novel_fold"),
               "all_554": pd.Series(True, index=wide.index)}
    # Every subset is reported pooled *and* split designed/natural. It is not
    # optional here: the homology-free subsets are dominated by de novo designs
    # (they have no homologs anywhere, by construction), and structure
    # predictors find their idealised backbones easy — so a pooled number is
    # mostly a statement about designed proteins.
    headline_rows = []
    for name, mask in subsets.items():
        for split, split_mask in (("all", pd.Series(True, index=wide.index)),
                                  ("natural", wide["designed"] == 0),
                                  ("designed", wide["designed"] == 1)):
            subset = wide[mask & split_mask]
            n_proteins = subset[["dataset", "stem"]].drop_duplicates().shape[0]
            if subset.empty:
                print(f"[headline] {name!r}/{split}: empty — reported with n=0")
                headline_rows.append(pd.DataFrame({
                    "range": "all", "cut": "R", "stratum": name, "split": split,
                    "predictor": [p for p in PREDICTOR_ORDER if p in wide.columns],
                    "n": 0, "n_valid": 0, "mean": np.nan, "sem": np.nan,
                }))
                continue
            if split == "all":
                print(f"[headline] {name!r}: {n_proteins} proteins "
                      f"({int(subset[(subset['range'] == 'all') & (subset['cut'] == 'R')]['designed'].sum())}"
                      f" designed)")
            means = stratum_means(subset.assign(stratum=name),
                                  ["range", "cut", "stratum"])
            deltas_here = deltas_vs_marinfold(subset.assign(stratum=name),
                                              ["range", "cut", "stratum"])
            merged = means.merge(
                deltas_here.rename(columns={"comparator": "predictor"}),
                on=["range", "cut", "stratum", "predictor"], how="left",
            )
            merged["split"] = split
            headline_rows.append(merged)
    headline = pd.concat(headline_rows, ignore_index=True)
    headline = headline.rename(columns={"stratum": "subset"})
    headline.to_csv(args.out_dir / "headline.csv", index=False)

    # (4) the continuous version of the same question, using every protein
    #     with a covered hit rather than just the tails.
    slopes = identity_slopes(wide)
    slopes.to_csv(args.out_dir / "identity_slopes.csv", index=False)

    # (4b) the inferential claim: is the change in MarinFold's standing between
    #      the full set and the homology-free subset bigger than sampling noise?
    #      Run under both homology definitions, since that threshold is the
    #      experiment's one judgement call.
    interactions = pd.concat([
        interaction_test(wide, wide["stratum"] == STRATUM_NO_HIT,
                         label="no_homolog"),
        interaction_test(wide, wide["n_hits"] == 0, label="no_hit_at_all"),
    ], ignore_index=True)
    interactions.to_csv(args.out_dir / "interaction.csv", index=False)

    # (5) counts, for the README and the plots' annotations.
    one_row_per_protein = wide[(wide["range"] == "all") & (wide["cut"] == "R")]
    counts = (one_row_per_protein
              .groupby(["stratum", "dataset"], observed=False).size()
              .unstack(fill_value=0))
    counts["total"] = counts.sum(axis=1)
    counts.to_csv(args.out_dir / "strata_counts.csv")

    # (6) sequence novelty vs *structural* novelty — two different leakage
    #     channels, and a protein can be novel on one and redundant on the other.
    cross = (one_row_per_protein
             .assign(fold_verdict=one_row_per_protein["fold_verdict"]
                     .fillna("unlabelled").replace("", "unlabelled"))
             .groupby(["stratum", "fold_verdict"], observed=False)
             .size().unstack(fill_value=0))
    cross.to_csv(args.out_dir / "sequence_vs_fold_novelty.csv")

    n_units = wide[["dataset", "stem"]].drop_duplicates().shape[0]
    summary = {
        "eval_units": n_units,
        "predictors": [p for p in PREDICTOR_ORDER if p in wide.columns],
        "stratum_counts": counts["total"].to_dict(),
        "marinfold_rows": str(args.marinfold),
        "n_bootstrap": N_BOOTSTRAP,
    }
    (args.out_dir / "stratify_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"eval units: {n_units}")
    print("\nstratum counts:")
    print(counts.to_string())
    print("\nsequence novelty x Foldseek fold novelty (vs AFDB train reps):")
    print(cross.to_string())
    show = headline[(headline["range"] == "all") & (headline["cut"] == "R")
                    & (headline["split"].isin(["all", "natural"]))]
    print("\nR-precision (all ranges), by subset "
          "(split=natural excludes the de novo designs):")
    print(show[["subset", "split", "predictor", "n", "mean",
                "delta_marinfold_minus_comparator", "ci_lo", "ci_hi"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    slope_show = slopes[(slopes["range"] == "all") & (slopes["cut"] == "R")
                        & (slopes["split"].isin(["all", "natural"]))]
    print("\nSpearman rho (R-precision vs best training identity):")
    print(slope_show[["split", "predictor", "n", "spearman_rho", "ci_lo", "ci_hi"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    inter_show = interactions[(interactions["range"] == "all")
                              & (interactions["cut"] == "R")
                              & (interactions["split"].isin(["all", "natural"]))]
    print("\nInteraction: (MarinFold - baseline | homology-free) "
          "- (MarinFold - baseline | has a homolog)")
    print(inter_show[["subset_definition", "split", "comparator", "n_homology_free",
                      "n_with_homolog", "delta_homology_free", "delta_with_homolog",
                      "effect", "ci_lo", "ci_hi"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
