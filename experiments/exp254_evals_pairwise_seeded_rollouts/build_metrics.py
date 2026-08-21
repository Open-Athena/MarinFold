# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Phase 3 -- score every arm against #245's eval-val ground truth.

Four MarinFold readouts come out of this, all from the same 100 rollouts per
protein per arm:

``<arm> consensus``
    rank every candidate pair by how many of the 100 rollouts emitted it -- the
    exp82 recipe. For the seeded arm this is reported twice: once with the
    pre-filled seed voting for itself (``seeded``), and once with that vote
    removed (``seeded (seed vote removed)``), because the ``+1`` is a property of
    the construction rather than something the model predicted.

``<arm> oracle best-of-100``
    per protein, the single best of the 100 rollouts. Each rollout supplies its
    own short, order-preserving contact list and precision is taken on its first
    R entries; the reported value is the max over rollouts. **Not a deployable
    recipe** -- picking the best rollout needs the ground truth -- so it is a
    headroom diagnostic and is labelled as one, never mixed into the frontier
    (#180's convention).

Plus ``pairwise`` itself, scored from phase 1's dense matrices, for context.

The metric implementation below is copied verbatim from exp89's
``compute_metrics.py`` (as exp82's ``build_oracle_best_rollout.py`` also does).
It must stay identical: a number computed by a re-derived metric is not
comparable to the published ones.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import EXPECTED_UNITS, load_ground_truth, load_targets

# --- verbatim from exp89 compute_metrics.py (DO NOT EDIT -- must stay identical) ---
RANGES: dict[str, tuple[int, int | None]] = {
    "all": (6, None), "short": (6, 11), "medium": (12, 23), "long": (24, None),
}
CUTS = (
    ("L", lambda L, c: L),
    ("L/2", lambda L, c: max(1, L // 2)),
    ("L/5", lambda L, c: max(1, L // 5)),
    ("R", lambda L, c: c),
)
MIN_DEG, MIN_SEP = 0.001, 6


def true_matrix(L: int, contacts) -> np.ndarray:
    m = np.zeros((L, L), bool)
    for i, j, d in contacts:
        i, j = int(i), int(j)
        if d >= MIN_DEG and (j - i) >= MIN_SEP and i < j < L:
            m[i, j] = True
    return m


def resolved_pairs(resolved: np.ndarray):
    a, b = np.triu_indices(len(resolved), k=1)
    i, j = resolved[a], resolved[b]
    return i, j, (j - i)


def metric_rows(score, tmat, pi, pj, psep, L, *, with_precision: bool) -> list[dict]:
    """precision@{L,L/2,L/5,R} (optional) + AUC, per range."""
    cs, cg = score[pi, pj], tmat[pi, pj].astype(int)
    rows: list[dict] = []
    for rng, (lo, hi) in RANGES.items():
        inr = psep >= lo
        if hi is not None:
            inr = inr & (psep <= hi)
        s, g = cs[inr], cg[inr]
        nc, nt = int(s.size), int(g.sum())
        if with_precision:
            order = np.argsort(-s, kind="mergesort") if nc else None
            gs = g[order] if nc else None
            for cut, fn in CUTS:
                tgt = int(fn(L, nt))
                if nc == 0 or tgt <= 0:
                    rows.append(dict(range=rng, cut=cut, precision=float("nan"),
                                     n_candidate=nc, n_true=nt, n_top=0))
                else:
                    top = min(tgt, nc)
                    rows.append(dict(range=rng, cut=cut, precision=float(gs[:top].sum()) / top,
                                     n_candidate=nc, n_true=nt, n_top=top))
        auc = float(roc_auc_score(g, s)) if (nc and 0 < nt < nc) else float("nan")
        rows.append(dict(range=rng, cut="AUC", precision=auc,
                         n_candidate=nc, n_true=nt, n_top=nc))
    return rows
# --- end verbatim ---


def load_detail(arm_dir: Path) -> pd.DataFrame:
    parts = sorted(arm_dir.glob("detail-part-*.parquet"))
    assert parts, f"no detail parts under {arm_dir}"
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    print(f"[metrics] {arm_dir.name}: {len(df):,} rollout-contact rows "
          f"from {len(parts)} parts")
    return df


def vote_matrix(group: pd.DataFrame, L: int) -> np.ndarray:
    """Symmetric ``[L, L]`` per-pair occurrence count over the rollouts.

    The detail rows are already deduplicated within a rollout, so a plain count
    is the same vote a scan of the rollout texts would produce.
    """
    m = np.zeros((L, L), dtype=np.float64)
    np.add.at(m, (group["i"].to_numpy(), group["j"].to_numpy()), 1.0)
    return m + m.T


def rollout_precisions(group: pd.DataFrame, resolved_mask: np.ndarray,
                       tmat: np.ndarray, pi, pj, psep,
                       lo: int, hi: int | None) -> tuple[dict[int, float], int]:
    """R-precision of each individual rollout of one protein, in one range.

    A rollout is its own ranking: the order-preserving list of contacts it
    emitted, filtered to the resolved-residue candidate universe and to the
    range, cut at the number of true contacts R. This is exp82's
    ``build_oracle_best_rollout.py`` definition, factored out so the oracle
    (max over rollouts) and the seed-conditioning analysis (mean, split by
    whether the seed was right) read the same numbers.

    Returns ``({rollout: precision}, n_true)``. An empty mapping with
    ``n_true > 0`` means no rollout put anything scorable in this range.
    """
    cg_all = tmat[pi, pj].astype(int)
    in_range_gt = psep >= lo
    if hi is not None:
        in_range_gt = in_range_gt & (psep <= hi)
    n_true = int(cg_all[in_range_gt].sum())
    if n_true <= 0:
        return {}, n_true

    ii, jj = group["i"].to_numpy(), group["j"].to_numpy()
    sep = np.abs(jj.astype(np.int64) - ii.astype(np.int64))
    in_range = sep >= lo
    if hi is not None:
        in_range = in_range & (sep <= hi)
    keep = resolved_mask[ii] & resolved_mask[jj] & in_range
    if not keep.any():
        return {}, n_true

    sub = pd.DataFrame({"rollout": group["rollout"].to_numpy()[keep],
                        "rank": group["rank"].to_numpy()[keep],
                        "i": ii[keep], "j": jj[keep]})
    out: dict[int, float] = {}
    for k, gk in sub.groupby("rollout", sort=False):
        gk = gk.sort_values("rank")
        top = min(n_true, len(gk))
        if top == 0:
            continue
        out[int(k)] = float(
            tmat[gk["i"].to_numpy()[:top], gk["j"].to_numpy()[:top]].sum()
        ) / top
    return out, n_true


def oracle_best(group: pd.DataFrame, rec: dict, resolved_mask: np.ndarray,
                tmat: np.ndarray, pi, pj, psep) -> list[dict]:
    """Per-range R-precision of the single best rollout for one protein."""
    out: list[dict] = []
    for rng, (lo, hi) in RANGES.items():
        precisions, n_true = rollout_precisions(group, resolved_mask, tmat,
                                                pi, pj, psep, lo, hi)
        if n_true <= 0:
            out.append(dict(range=rng, cut="R", precision=float("nan"),
                            n_true=n_true, best_rollout=-1))
            continue
        if not precisions:
            out.append(dict(range=rng, cut="R", precision=0.0,
                            n_true=n_true, best_rollout=-1))
            continue
        best_k = max(precisions, key=precisions.__getitem__)
        out.append(dict(range=rng, cut="R", precision=precisions[best_k],
                        n_true=n_true, best_rollout=best_k))
    return out


def seed_conditioning(detail: pd.DataFrame, seeds: pd.DataFrame | None, gt,
                      arm: str) -> list[dict]:
    """Every rollout's own R-precision, tagged with whether its seed was right.

    This is the decomposition that explains whichever way the headline goes. A
    seeded rollout inherits one asserted contact; if seeding does anything, a
    rollout given a TRUE contact should beat one given a false one, and the
    i.i.d. arm should sit between them. ``seed_correct`` is ``None`` for the
    i.i.d. arm, whose rollouts were given nothing.
    """
    seed_lookup: dict[tuple[str, str], dict[int, tuple[int, int]]] = {}
    groups = [] if seeds is None else seeds.groupby(["dataset", "stem"], sort=False)
    for (dataset, stem), group in groups:
        seed_lookup[(dataset, stem)] = {
            int(r): (int(i), int(j))
            for r, i, j in zip(group["rank"], group["i"], group["j"])
        }

    rows: list[dict] = []
    for (dataset, stem), group in detail.groupby(["dataset", "stem"], sort=False):
        rec = gt[(dataset, stem)]
        L = rec["L"]
        resolved = np.asarray(rec["resolved"], dtype=np.int64)
        resolved_mask = np.zeros(L, dtype=bool)
        resolved_mask[resolved] = True
        tmat = true_matrix(L, rec["contacts"])
        pi, pj, psep = resolved_pairs(resolved)
        lo, hi = RANGES["all"]
        precisions, n_true = rollout_precisions(group, resolved_mask, tmat,
                                                pi, pj, psep, lo, hi)
        if n_true <= 0:
            continue
        per_rollout_seed = seed_lookup.get((dataset, stem), {})
        for rollout, precision in precisions.items():
            pair = per_rollout_seed.get(rollout)
            seed_correct = None
            if pair is not None:
                i, j = pair
                seed_correct = bool(tmat[i, j]
                                    and resolved_mask[i] and resolved_mask[j])
            rows.append(dict(dataset=dataset, stem=stem, L=L, arm=arm,
                             rollout=rollout, precision=precision,
                             n_true=n_true, seed_correct=seed_correct))
    return rows


def score_consensus(detail: pd.DataFrame, gt, label: str) -> list[dict]:
    rows: list[dict] = []
    for (dataset, stem), group in detail.groupby(["dataset", "stem"], sort=False):
        rec = gt[(dataset, stem)]
        L = rec["L"]
        resolved = np.asarray(rec["resolved"], dtype=np.int64)
        tmat = true_matrix(L, rec["contacts"])
        pi, pj, psep = resolved_pairs(resolved)
        score = vote_matrix(group, L)
        for row in metric_rows(score, tmat, pi, pj, psep, L, with_precision=True):
            rows.append(dict(dataset=dataset, stem=stem, L=L, predictor=label, **row))
    return rows


def score_oracle(detail: pd.DataFrame, gt, label: str) -> list[dict]:
    rows: list[dict] = []
    for (dataset, stem), group in detail.groupby(["dataset", "stem"], sort=False):
        rec = gt[(dataset, stem)]
        L = rec["L"]
        resolved = np.asarray(rec["resolved"], dtype=np.int64)
        resolved_mask = np.zeros(L, dtype=bool)
        resolved_mask[resolved] = True
        tmat = true_matrix(L, rec["contacts"])
        pi, pj, psep = resolved_pairs(resolved)
        for row in oracle_best(group, rec, resolved_mask, tmat, pi, pj, psep):
            rows.append(dict(dataset=dataset, stem=stem, L=L, predictor=label, **row))
    return rows


def score_matrices(matrix_dir: Path, gt, targets, label: str) -> list[dict]:
    rows: list[dict] = []
    for target in targets:
        path = matrix_dir / f"{target.key}.npz"
        assert path.exists(), f"missing pairwise matrix {path}"
        rec = gt[(target.dataset, target.stem)]
        L = rec["L"]
        score = np.load(path)["score"].astype(np.float64)
        assert score.shape == (L, L), f"{target.stem}: {score.shape} != ({L}, {L})"
        tmat = true_matrix(L, rec["contacts"])
        pi, pj, psep = resolved_pairs(np.asarray(rec["resolved"], dtype=np.int64))
        for row in metric_rows(score, tmat, pi, pj, psep, L, with_precision=True):
            rows.append(dict(dataset=target.dataset, stem=target.stem, L=L,
                             predictor=label, **row))
    return rows


def paired_delta(per_protein: pd.DataFrame, a: str, b: str, rng_seed: int = 0,
                 n_boot: int = 10_000) -> list[dict]:
    """Bootstrap CI on the per-protein ``a - b`` difference, per range, cut=R."""
    rng = np.random.default_rng(rng_seed)
    out: list[dict] = []
    for rng_name in ("all", "long"):
        sel = per_protein[(per_protein["cut"] == "R")
                          & (per_protein["range"] == rng_name)]
        wide = sel.pivot_table(index="stem", columns="predictor", values="precision")
        if a not in wide.columns or b not in wide.columns:
            continue
        d = (wide[a] - wide[b]).dropna().to_numpy()
        idx = rng.integers(0, len(d), size=(n_boot, len(d)))
        means = d[idx].mean(axis=1)
        out.append(dict(range=rng_name, a=a, b=b, n=len(d), mean_delta=float(d.mean()),
                        lo=float(np.percentile(means, 2.5)),
                        hi=float(np.percentile(means, 97.5)),
                        frac_a_wins=float((d > 0).mean())))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True,
                    help="run directory holding pairwise/, iid/, seeded/")
    ap.add_argument("--out", type=Path, required=True, help="data/ output directory")
    ap.add_argument("--n-rollouts", type=int, default=100)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    gt = load_ground_truth()
    targets = load_targets()
    assert len(targets) == EXPECTED_UNITS

    rows: list[dict] = []
    rows += score_matrices(args.run / "pairwise", gt, targets, "pairwise")

    iid = load_detail(args.run / "iid")
    seeded = load_detail(args.run / "seeded")
    for name, detail in (("iid", iid), ("seeded", seeded)):
        got = detail.groupby(["dataset", "stem"])["rollout"].nunique()
        assert len(got) == EXPECTED_UNITS, f"{name}: {len(got)} proteins scored"
        assert (got == args.n_rollouts).all(), (
            f"{name}: proteins with != {args.n_rollouts} rollouts: "
            f"{got[got != args.n_rollouts].to_dict()}"
        )

    rows += score_consensus(iid, gt, "iid consensus")
    rows += score_consensus(seeded, gt, "seeded consensus")
    rows += score_consensus(seeded[~seeded["is_seed"]], gt,
                            "seeded consensus (seed vote removed)")
    rows += score_oracle(iid, gt, "iid oracle best-of-N")
    rows += score_oracle(seeded, gt, "seeded oracle best-of-N")

    per_protein = pd.DataFrame(rows)
    dest = args.out / "exp254_per_protein.csv.gz"
    per_protein.to_csv(dest, index=False)
    print(f"[metrics] wrote {len(per_protein):,} rows -> {dest}")

    headline = (
        per_protein[per_protein["cut"].isin(["R", "AUC"])]
        .groupby(["predictor", "range", "cut"])["precision"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "value", "count": "n"})
    )
    headline.to_csv(args.out / "exp254_headline.csv", index=False)
    print(headline[headline["range"].isin(["all", "long"])].to_string(index=False))

    seeds = pd.read_parquet(args.run / "seeds.parquet")
    conditioning = pd.DataFrame(
        seed_conditioning(iid, None, gt, "iid")
        + seed_conditioning(seeded, seeds, gt, "seeded")
    )
    conditioning.to_csv(args.out / "exp254_seed_conditioning.csv.gz", index=False)
    summary = (
        conditioning.groupby(["arm", "seed_correct"], dropna=False)["precision"]
        .agg(["mean", "count"]).reset_index()
    )
    print("\n[metrics] per-rollout R-precision (all ranges), by seed correctness:")
    print(summary.to_string(index=False))
    print(f"[metrics] seeds that are true contacts: "
          f"{conditioning[conditioning.arm == 'seeded'].seed_correct.mean():.4f}")

    comparisons = [
        ("seeded consensus", "iid consensus"),
        ("seeded consensus (seed vote removed)", "iid consensus"),
        ("seeded oracle best-of-N", "iid oracle best-of-N"),
        ("iid oracle best-of-N", "iid consensus"),
        ("iid consensus", "pairwise"),
    ]
    deltas = [d for a, b in comparisons for d in paired_delta(per_protein, a, b)]
    deltas_df = pd.DataFrame(deltas)
    deltas_df.to_csv(args.out / "exp254_paired_deltas.csv", index=False)
    print("\n[metrics] paired per-protein deltas (cut=R):")
    print(deltas_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
