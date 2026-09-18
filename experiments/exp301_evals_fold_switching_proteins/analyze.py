#!/usr/bin/env python
# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""exp301 analysis — fold preference, bimodality, ΔNLL, and the memorization join.

Reads the worker's three parquet families and produces the tables the README
reports. Nothing here is computed on the global contact map; every fold number
lives on the **discriminative universe** ``A`` (fold1-unique) and ``B``
(fold2-unique) that ``prepare_inputs.py`` built.

Five outputs, in the order they should be read:

1. ``calibration.csv`` — **the gate**. The eval-val proteins that rode along in
   the targets file, scored by this worker, paired against exp277's published
   per-protein R-precision. If this does not land at zero, nothing below means
   anything.
2. ``fold_preference.csv`` — per pair: mean φ with a seed error bar, per-fold
   R-precision and AUC, contacts per rollout, finish rate.
3. ``bimodality.csv`` — is the φ spread more than sampling noise? Tested against
   a binomial null built from the pair's own recalls, because a unimodal model
   spreads φ on its own and that spread is not evidence of two folds.
4. ``delta_nll.csv`` — teacher-forced NLL of both folds under matched
   realizations, per statement.
5. ``memorization.csv`` — φ joined to M3's training-fold label.

    uv run python analyze.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

sys.path.insert(0, str(HERE.parent / "exp89_evals_contacts_v1_model_on_eval_set"))
from compute_metrics import metric_rows, resolved_pairs  # noqa: E402

DEFAULT_SCORES = Path("/data/exp301/scores/exp277")
EXP277_PER_PROTEIN = (
    HERE.parent / "exp277_models_single_mpnn_pilot/data/eval_rollout_v2/figure_per_protein.csv"
)
#: #204 measured four evaluations of one unchanged checkpoint spanning 0.0023,
#: so differences under this are ties, in this experiment as everywhere else.
TIE_THRESHOLD = 0.005


def load_parquets(root: Path, kind: str) -> pd.DataFrame:
    files = sorted((root / kind).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no {kind} parquet under {root / kind}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def load_universe() -> dict[str, dict]:
    return {
        r["pair_id"]: r
        for r in (json.loads(line) for line in (DATA / "foldswitch_universe.jsonl").open())
    }


def load_targets() -> dict[str, dict]:
    return {r["pair_id"]: r for r in pd.read_parquet(DATA / "eval_targets.parquet").to_dict("records")}


# --------------------------------------------------------------------------
# Scoring one unit's vote matrix against one contact set
# --------------------------------------------------------------------------
def score_against(votes: pd.DataFrame, contacts, resolved: np.ndarray, L: int) -> dict:
    """R-precision and AUC of a vote matrix against one fold, exp89's implementation."""
    score = np.zeros((L, L), float)
    score[votes["i"].to_numpy(), votes["j"].to_numpy()] = votes["votes"].to_numpy()
    truth = np.zeros((L, L), bool)
    for i, j in contacts:
        truth[int(i), int(j)] = True
    pi, pj, psep = resolved_pairs(resolved)
    rows = metric_rows(score, truth, pi, pj, psep, L, with_precision=True)
    out = {}
    for row in rows:
        if row["range"] in ("all", "long") and row["cut"] in ("R", "AUC"):
            out[f"{row['cut']}_{row['range']}"] = row["precision"]
    return out


# --------------------------------------------------------------------------
# 1. The calibration gate
# --------------------------------------------------------------------------
def calibration_gate(votes: pd.DataFrame, targets: dict) -> pd.DataFrame:
    published = {
        row["stem"]: float(row["value"])
        for row in csv.DictReader(EXP277_PER_PROTEIN.open())
        if row["predictor"] == "MarinFold exp277" and row["subset"] == "eval-val"
    }
    rows = []
    for pair_id, target in targets.items():
        if target["role"] != "calibration":
            continue
        stem = target["fold1"]
        unit = votes[votes["pair_id"] == pair_id]
        if unit.empty:
            continue
        L = int(target["L"])
        resolved = np.asarray(sorted(int(p) for p in target["common_positions"]))
        # Seeds are independent repeats of the published recipe; average them.
        per_seed = [
            score_against(unit[unit["seed"] == s], target["contacts_fold1"], resolved, L)
            for s in sorted(unit["seed"].unique())
        ]
        ours = float(np.mean([m["R_all"] for m in per_seed]))
        rows.append({
            "stem": stem, "L": L, "ours_R_all": round(ours, 4),
            "published_R_all": published.get(stem),
            "delta": round(ours - published[stem], 4) if stem in published else None,
            "n_seeds": len(per_seed),
            "seed_sd": round(float(np.std([m["R_all"] for m in per_seed], ddof=1)), 4)
            if len(per_seed) > 1 else None,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 2. Fold preference
# --------------------------------------------------------------------------
def fold_preference(rollouts: pd.DataFrame, votes: pd.DataFrame,
                    universe: dict, targets: dict) -> pd.DataFrame:
    rows = []
    fold_switch = rollouts[rollouts["role"] == "foldswitch"]
    for pair_id, unit in fold_switch.groupby("pair_id"):
        record, target = universe[pair_id], targets[pair_id]
        L = int(target["L"])
        resolved = np.asarray(sorted(int(p) for p in target["common_positions"]))
        n_a, n_b = int(unit["n_a"].iloc[0]), int(unit["n_b"].iloc[0])
        n_a_fs, n_b_fs = int(unit["n_a_fs"].iloc[0]), int(unit["n_b_fs"].iloc[0])

        phi = unit["n_hit_a"] / max(n_a, 1) - unit["n_hit_b"] / max(n_b, 1)
        per_seed_phi = unit.groupby("seed").apply(
            lambda g: (g["n_hit_a"] / max(n_a, 1) - g["n_hit_b"] / max(n_b, 1)).mean(),
            include_groups=False)
        # FS-restricted φ: the same quantity on the contacts the fold switch
        # actually moved, which for a long protein is a small minority of the
        # discriminative set.
        phi_fs = (unit["n_hit_a_fs"] / n_a_fs if n_a_fs else 0.0) - \
                 (unit["n_hit_b_fs"] / n_b_fs if n_b_fs else 0.0)

        unit_votes = votes[votes["pair_id"] == pair_id]
        per_seed_scores = {1: [], 2: []}
        for seed in sorted(unit_votes["seed"].unique()):
            sv = unit_votes[unit_votes["seed"] == seed]
            per_seed_scores[1].append(score_against(sv, target["contacts_fold1"], resolved, L))
            per_seed_scores[2].append(score_against(sv, target["contacts_fold2"], resolved, L))

        row = {
            "pair_id": pair_id, "fold1": record["fold1"], "fold2": record["fold2"],
            "tier": record["tier"], "seq_class": record["seq_class"], "L": L,
            "n_a": n_a, "n_b": n_b, "n_a_fs": n_a_fs, "n_b_fs": n_b_fs,
            "jaccard": round(record["jaccard"], 4),
            "recall_a": round(float((unit["n_hit_a"] / max(n_a, 1)).mean()), 4),
            "recall_b": round(float((unit["n_hit_b"] / max(n_b, 1)).mean()), 4),
            "phi": round(float(phi.mean()), 4),
            "phi_seed_sd": round(float(per_seed_phi.std(ddof=1)), 4) if len(per_seed_phi) > 1 else None,
            "phi_fs": round(float(np.mean(phi_fs)), 4) if (n_a_fs and n_b_fs) else None,
            "prefers": "fold1" if phi.mean() > 0 else "fold2",
            "contacts_per_rollout": round(float(unit["n_pred"].mean()), 1),
            "frac_finished": round(float(unit["finished"].mean()), 4),
            "budget_capped": bool(unit["budget_capped"].iloc[0]),
        }
        for fold in (1, 2):
            for key in ("R_all", "R_long", "AUC_all"):
                row[f"fold{fold}_{key}"] = round(
                    float(np.mean([m[key] for m in per_seed_scores[fold]])), 4)
        row["R_all_margin"] = round(row["fold1_R_all"] - row["fold2_R_all"], 4)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("phi").reset_index(drop=True)


# --------------------------------------------------------------------------
# 3. Bimodality, against a binomial null
# --------------------------------------------------------------------------
def bimodality(rollouts: pd.DataFrame) -> pd.DataFrame:
    """Is the per-rollout φ spread more than independent sampling would give?

    A model with one fold in its distribution still produces a spread of φ: each
    rollout draws a different subset of contacts. Under that null the hit counts
    are binomial, so φ has variance ``p_a(1-p_a)/|A| + p_b(1-p_b)/|B|``. Excess
    variance over that is what "the model samples two folds" would look like.
    Reported as a dispersion ratio alongside a 2-vs-1-component mixture BIC.
    """
    from sklearn.mixture import GaussianMixture

    rows = []
    for pair_id, unit in rollouts[rollouts["role"] == "foldswitch"].groupby("pair_id"):
        n_a, n_b = int(unit["n_a"].iloc[0]), int(unit["n_b"].iloc[0])
        if n_a == 0 or n_b == 0:
            continue
        pa = unit["n_hit_a"] / n_a
        pb = unit["n_hit_b"] / n_b
        phi = (pa - pb).to_numpy()
        null_var = (pa.mean() * (1 - pa.mean()) / n_a) + (pb.mean() * (1 - pb.mean()) / n_b)
        observed_var = float(phi.var(ddof=1))
        x = phi.reshape(-1, 1)
        bic1 = GaussianMixture(1, random_state=0).fit(x).bic(x)
        bic2 = GaussianMixture(2, random_state=0, n_init=3).fit(x).bic(x)
        rows.append({
            "pair_id": pair_id, "n_rollouts": len(phi),
            "phi_mean": round(float(phi.mean()), 4),
            "phi_sd": round(float(np.sqrt(observed_var)), 4),
            "null_sd": round(float(np.sqrt(null_var)), 4),
            "dispersion": round(float(observed_var / null_var), 3) if null_var > 0 else None,
            "bic_1": round(bic1, 1), "bic_2": round(bic2, 1),
            "bic_favours_2": bool(bic2 < bic1),
            "frac_phi_negative": round(float((phi < 0).mean()), 4),
        })
    return pd.DataFrame(rows).sort_values("dispersion", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# 4. ΔNLL
# --------------------------------------------------------------------------
def delta_nll(nll: pd.DataFrame) -> pd.DataFrame:
    nll = nll.copy()
    nll["nll_per_token"] = -nll["sum_logprob"] / nll["n_statement_tokens"].clip(lower=1)
    nll["nll_per_statement"] = -nll["sum_logprob"] / nll["n_statements"].clip(lower=1)
    rows = []
    for (pair_id, variant), grp in nll.groupby(["pair_id", "variant"]):
        f1 = grp[grp["fold"] == 1]
        f2 = grp[grp["fold"] == 2]
        if f1.empty or f2.empty:
            continue
        rows.append({
            "pair_id": pair_id, "variant": variant,
            "n_realizations": len(f1),
            "nll_tok_fold1": round(float(f1["nll_per_token"].mean()), 4),
            "nll_tok_fold2": round(float(f2["nll_per_token"].mean()), 4),
            "delta_nll_tok": round(float(f1["nll_per_token"].mean() - f2["nll_per_token"].mean()), 4),
            "nll_stmt_fold1": round(float(f1["nll_per_statement"].mean()), 4),
            "nll_stmt_fold2": round(float(f2["nll_per_statement"].mean()), 4),
            "delta_nll_stmt": round(
                float(f1["nll_per_statement"].mean() - f2["nll_per_statement"].mean()), 4),
            "prefers": "fold1" if f1["nll_per_token"].mean() < f2["nll_per_token"].mean() else "fold2",
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 5. The memorization join
# --------------------------------------------------------------------------
def memorization(preference: pd.DataFrame) -> pd.DataFrame:
    labels_path = DATA / "training_fold_labels.csv"
    if not labels_path.exists():
        return pd.DataFrame()
    labels = pd.read_csv(labels_path)
    keep = ["pair_id", "training_fold", "hit_identity", "hit_arm", "recall_a", "recall_b"]
    merged = preference.merge(
        labels[keep].rename(columns={"recall_a": "train_recall_a", "recall_b": "train_recall_b"}),
        on="pair_id", how="left")
    return merged


# --------------------------------------------------------------------------
# 6. M4 — the conditioning dose-response
# --------------------------------------------------------------------------
def conditioning(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-pair and population dose-response, plus k*.

    ``phi`` here is computed on the REMAINING sets (``A\G``, ``B\G``) exactly as
    the worker recorded them, so the given contacts cannot inflate it. The
    population curve is a mean over pairs of per-pair means, not a mean over
    rollouts, so a pair with many discriminative contacts does not dominate.
    """
    files = sorted(root.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no conditioning parquet under {root}")
    raw = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    raw["phi"] = (raw["n_hit_a_rem"] / raw["n_a_rem"].clip(lower=1)
                  - raw["n_hit_b_rem"] / raw["n_b_rem"].clip(lower=1))
    raw["echo_rate"] = raw["n_given_echoed"] / raw["k"].clip(lower=1)

    # k as a FRACTION of the fold's own discriminative set, not just an absolute
    # count. A dose of 10 is 62% of a pair with |B|=16 and 3% of one with
    # |B|=374, so an absolute-k curve silently mixes two very different asks and
    # will read as a lower k* than any large protein actually achieves.
    size_b = raw[raw["k"] == 0].groupby("pair_id")["n_b_rem"].first()
    raw["k_frac"] = raw["k"] / raw["pair_id"].map(size_b).clip(lower=1)

    per_pair = (raw.groupby(["pair_id", "arm", "k"])
                .agg(phi=("phi", "mean"), n_pred=("n_pred", "mean"),
                     echo_rate=("echo_rate", "mean"), finished=("finished", "mean"),
                     k_frac=("k_frac", "first"), n_b=("n_b_rem", "first"),
                     n_rollouts=("phi", "size"))
                .reset_index())

    rows = []
    for (arm, k), grp in per_pair.groupby(["arm", "k"]):
        # Paired against each pair's own k=0, so the curve is a within-pair
        # change rather than a mix of pairs that entered at different baselines.
        base = per_pair[(per_pair["arm"] == arm) & (per_pair["k"] == 0)][["pair_id", "phi"]]
        merged = grp.merge(base.rename(columns={"phi": "phi_0"}), on="pair_id", how="inner")
        boot = np.random.default_rng(0).choice(
            merged["phi"].to_numpy(), (4000, len(merged))).mean(1)
        rows.append({
            "arm": arm, "k": int(k), "n_pairs": len(merged),
            "phi": round(float(merged["phi"].mean()), 4),
            "phi_lo": round(float(np.percentile(boot, 2.5)), 4),
            "phi_hi": round(float(np.percentile(boot, 97.5)), 4),
            "delta_vs_k0": round(float((merged["phi"] - merged["phi_0"]).mean()), 4),
            "frac_negative": round(float((merged["phi"] < 0).mean()), 4),
            "echo_rate": round(float(merged["echo_rate"].mean()), 4),
            "contacts_per_rollout": round(float(merged["n_pred"].mean()), 1),
            "k_frac_median": round(float(merged["k_frac"].median()), 4),
        })
    curve = pd.DataFrame(rows).sort_values(["arm", "k"]).reset_index(drop=True)
    return per_pair, curve


def k_star(curve: pd.DataFrame) -> int | None:
    """Smallest fold2-seeded dose whose mean fold score is negative."""
    seeded = curve[(curve["arm"] == "seed_b") & (curve["phi"] < 0)].sort_values("k")
    return int(seeded["k"].iloc[0]) if not seeded.empty else None


def k_star_per_pair(per_pair: pd.DataFrame) -> pd.DataFrame:
    """Each pair's own flip point, in contacts and as a fraction of its |B|.

    The population curve answers "what dose moves the average pair"; this
    answers "what did each pair need", which is the quantity that generalises
    across protein size. Pairs that never flip are kept with a null k*, because
    dropping them would bias the summary toward the easy ones.
    """
    rows = []
    for pair_id, grp in per_pair[per_pair["arm"] == "seed_b"].groupby("pair_id"):
        grp = grp.sort_values("k")
        baseline = grp[grp["k"] == 0]["phi"]
        phi_0 = float(baseline.iloc[0]) if len(baseline) else float("nan")
        # A pair already preferring fold2 unconditioned (phi_0 < 0) has nothing
        # to flip. Counting it as "flipped at k*=0" would drag the median k*
        # toward zero with pairs conditioning never had to move.
        needs_flip = phi_0 > 0
        flipped = grp[(grp["k"] > 0) & (grp["phi"] < 0)] if needs_flip else grp.iloc[0:0]
        rows.append({
            "pair_id": pair_id,
            "phi_k0": round(phi_0, 4),
            "n_b": int(grp["n_b"].iloc[0]),
            "needs_flip": needs_flip,
            "k_star": int(flipped["k"].iloc[0]) if not flipped.empty else None,
            "k_star_frac": round(float(flipped["k_frac"].iloc[0]), 4) if not flipped.empty else None,
            "max_k_tested": int(grp["k"].max()),
            "flipped": not flipped.empty,
        })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    ap.add_argument("--conditioning", type=Path,
                    default=Path("/data/exp301/conditioning/exp277"),
                    help="M4 output root; skipped when absent")
    args = ap.parse_args()

    rollouts = load_parquets(args.scores, "rollouts")
    votes = load_parquets(args.scores, "votes")
    universe, targets = load_universe(), load_targets()
    n_pairs = rollouts[rollouts["role"] == "foldswitch"]["pair_id"].nunique()
    print(f"loaded {len(rollouts)} rollout rows over {rollouts['pair_id'].nunique()} units "
          f"({n_pairs} fold-switch pairs)\n")

    cal = calibration_gate(votes, targets)
    if not cal.empty:
        cal.to_csv(DATA / "calibration.csv", index=False)
        paired = cal.dropna(subset=["delta"])
        mean_delta = paired["delta"].mean()
        se = paired["delta"].std(ddof=1) / np.sqrt(len(paired))
        verdict = "PASS" if abs(mean_delta) < TIE_THRESHOLD else "INVESTIGATE"
        print(f"[gate] calibration vs exp277 published, paired over {len(paired)} eval-val proteins")
        print(f"       mean delta {mean_delta:+.4f} +/- {se:.4f} (tie threshold {TIE_THRESHOLD}) -> {verdict}\n")

    pref = fold_preference(rollouts, votes, universe, targets)
    pref.to_csv(DATA / "fold_preference.csv", index=False)
    n1 = int((pref["phi"] > 0).sum())
    print(f"[M1] fold preference over {len(pref)} pairs")
    print(f"     prefers fold1: {n1}/{len(pref)} = {n1 / len(pref):.0%}   "
          f"mean phi {pref['phi'].mean():+.4f}  median {pref['phi'].median():+.4f}")
    print(f"     recall on A {pref['recall_a'].mean():.4f} vs B {pref['recall_b'].mean():.4f}")
    print(f"     R-precision: fold1 {pref['fold1_R_all'].mean():.4f} vs "
          f"fold2 {pref['fold2_R_all'].mean():.4f}  (margin {pref['R_all_margin'].mean():+.4f})")
    by_tier = pref.groupby("tier")["phi"].agg(["count", "mean"])
    print(f"     by tier:\n{by_tier.to_string()}\n")

    bim = bimodality(rollouts)
    if not bim.empty:
        bim.to_csv(DATA / "bimodality.csv", index=False)
        print(f"[M1b] bimodality over {len(bim)} pairs")
        print(f"      dispersion vs binomial null: median {bim['dispersion'].median():.2f}  "
              f">2x: {(bim['dispersion'] > 2).sum()}/{len(bim)}")
        print(f"      BIC favours 2 components: {bim['bic_favours_2'].sum()}/{len(bim)}\n")

    try:
        nll = load_parquets(args.scores, "nll")
    except FileNotFoundError:
        nll = pd.DataFrame()
    if not nll.empty:
        dn = delta_nll(nll)
        dn.to_csv(DATA / "delta_nll.csv", index=False)
        for variant in sorted(dn["variant"].unique()):
            sub = dn[dn["variant"] == variant]
            n1n = int((sub["delta_nll_tok"] < 0).sum())
            print(f"[M2] {variant:8s} n={len(sub)}  mean dNLL/token {sub['delta_nll_tok'].mean():+.4f}  "
                  f"fold1 more likely in {n1n}/{len(sub)} pairs")
        print()

    mem = memorization(pref)
    if not mem.empty and "training_fold" in mem:
        mem.to_csv(DATA / "memorization.csv", index=False)
        decided = mem[mem["training_fold"].isin(["fold1", "fold2"])]
        print(f"[M3] memorization join, {len(decided)} pairs with a decided training fold")
        if not decided.empty:
            agree = (decided["prefers"] == decided["training_fold"]).mean()
            print(f"     model preference matches the training fold in {agree:.0%} of them")
            print(decided.groupby("training_fold")["phi"].agg(["count", "mean"]).to_string())
    cond_root = args.conditioning
    if cond_root and cond_root.exists():
        per_pair, curve = conditioning(cond_root)
        per_pair.to_csv(DATA / "conditioning_per_pair.csv", index=False)
        curve.to_csv(DATA / "conditioning_curve.csv", index=False)
        print("[M4] conditioning dose-response (phi on the REMAINING sets)")
        for arm in ("seed_b", "seed_a"):
            sub = curve[curve["arm"] == arm]
            trail = "  ".join(f"k{int(r.k)}:{r.phi:+.3f}" for r in sub.itertuples())
            print(f"     {arm}: {trail}")
        ks = k_star(curve)
        print(f"     k* (smallest fold2-seeded dose with mean phi < 0): {ks if ks is not None else 'not reached'}")
        ks_pair = k_star_per_pair(per_pair)
        ks_pair.to_csv(DATA / "conditioning_k_star.csv", index=False)
        need = ks_pair[ks_pair["needs_flip"]]
        flipped = need[need["flipped"]]
        already = ks_pair[~ks_pair["needs_flip"]]
        print(f"     pairs already preferring fold2 at k=0 (nothing to flip): {len(already)}")
        print(f"     of the {len(need)} that need flipping, {len(flipped)} do at some tested dose")
        if not flipped.empty:
            print(f"     per-pair k*: median {flipped['k_star'].median():.0f} contacts "
                  f"= {flipped['k_star_frac'].median():.1%} of that pair's |B|")
        for k in sorted(set(curve["k"])):
            b = curve[(curve.arm == "seed_b") & (curve.k == k)]
            a = curve[(curve.arm == "seed_a") & (curve.k == k)]
            if len(a) and len(b):
                print(f"     k={k:2d}  asymmetry (seed_a - seed_b) = "
                      f"{float(a['phi'].iloc[0]) - float(b['phi'].iloc[0]):+.4f}"
                      f"   echo {float(b['echo_rate'].iloc[0]):.2f}")
    print(f"\nwrote tables to {DATA}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
