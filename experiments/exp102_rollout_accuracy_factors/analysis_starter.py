# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Starter analysis for MarinFold issue #102 — what differentiates high-accuracy
rollouts from average ones? Runs entirely on **CPU** from the published exp102
parquets (no GPU, no model).

This is a scaffold, not a conclusion: it builds the two workhorse tables and
shows one worked example per question in the issue, with the numbers printed so
you can see the analysis is wired correctly. Extend from here.

The key object is ``contacts_frame`` — one row per (rollout, predicted contact),
carrying everything the order/confidence/amino-acid questions need:

    entry_id r rank  i j  sep band  logprob  correct  rollout_f1 rollout_q  L aa_i aa_j

  * ``rank``     — 0-based emission order within the rollout (0 = predicted first)
  * ``sep``      — |i-j| sequence separation; ``band`` in {short,med,long}
  * ``logprob``  — the contact's emission logprob (higher = more confident)
  * ``correct``  — is (i,j) a ground-truth contact for this target?
  * ``rollout_f1``/``rollout_q`` — the parent rollout's all-band F1 and its
    per-target F1 quartile (Q4 = the high-accuracy rollouts)

Run:  uv run python analysis_starter.py            # local run dir
      uv run python analysis_starter.py --source hf # published bucket
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

BUCKET = ("hf://buckets/open-athena/MarinFold/data/"
          "contacts-v1-rollouts-ordered-exp102")


def load(source: str, run: str, targets: str):
    """Return (metrics_df, targets_df). ``source`` = 'hf' (published bucket) or
    'local' (aggregate the per-target parquet files under ``run``)."""
    if source == "hf":
        return (pd.read_parquet(f"{BUCKET}/rollout_metrics_ordered.parquet"),
                pd.read_parquet(f"{BUCKET}/targets.parquet"))
    mdir = f"{run}/rollout_metrics_ordered"
    parts = []
    for f in sorted(os.listdir(mdir)):
        if not f.endswith(".parquet"):
            continue
        df = pd.read_parquet(f"{mdir}/{f}")
        if "entry_id" not in df.columns:
            df.insert(0, "entry_id", f[: -len(".parquet")])
        parts.append(df)
    return pd.concat(parts, ignore_index=True), pd.read_parquet(targets)


def band_of(sep: int) -> str:
    if sep <= 11:
        return "short"
    if sep <= 23:
        return "med"
    return "long"


def contacts_frame(m: pd.DataFrame, t: pd.DataFrame) -> pd.DataFrame:
    """Explode every rollout into one row per predicted contact (see module doc).

    ~200k rollouts x ~50 contacts ~= 10M rows; a few GB in pandas. Pass a
    subset of ``m`` (e.g. one entry_id, or ``m.sample(...)``) if memory-bound.
    """
    gt = {r.entry_id: {(int(i), int(j)) for i, j in r.gt_contacts} for r in t.itertuples()}
    seq = {r.entry_id: r.sequence for r in t.itertuples()}
    Lmap = {r.entry_id: int(r.L) for r in t.itertuples()}

    # per-target F1 quartile label (Q1..Q4) so "high-accuracy" is defined
    # relative to each target's own rollout distribution.
    m = m.copy()
    m["rollout_q"] = (m.groupby("entry_id")["all_f1"]
                      .transform(lambda s: pd.qcut(s.rank(method="first"), 4,
                                                   labels=["Q1", "Q2", "Q3", "Q4"])))

    rows = []
    for e in m.itertuples():
        pred = e.pred
        lp = e.pred_logprob
        g = gt[e.entry_id]
        s = seq[e.entry_id]
        for rank in range(len(lp)):
            i, j = int(pred[2 * rank]), int(pred[2 * rank + 1])
            sep = abs(i - j)
            rows.append((e.entry_id, e.r, rank, i, j, sep, band_of(sep),
                         float(lp[rank]), (i, j) in g, float(e.all_f1),
                         e.rollout_q, Lmap[e.entry_id],
                         s[i] if i < len(s) else "?", s[j] if j < len(s) else "?"))
    return pd.DataFrame(rows, columns=[
        "entry_id", "r", "rank", "i", "j", "sep", "band", "logprob", "correct",
        "rollout_f1", "rollout_q", "L", "aa_i", "aa_j"])


# ---------------------------------------------------------------------------
# Worked examples — one per question in issue #102. Each returns a small frame.
# ---------------------------------------------------------------------------

def q_length(m: pd.DataFrame) -> pd.Series:
    """Are high-accuracy rollouts just longer? Within-target Spearman of a
    rollout's F1 vs its length (n_gen_tokens) and vs n_pred, averaged over
    targets (so target size doesn't drive the correlation)."""
    def sp(g):
        return pd.Series({
            "rho_f1_vs_ntok": g["all_f1"].corr(g["n_gen_tokens"], method="spearman"),
            "rho_f1_vs_npred": g["all_f1"].corr(g["n_pred"], method="spearman"),
        })
    return m.groupby("entry_id").apply(sp, include_groups=False).mean()


def q_order_bias_by_band(cf: pd.DataFrame) -> pd.DataFrame:
    """Do high-accuracy rollouts front-load a particular separation band?
    Mean normalized emission rank (0=first,1=last) of each band, split by the
    rollout's per-target F1 quartile. Lower = emitted earlier."""
    cf = cf.copy()
    cf["norm_rank"] = cf.groupby(["entry_id", "r"])["rank"].transform(
        lambda s: s / max(len(s) - 1, 1))
    return (cf.pivot_table(index="rollout_q", columns="band", values="norm_rank",
                           aggfunc="mean", observed=True)
            .loc[["Q1", "Q2", "Q3", "Q4"], ["short", "med", "long"]])


def q_confidence_order(cf: pd.DataFrame) -> pd.DataFrame:
    """Is there an emission-order/confidence signature? Within each rollout,
    Spearman(rank, logprob) — negative means confident contacts emitted first —
    averaged per F1 quartile. Also correctness rate of the first-3 vs last-3."""
    def per_rollout(g):
        g = g.sort_values("rank")
        rho = g["rank"].corr(g["logprob"], method="spearman") if len(g) > 3 else np.nan
        first3 = g.head(3)["correct"].mean()
        last3 = g.tail(3)["correct"].mean()
        return pd.Series({"rho_rank_logprob": rho, "acc_first3": first3, "acc_last3": last3})
    per = cf.groupby(["entry_id", "r", "rollout_q"], observed=True).apply(
        per_rollout, include_groups=False).reset_index()
    return per.groupby("rollout_q", observed=True)[
        ["rho_rank_logprob", "acc_first3", "acc_last3"]].mean().loc[["Q1", "Q2", "Q3", "Q4"]]


def q_within_target_informative(cf: pd.DataFrame, entry_id: str) -> pd.DataFrame:
    """Within ONE target, is there a highly-informative contact whose early
    prediction marks the good rollouts? For each GT contact, correlate (across
    the target's rollouts) 'was it predicted, and how early' with rollout F1.
    Returns the top GT contacts by |correlation|."""
    sub = cf[(cf.entry_id == entry_id) & (cf.correct)]
    # earliness = 1 - norm_rank if predicted, else 0 (not predicted this rollout)
    n_roll = cf[cf.entry_id == entry_id][["r"]].drop_duplicates().shape[0]
    recs = []
    f1_by_r = (cf[cf.entry_id == entry_id].drop_duplicates("r").set_index("r")["rollout_f1"])
    for (i, j), g in sub.groupby(["i", "j"]):
        earliness = pd.Series(0.0, index=f1_by_r.index)
        nr = g.groupby("r")["rank"].min()
        maxrank = cf[cf.entry_id == entry_id].groupby("r")["rank"].max()
        earliness.loc[nr.index] = 1 - (nr / maxrank.loc[nr.index]).values
        rho = np.corrcoef(earliness.values, f1_by_r.values)[0, 1]
        recs.append((i, j, int(g["r"].nunique()), g["r"].nunique() / n_roll, rho))
    out = pd.DataFrame(recs, columns=["i", "j", "n_predicted", "hit_rate", "corr_earliness_f1"])
    return out.reindex(out.corr_earliness_f1.abs().sort_values(ascending=False).index).head(10)


def q_aa_composition(cf: pd.DataFrame) -> pd.DataFrame:
    """Do high- vs low-accuracy rollouts predict contacts between different
    amino acids? Enrichment of each residue (either endpoint) in Q4 vs Q1."""
    def comp(qs):
        s = pd.concat([cf.loc[cf.rollout_q == qs, "aa_i"], cf.loc[cf.rollout_q == qs, "aa_j"]])
        return s.value_counts(normalize=True)
    q4, q1 = comp("Q4"), comp("Q1")
    aa = sorted(set(q4.index) | set(q1.index))
    df = pd.DataFrame({"Q4_frac": q4.reindex(aa).fillna(0),
                       "Q1_frac": q1.reindex(aa).fillna(0)})
    df["log2_enrich_Q4"] = np.log2((df.Q4_frac + 1e-6) / (df.Q1_frac + 1e-6))
    return df.sort_values("log2_enrich_Q4", ascending=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["local", "hf"], default="local")
    ap.add_argument("--run", default="data/runs/full")
    ap.add_argument("--targets", default="data/targets.parquet",
                    help="targets.parquet (local source only)")
    ap.add_argument("--max-targets", type=int, default=None,
                    help="use only the first N targets (faster while the run fills in)")
    a = ap.parse_args()

    m, t = load(a.source, a.run, a.targets)
    if a.max_targets:
        keep = sorted(m.entry_id.unique())[: a.max_targets]
        m, t = m[m.entry_id.isin(keep)], t[t.entry_id.isin(keep)]
    print(f"loaded {len(m):,} rollouts across {m.entry_id.nunique()} targets", flush=True)

    print("\n[Q: length] within-target Spearman of rollout F1 vs length "
          "(mean over targets):")
    print(q_length(m).round(3).to_string())

    cf = contacts_frame(m, t)
    print(f"\ncontacts_frame: {len(cf):,} (rollout, contact) rows")

    print("\n[Q: order bias] mean normalized emission rank by band x F1 quartile "
          "(lower = emitted earlier):")
    print(q_order_bias_by_band(cf).round(3).to_string())

    print("\n[Q: confidence/order] per-rollout rank~logprob corr + first-3/last-3 "
          "correctness, by F1 quartile:")
    print(q_confidence_order(cf).round(3).to_string())

    print("\n[Q: amino acids] residue enrichment in Q4 (high-F1) vs Q1 rollouts "
          "(top+bottom 5):")
    aa = q_aa_composition(cf)
    print(pd.concat([aa.head(5), aa.tail(5)]).round(3).to_string())

    ex = m.groupby("entry_id")["all_f1"].std().idxmax()  # target with most variance
    print(f"\n[Q: within-target] most informative GT contacts for {ex} "
          f"(highest-variance target):")
    print(q_within_target_informative(cf, ex).round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
