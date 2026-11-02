# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0
"""Step-1 control: base contacts-v1 E8 model *calibrated* contact-logprob-matrix
R-precision on a length-stratified sample of the exp98 TRAIN proteins.

This measures whether the base model's own one-shot calibrated readout (the
exp82/exp89 canonical symmetrized geo-mean logprob of `<contact> <pi> <pj>`,
ranked) beats a training-free consensus vote over rollouts (prior probe:
all-band R-precision 0.224 @K=16).

Pipeline is the *canonical* one, reused verbatim (nothing reimplemented):
  * prefix + position ids  : exp89 score_eval_set.prefix_and_positions
  * [L,L] score matrix      : exp89 score_eval_set.score_matrix (-> exp82 Scorer)
  * true matrix / pair univ : exp89 compute_metrics.true_matrix / resolved_pairs
  * R-precision + AUC        : exp89 compute_metrics.metric_rows (all + long ranges)

The only difference vs a normal exp89 run is the INPUT set (exp98 train proteins
with precomputed GT contacts) instead of the exp89 eval-set manifests.

    uv run python step1_base_matrix.py --limit 10          # validate
    uv run python step1_base_matrix.py                      # full ~150
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd

REPO = Path("/home/bizon/git/MarinFold/.claude/worktrees/protein-rollout-post-training-e203d8")
EXP82 = REPO / "experiments/exp82_evals_contacts_v1_contact_prediction"
EXP89 = REPO / "experiments/exp89_evals_contacts_v1_model_on_eval_set"
# exp82 first so `score_eval_set`'s `from eval_contact_prediction import ...` resolves.
sys.path.insert(0, str(EXP82))
sys.path.insert(0, str(EXP89))

# --- canonical, reused verbatim ------------------------------------------------
from eval_contact_prediction import Scorer  # noqa: E402  (exp82 HF/CUDA scorer)
from score_eval_set import prefix_and_positions, score_matrix  # noqa: E402  (exp89 helpers)
from compute_metrics import metric_rows, resolved_pairs, true_matrix  # noqa: E402  (exp89 metrics)

DEFAULT_MODEL = "/home/bizon/exp89_export/hf_step35679"  # E8 step-35679 fp32 HF export
DEFAULT_TARGETS = str(
    EXP82.parent / "exp98_data_generate_rollouts_contacts_v1_train/data/targets.parquet"
)
# Fixed length bands for stratified sampling + per-band reporting (L<=512).
BAND_EDGES = [0, 100, 150, 200, 250, 350, 513]


def band_of(L: int) -> str:
    for lo, hi in zip(BAND_EDGES[:-1], BAND_EDGES[1:]):
        if lo < L <= hi:
            return f"{lo+1 if lo else BAND_EDGES[0]}-{hi if hi < 513 else 512}"
    return "other"


def stratified_order(df: pd.DataFrame, n: int, seed: int, max_len: int) -> pd.DataFrame:
    """Proportional-by-length stratified sample of size ~n, ordered so that a
    prefix of the result (``--limit``) still spans the length range."""
    df = df[df.L <= max_len].copy()
    df["band"] = df.L.apply(band_of)
    rng = np.random.default_rng(seed)
    bands = [f"{lo+1 if lo else BAND_EDGES[0]}-{hi if hi < 513 else 512}"
             for lo, hi in zip(BAND_EDGES[:-1], BAND_EDGES[1:])]
    picks: dict[str, list] = {}
    total = len(df)
    for b in bands:
        sub = df[df.band == b]
        if len(sub) == 0:
            picks[b] = []
            continue
        k = max(1, round(n * len(sub) / total))
        idx = rng.choice(sub.index.to_numpy(), size=min(k, len(sub)), replace=False)
        picks[b] = list(idx)
    # round-robin interleave across bands so a prefix spans lengths
    ordered_idx: list = []
    pos = {b: 0 for b in bands}
    while len(ordered_idx) < sum(len(v) for v in picks.values()):
        for b in bands:
            if pos[b] < len(picks[b]):
                ordered_idx.append(picks[b][pos[b]])
                pos[b] += 1
    return df.loc[ordered_idx].reset_index(drop=True)


def extract(rows: list[dict]) -> dict:
    """Pull the all/long R-precision + AUC (and universe sizes) out of metric_rows."""
    d = {(r["range"], r["cut"]): r for r in rows}
    out = {}
    for band in ("all", "long"):
        out[f"{band}_R"] = d[(band, "R")]["precision"]
        out[f"{band}_AUC"] = d[(band, "AUC")]["precision"]
        out[f"{band}_ntrue"] = d[(band, "R")]["n_true"]
        out[f"{band}_ncand"] = d[(band, "R")]["n_candidate"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--targets", default=DEFAULT_TARGETS)
    ap.add_argument("--out", required=True, help="per-protein results CSV")
    ap.add_argument("--n", type=int, default=150, help="stratified sample size")
    ap.add_argument("--limit", type=int, default=None, help="score only first M of the sample")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    df = pd.read_parquet(a.targets)
    sample = stratified_order(df, a.n, a.seed, a.max_len)
    if a.limit:
        sample = sample.iloc[: a.limit]
    print(f"targets={len(df)}  stratified sample={len(sample)} (n={a.n}, seed={a.seed}, "
          f"max_len={a.max_len}); model={a.model}", flush=True)
    print("band counts:", sample.band.value_counts().to_dict(), flush=True)

    done: set[str] = set()
    prior: list[dict] = []
    if a.resume and os.path.exists(a.out):
        prev = pd.read_csv(a.out)
        done = set(prev.entry_id.astype(str))
        prior = prev.to_dict("records")
        print(f"resume: {len(done)} already scored in {a.out}", flush=True)

    scorer = Scorer(a.model)  # exp82 Scorer: bf16 on cuda, batch=16 (canonical default)
    import torch
    print(f"model loaded (dtype={next(scorer.model.parameters()).dtype}, "
          f"device={scorer.device}, vocab={scorer.model.config.vocab_size})", flush=True)

    recs: list[dict] = list(prior)
    t_all = time.time()
    for k, r in sample.iterrows():
        entry = str(r["entry_id"])
        if entry in done:
            continue
        seq = r["sequence"]
        L = int(r["L"])
        built = prefix_and_positions(entry, seq)
        if built is None:
            print(f"  {entry} L={L}: build_document -> None; skipping", flush=True)
            continue
        prefix, seq_positions, Lb = built
        if Lb != L:
            print(f"  {entry}: built L={Lb} != parquet L={L}; using built L", flush=True)
            L = Lb
        t0 = time.time()
        try:
            sym = score_matrix(scorer, prefix, seq_positions)  # [L,L] float32, canonical
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"  {entry} L={L}: CUDA OOM; skipping", flush=True)
            continue
        dt = time.time() - t0
        # GT true-matrix over the same universe metric_rows expects.
        contacts = [(int(a_), int(b_), 1.0) for a_, b_ in r["gt_contacts"]]
        tmat = true_matrix(L, contacts)
        resolved = np.arange(L, dtype=np.int64)  # exp98 seqs fully resolved
        pi, pj, psep = resolved_pairs(resolved)
        rows = metric_rows(sym.astype(np.float64), tmat, pi, pj, psep, L, with_precision=True)
        m = extract(rows)
        rec = dict(entry_id=entry, L=L, band=r["band"], n_gt=int(r["n_gt"]),
                   global_plddt=float(r["global_plddt"]), seconds=round(dt, 3), **m)
        recs.append(rec)
        # incremental save (crash-safe / resumable)
        pd.DataFrame(recs).to_csv(a.out, index=False)
        torch.cuda.empty_cache()
        n_done = len(recs) - len(prior)
        print(f"  [{n_done}] {entry} L={L:>3} band={r['band']:>7} n_gt={int(r['n_gt']):>3}  "
              f"all_R={m['all_R']:.3f} long_R={m['long_R'] if not np.isnan(m['long_R']) else float('nan'):.3f}  "
              f"all_AUC={m['all_AUC']:.3f} long_AUC={m['long_AUC'] if not np.isnan(m['long_AUC']) else float('nan'):.3f}  "
              f"{dt:.1f}s", flush=True)

    res = pd.DataFrame(recs)
    res.to_csv(a.out, index=False)
    print(f"\nwrote {len(res)} rows -> {a.out}  (wall {time.time()-t_all:.0f}s)", flush=True)

    def meanofcol(c):
        v = pd.to_numeric(res[c], errors="coerce")
        return float(np.nanmean(v)), int(v.notna().sum())

    print("\n=== AGGREGATE (mean over proteins; NaN long-band = proteins with 0 long GT) ===")
    for c in ("all_R", "long_R", "all_AUC", "long_AUC"):
        mu, nn = meanofcol(c)
        print(f"  {c:<9} mean={mu:.4f}  (n={nn})")
    print(f"  n_proteins_scored = {len(res)}")

    print("\n=== per length band ===")
    for b, g in res.groupby("band"):
        print(f"  {b:>8}  n={len(g):>3}  "
              f"all_R={np.nanmean(pd.to_numeric(g.all_R)):.3f}  "
              f"long_R={np.nanmean(pd.to_numeric(g.long_R)):.3f}  "
              f"all_AUC={np.nanmean(pd.to_numeric(g.all_AUC)):.3f}  "
              f"long_AUC={np.nanmean(pd.to_numeric(g.long_AUC)):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
