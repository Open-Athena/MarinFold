# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Score the #160 eval: contact accuracy (#89 metrics) + retraction diagnostics.

Consumes what ``score_backtracking_worker.py`` wrote — folded vote triplets and
per-rollout edit lists — and produces the two numbers issue #160 turns on:

1. **Did backtracking cost contact accuracy?** Per-protein precision@{L,L/2,L/5,R}
   and AUC over the #89 candidate universe, for every arm, using metric functions
   copied verbatim from exp89's ``compute_metrics.py``. exp82's own ``metrics()``
   disagrees with exp89's by up to 0.4/protein on small proteins, so anything
   compared across predictors has to go through exp89's implementation.
2. **Is the model's own retraction FP-enriched?** ``retraction_diagnostics``
   applied to each rollout's edit list, aggregated per protein and then pooled.
   ``enrichment = P(FP | retracted) / P(FP)`` is the pass/fail; 1.0 is no signal.

**The universe question.** Truth is only defined on pairs the #89 universe
covers: both residues resolved in the experimental structure and ``|i-j| >= 6``.
A rollout will also emit statements outside that (unresolved residues, near-
diagonal pairs), and those have no ground-truth answer. The **primary**
diagnostic therefore scores in-universe statements only; the unrestricted
variant is reported beside it as a sensitivity check, since the #159 corpus
reference (5.85x) was computed with no restriction at all — its proteins are
predicted structures where every residue is resolved. Caveat that comes with
the restriction: retraction *distance* is then counted in in-universe
statements, so it runs slightly shorter than the corpus's all-statement number.

    uv run python score_eval.py --gt /home/bizon/exp160_eval/gt_universe.jsonl \\
        --scores gs://…/eval/scores --labels exp160-bt50,exp120-base --out-dir data
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score

from retraction_diagnostics import DocDiagnostics, aggregate, diagnose_document, format_report

# --- verbatim from exp89 compute_metrics.py (DO NOT EDIT — must stay identical) ---
RANGES = {"all": (6, None), "short": (6, 11), "medium": (12, 23), "long": (24, None)}
CUTS = (("L", lambda L, c: L), ("L/2", lambda L, c: max(1, L // 2)),
        ("L/5", lambda L, c: max(1, L // 5)), ("R", lambda L, c: c))
MIN_DEG, MIN_SEP = 0.001, 6
STRATA_COLS = ["neff_tier", "fold_verdict", "seq_leakage", "msa_neff", "length"]


def true_matrix(L, contacts):
    m = np.zeros((L, L), bool)
    for i, j, d in contacts:
        i, j = int(i), int(j)
        if d >= MIN_DEG and (j - i) >= MIN_SEP and i < j < L:
            m[i, j] = True
    return m


def resolved_pairs(resolved):
    a, b = np.triu_indices(len(resolved), k=1)
    i, j = resolved[a], resolved[b]
    return i, j, (j - i)


def metric_rows(score, tmat, pi, pj, psep, L, *, with_precision):
    cs, cg = score[pi, pj], tmat[pi, pj].astype(int)
    rows = []
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


def stamp(rows, *, rec, model, mode, predictor):
    strata = rec.get("strata", {}) or {}
    base = dict(dataset=rec["dataset"], stem=rec["stem"], n_residues=rec["L"],
                model=model, mode=mode, predictor=predictor)
    for k in STRATA_COLS:
        base[k] = strata.get(k)
    return [{**base, **r} for r in rows]
# --- end verbatim ---


def list_parts(prefix: str) -> list[str]:
    """Every parquet part under a prefix, as reopenable URIs."""
    import fsspec

    fs, _ = fsspec.core.url_to_fs(prefix)
    try:
        parts = fs.glob(f"{prefix.rstrip('/')}/*.parquet")
    except FileNotFoundError:
        return []
    return sorted(fs.unstrip_protocol(p) for p in parts)


def read_table(uri: str, **kw):
    import fsspec

    with fsspec.open(uri, "rb") as fh:
        return pq.read_table(fh, **kw)


def load_vote_matrices(prefix: str, gt: list[dict]) -> dict[tuple[str, str], np.ndarray]:
    """Rebuild the ``[L, L]`` vote matrices from the sparse triplet parts."""
    dims = {(r["dataset"], r["stem"]): r["L"] for r in gt}
    mats: dict[tuple[str, str], np.ndarray] = {}
    # Which part first supplied each protein. A protein appearing in two parts
    # means two independent sets of rollouts were written for it (a smoke run
    # left behind under the same prefix, or two shardings of the same label),
    # and since cells are *assigned* rather than accumulated the result would be
    # a silent blend of both. Refuse instead.
    origin: dict[tuple[str, str], str] = {}
    parts = list_parts(prefix)
    if not parts:
        raise SystemExit(f"no parquet parts under {prefix}")
    for uri in parts:
        t = read_table(uri).to_pydict()
        for d, s, i, j, v in zip(t["dataset"], t["stem"], t["i"], t["j"], t["votes"]):
            key = (d, s)
            L = dims.get(key)
            if L is None:
                continue                                  # not in the GT universe
            m = mats.get(key)
            if m is None:
                m = mats[key] = np.zeros((L, L), np.int32)
                origin[key] = uri
            elif origin[key] != uri:
                raise SystemExit(
                    f"[score] {key[0]}__{key[1]} appears in two parts:\n"
                    f"  {origin[key]}\n  {uri}\n"
                    "Two rollout sets for one protein cannot be merged cell-wise; "
                    "clear the stale parts and rescore."
                )
            m[i, j] = v
    print(f"[score] {prefix}: {len(parts)} parts -> {len(mats)} proteins")
    return mats


def universe(rec: dict) -> tuple[set[int], frozenset[tuple[int, int]]]:
    """The residues truth is defined on, and the true contact pairs among them."""
    resolved = set(int(x) for x in rec["resolved"])
    tmat = true_matrix(rec["L"], rec["contacts"])
    ii, jj = np.nonzero(tmat)
    return resolved, frozenset((int(a), int(b)) for a, b in zip(ii, jj))


def _sum_diags(diags: list[DocDiagnostics]) -> DocDiagnostics:
    """Fold a protein's per-rollout diagnostics into one, so ``aggregate`` pools proteins.

    Aggregating at protein level (rather than treating all 55k rollouts as
    independent documents) is what makes the bootstrap below a resample over
    proteins, which is the unit that actually varies.
    """
    out = DocDiagnostics()
    for d in diags:
        out.n_statements += d.n_statements
        out.n_contact_statements += d.n_contact_statements
        out.n_retract_statements += d.n_retract_statements
        out.retracted_fp += d.retracted_fp
        out.retracted_tp += d.retracted_tp
        out.kept_fp += d.kept_fp
        out.kept_tp += d.kept_tp
        out.distances.extend(d.distances)
        out.n_recovered += d.n_recovered
        out.n_retracted_pairs += d.n_retracted_pairs
        out.n_retract_absent += d.n_retract_absent
    return out


def diagnose_label(prefix: str, gt: list[dict]) -> dict:
    """Retraction diagnostics for one arm, in-universe (primary) and unrestricted."""
    info = {(r["dataset"], r["stem"]): universe(r) for r in gt}
    per_protein: dict[tuple[str, str], list[DocDiagnostics]] = defaultdict(list)
    per_protein_all: dict[tuple[str, str], list[DocDiagnostics]] = defaultdict(list)
    rollout_rows = []

    parts = list_parts(prefix)
    if not parts:
        raise SystemExit(f"no parquet parts under {prefix}")
    for uri in parts:
        t = read_table(uri).to_pydict()
        for n in range(len(t["stem"])):
            key = (t["dataset"][n], t["stem"][n])
            if key not in info:
                continue
            resolved, gt_pairs = info[key]
            kinds, iis, jjs = t["kind"][n], t["i"][n], t["j"][n]
            statements, in_universe = [], []
            for k, i, j in zip(kinds, iis, jjs):
                a, b = (i, j) if i <= j else (j, i)
                stmt = ("contact" if k == 0 else "retract", a, b)
                statements.append(stmt)
                if a in resolved and b in resolved and (b - a) >= MIN_SEP:
                    in_universe.append(stmt)
            d = diagnose_document(in_universe, gt_pairs)
            per_protein[key].append(d)
            per_protein_all[key].append(diagnose_document(statements, gt_pairs))
            rollout_rows.append(dict(
                dataset=key[0], stem=key[1], L=t["L"][n], rollout=t["rollout"][n],
                finished=t["finished"][n], n_tokens=t["n_tokens"][n],
                n_statements=len(statements), n_in_universe=len(in_universe),
                n_contact=d.n_contact_statements, n_retract=d.n_retract_statements,
                retracted_fp=d.retracted_fp, retracted_tp=d.retracted_tp,
                kept_fp=d.kept_fp, kept_tp=d.kept_tp, n_recovered=d.n_recovered,
                n_unmapped=t["n_unmapped"][n], n_retract_absent=t["n_retract_absent"][n],
                n_reemit=t["n_reemit"][n], n_redundant_contact=t["n_redundant_contact"][n],
            ))

    folded = {k: _sum_diags(v) for k, v in per_protein.items()}
    folded_all = {k: _sum_diags(v) for k, v in per_protein_all.items()}
    return {
        "per_protein": folded,
        "summary": aggregate(list(folded.values())),
        "summary_unrestricted": aggregate(list(folded_all.values())),
        "rollouts": pd.DataFrame(rollout_rows),
    }


def bootstrap_enrichment(per_protein: dict, *, draws: int = 2000, seed: int = 0) -> tuple:
    """Percentile CI for pooled enrichment, resampling **proteins** with replacement.

    Pooled enrichment is a ratio of sums, so the protein-to-protein variation is
    the only thing that makes it uncertain; resampling rollouts within a protein
    would understate it.
    """
    keys = sorted(per_protein)
    rfp = np.array([per_protein[k].retracted_fp for k in keys], float)
    rtp = np.array([per_protein[k].retracted_tp for k in keys], float)
    kfp = np.array([per_protein[k].kept_fp for k in keys], float)
    ktp = np.array([per_protein[k].kept_tp for k in keys], float)

    def enrich(idx):
        a, b, c, d = rfp[idx].sum(), rtp[idx].sum(), kfp[idx].sum(), ktp[idx].sum()
        n_ret, n_emit, n_fp = a + b, a + b + c + d, a + c
        if not (n_ret and n_emit and n_fp):
            return float("nan")
        return (a / n_ret) / (n_fp / n_emit)

    rng = np.random.default_rng(seed)
    all_idx = np.arange(len(keys))
    point = enrich(all_idx)
    boot = np.array([enrich(rng.integers(0, len(keys), len(keys))) for _ in range(draws)])
    boot = boot[np.isfinite(boot)]
    if boot.size == 0:
        return point, float("nan"), float("nan")
    return point, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--scores", required=True, help="prefix holding <label>/{votes,streams}")
    ap.add_argument("--labels", required=True, help="comma-separated arm labels")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--skip-diagnostics", action="store_true")
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="score what is present instead of refusing a partial run")
    a = ap.parse_args()

    gt = [json.loads(line) for line in a.gt.open()]
    labels = [x for x in a.labels.split(",") if x]
    a.out_dir.mkdir(parents=True, exist_ok=True)

    metric_records, diag_records, rollout_frames = [], [], []
    for label in labels:
        mats = load_vote_matrices(f"{a.scores.rstrip('/')}/{label}/votes", gt)
        missing = [f"{r['dataset']}__{r['stem']}" for r in gt
                   if (r["dataset"], r["stem"]) not in mats]
        if missing:
            msg = f"{label}: {len(missing)}/{len(gt)} proteins missing (e.g. {missing[:3]})"
            if not a.allow_incomplete:
                raise SystemExit(f"[score] INCOMPLETE — {msg}; rerun the shard or pass "
                                 "--allow-incomplete")
            print(f"[score] WARNING {msg}")

        for rec in gt:
            key = (rec["dataset"], rec["stem"])
            if key not in mats:
                continue
            score = mats[key].astype(np.float64)
            resolved = np.asarray(rec["resolved"], dtype=np.int64)
            tmat = true_matrix(rec["L"], rec["contacts"])
            pi, pj, psep = resolved_pairs(resolved)
            metric_records += stamp(
                metric_rows(score, tmat, pi, pj, psep, rec["L"], with_precision=True),
                rec=rec, model=label, mode="single_seq", predictor="lm")

        if a.skip_diagnostics:
            continue
        diag = diagnose_label(f"{a.scores.rstrip('/')}/{label}/streams", gt)
        point, lo, hi = bootstrap_enrichment(diag["per_protein"])
        # Compute spent, so the accuracy comparison can be read honestly: the
        # two arms are matched on rollout *count*, and retraction lengthens
        # documents, so the backtracking arm gets somewhat more tokens. A
        # strictly token-matched comparison would give it fewer rollouts.
        roll = diag["rollouts"]
        print(f"\n=== {label}: rollout compute ===")
        print(f"rollouts:             {len(roll):,} "
              f"({100 * (~roll.finished.astype(bool)).mean():.2f}% hit the token budget)")
        print(f"mean tokens/rollout:  {roll.n_tokens.mean():.0f}  "
              f"(total {roll.n_tokens.sum() / 1e6:.1f}M)")
        print(f"mean statements/rollout: {roll.n_statements.mean():.1f} "
              f"({roll.n_in_universe.mean():.1f} in-universe)")
        print(f"\n=== {label}: retraction diagnostics (in-universe) ===")
        print(format_report(diag["summary"]))
        print(f"enrichment 95% CI (protein bootstrap): [{lo:.2f}, {hi:.2f}]  point {point:.2f}")
        # Enrichment is bounded by 1 / FP-base-rate: a model whose emissions are
        # mostly wrong has little headroom above 1.0 no matter how well it
        # discriminates. The #159 corpus reaches 5.85x from a base rate of
        # 0.166 (ceiling 6.02x); rollouts on the experimental eval set are far
        # less precise, so their ceiling is much lower and the raw enrichments
        # are NOT comparable. Fraction-of-ceiling is.
        base = diag["summary"].get("fp_base_rate", float("nan"))
        ceiling = (1.0 / base) if base else float("nan")
        if ceiling == ceiling and point == point:
            print(f"ceiling (1/base rate):{ceiling:.2f}x  -> achieved "
                  f"{100 * (point - 1) / (ceiling - 1):.0f}% of the available headroom "
                  f"(#159 corpus: 5.85x of 6.02x = 97%)")
        print(f"--- {label}: unrestricted (all emitted statements) ---")
        print(format_report(diag["summary_unrestricted"]))
        rec = {"model": label, **diag["summary"],
               "enrichment_ci_low": lo, "enrichment_ci_high": hi,
               **{f"unrestricted_{k}": v for k, v in diag["summary_unrestricted"].items()}}
        diag_records.append(rec)
        frame = diag["rollouts"]
        frame.insert(0, "model", label)
        rollout_frames.append(frame)

    rows = pd.DataFrame(metric_records)
    rows_path = a.out_dir / "exp160_rows.csv.gz"
    rows.to_csv(rows_path, index=False)
    print(f"\n[score] wrote {len(rows)} metric rows -> {rows_path}")

    summary = (rows.groupby(["model", "range", "cut"])["precision"].mean().reset_index()
               .rename(columns={"precision": "mean_precision"}))
    summary.to_csv(a.out_dir / "exp160_summary.csv", index=False)
    print(summary[summary.cut.isin(["R", "AUC", "L"])]
          .pivot_table(index="model", columns=["range", "cut"], values="mean_precision")
          .round(4).to_string())

    if diag_records:
        pd.DataFrame(diag_records).to_csv(a.out_dir / "exp160_retraction.csv", index=False)
        pd.concat(rollout_frames).to_csv(a.out_dir / "exp160_rollouts.csv.gz", index=False)
        print(f"[score] wrote retraction diagnostics -> {a.out_dir / 'exp160_retraction.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
