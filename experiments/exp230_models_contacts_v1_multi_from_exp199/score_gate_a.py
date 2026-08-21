# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0
"""Gate A's reducer: vote parquets -> R-precision -> the paired base/fine-tune verdict.

``_exp82_score_rollout_worker.py`` emits sparse upper-triangle vote triplets
(``dataset``/``stem``/``L``/``i``/``j``/``votes``), not metrics.  Turning those
into R-precision is a separate step, and it is a step where a well-meaning
re-implementation silently makes the number incomparable with #180's frontier
table.  So the metric block below is copied **verbatim** from exp82's
``build_rollout_rows.py``, which itself carries it verbatim from exp89's
``compute_metrics.py`` under a DO-NOT-EDIT banner.  The chain is the point: this
file adds no measurement of its own, it only reduces.

What Gate A actually asks is a **paired** question -- did *this* fine-tune lose
accuracy against *this* base -- so the headline is the per-protein difference and
its confidence interval, not two independently-quoted means.  Both models are
scored in the same run by the same worker (see ``run_gpu_node_eval.sh``), which
is what makes the pairing legitimate.  #209 is the cautionary tale: exp199 reads
0.6103 here against its published 0.5873, and that 0.023 is the eval pipeline,
not the accelerator -- so a fine-tune scored here must never be compared against
a published number scored elsewhere.

    python score_gate_a.py --rprec ~/exp230_data/eval/rprec \\
        --targets ~/exp230_data/eval554_targets.parquet \\
        --base base --finetune finetune --out ~/exp230_data/eval/gate_a
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

# --- verbatim from exp82 build_rollout_rows.py / exp89 compute_metrics.py ---
# --- (DO NOT EDIT -- must stay identical for #180 comparability) -------------
RANGES = {"all": (6, None), "short": (6, 11), "medium": (12, 23), "long": (24, None)}
CUTS = (("L", lambda L, c: L), ("L/2", lambda L, c: max(1, L // 2)),
        ("L/5", lambda L, c: max(1, L // 5)), ("R", lambda L, c: c))
MIN_SEP = 6
# ---------------------------------------------------------------------------

EXPECTED_UNITS = 554


def load_votes(root: str, label: str) -> tuple[dict, dict]:
    """Sparse triplets -> per-unit dense score matrices, with exp82's dupe check."""
    import fsspec

    uri = f"{root.rstrip('/')}/{label}"
    fs = fsspec.core.url_to_fs(uri)[0]
    parts = sorted(fs.glob(f"{uri}/*.parquet"))
    if not parts:
        raise SystemExit(f"no vote parts under {uri}")

    tri: dict = defaultdict(list)
    lengths: dict = {}
    seen_in_part: dict = defaultdict(set)
    for p in parts:
        d = pq.read_table(fs.unstrip_protocol(p)).to_pydict()
        for ds, stem, L, i, j, v in zip(d["dataset"], d["stem"], d["L"],
                                        d["i"], d["j"], d["votes"]):
            key = (ds, stem)
            tri[key].append((int(i), int(j), int(v)))
            lengths[key] = int(L)
            seen_in_part[key].add(p)

    # A protein in two parts means a retried shard double-counted it and the
    # votes would be summed. exp82 fails rather than averages; so do we.
    dupes = {k: sorted(v) for k, v in seen_in_part.items() if len(v) > 1}
    if dupes:
        raise SystemExit(f"!! {len(dupes)} protein(s) in more than one part, e.g. "
                         f"{list(dupes.items())[:2]}")

    mats = {}
    for key, trips in tri.items():
        L = lengths[key]
        M = np.zeros((L, L), np.float32)
        for i, j, v in trips:
            M[i, j] = v
            M[j, i] = v
        # float16 on purpose: that is what fetch_cw_scores.py persists, and the
        # cast decides tie-breaking among equal vote counts.
        mats[key] = M.astype(np.float16)
    return mats, lengths


def load_truth(path: str) -> dict:
    t = pq.read_table(path).to_pylist()
    out = {}
    for r in t:
        gt = {(int(i), int(j)) if int(i) < int(j) else (int(j), int(i))
              for i, j in r["gt_contacts"]}
        out[(r["dataset"], r["stem"])] = dict(
            L=int(r["L"]), gt=gt,
            in_legacy554=bool(r.get("in_legacy554", True)),
            in_eval2=bool(r.get("in_eval2", False)),
            designed_any=bool(r.get("designed_any", False)),
            passes_30=bool(r.get("passes_30", False)))
    return out


def cuts_of(truth: dict) -> dict:
    """The reported slices.

    eval2 POOLED is 75% de novo design -- designs are what survive a homology
    filter -- so exp226's standing rule is to lead with eval2-natural. On exp199
    the two read 0.545 and 0.337, so this is a difference in conclusion, not in
    presentation.
    """
    return {
        "legacy554": {k for k, v in truth.items() if v["in_legacy554"]},
        "eval2": {k for k, v in truth.items() if v["in_eval2"]},
        "eval2_natural": {k for k, v in truth.items()
                          if v["in_eval2"] and not v["designed_any"]},
        "eval2_lt30": {k for k, v in truth.items()
                       if v["in_eval2"] and v["passes_30"]},
    }


def metrics_for(score, gt: set, L: int) -> dict:
    """exp89's metric, restricted to what Gate A quotes."""
    pi, pj = np.triu_indices(L, k=1)
    psep = pj - pi
    tmat = np.zeros((L, L), bool)
    for i, j in gt:
        if 0 <= i < j < L and (j - i) >= MIN_SEP:
            tmat[i, j] = True

    cs, cg = score[pi, pj].astype(np.float32), tmat[pi, pj].astype(int)
    out = {}
    for rng, (lo, hi) in RANGES.items():
        inr = psep >= lo
        if hi is not None:
            inr = inr & (psep <= hi)
        s, g = cs[inr], cg[inr]
        nc, nt = int(s.size), int(g.sum())
        if nc == 0:
            continue
        order = np.argsort(-s, kind="mergesort")     # stable: ties by index
        gs = g[order]
        for cut, fn in CUTS:
            tgt = int(fn(L, nt))
            if tgt <= 0:
                out[f"{rng}:{cut}"] = float("nan")
            else:
                top = min(tgt, nc)
                out[f"{rng}:{cut}"] = float(gs[:top].sum()) / top
        out[f"{rng}:n_true"] = nt
    return out


def paired_report(a: dict, b: dict, key="all:R", n_boot=10000, seed=230,
                  only: set | None = None) -> dict:
    """Per-protein difference (b - a) with a bootstrap CI over the shared units."""
    units = sorted(set(a) & set(b))
    if only is not None:
        units = [u for u in units if u in only]
    if not units:
        return dict(n=0)
    da = np.array([a[u][key] for u in units], float)
    db = np.array([b[u][key] for u in units], float)
    ok = ~(np.isnan(da) | np.isnan(db))
    da, db = da[ok], db[ok]
    diff = db - da
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    boots = diff[idx].mean(axis=1)
    return dict(
        n=int(len(diff)),
        base_mean=float(da.mean()), finetune_mean=float(db.mean()),
        delta_mean=float(diff.mean()),
        delta_ci95=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        delta_median=float(np.median(diff)),
        frac_finetune_better=float((diff > 0).mean()),
        n_worse=int((diff < 0).sum()), n_better=int((diff > 0).sum()),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rprec", required=True, help="dir holding <label>/ vote parts")
    ap.add_argument("--targets", required=True)
    ap.add_argument("--base", default="base")
    ap.add_argument("--finetune", default="finetune")
    ap.add_argument("--out", default=None)
    ap.add_argument("--expect", type=int, default=EXPECTED_UNITS)
    a = ap.parse_args()

    truth = load_truth(a.targets)
    per_label = {}
    for label in (a.base, a.finetune):
        mats, _ = load_votes(a.rprec, label)
        missing = [u for u in mats if u not in truth]
        if missing:
            raise SystemExit(f"{label}: {len(missing)} scored units absent from targets, "
                             f"e.g. {missing[:3]}")
        per_label[label] = {u: metrics_for(m, truth[u]["gt"], truth[u]["L"])
                            for u, m in mats.items()}
        n = len(per_label[label])
        flag = "OK" if n == a.expect else f"INCOMPLETE (expected {a.expect})"
        print(f"[{label}] {n} units {flag}")

    cuts = cuts_of(truth)
    rep = {"per_label_mean": {
        lab: {k: float(np.nanmean([v.get(k, np.nan) for v in d.values()]))
              for k in ("all:R", "all:L", "all:L/5", "long:R", "short:R")}
        for lab, d in per_label.items()}}

    rep["cuts"] = {}
    print(f"\n{'cut':<16}{'n':>5}{'base':>9}{'finetune':>10}{'delta':>9}"
          f"{'  95% CI':>18}   verdict")
    for cut, keys in cuts.items():
        r = paired_report(per_label[a.base], per_label[a.finetune], only=keys)
        if not r.get("n"):
            continue
        r_long = paired_report(per_label[a.base], per_label[a.finetune],
                               key="long:R", only=keys)
        rep["cuts"][cut] = {"all_R": r, "long_R": r_long}
        lo, hi = r["delta_ci95"]
        verdict = ("no significant loss" if lo > -0.005 else
                   "REGRESSION" if hi < 0 else "inconclusive")
        print(f"{cut:<16}{r['n']:>5}{r['base_mean']:>9.4f}{r['finetune_mean']:>10.4f}"
              f"{r['delta_mean']:>+9.4f}   [{lo:+.4f},{hi:+.4f}]   {verdict}")

    # The headline Gate A verdict is the legacy 554, for continuity with every
    # published figure; eval2-natural is the honest low-homology readout.
    p = rep["cuts"].get("legacy554", {}).get("all_R")
    if p:
        lo, hi = p["delta_ci95"]
        verdict = ("NO SIGNIFICANT LOSS" if lo > -0.005 else
                   "REGRESSION" if hi < 0 else "INCONCLUSIVE")
        print(f"\nGATE A (legacy 554): base {p['base_mean']:.4f} -> finetune "
              f"{p['finetune_mean']:.4f}  delta {p['delta_mean']:+.4f} "
              f"[{lo:+.4f},{hi:+.4f}]  => {verdict}")
    print("        #204 noise floor 0.0023 | #209: exp199 reads ~0.6103 here, "
          "never its published 0.5873")
    print("        lead with eval2_natural, NOT eval2 pooled (75% de novo design)")

    if a.out:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "gate_a.json").write_text(json.dumps(rep, indent=2))
        rows = []
        for lab, d in per_label.items():
            for (ds, stem), m in d.items():
                rows.append(dict(label=lab, dataset=ds, stem=stem, **m))
        import pyarrow as pa
        pq.write_table(pa.Table.from_pylist(rows), out / "gate_a_per_protein.parquet")
        print(f"wrote {out}/gate_a.json and gate_a_per_protein.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
