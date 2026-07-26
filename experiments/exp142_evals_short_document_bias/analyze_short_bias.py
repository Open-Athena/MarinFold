# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Assess the "short-document bias": does the tuned contacts-v1 1.5B emit far
fewer contacts / far shorter documents than the ground truth, and does it worsen
with length?

Reads the eval rollout run (per-rollout ``rollout_metrics/*.parquet``) + the
targets, and produces:
  * a per-protein table (L, n_gt, contacts emitted, pred/gt, tokens, finished,
    recall/precision), sorted by L;
  * pooled summaries; and
  * a 4-panel diagnostic figure.

    uv run python analyze_short_bias.py --run data/eval/runs/probe \
        --targets data/eval/targets.parquet --out-prefix data/eval/short_bias
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pyarrow.parquet as pq


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="data/eval/runs/probe")
    ap.add_argument("--targets", default="data/eval/targets.parquet")
    ap.add_argument("--out-prefix", default="data/eval/short_bias")
    a = ap.parse_args()

    tgt = {t["entry_id"]: t for t in pq.read_table(a.targets).to_pylist()}
    files = sorted(glob.glob(f"{a.run}/rollout_metrics/*.parquet"),
                   key=lambda f: tgt[os.path.basename(f)[:-8]]["L"])

    rows = []
    for f in files:
        entry = os.path.basename(f)[:-8]
        t = tgt[entry]
        L, n_gt = int(t["L"]), int(t["n_gt"])
        d = pq.read_table(f).to_pandas()
        npred = d["n_pred"].to_numpy()
        stmts = d["n_contact_stmts"].to_numpy()
        ntok = d["n_gen_tokens"].to_numpy()
        ratio = npred / n_gt
        rows.append(dict(
            entry=entry, dataset=t["dataset"], L=L, n_gt=n_gt,
            n_roll=len(d),
            frac_finished=float(d["finished"].mean()),
            # contacts emitted vs GT
            npred_mean=float(npred.mean()), npred_med=float(np.median(npred)),
            npred_min=int(npred.min()), npred_max=int(npred.max()),
            pred_gt_mean=float(ratio.mean()), pred_gt_med=float(np.median(ratio)),
            frac_lt_half=float((ratio < 0.5).mean()),   # rollouts with <50% of GT count
            frac_ge_80=float((ratio >= 0.8).mean()),
            # duplication: raw statements vs unique valid preds
            dup_ratio=float((stmts / np.maximum(npred, 1)).mean()),
            # document length
            tok_mean=float(ntok.mean()), tok_med=float(np.median(ntok)),
            full_doc_tok=3 * n_gt + 1,                  # tokens a complete GT doc needs
            tok_completeness=float(ntok.mean() / (3 * n_gt + 1)),
            # quality
            all_rec=float(d["all_rec"].mean()), all_prec=float(d["all_prec"].mean()),
            long_rec=float(d["long_rec"].mean()), long_prec=float(d["long_prec"].mean()),
            best_f1=float(d["all_f1"].max()),
            _npred=npred, _ratio=ratio, _ntok=ntok,
        ))

    # ---- per-protein table ----
    print(f"\n{'entry':<22}{'L':>5}{'n_gt':>6}{'pred':>7}{'pred/gt':>9}{'<50%':>7}"
          f"{'tok':>7}{'fin':>6}{'a_rec':>7}{'a_prec':>8}{'l_rec':>7}{'bestF1':>8}")
    print("-" * 105)
    for r in rows:
        print(f"{r['entry']:<22}{r['L']:>5}{r['n_gt']:>6}{r['npred_mean']:>7.1f}"
              f"{r['pred_gt_mean']:>9.2f}{r['frac_lt_half']*100:>6.0f}%{r['tok_mean']:>7.0f}"
              f"{r['frac_finished']*100:>5.0f}%{r['all_rec']:>7.2f}{r['all_prec']:>8.2f}"
              f"{r['long_rec']:>7.2f}{r['best_f1']:>8.2f}")

    # ---- pooled ----
    all_ratio = np.concatenate([r["_ratio"] for r in rows])
    Ls = np.array([r["L"] for r in rows])
    pgm = np.array([r["pred_gt_mean"] for r in rows])
    print("-" * 105)
    print(f"POOLED over {len(all_ratio)} rollouts ({len(rows)} proteins):")
    print(f"  pred/gt  : mean {all_ratio.mean():.2f}  median {np.median(all_ratio):.2f}"
          f"  p10 {np.percentile(all_ratio,10):.2f}  p90 {np.percentile(all_ratio,90):.2f}")
    print(f"  rollouts with <50% of GT contact count: {100*(all_ratio<0.5).mean():.1f}%")
    print(f"  frac_finished (pooled): {100*np.mean([r['frac_finished'] for r in rows]):.1f}%")
    if len(rows) >= 3:
        rho = np.corrcoef(Ls, pgm)[0, 1]
        # simple OLS slope of per-protein pred/gt on L
        slope = np.polyfit(Ls, pgm, 1)[0]
        print(f"  pred/gt vs L: corr {rho:+.2f}, slope {slope:+.2e} per residue "
              f"(short {pgm[Ls.argmin()]:.2f} @L{Ls.min()} -> long {pgm[Ls.argmax()]:.2f} @L{Ls.max()})")

    # ---- figure ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        order = list(range(len(rows)))

        # (1) pred/gt ratio distribution per protein vs L
        ax = axes[0, 0]
        bp = ax.boxplot([rows[i]["_ratio"] for i in order], positions=[rows[i]["L"] for i in order],
                        widths=8, showfliers=False, patch_artist=True, manage_ticks=False)
        for b in bp["boxes"]:
            b.set(facecolor="#4C72B0", alpha=0.5)
        ax.axhline(1.0, color="crimson", ls="--", lw=1.5, label="parity (pred=GT count)")
        ax.axhline(0.5, color="gray", ls=":", lw=1, label="half of GT")
        ax.set_xlabel("protein length L"); ax.set_ylabel("contacts emitted / n_gt  (per rollout)")
        ax.set_title("Contacts emitted vs ground truth"); ax.legend(fontsize=8)
        ax.set_ylim(0, max(1.6, min(3.0, all_ratio.max())))

        # (2) mean contacts emitted vs n_gt (parity)
        ax = axes[0, 1]
        ng = np.array([r["n_gt"] for r in rows]); npm = np.array([r["npred_mean"] for r in rows])
        sc = ax.scatter(ng, npm, c=Ls, cmap="viridis", s=70, zorder=3)
        mx = max(ng.max(), npm.max()) * 1.05
        ax.plot([0, mx], [0, mx], "crimson", ls="--", label="y=x (parity)")
        for r in rows:
            ax.annotate(str(r["L"]), (r["n_gt"], r["npred_mean"]), fontsize=7,
                        xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("n_gt (ground-truth contacts)"); ax.set_ylabel("mean contacts emitted")
        ax.set_title("Mean contacts emitted vs GT count"); ax.legend(fontsize=8)
        plt.colorbar(sc, ax=ax, label="L")

        # (3) recall & precision vs L
        ax = axes[1, 0]
        ax.plot(Ls, [r["all_rec"] for r in rows], "o-", label="recall (all, sep>=6)", color="#4C72B0")
        ax.plot(Ls, [r["long_rec"] for r in rows], "s-", label="recall (long, sep>=24)", color="#55A868")
        ax.plot(Ls, [r["all_prec"] for r in rows], "^--", label="precision (all)", color="#C44E52")
        ax.set_xlabel("protein length L"); ax.set_ylabel("mean per-rollout metric")
        ax.set_title("Rollout accuracy vs length"); ax.legend(fontsize=8); ax.set_ylim(0, None)

        # (4) generated tokens vs full-GT-document tokens
        ax = axes[1, 1]
        fd = np.array([r["full_doc_tok"] for r in rows]); tm = np.array([r["tok_mean"] for r in rows])
        ax.scatter(fd, tm, c=Ls, cmap="viridis", s=70, zorder=3)
        mx = max(fd.max(), tm.max()) * 1.05
        ax.plot([0, mx], [0, mx], "crimson", ls="--", label="y=x (complete doc)")
        ax.set_xlabel("tokens for a complete GT doc (3*n_gt+1)")
        ax.set_ylabel("mean generated tokens")
        ax.set_title("Document length: generated vs complete"); ax.legend(fontsize=8)

        fig.suptitle("contacts-v1 1.5B (step-35679) — short-document-bias probe on 12 eval proteins",
                     fontsize=12)
        fig.tight_layout()
        os.makedirs(os.path.dirname(a.out_prefix) or ".", exist_ok=True)
        fig.savefig(f"{a.out_prefix}.png", dpi=130)
        print(f"\nwrote {a.out_prefix}.png")
    except Exception as e:  # noqa: BLE001
        print(f"(plot skipped: {e!r})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
