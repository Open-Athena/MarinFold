#!/usr/bin/env python
# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Step D — figures for exp224.

* ``contact_maps.png`` — the headline. For each arm, one square map with the
  model's vote fraction above the diagonal and the pyconfind ground truth below
  it, so prediction and truth are read against each other without flipping
  between panels. The permutation's cut point is marked on the CP panel.
* ``top_l_overlay.png`` — what a user actually consumes: the top-L predicted
  pairs, coloured by whether they are right, over the greyed-out truth.
* ``cp_in_wt_frame.png`` — the CP prediction re-indexed into WT coordinates and
  laid against the WT prediction. Same fold, same frame: any difference here is
  the permutation.
* ``permutation_contrast.png`` — R-precision and AUC for pairs the permutation
  moved vs pairs it left alone, both arms, with seed error bars.
* ``separation_profile.png`` — accuracy against sequence separation, which is
  the axis the permutation actually rewrites.

    uv run python plot.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_summary import save_plot_with_meta  # noqa: E402

sys.path.insert(0, str(HERE.parent / "exp89_evals_contacts_v1_model_on_eval_set"))
from compute_metrics import MIN_DEG, MIN_SEP, true_matrix  # noqa: E402

DATA, PLOTS = HERE / "data", HERE / "plots"

# dataviz reference palette. Sequential = one hue (blue) light->dark; the second
# sequential context takes the next categorical slot's hue (orange).
BLUE, ORANGE = "#2a78d6", "#eb6834"
BLUE_RAMP = ["#ffffff", "#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
ORANGE_RAMP = ["#ffffff", "#fbdccd", "#f7b997", "#f28f61", "#eb6834", "#b94d22", "#7d3315"]
GOOD, CRITICAL = "#0ca30c", "#d03b3b"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
GT_INK = "#3f3e3b"

CMAP_BLUE = LinearSegmentedColormap.from_list("mf_blue", BLUE_RAMP)
CMAP_ORANGE = LinearSegmentedColormap.from_list("mf_orange", ORANGE_RAMP)

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.6,
    "legend.frameon": False, "figure.facecolor": "white",
})

# Both contact-map panels show the SAME measure (vote fraction) for two
# molecules, so they share one sequential ramp — the panels are small multiples,
# not two different quantities. Splitting them across two hues would make the
# confidence difference between CP and WT, which is a real result, read as a
# palette artefact. The categorical hues below are for the summary/contrast
# charts, where CP and WT are series rather than facets.
ARMS = [("cp_1un2", "CP", ORANGE, CMAP_BLUE), ("wt_1fvk", "WT", BLUE, CMAP_BLUE)]


def load():
    units = json.loads((DATA / "units.json").read_text())
    cpmap = json.loads((DATA / "cp_wt_map.json").read_text())
    return units, cpmap


def mean_score(unit: str, scores_root: Path) -> np.ndarray:
    """Vote fraction averaged over seeds (each seed is 100 rollouts)."""
    mats = [np.load(p / f"{unit}.npz")["score"].astype(np.float64)
            for p in sorted(scores_root.glob("seed*")) if (p / f"{unit}.npz").exists()]
    return np.mean(mats, axis=0) / 100.0


def gt_mask(rec) -> np.ndarray:
    return true_matrix(rec["L"], rec["contacts"]) | true_matrix(rec["L"], rec["contacts"]).T


def resolved_mask(rec) -> np.ndarray:
    m = np.zeros(rec["L"], bool)
    m[np.array(rec["resolved_positions"])] = True
    return m


def _sep_mask(L: int) -> np.ndarray:
    i, j = np.indices((L, L))
    return np.abs(i - j) >= MIN_SEP


def split_map(ax, score, gt, L, *, cmap, title, cut=None, seg_labels=None):
    """Prediction above the diagonal, ground truth below it."""
    ok = _sep_mask(L)
    upper = np.where(np.triu(np.ones((L, L), bool), 1) & ok, score, np.nan)
    lower = np.tril(np.ones((L, L), bool), -1) & ok & gt
    ax.imshow(upper, cmap=cmap, vmin=0, vmax=1, interpolation="nearest", origin="upper")
    yy, xx = np.nonzero(lower)
    ax.scatter(xx, yy, s=0.7, c=GT_INK, marker="s", linewidths=0)
    ax.plot([0, L - 1], [0, L - 1], color=MUTED, lw=0.5, alpha=0.6)
    if cut is not None:
        for c in cut:
            ax.axhline(c - 0.5, color=CRITICAL, lw=0.7, ls="--", alpha=0.85)
            ax.axvline(c - 0.5, color=CRITICAL, lw=0.7, ls="--", alpha=0.85)
    ax.set_xlim(-0.5, L - 0.5)
    ax.set_ylim(L - 0.5, -0.5)
    ax.set_title(title, pad=18 if seg_labels else 6)
    ax.set_xlabel("residue index")
    ax.set_ylabel("residue index")
    ax.tick_params(length=2, labelsize=7)
    if seg_labels:
        # Above the map, under the title — the x-axis is already taken.
        for pos, txt in seg_labels:
            ax.text(pos / L, 1.012, txt, ha="center", va="bottom",
                    fontsize=7, color=CRITICAL, transform=ax.transAxes)


def fig_contact_maps(units, cpmap, scores_root):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 5.0))
    seg_a, seg_b = cpmap["seg_a"], cpmap["seg_b"]
    for ax, (unit, tag, _, cmap) in zip(axes, ARMS):
        rec = units[unit]
        sc, gt = mean_score(unit, scores_root), gt_mask(rec)
        if tag == "CP":
            cut = [seg_a["cp_end"], seg_b["cp_start"], seg_b["cp_end"]]
            labels = [(seg_a["cp_end"] / 2, "WT 100–189"),
                      ((seg_b["cp_start"] + seg_b["cp_end"]) / 2, "WT 1–99")]
        else:
            cut, labels = None, None
        split_map(ax, sc, gt, rec["L"], cmap=cmap,
                  title=f"{tag} — {rec['label']}, L={rec['L']}", cut=cut, seg_labels=labels)
    fig.suptitle("MarinFold contact prediction (upper) vs pyconfind ground truth (lower)",
                 y=1.02, fontsize=11)
    sm = plt.cm.ScalarMappable(cmap=CMAP_BLUE, norm=mpl.colors.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=axes, fraction=0.022, pad=0.02)
    cb.set_label("predicted contact frequency (fraction of rollouts)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    cb.outline.set_visible(False)
    fig.text(0.5, -0.05,
             "One shared colour scale, so the panels are directly comparable: the permutant's "
             "map is fainter because the model is genuinely less confident about it, not "
             "because of the palette.\nRed dashes mark the permutation's cut points.",
             ha="center", fontsize=7.5, color=MUTED)
    save_plot_with_meta(fig, PLOTS / "contact_maps.png", caption=
        "Prediction (upper triangle) vs pyconfind truth (lower), one shared colour scale. The permutant's map is fainter because the model is less confident.")
    plt.close(fig)


def fig_top_l_overlay(units, cpmap, scores_root):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 5.0))
    for ax, (unit, tag, colour, _) in zip(axes, ARMS):
        rec = units[unit]
        L = rec["L"]
        sc, gt = mean_score(unit, scores_root), gt_mask(rec)
        res = resolved_mask(rec)
        i, j = np.triu_indices(L, k=MIN_SEP)
        cand = res[i] & res[j]
        i, j = i[cand], j[cand]
        order = np.argsort(-sc[i, j], kind="mergesort")[:L]
        ti, tj = i[order], j[order]
        hit = gt[ti, tj]
        gi, gj = np.nonzero(np.triu(gt, MIN_SEP))
        for a, b in ((gi, gj), (gj, gi)):
            ax.scatter(b, a, s=1.6, c=GRID, marker="s", linewidths=0)
        for a, b, m, c, lbl in ((ti, tj, hit, GOOD, "correct"),
                                (ti, tj, ~hit, CRITICAL, "wrong")):
            ax.scatter(b[m], a[m], s=3.2, c=c, marker="s", linewidths=0, label=lbl)
            ax.scatter(a[m], b[m], s=3.2, c=c, marker="s", linewidths=0)
        ax.plot([0, L - 1], [0, L - 1], color=MUTED, lw=0.5, alpha=0.6)
        prec = hit.mean()
        ax.set_xlim(-0.5, L - 0.5)
        ax.set_ylim(L - 0.5, -0.5)
        ax.set_title(f"{tag} — top-{L} predictions · precision {prec:.3f}", pad=6)
        ax.set_xlabel("residue index")
        ax.set_ylabel("residue index")
        ax.tick_params(length=2, labelsize=7)
    handles = [Patch(facecolor=GRID, label="true contact (all)"),
               Patch(facecolor=GOOD, label="predicted, correct"),
               Patch(facecolor=CRITICAL, label="predicted, wrong")]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04),
               fontsize=8)
    fig.suptitle("Top-L predicted contacts against the truth", y=1.0, fontsize=11)
    save_plot_with_meta(fig, PLOTS / "top_l_overlay.png", caption=
        "The top-L predicted pairs, coloured by correctness, over the greyed-out truth. Precision 0.47 (CP) vs 0.57 (WT).")
    plt.close(fig)


def fig_cp_in_wt_frame(units, cpmap, scores_root):
    """Re-index the CP prediction into WT coordinates: same fold, one frame."""
    cp, wt = units["cp_1un2"], units["wt_1fvk"]
    c2w = cpmap["cp_to_wt"]
    Lw = wt["L"]
    cp_sc = mean_score("cp_1un2", scores_root)
    wt_sc = mean_score("wt_1fvk", scores_root)
    cp_in_wt = np.full((Lw, Lw), np.nan)
    idx = [(c, w) for c, w in enumerate(c2w) if w is not None]
    ci = np.array([c for c, _ in idx])
    wi = np.array([w for _, w in idx])
    cp_in_wt[np.ix_(wi, wi)] = cp_sc[np.ix_(ci, ci)]

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 5.0))
    gt = gt_mask(wt)
    ok = _sep_mask(Lw)
    # Same measure, two facets -> one ramp (see the note on ARMS).
    for ax, (mat, tag, cmap) in zip(axes, [(cp_in_wt, "CP prediction, re-indexed to WT", CMAP_BLUE),
                                           (wt_sc, "WT prediction", CMAP_BLUE)]):
        upper = np.where(np.triu(np.ones((Lw, Lw), bool), 1) & ok, mat, np.nan)
        ax.imshow(upper, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        yy, xx = np.nonzero(np.tril(np.ones((Lw, Lw), bool), -1) & ok & gt)
        ax.scatter(xx, yy, s=0.7, c=GT_INK, marker="s", linewidths=0)
        ax.plot([0, Lw - 1], [0, Lw - 1], color=MUTED, lw=0.5, alpha=0.6)
        for c in (cpmap["seg_b"]["wt_end"],):
            ax.axhline(c - 0.5, color=CRITICAL, lw=0.7, ls="--", alpha=0.85)
            ax.axvline(c - 0.5, color=CRITICAL, lw=0.7, ls="--", alpha=0.85)
        ax.set_xlim(-0.5, Lw - 0.5)
        ax.set_ylim(Lw - 0.5, -0.5)
        ax.set_title(tag, pad=6)
        ax.set_xlabel("wild-type residue index")
        ax.set_ylabel("wild-type residue index")
        ax.tick_params(length=2, labelsize=7)
    fig.suptitle("Both predictions in wild-type coordinates (lower triangle: WT truth)",
                 y=1.0, fontsize=11)
    fig.text(0.5, -0.03,
             "Red dashes mark the T99/Q100 cut. Off-diagonal blocks straddling it are the "
             "contacts the permutation moved to a different sequence separation.",
             ha="center", fontsize=7.5, color=MUTED)
    save_plot_with_meta(fig, PLOTS / "cp_in_wt_frame.png", caption=
        "Both predictions re-indexed into wild-type coordinates. The block straddling the T99/Q100 cut is visibly thinner in the permutant.")
    plt.close(fig)


def fig_permutation_contrast(contrast: pd.DataFrame):
    classes = ["within-segment", "cross-segment"]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
    for ax, (metric, name) in zip(axes, [("r_precision", "R-precision"), ("auc", "AUC")]):
        x = np.arange(len(classes))
        w = 0.34
        for k, (arm, colour, lbl) in enumerate([("wt", BLUE, "WT (1FVK)"),
                                                ("cp", ORANGE, "CP (1UN2)")]):
            mu = [contrast[(contrast.pair_class == c) & (contrast.arm == arm)][metric].mean()
                  for c in classes]
            sd = [contrast[(contrast.pair_class == c) & (contrast.arm == arm)][metric].std()
                  for c in classes]
            ax.bar(x + (k - 0.5) * w, mu, w * 0.92, yerr=sd, capsize=2.5,
                   color=colour, label=lbl,
                   error_kw=dict(lw=0.8, ecolor=MUTED))
            for xi, m, s in zip(x + (k - 0.5) * w, mu, sd):
                ax.text(xi, m + s + 0.022, f"{m:.3f}", ha="center", fontsize=7, color=INK)
        ax.set_xticks(x)
        ax.set_xticklabels(["separation\nUNCHANGED", "separation\nCHANGED"], fontsize=8)
        ax.set_ylabel(name)
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Same 3D contacts, same model — split by whether the permutation moved the pair",
                 y=1.04, fontsize=10.5)
    fig.text(0.5, -0.10,
             "Error bars are the standard deviation over independent rollout seeds.",
             ha="center", fontsize=7.5, color=MUTED)
    save_plot_with_meta(fig, PLOTS / "permutation_contrast.png", caption=
        "Same pairs and same model, split by whether the permutation moved them. Moved pairs lose 2.4x more R-precision.")
    plt.close(fig)


def fig_summary(per_unit: pd.DataFrame):
    """Headline: the permutation costs accuracy; the extra residues do not."""
    order = ["cp_1un2", "ctrl_identity", "wt_1fvk", "wt_1dsb", "wt_1a2j"]
    names = ["CP 1UN2\npermuted", "control\nWT + linker\n+ tail", "WT 1FVK\n1.7 Å",
             "WT 1DSB\n2.0 Å", "WT 1A2J\n2.0 Å"]
    colours = [ORANGE, BLUE, BLUE, BLUE, BLUE]
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.0))
    for ax, (rng, name) in zip(axes, [("all", "R-precision (all ranges)"),
                                      ("long", "R-precision (long range, |i−j|≥24)")]):
        mu, sd = [], []
        for u in order:
            v = per_unit[(per_unit.unit == u) & (per_unit.range == rng)
                         & (per_unit.cut == "R")].precision
            mu.append(v.mean())
            sd.append(v.std())
        x = np.arange(len(order))
        ax.bar(x, mu, 0.62, yerr=sd, capsize=3, color=colours,
               error_kw=dict(lw=0.9, ecolor=MUTED))
        for xi, m, s in zip(x, mu, sd):
            ax.text(xi, m + s + 0.018, f"{m:.3f}", ha="center", fontsize=7.5, color=INK)
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=7.5, linespacing=1.35)
        ax.set_ylabel(name, fontsize=8.5)
        ax.set_ylim(0, 0.80)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", length=0, pad=4)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    handles = [Patch(facecolor=ORANGE, label="circularly permuted"),
               Patch(facecolor=BLUE, label="native residue order")]
    axes[0].legend(handles=handles, fontsize=8, loc="upper right")
    fig.suptitle("The re-ordering costs accuracy — the extra 8 residues do not", y=1.02,
                 fontsize=11)
    fig.subplots_adjust(wspace=0.22)
    fig.text(0.5, -0.13,
             "The control has the CP construct's exact length and non-native residues "
             "(GGGTG linker, LIK tail) in wild-type order. Error bars: sd over 10 rollout seeds.",
             ha="center", fontsize=7.5, color=MUTED)
    save_plot_with_meta(fig, PLOTS / "summary.png", caption=
        "R-precision for the permutant, a length/composition control, and three wild-type crystals. The re-ordering costs ~0.09; the extra 8 residues cost nothing.")
    plt.close(fig)


def fig_separation_profile(units, scores_root):
    """Accuracy by separation, and how the permutation redistributes contacts.

    The right panel is the mechanism: the permutation does not change which
    residues touch in 3D, it changes where those touches land on the |i−j| axis.
    (AUC is deliberately not plotted — it sits at 0.97–0.99 in every bin for both
    arms, so a panel of it is a flat line at the ceiling. The numbers are in
    ``per_unit_metrics.csv.gz`` and the README table.)
    """
    bins = [(6, 11), (12, 23), (24, 47), (48, 95), (96, 10_000)]
    names = ["6–11", "12–23", "24–47", "48–95", "96+"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    for unit, tag, colour, _ in ARMS:
        rec = units[unit]
        L = rec["L"]
        gt = true_matrix(L, rec["contacts"])
        res = np.array(rec["resolved_positions"])
        a, b = np.triu_indices(len(res), k=1)
        pi, pj = res[a], res[b]
        psep = pj - pi
        precs, counts = [], []
        sc = mean_score(unit, scores_root)
        s_all, g_all = sc[pi, pj], gt[pi, pj].astype(int)
        for lo, hi in bins:
            m = (psep >= lo) & (psep <= hi)
            s, g = s_all[m], g_all[m]
            nt = int(g.sum())
            counts.append(nt)
            if nt == 0 or nt == len(g):
                precs.append(np.nan)
                continue
            gs = g[np.argsort(-s, kind="mergesort")]
            precs.append(gs[:nt].sum() / nt)
        lbl = f"{tag} ({rec['pdb']})"
        axes[0].plot(names, precs, marker="o", ms=5, lw=2, color=colour, label=lbl)
        axes[1].plot(names, counts, marker="o", ms=5, lw=2, color=colour, label=lbl)
    axes[0].set_ylabel("R-precision")
    axes[0].set_ylim(0, 1.0)
    axes[1].set_ylabel("number of true contacts")
    axes[1].set_ylim(bottom=0)
    for ax in axes:
        ax.set_xlabel("sequence separation |i−j|")
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    axes[0].legend(fontsize=8)
    axes[0].set_title("accuracy", fontsize=9, color=MUTED)
    axes[1].set_title("where the contacts sit", fontsize=9, color=MUTED)
    fig.suptitle("Sequence separation, in each molecule's own coordinates", y=1.03,
                 fontsize=10.5)
    fig.text(0.5, -0.10,
             "Same protein, same fold: the permutation moves ~47% of the contacts to a "
             "different separation (right), and those are the ones the model gets wrong (left).",
             ha="center", fontsize=7.5, color=MUTED)
    save_plot_with_meta(fig, PLOTS / "separation_profile.png", caption=
        "Accuracy by sequence separation (left) and where the true contacts sit on that axis (right), in each molecule's own coordinates.")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path, default=HERE / "_scratch" / "scores")
    args = ap.parse_args()
    PLOTS.mkdir(exist_ok=True)
    units, cpmap = load()
    contrast = pd.read_csv(DATA / "permutation_contrast.csv")
    per_unit = pd.read_csv(DATA / "per_unit_metrics.csv.gz")
    fig_summary(per_unit)
    fig_contact_maps(units, cpmap, args.scores)
    fig_top_l_overlay(units, cpmap, args.scores)
    fig_cp_in_wt_frame(units, cpmap, args.scores)
    fig_permutation_contrast(contrast)
    fig_separation_profile(units, args.scores)
    print(f"wrote {len(list(PLOTS.glob('*.png')))} figures to {PLOTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
