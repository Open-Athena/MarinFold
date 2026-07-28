# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""The two figures for issue #169.

1. ``where_we_stand_rprecision.png`` — R-precision distribution over the 554
   eval proteins for the #169 checkpoints next to every structure predictor.
   Answers "where does the current best MarinFold model sit?".
2. ``loss_vs_rprecision.png`` — val loss against R-precision with paired 95%
   CIs. Answers the question #169 actually poses: at a 0.008-nat separation,
   does lower val loss still buy contact accuracy?

Colour does one job in each figure — group identity — so both use two slots of
the data-viz reference palette (slots 1 and 2, documented as clearing the
all-pairs CVD and normal-vision floors in both modes). Every box and point is
also directly labelled, so identity never rests on colour alone.

    uv run python plot_results.py --rows data/exp169_rows.csv.gz \\
        --exp89-csv ../exp89_.../data/contact_precision_all.csv --out-dir plots
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# One implementation of the paired statistic, shared with the table.
from summarize_results import paired_differences  # noqa: E402

# data-viz reference palette, light mode: slot 1 (blue) and slot 2 (orange).
LM_COLOR = "#eb6834"
LM_PRIOR_COLOR = "#f7bda4"       # same hue, recessive step — already-published rows
STRUCTURE_COLOR = "#2a78d6"
TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED = "#0b0b0b", "#52514e", "#8a8880"
GRID = "#d8d7d2"

# (row key in the per-protein table, mode, predictor, display label, group)
# `group` drives colour; the axis label carries identity.
PANEL = [
    ("marinfold-cv1-exp75-rollout", "single_seq", "lm",
     "MarinFold #61/#75\n1.5B · E8 · 2.7566", "lm_prior"),
    ("exp117_e16_final_step35679", "single_seq", "lm",
     "MarinFold #117\n1.5B · final · 2.7037", "lm"),
    ("exp117_e16_early_step33450", "single_seq", "lm",
     "MarinFold #117\n1.5B · early · 2.6961", "lm"),
    ("exp146_3b_e8_step17839", "single_seq", "lm",
     "MarinFold #146\n3B · E8 · 2.7025", "lm"),
    ("protenix-v2", "single_seq", "structure", "Protenix-v2\nsingle-seq", "structure"),
    ("protenix-v2", "msa", "structure", "Protenix-v2\n+ MSA", "structure"),
    ("esmfold", "single_seq", "structure", "ESMFold", "structure"),
    ("esmfold2", "single_seq", "structure", "ESMFold2", "structure"),
]
GROUP_COLOR = {"lm": LM_COLOR, "lm_prior": LM_PRIOR_COLOR, "structure": STRUCTURE_COLOR}
GROUP_LEGEND = [
    (LM_PRIOR_COLOR, "MarinFold LM — previously published"),
    (LM_COLOR, "MarinFold LM — evaluated in #169"),
    (STRUCTURE_COLOR, "structure predictor"),
]

# Checkpoints on the loss axis: key -> (label, val loss, parameter count).
LOSS_POINTS = [
    ("marinfold-cv1-exp75-rollout", "#61/#75 E8", 2.756602, "1.5B"),
    ("exp117_e16_final_step35679", "#117 final", 2.703709, "1.5B"),
    ("exp117_e16_early_step33450", "#117 early stop", 2.696074, "1.5B"),
    ("exp146_3b_e8_step17839", "#146 3B", 2.702478, "3B"),
]
SIZE_COLOR = {"1.5B": STRUCTURE_COLOR, "3B": LM_COLOR}


def _values(rows: pd.DataFrame, model: str, mode: str, predictor: str) -> np.ndarray:
    sel = rows[(rows["model"] == model) & (rows["mode"] == mode)
               & (rows["predictor"] == predictor)]
    v = sel["precision"].to_numpy(dtype=float)
    return v[np.isfinite(v)]


def _write_meta(path: Path, meta: dict) -> None:
    path.with_suffix(path.suffix + ".meta.json").write_text(json.dumps(meta, indent=2))


def plot_where_we_stand(rows: pd.DataFrame, out: Path) -> None:
    sub = rows[(rows["cut"] == "R") & (rows["range"] == "all")]
    series, labels, colors, means, counts = [], [], [], [], []
    for model, mode, predictor, label, group in PANEL:
        v = _values(sub, model, mode, predictor)
        if v.size == 0:
            print(f"!! no rows for {model}/{mode}/{predictor} — dropping from the figure")
            continue
        series.append(v)
        labels.append(label)
        colors.append(GROUP_COLOR[group])
        means.append(float(v.mean()))
        counts.append(int(v.size))
    if len(set(counts)) != 1:
        print(f"!! uneven protein counts across boxes: {dict(zip(labels, counts))}")

    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.set_facecolor("white")
    positions = np.arange(len(series))
    box = ax.boxplot(series, positions=positions, widths=0.62, patch_artist=True,
                     showmeans=True, showfliers=True,
                     meanprops=dict(marker="D", markerfacecolor="white",
                                    markeredgecolor=TEXT_PRIMARY, markersize=5.5),
                     medianprops=dict(color=TEXT_PRIMARY, linewidth=1.4),
                     whiskerprops=dict(color=TEXT_SECONDARY, linewidth=1.0),
                     capprops=dict(color=TEXT_SECONDARY, linewidth=1.0),
                     flierprops=dict(marker=".", markersize=2.5, markerfacecolor=TEXT_MUTED,
                                     markeredgecolor="none", alpha=0.35))
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
        # 2px surface ring so adjacent fills never touch.
        patch.set_edgecolor("white")
        patch.set_linewidth(2.0)

    for x, mean in zip(positions, means):
        ax.text(x, 1.04, f"{mean:.3f}", ha="center", va="bottom", fontsize=9.5,
                color=TEXT_PRIMARY, fontweight="normal")

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8.5, color=TEXT_SECONDARY)
    ax.set_ylabel("R-precision  (top-R ranked pairs, R = #true contacts)",
                  fontsize=10.5, color=TEXT_SECONDARY)
    ax.set_ylim(-0.02, 1.12)
    ax.grid(axis="y", color=GRID, alpha=0.8, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.set_title(f"Contact R-precision, all ranges (sep ≥ 6) · n = {counts[0]} proteins",
                 fontsize=13, color=TEXT_PRIMARY, pad=14)
    # Legend as a footer row under the axis labels — the plot area itself is full
    # of boxes and the mean labels sit along its top edge, so there is no
    # in-axes corner a legend can occupy without colliding with data.
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, alpha=0.85, edgecolor="white")
               for c, _ in GROUP_LEGEND]
    fig.legend(handles, [t for _, t in GROUP_LEGEND], loc="lower center", frameon=False,
               fontsize=9, labelcolor=TEXT_SECONDARY, ncols=3, bbox_to_anchor=(0.5, 0.045))
    fig.text(0.5, 0.008,
             "box = median & IQR · whiskers = 1.5×IQR · ◆ = mean, also printed above each box · "
             "exp89 metric implementation",
             ha="center", fontsize=8, color=TEXT_MUTED)
    fig.tight_layout(rect=(0, 0.115, 1, 1))
    fig.savefig(out, dpi=170)
    plt.close(fig)
    _write_meta(out, dict(figure="where_we_stand_rprecision",
                          metric="R-precision, range=all",
                          n_proteins=counts[0],
                          means=dict(zip(labels, means))))
    print(f"wrote {out}")


def plot_loss_vs_rprecision(rows: pd.DataFrame, out: Path) -> None:
    """Two panels, because the question has two scales.

    Left: the loss -> accuracy relationship across the whole 0.06-nat span the
    project has measured. Right: the #169 checkpoints alone, as *paired* deltas
    against the #117 final baseline — at a 0.008-nat separation the unpaired CIs
    on the left are an order of magnitude too wide to resolve anything, and only
    the paired statistic can say whether these checkpoints differ at all.
    """
    sub = rows[(rows["cut"] == "R") & (rows["range"] == "all")]
    xs, ys, errs, labels, colors, sizes = [], [], [], [], [], []
    for key, label, loss, size in LOSS_POINTS:
        v = _values(sub, key, "single_seq", "lm")
        if v.size == 0:
            print(f"!! no rows for {key} — dropping from the loss figure")
            continue
        xs.append(loss)
        ys.append(float(v.mean()))
        errs.append(1.96 * float(v.std(ddof=1)) / np.sqrt(v.size))
        labels.append(label)
        colors.append(SIZE_COLOR[size])
        sizes.append(size)

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.6, 5.4),
                                 gridspec_kw=dict(width_ratios=[1.25, 1.0]))

    # --- left: loss vs R-precision -----------------------------------------
    ax.set_facecolor("white")
    # Trend across the 1.5B family only — the 3B is a different capacity and
    # should not be fitted as if it were another point on one model's curve.
    fam = [i for i, s in enumerate(sizes) if s == "1.5B"]
    if len(fam) >= 2:
        fx = np.array([xs[i] for i in fam])
        fy = np.array([ys[i] for i in fam])
        slope, intercept = np.polyfit(fx, fy, 1)
        grid = np.linspace(min(xs) - 0.006, max(xs) + 0.006, 50)
        ax.plot(grid, slope * grid + intercept, color=STRUCTURE_COLOR, linewidth=2.0,
                alpha=0.30, zorder=1)
        ax.text(0.03, 0.94, f"1.5B: {abs(slope):.1f} R-precision per nat of loss reduction",
                transform=ax.transAxes, fontsize=9, color=TEXT_SECONDARY, va="top")

    ax.errorbar(xs, ys, yerr=errs, fmt="none", ecolor=TEXT_MUTED, elinewidth=1.2,
                capsize=4, zorder=2)
    # Stagger labels: the three #169 checkpoints sit within 0.008 nats of each
    # other, so a fixed offset would overprint them.
    offsets = _label_offsets(xs)
    for x, y, label, color, (dx, dy) in zip(xs, ys, labels, colors, offsets):
        ax.scatter([x], [y], s=110, color=color, edgecolor="white", linewidth=2.0, zorder=3)
        ax.annotate(f"{label}  {y:.3f}", (x, y), textcoords="offset points",
                    xytext=(dx, dy), ha="center", fontsize=9, color=TEXT_PRIMARY, zorder=4,
                    arrowprops=dict(arrowstyle="-", color=GRID, linewidth=1.0,
                                    shrinkA=2, shrinkB=6))

    ax.invert_xaxis()          # better model to the right
    # Headroom for the staggered labels: LABEL_SLOTS reach ~56 px from a point,
    # so without padding the outermost label lands on the axis.
    lo = min(y - e for y, e in zip(ys, errs))
    hi = max(y + e for y, e in zip(ys, errs))
    pad = max(hi - lo, 1e-3) * 0.38
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("eval/tokenized/contacts-v1-val/loss   (lower is better →)",
                  fontsize=10, color=TEXT_SECONDARY)
    ax.set_ylabel("mean R-precision  (n = 554)", fontsize=10, color=TEXT_SECONDARY)
    ax.set_title("A · across the whole measured loss range", fontsize=11,
                 color=TEXT_PRIMARY, pad=10, loc="left")
    handles = [plt.Line2D([], [], marker="o", linestyle="none", markersize=9,
                          markerfacecolor=SIZE_COLOR[s], markeredgecolor="white", label=s)
               for s in ("1.5B", "3B")]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=9,
              labelcolor=TEXT_SECONDARY, title="parameters",
              title_fontproperties=dict(size=9))

    # --- right: paired deltas vs the #117 final baseline ---------------------
    # Only the #169 checkpoints. The #61/#75 delta is ~10x larger and would set
    # the x-range so wide that the two comparisons this panel exists to answer
    # would collapse onto the zero line; its size is quoted as text instead.
    baseline = "exp117_e16_final_step35679"
    compared = ("exp117_e16_early_step33450", "exp146_3b_e8_step17839")
    paired = paired_differences(rows, cut="R", rng="all")

    def _delta(key: str):
        row = paired[((paired.model_a == baseline) & (paired.model_b == key))
                     | ((paired.model_a == key) & (paired.model_b == baseline))]
        if row.empty:
            return None
        row = row.iloc[0]
        flip = 1.0 if row.model_a == key else -1.0
        win = (row.win_rate_a if row.model_a == key
               else 1.0 - row.win_rate_a - row.tie_rate)
        return flip * row.mean_delta, flip * row.ci_low, flip * row.ci_high, win

    entries = []
    for key, label, _, size in LOSS_POINTS:
        if key not in compared:
            continue
        d = _delta(key)
        if d is not None:
            entries.append((label, size, *d))
    prior = _delta("marinfold-cv1-exp75-rollout")

    bx.set_facecolor("white")
    if entries:
        ypos = np.arange(len(entries))[::-1]
        for y, (label, size, delta, lo, hi, win) in zip(ypos, entries):
            lo, hi = min(lo, hi), max(lo, hi)
            bx.plot([lo, hi], [y, y], color=TEXT_MUTED, linewidth=1.4,
                    solid_capstyle="butt", zorder=2)
            for edge in (lo, hi):
                bx.plot([edge, edge], [y - 0.09, y + 0.09], color=TEXT_MUTED,
                        linewidth=1.4, zorder=2)
            bx.scatter([delta], [y], s=110, color=SIZE_COLOR[size], edgecolor="white",
                       linewidth=2.0, zorder=3)
            resolved = "resolved" if lo * hi > 0 else "not resolved"
            bx.annotate(f"{delta:+.4f}  ({resolved}; wins {win:.0%})",
                        (delta, y), textcoords="offset points", xytext=(0, 16),
                        ha="center", fontsize=9, color=TEXT_PRIMARY)
        bx.axvline(0.0, color=TEXT_SECONDARY, linewidth=1.2, zorder=1)
        bx.text(0.0, len(entries) - 0.42, "#117 final", ha="center", va="bottom",
                fontsize=8.5, color=TEXT_SECONDARY)
        bx.set_yticks(ypos)
        bx.set_yticklabels([e[0] for e in entries], fontsize=9.5, color=TEXT_SECONDARY)
        bx.set_ylim(-0.7, len(entries) - 0.15)
        span = max(abs(v) for e in entries for v in (e[3], e[4])) * 2.4
        bx.set_xlim(-span, span)
        if prior is not None:
            bx.text(0.5, 0.02,
                    f"for scale, the previous generation (#61/#75) sits at "
                    f"{prior[0]:+.3f} on this axis",
                    transform=bx.transAxes, ha="center", fontsize=8.5, color=TEXT_MUTED)
    bx.set_xlabel("Δ R-precision vs #117 final  (paired, per protein)",
                  fontsize=10, color=TEXT_SECONDARY)
    bx.set_title("B · is the difference resolvable?", fontsize=11,
                 color=TEXT_PRIMARY, pad=10, loc="left")

    for axis in (ax, bx):
        axis.grid(color=GRID, alpha=0.8, linewidth=0.8)
        axis.set_axisbelow(True)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            axis.spines[spine].set_color(GRID)
        axis.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    bx.grid(axis="y", visible=False)

    fig.suptitle("Does lower val loss still buy contact accuracy?", fontsize=13.5,
                 color=TEXT_PRIMARY, x=0.5, y=0.985)
    fig.text(0.5, 0.008,
             "A: error bars = 95% CI of the mean (unpaired).   "
             "B: error bars = 95% CI of the mean per-protein difference (paired, n = 554) — "
             "the same 554 proteins score every checkpoint, so this is ~10× tighter.",
             ha="center", fontsize=8, color=TEXT_MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(out, dpi=170)
    plt.close(fig)
    _write_meta(out, dict(
        figure="loss_vs_rprecision",
        panel_a=[dict(label=lbl, loss=x, r_precision=y, ci95=e)
                 for lbl, x, y, e in zip(labels, xs, ys, errs)],
        panel_b=[dict(label=l, delta=d, ci_low=lo, ci_high=hi, win_rate=w)
                 for l, _, d, lo, hi, w in entries]))
    print(f"wrote {out}")


# Vertical label slots, alternating above/below and stepping outward. Three of
# the four checkpoints sit within 0.008 nats of each other, so any fixed offset
# — and any simple above/below alternation — overprints them.
LABEL_SLOTS = ((0, 18), (0, -30), (0, 46), (0, -56))


def _label_offsets(xs: list[float]) -> list[tuple[int, int]]:
    """Assign each point a distinct vertical label slot, ordered along x."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    offsets: list[tuple[int, int]] = [(0, 18)] * len(xs)
    for rank, i in enumerate(order):
        offsets[i] = LABEL_SLOTS[rank % len(LABEL_SLOTS)]
    return offsets


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=Path, required=True, help="this experiment's rows CSV")
    ap.add_argument("--exp89-csv", type=Path, required=True,
                    help="exp89 contact_precision_all.csv — the structure-predictor rows")
    ap.add_argument("--prior-rows", type=Path, default=None,
                    help="exp82 where_we_stand_rows.csv.gz — the #61/#75 rollout rows")
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    frames = [pd.read_csv(a.rows), pd.read_csv(a.exp89_csv)]
    if a.prior_rows:
        frames.append(pd.read_csv(a.prior_rows))
    rows = pd.concat(frames, ignore_index=True)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    plot_where_we_stand(rows, a.out_dir / "where_we_stand_rprecision.png")
    plot_loss_vs_rprecision(rows, a.out_dir / "loss_vs_rprecision.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
