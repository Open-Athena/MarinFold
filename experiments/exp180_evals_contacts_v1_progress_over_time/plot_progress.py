# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Three figures on contacts-v1 progress over time.

1. ``rprecision_frontier.png`` — R-precision (all ranges, 554-protein eval set)
   of the best contacts-v1 model trained up to each date.
2. ``val_loss_frontier.png`` — the same staircase on held-out
   `contacts-v1-val` loss, over every finished contacts-v1 training run.
3. ``rprecision_vs_val_loss.png`` — the two axes against each other.

Colour does one job in every figure: it separates the two contact-inference
recipes (they are ~0.086 R-precision apart on the same weights, so the reader
must never have to guess which one a point is). Two slots of the data-viz
reference palette, slot 1 (blue) and slot 2 (orange); every point is also
directly labelled, so identity never rests on colour alone.

    uv run python plot_progress.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).parent

# data-viz reference palette, light mode.
ROLLOUT_C = "#2a78d6"     # slot 1 — the settled exp82 rollout recipe
PAIRWISE_C = "#eb6834"    # slot 2 — the original exp89 pairwise scorer
FRONTIER_C = "#0b0b0b"
CLOUD_C = "#898781"
TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, SURFACE = "#e1e0d9", "#fcfcfb"

RECIPE_LABEL = {
    "rollout": "rollout + resample, n=100 (exp82; top-k off)",
    "pairwise": "pairwise scoring (exp89 original)",
}
RECIPE_COLOR = {"rollout": ROLLOUT_C, "pairwise": PAIRWISE_C}
RECIPE_MARKER = {"rollout": "o", "pairwise": "^"}


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, alpha=0.9, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#c3c2b7")
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)


def _date_axis(ax, lo=None, hi=None, pad_lo=4.0, pad_hi=4.5) -> None:
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_minor_locator(mdates.DayLocator())
    # Explicit margins: several labels sit outboard of the first and last
    # marker, and matplotlib's default margin is not enough to hold them.
    if lo is not None:
        ax.set_xlim(pd.Timestamp(lo) - pd.Timedelta(days=pad_lo),
                    pd.Timestamp(hi) + pd.Timedelta(days=pad_hi))


def _write_meta(path: Path, meta: dict, caption: str) -> None:
    """Sidecar for ``build_summary.py``'s plot appendix + the numbers behind the figure.

    Keep ``caption`` to at most two printed lines — longer captions silently
    overlap the plot on the generated slide.
    """
    meta = dict(caption=caption, script="plot_progress.py", args=[], **meta)
    path.with_suffix(path.suffix + ".meta.json").write_text(json.dumps(meta, indent=2))


def _running_best(values: np.ndarray, mode: str) -> np.ndarray:
    return np.maximum.accumulate(values) if mode == "max" else np.minimum.accumulate(values)


def _draw_baselines(ax, baselines: pd.DataFrame, xfrac: float = 0.99,
                    ha: str = "right") -> None:
    """Structure predictors as dotted reference lines on an R-precision axis.

    They are context, not series: one recessive ink and a dotted stroke for all
    four, identity carried by a direct label rather than by colour — so the two
    colour slots stay reserved for the inference recipes.
    """
    for _, b in baselines.sort_values("r_precision").iterrows():
        ax.axhline(b["r_precision"], color=TEXT_MUTED, linewidth=1.2,
                   linestyle=(0, (2, 3)), zorder=1)
        ax.annotate(f"{b['label']}  {b['r_precision']:.3f}",
                    xy=(xfrac, b["r_precision"]), xycoords=("axes fraction", "data"),
                    xytext=(0, 4), textcoords="offset points",
                    ha=ha, va="bottom", fontsize=8.5, color=TEXT_MUTED, zorder=2,
                    # ESMFold2 (0.786) and Protenix-MSA (0.812) are 0.026 apart,
                    # so the upper line would otherwise strike through the
                    # lower label; a surface-coloured patch keeps text legible.
                    bbox=dict(boxstyle="square,pad=0.16", facecolor=SURFACE,
                              edgecolor="none"))


# ---------------------------------------------------------------------------
# Figure 1 — R-precision of the best model trained to date
# ---------------------------------------------------------------------------
# Label placement is hand-set: four near-chance checkpoints stack up at
# y ~ 0.03 and the three post-#117 points sit 0.02 apart. Offsets keep every
# label inside the axes and off the frontier's vertical risers.
RP_LABEL_OFFSET = {
    "#67 quick 1.5B": (10, 44),
    "#75 E1": (-52, 6),
    "#75 E2": (-26, 22),
    "#75 E4": (38, 10),
    "#61/#75 E8": (-62, 16),
    "#120 re-epoch": (-58, 18),
    "#117 E8 bs64": (-52, -22),
    "#117 E16 final": (-58, 26),
    "#146 3B E8": (6, 24),
    "#160 backtracking": (0, -28),
    "#155 3-way restart": (28, -22),
}
# Where the second (pairwise) reading of a dual-measured checkpoint goes.
RP_TWIN_OFFSET = {"#61/#75 E8": (-44, -8), "#120 re-epoch": (-42, -10)}
# The #117 early-stop checkpoint is step 33450 of the run that ends at 35679,
# so at day resolution it plots on top of the final. Both markers stay (they
# genuinely are indistinguishable); one label carries both numbers.
RP_LABEL_TEXT = {
    "#117 E16 final": "#117 E16 final  0.534\n(early-stop ckpt, same run: 0.532)",
}
RP_LABEL_SKIP = {"#117 E16 early stop"}


def plot_rprecision_frontier(rp: pd.DataFrame, baselines: pd.DataFrame, out: Path) -> None:
    rp = rp.sort_values(["date", "r_precision"]).reset_index(drop=True)

    # The frontier is the running max over the best measurement available for
    # each checkpoint — literally "the best contact accuracy we had by then".
    per_ckpt = (rp.groupby(["date", "label"], as_index=False)["r_precision"].max()
                  .sort_values("date"))
    fx = per_ckpt["date"].to_numpy()
    fy = _running_best(per_ckpt["r_precision"].to_numpy(), "max")

    fig, ax = plt.subplots(figsize=(11.6, 7.2))
    _style(ax)
    _date_axis(ax, rp["date"].min(), rp["date"].max(), pad_lo=5.0, pad_hi=6.0)
    _draw_baselines(ax, baselines)

    ax.step(fx, fy, where="post", color=FRONTIER_C, linewidth=2.0, zorder=2,
            label="best model trained to date")

    for recipe, sub in rp.groupby("inference"):
        ax.scatter(sub["date"], sub["r_precision"], s=120,
                   marker=RECIPE_MARKER[recipe], color=RECIPE_COLOR[recipe],
                   edgecolor="white", linewidth=2.0, zorder=4,
                   label=RECIPE_LABEL[recipe])

    # The headline label goes on each checkpoint's *best* reading; a checkpoint
    # measured under both recipes gets a bare value on its lower (pairwise) one.
    headline = set(rp.groupby("label")["r_precision"].idxmax())
    for i, r in rp.iterrows():
        if r["label"] in RP_LABEL_SKIP:
            continue
        if i in headline:
            text = RP_LABEL_TEXT.get(r["label"], f"{r['label']}  {r['r_precision']:.3f}")
            dx, dy = RP_LABEL_OFFSET.get(r["label"], (0, 20))
        else:
            text = f"{r['r_precision']:.3f}"
            dx, dy = RP_TWIN_OFFSET.get(r["label"], (4, -26))
        ax.annotate(text, (r["date"], r["r_precision"]), textcoords="offset points",
                    xytext=(dx, dy), ha="center", fontsize=9, color=TEXT_PRIMARY,
                    zorder=5,
                    arrowprops=dict(arrowstyle="-", color="#c3c2b7", linewidth=1.0,
                                    shrinkA=1, shrinkB=7))

    ax.set_ylim(-0.03, 0.88)
    ax.set_ylabel("R-precision, all ranges  (mean over 554 proteins)",
                  fontsize=10.5, color=TEXT_SECONDARY)
    ax.set_title("Best contacts-v1 model trained to date - contact R-precision",
                 fontsize=13.5, color=TEXT_PRIMARY, pad=12, loc="left")
    # Bottom-right is the one region with no marks, no labels and no dotted
    # baseline running through it.
    ax.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY)
    fig.text(0.5, 0.030,
             "Step line = running best; dotted lines = structure predictors, same 554 proteins and same metric.",
             ha="center", fontsize=8, color=TEXT_MUTED)
    fig.text(0.5, 0.010,
             "Identical weights score ~0.086 higher under rollout than under pairwise "
             "(#61/#75 E8: 0.339 -> 0.425; #120: 0.350 -> 0.436), so the recipes are drawn apart, never merged.",
             ha="center", fontsize=8, color=TEXT_MUTED)
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    _write_meta(out, dict(
        figure="rprecision_frontier", metric="R-precision, range=all, n=554",
        baselines={b["label"]: float(b["r_precision"]) for _, b in baselines.iterrows()},
        frontier=[dict(date=str(pd.Timestamp(d).date()), r_precision=float(v))
                  for d, v in zip(fx, fy)]),
        caption="Best contacts-v1 model to date on contact R-precision. Two jumps, "
                "both from the base model: #75 E8 (Jun 21) and #117 E16 (Jul 22).")
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# Figure 2 — best held-out contacts-v1 val loss to date
# ---------------------------------------------------------------------------
VL_LABEL_OFFSET = {
    "prot-exp75-cv1-1_5b-e1-lr8.75e-5-wd0p05-v1": (56, 12),
    "prot-exp75-cv1-1_5b-e2-lr7e-4-wd0p1-v1": (48, 14),
    "prot-exp75-cv1-1_5b-e8-lr1e-3-wd0p1-v1": (62, 14),
    "prot-exp75-cv1-1_5b-e8-lr1e-3-wd0p2-v1": (78, 14),
    "plm-exp108-cv1-3b-e16-lr2e-3-wd0p2-v3": (-10, -34),
    "exp120-cv1-1_5b-orig-lr3e-4-e1-cos": (-6, 26),
    "prot-exp117-cv1-s02-1_5b-e16-lr3p162e-3-wd0p2-bs256-europe-west4": (-46, -38),
    "prot-exp146-cv1-s01-3b-e8-lr3p162e-3-wd0p4-bs256-us-east1": (14, 26),
}
# Short human names for the frontier steps worth calling out.
VL_NAMES = {
    "prot-exp75-cv1-1_5b-e1-lr8.75e-5-wd0p05-v1": "#75 sweep opens (E1)",
    "prot-exp75-cv1-1_5b-e2-lr7e-4-wd0p1-v1": "#75 E2 rung",
    "prot-exp75-cv1-1_5b-e8-lr1e-3-wd0p1-v1": "#75 first E8",
    "prot-exp75-cv1-1_5b-e8-lr1e-3-wd0p2-v1": "#61/#75 E8 winner",
    "plm-exp108-cv1-3b-e16-lr2e-3-wd0p2-v3": "#108 3B, CoreWeave H100",
    "exp120-cv1-1_5b-orig-lr3e-4-e1-cos": "#120 re-epoch",
    "prot-exp117-cv1-s02-1_5b-e16-lr3p162e-3-wd0p2-bs256-europe-west4": "#117 E16 final",
    "prot-exp146-cv1-s01-3b-e8-lr3p162e-3-wd0p4-bs256-us-east1": "#146 3B E8",
}
YTOP = 3.26


def plot_val_loss_frontier(runs: pd.DataFrame, rp: pd.DataFrame, out: Path) -> None:
    done = runs[(runs["state"] == "finished") & (~runs["excluded"])].sort_values("finished")
    fx = done["finished"].to_numpy()
    fy = _running_best(done["val_loss"].to_numpy(), "min")

    live = runs[(runs["state"] == "running") & (~runs["excluded"])]

    fig, ax = plt.subplots(figsize=(11.6, 7.2))
    _style(ax)
    _date_axis(ax, done["finished"].min(), runs["finished"].max(),
               pad_lo=3.0, pad_hi=6.0)

    n_above = int((done["val_loss"] > YTOP).sum())
    ax.scatter(done["finished"], done["val_loss"], s=26, color=CLOUD_C, alpha=0.45,
               linewidth=0, zorder=1, label=f"finished training run (n={len(done)})")
    ax.step(fx, fy, where="post", color=FRONTIER_C, linewidth=2.0, zorder=3,
            label="best val loss to date")

    # Ring the checkpoints that also carry a contact score, so the two figures
    # can be read against each other.
    scored = rp.dropna(subset=["val_loss"]).drop_duplicates("label")
    ax.scatter(scored["date"], scored["val_loss"], s=115, marker="o",
               facecolor="none", edgecolor=ROLLOUT_C, linewidth=2.0, zorder=4,
               label="also scored on the 554-protein benchmark")

    if len(live):
        ax.scatter(live["finished"], live["val_loss"], s=90, marker="D",
                   facecolor="none", edgecolor=PAIRWISE_C, linewidth=1.8, zorder=4,
                   label="still training (not on the frontier)")
        best_live = live.loc[live["val_loss"].idxmin()]
        ax.annotate(f"#155 3-way mix, in flight  {best_live['val_loss']:.4f}",
                    (best_live["finished"], best_live["val_loss"]),
                    textcoords="offset points", xytext=(30, -38), ha="center",
                    fontsize=9, color=TEXT_PRIMARY,
                    arrowprops=dict(arrowstyle="-", color="#c3c2b7", linewidth=1.0,
                                    shrinkA=1, shrinkB=7))

    # Label the frontier steps worth naming, plus whichever step is current.
    steps, prev = [], np.inf
    for x, y, name in zip(fx, fy, done["name"]):
        if y < prev - 1e-12:
            steps.append((x, y, name))
            prev = y
    for i, (x, y, name) in enumerate(steps):
        if name not in VL_NAMES and i != len(steps) - 1:
            continue
        dx, dy = VL_LABEL_OFFSET.get(name, (0, 20))
        ax.annotate(f"{VL_NAMES.get(name, name)}  {y:.4f}", (x, y),
                    textcoords="offset points", xytext=(dx, dy), ha="center",
                    fontsize=9, color=TEXT_PRIMARY, zorder=5,
                    arrowprops=dict(arrowstyle="-", color="#c3c2b7", linewidth=1.0,
                                    shrinkA=1, shrinkB=7))

    ax.set_ylim(2.632, YTOP)
    ax.set_ylabel("held-out contacts-v1 val loss  (nats/token, lower is better)",
                  fontsize=10.5, color=TEXT_SECONDARY)
    ax.set_title("Best contacts-v1 model trained to date - validation loss",
                 fontsize=13.5, color=TEXT_PRIMARY, pad=12, loc="left")
    ax.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY)
    fig.text(0.5, 0.030,
             "Every finished contacts-v1 run in W&B (open-athena/MarinFold + eric-czech/marin exp75/117/146/153); "
             f"{n_above} runs sit above the top of the axis.",
             ha="center", fontsize=8, color=TEXT_MUTED)
    fig.text(0.5, 0.010,
             "Preempted/crashed runs are excluded - their last logged loss is mid-training, not a trained model.",
             ha="center", fontsize=8, color=TEXT_MUTED)
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    _write_meta(out, dict(
        figure="val_loss_frontier", n_runs=len(done), n_above_axis=n_above,
        frontier=[dict(date=str(pd.Timestamp(x).date()), val_loss=float(y), run=n)
                  for x, y, n in steps]),
        caption="Best held-out contacts-v1 val loss to date over 153 finished runs. "
                "The loss frontier has many more steps than the accuracy one.")
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# Figure 3 — R-precision against val loss
# ---------------------------------------------------------------------------
S3_OFFSET = {
    ("#75 E1", "pairwise"): (16, -32),
    ("#75 E2", "pairwise"): (2, 26),
    ("#75 E4", "pairwise"): (-6, -32),
    ("#67 quick 1.5B", "pairwise"): (2, 52),
    ("#61/#75 E8", "pairwise"): (-58, -20),
    ("#61/#75 E8", "rollout"): (-64, 22),
    ("#120 re-epoch", "pairwise"): (-14, -36),
    ("#120 re-epoch", "rollout"): (-46, 44),
    ("#117 E8 bs64", "pairwise"): (74, -18),
    ("#117 E16 final", "rollout"): (-46, 34),
    ("#117 E16 early stop", "rollout"): (18, 34),
    ("#146 3B E8", "rollout"): (56, -14),
}


def plot_rprecision_vs_loss(rp: pd.DataFrame, baselines: pd.DataFrame, out: Path) -> None:
    sub = rp.dropna(subset=["val_loss"]).copy()

    fig, ax = plt.subplots(figsize=(11.2, 7.4))
    _style(ax)
    # Better model to the right, matching exp169's panel A. Set explicitly
    # rather than via invert_xaxis() so the margins hold the outboard labels.
    ax.set_xlim(sub["val_loss"].max() + 0.022, sub["val_loss"].min() - 0.020)
    # Structure predictors have no LM validation loss, so they can only be
    # horizontal reference lines here — which is exactly the useful reading:
    # how far along the loss axis a contacts-v1 model would have to get.
    # Labelled at the left, where the panel is empty above R = 0.55.
    _draw_baselines(ax, baselines, xfrac=0.018, ha="left")

    for recipe, s in sub.groupby("inference"):
        s = s.sort_values("val_loss", ascending=False)
        ax.plot(s["val_loss"], s["r_precision"], color=RECIPE_COLOR[recipe],
                linewidth=1.6, alpha=0.35, zorder=1)
        ax.scatter(s["val_loss"], s["r_precision"], s=130,
                   marker=RECIPE_MARKER[recipe], color=RECIPE_COLOR[recipe],
                   edgecolor="white", linewidth=2.0, zorder=3,
                   label=RECIPE_LABEL[recipe])

    for _, r in sub.iterrows():
        dx, dy = S3_OFFSET.get((r["label"], r["inference"]), (0, 20))
        ax.annotate(f"{r['label']}\n{r['val_loss']:.4f} -> {r['r_precision']:.3f}",
                    (r["val_loss"], r["r_precision"]), textcoords="offset points",
                    xytext=(dx, dy), ha="center", fontsize=8.5, color=TEXT_PRIMARY,
                    zorder=4,
                    arrowprops=dict(arrowstyle="-", color="#c3c2b7", linewidth=1.0,
                                    shrinkA=1, shrinkB=8))

    # The phase change: a 0.17-nat drop (2.924 -> 2.757) takes R-precision from
    # ~0.03 to ~0.34. Everything to the left of it is at chance.
    ax.axvspan(2.757, 2.924, color=CLOUD_C, alpha=0.10, zorder=0)
    ax.text(2.8405, 0.15, "the phase change\n0.17 nats, ~10x R-precision",
            ha="center", fontsize=9, color=TEXT_MUTED, zorder=2)

    ax.set_xlabel("held-out contacts-v1 val loss   (lower is better ->)",
                  fontsize=10.5, color=TEXT_SECONDARY)
    ax.set_ylabel("R-precision, all ranges  (mean over 554 proteins)",
                  fontsize=10.5, color=TEXT_SECONDARY)
    ax.set_ylim(-0.04, 0.90)
    ax.set_title("Contact R-precision against contacts-v1 validation loss",
                 fontsize=13.5, color=TEXT_PRIMARY, pad=12, loc="left")
    ax.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY)
    fig.text(0.5, 0.030,
             "Dotted lines = structure predictors, same 554 proteins and metric; they have no LM validation loss, "
             "so they can only be horizontal here.",
             ha="center", fontsize=8, color=TEXT_MUTED)
    fig.text(0.5, 0.010,
             "#120's loss is `contacts-v1-val-orig`, the others `contacts-v1-val` - the same held-out split under "
             "different cache names (see README.md).",
             ha="center", fontsize=8, color=TEXT_MUTED)
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    _write_meta(out, dict(
        figure="rprecision_vs_val_loss",
        points=[dict(label=r["label"], inference=r["inference"],
                     val_loss=float(r["val_loss"]), r_precision=float(r["r_precision"]))
                for _, r in sub.iterrows()]),
        caption="R-precision against val loss. ~2 R-precision per nat across "
                "generations; flat inside one run (#117 early vs final).")
    print(f"wrote {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=HERE / "data")
    ap.add_argument("--out-dir", type=Path, default=HERE / "plots")
    a = ap.parse_args()

    rp = pd.read_csv(a.data_dir / "rprecision_checkpoints.csv", parse_dates=["date"])
    runs = pd.read_csv(a.data_dir / "val_loss_runs.csv", parse_dates=["started", "finished"])
    baselines = pd.read_csv(a.data_dir / "structure_baselines.csv")

    a.out_dir.mkdir(parents=True, exist_ok=True)
    plot_rprecision_frontier(rp, baselines, a.out_dir / "rprecision_frontier.png")
    plot_val_loss_frontier(runs, rp, a.out_dir / "val_loss_frontier.png")
    plot_rprecision_vs_loss(rp, baselines, a.out_dir / "rprecision_vs_val_loss.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
