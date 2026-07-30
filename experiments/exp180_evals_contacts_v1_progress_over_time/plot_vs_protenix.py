# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""``plots/marinfold_vs_protenix.png`` — the comparison, one point per protein.

The progress figures show where the frontier is; this one shows *which
proteins* it is made of. The current best contacts-v1 model (#117 E16 final)
against Protenix-v2 in single-sequence mode, R-precision (all ranges) on each
of the 554 eval proteins, so the y = x diagonal separates the two.

Colour, and only colour, encodes **sequence length** — a magnitude, so a
single-hue ordinal ramp (the data-viz reference blue). Length is the variable
that actually moves this comparison: the two predictors cross over above ~400
residues.

    uv run python plot_vs_protenix.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).parent
EXP169 = HERE / ".." / "exp169_evals_selected_checkpoints_117_146" / "data" / "exp169_rows.csv.gz"
EXP89 = (HERE / ".." / "exp89_evals_contacts_v1_model_on_eval_set" / "data"
         / "contact_precision_all.csv")

# Our best contacts-v1 model, same checkpoint that tops the progress figures.
MARINFOLD_MODEL = "exp117_e16_final_step35679"
MARINFOLD_LABEL = "MarinFold #117 E16 final"
PROTENIX = dict(model="protenix-v2", mode="single_seq", predictor="structure")
PROTENIX_LABEL = "Protenix-v2, single-sequence"

TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, SURFACE = "#e1e0d9", "#fcfcfb"

# Length bins. Chosen before looking at the deltas: the 400+ bin is "longer
# than almost anything in the eval set". They do double duty as the colour
# encoding and as the summary table, so a reader can find a table row in the
# cloud.
LENGTH_BINS = [(0, 100), (100, 200), (200, 400), (400, 10_000)]
BIN_LABEL = {(0, 100): "< 100", (100, 200): "100-200",
             (200, 400): "200-400", (400, 10_000): "> 400"}
# Ordinal ramp: four steps of the reference blue (250 / 400 / 550 / 700).
# Length is a magnitude, so one hue light->dark, and step 250 is the lightest
# allowed against the light surface for an ordinal (discrete) ramp. A
# continuous ramp was tried first and is unreadable at 554 overlapping points.
BIN_COLOR = {(0, 100): "#86b6ef", (100, 200): "#3987e5",
             (200, 400): "#1c5cab", (400, 10_000): "#0d366b"}
# One size for every point: colour alone carries the length bin, so nothing on
# the plot can be mistaken for a second encoding. The 17 longest proteins stay
# findable because their bin is the darkest step and is drawn last, on top.
MARKER_SIZE = 46


def load(exp169: Path, exp89: Path) -> pd.DataFrame:
    """One row per protein: both R-precisions, the length, and the strata."""
    mf = pd.read_csv(exp169)
    mf = mf[(mf["model"] == MARINFOLD_MODEL) & (mf["cut"] == "R")
            & (mf["range"] == "all")]
    px = pd.read_csv(exp89)
    px = px[(px["model"] == PROTENIX["model"]) & (px["mode"] == PROTENIX["mode"])
            & (px["predictor"] == PROTENIX["predictor"]) & (px["cut"] == "R")
            & (px["range"] == "all")]

    j = mf[["dataset", "stem", "n_residues", "n_true", "precision"]].merge(
        px[["dataset", "stem", "precision", "neff_tier", "fold_verdict"]],
        on=["dataset", "stem"], suffixes=("_mf", "_px"), validate="one_to_one")
    j = j.dropna(subset=["precision_mf", "precision_px"])
    if len(j) != 554:
        print(f"!! joined {len(j)} proteins, expected 554 — check the inputs")
    return j


def paired_stats(j: pd.DataFrame) -> dict:
    d = (j["precision_mf"] - j["precision_px"]).to_numpy()
    sem = d.std(ddof=1) / np.sqrt(d.size)
    return dict(
        n=int(d.size),
        mean_mf=float(j["precision_mf"].mean()),
        mean_px=float(j["precision_px"].mean()),
        mean_delta=float(d.mean()),
        ci_low=float(d.mean() - 1.96 * sem),
        ci_high=float(d.mean() + 1.96 * sem),
        win_rate=float((d > 0).mean()),
        tie_rate=float((d == 0).mean()),
        pearson=float(j["precision_mf"].corr(j["precision_px"])),
        spearman=float(j["precision_mf"].corr(j["precision_px"], method="spearman")),
    )


def by_length(j: pd.DataFrame) -> list[dict]:
    out = []
    for lo, hi in LENGTH_BINS:
        s = j[(j["n_residues"] >= lo) & (j["n_residues"] < hi)]
        if s.empty:
            continue
        out.append(dict(bin=BIN_LABEL[(lo, hi)], n=int(len(s)),
                        mf=float(s["precision_mf"].mean()),
                        px=float(s["precision_px"].mean()),
                        mf_wins=float((s["precision_mf"] > s["precision_px"]).mean())))
    return out


def plot(j: pd.DataFrame, out: Path) -> None:
    st, bins = paired_stats(j), by_length(j)

    # Two panels: a square scatter, and a text column. At 554 points there is
    # no in-axes region that reliably holds a stats block without covering
    # data, so the numbers get their own space instead of fighting for it.
    fig, (ax, tx) = plt.subplots(
        1, 2, figsize=(13.4, 8.2), gridspec_kw=dict(width_ratios=[1.0, 0.46]))

    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, alpha=0.9, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#c3c2b7")
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)

    # y = x, and which side means what. The diagonal is the whole point of the
    # figure, so it is drawn under the data and labelled on both sides.
    ax.plot([0, 1], [0, 1], color=TEXT_SECONDARY, linewidth=1.3,
            linestyle=(0, (5, 4)), zorder=2)
    corner = dict(fontsize=10.5, color=TEXT_SECONDARY, zorder=8,
                  bbox=dict(boxstyle="square,pad=0.3", facecolor=SURFACE,
                            edgecolor="none"))
    ax.text(0.02, 1.0, "MarinFold higher", ha="left", va="top", **corner)
    ax.text(1.0, 0.0, "Protenix-v2 higher", ha="right", va="bottom", **corner)

    # Longest bin drawn last so its 17 points sit on top of the pile.
    for i, (lo, hi) in enumerate(LENGTH_BINS):
        s = j[(j["n_residues"] >= lo) & (j["n_residues"] < hi)]
        if s.empty:
            continue
        ax.scatter(s["precision_px"], s["precision_mf"],
                   s=MARKER_SIZE, color=BIN_COLOR[(lo, hi)],
                   alpha=0.85, linewidth=0.7, edgecolor="white", zorder=3 + i)

    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_aspect("equal")
    ax.set_xlabel(f"{PROTENIX_LABEL}  -  R-precision", fontsize=11, color=TEXT_SECONDARY)
    ax.set_ylabel(f"{MARINFOLD_LABEL}  -  R-precision", fontsize=11, color=TEXT_SECONDARY)
    # No in-axes legend: the colour chips beside the length table in the right
    # panel are the legend, and an in-axes one both duplicates them and covers
    # the lower-left corner of the cloud.

    # --- text column ------------------------------------------------------
    tx.axis("off")
    y = 0.97
    tx.text(0.0, y, "554 eval proteins", fontsize=12.5, fontweight="bold",
            color=TEXT_PRIMARY, va="top")
    y -= 0.055
    tx.text(0.0, y, "R-precision, all ranges (sep >= 6)", fontsize=9.5,
            color=TEXT_MUTED, va="top")

    y -= 0.08
    for label, value in (("MarinFold #117", st["mean_mf"]),
                         ("Protenix-v2 SS", st["mean_px"])):
        tx.text(0.0, y, label, fontsize=10.5, color=TEXT_SECONDARY, va="top")
        tx.text(0.62, y, f"{value:.3f}", fontsize=10.5, color=TEXT_PRIMARY,
                va="top", family="DejaVu Sans Mono")
        y -= 0.05

    y -= 0.03
    for label, value in (
        ("paired delta", f"{st['mean_delta']:+.3f}"),
        ("95% CI", f"[{st['ci_low']:+.3f}, {st['ci_high']:+.3f}]"),
        ("MarinFold higher on", f"{st['win_rate']:.0%}"),
        ("Spearman", f"{st['spearman']:.2f}"),
    ):
        tx.text(0.0, y, label, fontsize=10, color=TEXT_SECONDARY, va="top")
        tx.text(0.62, y, value, fontsize=10, color=TEXT_PRIMARY, va="top",
                family="DejaVu Sans Mono")
        y -= 0.048

    y -= 0.05
    tx.text(0.0, y, "By sequence length", fontsize=11, fontweight="bold",
            color=TEXT_PRIMARY, va="top")
    y -= 0.042
    tx.text(0.0, y, "darker = longer", fontsize=8.5,
            color=TEXT_MUTED, va="top")
    y -= 0.05
    tx.text(0.0, y, f"{'':>9}{'n':>5}{'MF':>8}{'PX':>8}{'MF hi':>9}",
            fontsize=9, color=TEXT_MUTED, va="top", family="DejaVu Sans Mono")
    y -= 0.045
    for b, (lo, hi) in zip(bins, LENGTH_BINS):
        tx.text(0.0, y,
                f"{b['bin']:>9}{b['n']:>5}{b['mf']:>8.3f}{b['px']:>8.3f}"
                f"{b['mf_wins']:>8.0%}",
                fontsize=9, color=TEXT_PRIMARY, va="top",
                family="DejaVu Sans Mono")
        # Chip at the plot's marker size, so the legend matches the cloud.
        tx.scatter([-0.05], [y - 0.014], s=MARKER_SIZE,
                   color=BIN_COLOR[(lo, hi)], edgecolor="white",
                   linewidth=0.7, clip_on=False)
        y -= 0.045

    y -= 0.05
    tx.text(0.0, y,
            "MarinFold is lower on average. The\n"
            "difference narrows as length rises and\n"
            "changes sign in the > 400 bin, where\n"
            "n = 17. Both predictors decline with\n"
            "length; MarinFold declines less.",
            fontsize=9.5, color=TEXT_SECONDARY, va="top", linespacing=1.5)

    tx.set_xlim(-0.09, 1.0)
    tx.set_ylim(0, 1)

    fig.suptitle("Contact R-precision per protein: MarinFold #117 vs Protenix-v2 (single-sequence)",
                 fontsize=13.5, color=TEXT_PRIMARY, x=0.045, y=0.975, ha="left")
    fig.text(0.5, 0.022,
             "Each point is one eval protein. MarinFold reads contacts from sequence alone; Protenix-v2 here is "
             "also single-sequence (no MSA).",
             ha="center", fontsize=8, color=TEXT_MUTED)
    fig.text(0.5, 0.004,
             "MarinFold scored with exp82's rollout recipe (n=100, top-k off); both scored by exp89's "
             "compute_metrics on the same ground truth.",
             ha="center", fontsize=8, color=TEXT_MUTED)
    fig.tight_layout(rect=(0, 0.038, 1, 0.955))
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    plt.close(fig)

    meta = dict(
        caption="Per-protein R-precision, MarinFold #117 vs Protenix-v2 single-seq "
                "(n=554). Means 0.534 vs 0.603; the difference varies with length.",
        script="plot_vs_protenix.py", args=[],
        figure="marinfold_vs_protenix",
        marinfold=MARINFOLD_MODEL, protenix=PROTENIX,
        stats=st, by_length=bins)
    out.with_suffix(out.suffix + ".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out}")
    print(f"  MarinFold {st['mean_mf']:.4f}  Protenix-v2 SS {st['mean_px']:.4f}  "
          f"delta {st['mean_delta']:+.4f}  win rate {st['win_rate']:.1%}")
    for b in bins:
        print(f"  L {b['bin']:>8}: n={b['n']:3d}  mf={b['mf']:.3f}  px={b['px']:.3f}  "
              f"mf wins {b['mf_wins']:.0%}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp169-rows", type=Path, default=EXP169)
    ap.add_argument("--exp89-csv", type=Path, default=EXP89)
    ap.add_argument("--out-dir", type=Path, default=HERE / "plots")
    ap.add_argument("--csv-out", type=Path, default=HERE / "data" / "marinfold_vs_protenix.csv")
    a = ap.parse_args()

    j = load(a.exp169_rows, a.exp89_csv)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    a.csv_out.parent.mkdir(parents=True, exist_ok=True)
    j.to_csv(a.csv_out, index=False)
    print(f"wrote {a.csv_out} ({len(j)} proteins)")
    plot(j, a.out_dir / "marinfold_vs_protenix.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
