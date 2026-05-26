# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot distributional stats for v2 documents over the sample.

Reads ``data/sample_stats.csv`` (produced by
``_scripts/sample_stats.py``) and writes one PNG per plot to
``plots/``, plus a ``.meta.json`` sidecar per plot so the
``build_summary.py`` PDF builder can render the script + invocation
under each slide.

Knobs:

- ``--in``  CSV path (default ``data/sample_stats.csv``)
- ``--out`` plots directory (default ``plots/``)

Each plot lives in its own ``_plot_*`` function so it can be
re-derived in isolation; the orchestrator at the bottom calls them
all and writes the sidecars.
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP_ROOT = HERE.parent

sys.path.insert(0, str(EXP_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_summary import save_plot_with_meta


# Mode order/colors used everywhere so plots compose visually.
_MODE_ORDER = ("long", "medium", "short")
_MODE_COLORS = {"long": "#1f77b4", "medium": "#ff7f0e", "short": "#2ca02c"}
_MODE_LABEL = {"long": "long-range", "medium": "medium-range", "short": "short-range"}


def _load(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for c in df.columns:
        if c == "entry_id":
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def _plot_length_distribution(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["n_residues"], bins=20, color="#4c72b0", edgecolor="white")
    ax.axvline(df["n_residues"].median(), color="black", ls="--", lw=1,
               label=f"median = {df['n_residues'].median():.0f}")
    ax.set_xlabel("sequence length (residues)")
    ax.set_ylabel("# proteins")
    ax.set_title("Sample length distribution (FoldBench monomers, n=100)")
    ax.legend()
    save_plot_with_meta(
        fig, out / "01_length_distribution.png",
        caption=(
            "Sequence-length histogram of the 100 FoldBench monomer "
            "ground-truth CIFs used as the representative sample. "
            "Median ~ 220 residues; range 30..761."
        ),
    )
    plt.close(fig)


def _plot_statements_per_doc(df: pd.DataFrame, out: Path) -> None:
    """Boxplots of contact + distance statement counts per doc."""
    fig, ax = plt.subplots(figsize=(7, 4))
    data = [
        df["n_long_shown"],
        df["n_medium_shown"],
        df["n_short_shown"],
        df["n_distance_shown"],
    ]
    labels = ["long\ncontacts", "medium\ncontacts", "short\ncontacts", "distance\nstatements"]
    colors = [_MODE_COLORS["long"], _MODE_COLORS["medium"], _MODE_COLORS["short"], "#9467bd"]
    bp = ax.boxplot(
        data, tick_labels=labels, patch_artist=True, showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "white",
                   "markeredgecolor": "black", "markersize": 6},
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    ax.set_ylabel("# statements per doc")
    ax.set_title("Statements per doc, by type")
    ax.set_yscale("symlog", linthresh=10)
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    save_plot_with_meta(
        fig, out / "02_statements_per_doc.png",
        caption=(
            "Statements per document, by type. Distance statements "
            "dominate (mean ~1210); contacts are limited by the "
            "per-mode budget knob ``contact_f_range`` (default "
            "uniform(-0.1, 0.2), clamped to 0). Long-range contacts "
            "are the most numerous because their eligible pool is "
            "the largest. Symlog y-axis to keep both contact and "
            "distance scales legible on one panel."
        ),
    )
    plt.close(fig)


def _plot_fraction_captured(df: pd.DataFrame, out: Path) -> None:
    """For each mode, n_shown / n_eligible — limited to rows with eligible > 0."""
    fig, ax = plt.subplots(figsize=(7, 4))
    fracs = []
    labels = []
    for mode in _MODE_ORDER:
        e = df[f"n_{mode}_eligible"]
        s = df[f"n_{mode}_shown"]
        mask = e > 0
        f = (s[mask] / e[mask]).clip(upper=1.0).to_numpy()
        fracs.append(f)
        labels.append(f"{_MODE_LABEL[mode]}\n(n={int(mask.sum())})")
    bp = ax.boxplot(
        fracs, tick_labels=labels, patch_artist=True, showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "white",
                   "markeredgecolor": "black", "markersize": 6},
    )
    for patch, color in zip(bp["boxes"], [_MODE_COLORS[m] for m in _MODE_ORDER]):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    ax.set_ylim(-0.02, 1.05)
    ax.axhline(1.0, color="grey", ls=":", lw=1)
    ax.set_ylabel("fraction of eligible contacts shown")
    ax.set_title("Fraction of eligible contacts captured per mode")
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    save_plot_with_meta(
        fig, out / "03_fraction_captured.png",
        caption=(
            "What fraction of a protein's eligible contacts make it "
            "into the doc, broken out by mode. Medium- and short-"
            "range typically saturate (median ~1.0) because the "
            "per-mode budget can usually hold them all. Long-range "
            "is the bottleneck — median ~0.41, because long-range "
            "eligible pools can be huge (up to 1365 contacts on the "
            "longest protein here)."
        ),
    )
    plt.close(fig)


def _plot_token_composition(df: pd.DataFrame, out: Path) -> None:
    """Average token-budget breakdown across the sample."""
    contact_tokens = 3 * (
        df["n_long_shown"] + df["n_medium_shown"] + df["n_short_shown"]
    )
    distance_tokens = 6 * df["n_distance_shown"]
    means = {
        "fixed overhead\n(<task> <begin_seq> <begin_stmts> <plddt> <end>)": 5.0,
        "residue tokens": df["n_residues"].mean(),
        "<think> tokens": df["n_think_total"].mean(),
        "contact statements": contact_tokens.mean(),
        "distance statements": distance_tokens.mean(),
    }
    total = sum(means.values())
    colors = ["#bbbbbb", "#4c72b0", "#dd8452", "#55a467", "#9467bd"]
    fig, ax = plt.subplots(figsize=(7, 4))
    left = 0.0
    for (label, val), color in zip(means.items(), colors):
        frac = val / total
        ax.barh([0], [val], left=left, color=color, edgecolor="white",
                label=f"{label}: {val:.0f} ({100 * frac:.1f}%)")
        if frac > 0.03:
            ax.text(
                left + val / 2, 0,
                f"{100 * frac:.1f}%", ha="center", va="center",
                color="white", fontsize=9, fontweight="bold",
            )
        left += val
    ax.set_yticks([])
    ax.set_xlim(0, 8192)
    ax.set_xlabel("tokens per doc (mean across n=100)")
    ax.set_title("Token-budget breakdown (mean over the sample)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=9)
    save_plot_with_meta(
        fig, out / "04_token_composition.png",
        caption=(
            "Per-doc token budget, averaged across the sample. "
            "Distance statements eat ~89% of the context (6 tokens "
            "each at ~1210 statements per doc); contacts ~8%, "
            "residues ~3%, <think> ~0.1%. Docs pack to the budget "
            "almost exactly (mean total ~8190/8192)."
        ),
    )
    plt.close(fig)


def _plot_statements_vs_length(df: pd.DataFrame, out: Path) -> None:
    """One-panel-per-mode + distance scatter of statement count vs length."""
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
    panels = [
        ("n_long_shown",     "long-range contacts shown",   _MODE_COLORS["long"]),
        ("n_medium_shown",   "medium-range contacts shown", _MODE_COLORS["medium"]),
        ("n_short_shown",    "short-range contacts shown",  _MODE_COLORS["short"]),
        ("n_distance_shown", "distance statements shown",   "#9467bd"),
    ]
    for ax, (col, label, color) in zip(axes.ravel(), panels):
        ax.scatter(df["n_residues"], df[col], c=color, alpha=0.7, edgecolor="white", s=28)
        ax.set_xlabel("sequence length (residues)")
        ax.set_ylabel(label)
        ax.grid(True, linestyle=":", alpha=0.5)
    fig.suptitle("Per-statement-type count vs sequence length", y=1.0)
    fig.tight_layout()
    save_plot_with_meta(
        fig, out / "05_statements_vs_length.png",
        caption=(
            "Each scatter point is one document. Contact-statement "
            "counts rise with sequence length (more eligible "
            "contacts available), with substantial variance from "
            "the per-mode fraction sampling. Distance-statement "
            "count *falls* with sequence length — because residue "
            "tokens take a larger bite of the 8192 budget, leaving "
            "less room for statements after contacts have claimed "
            "their share."
        ),
    )
    plt.close(fig)


def _plot_context_fill(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(df["n_total_tokens"], bins=20, color="#7f7f7f", edgecolor="white")
    ax.axvline(8192, color="red", ls="--", lw=1, label="context_length = 8192")
    ax.set_xlim(8180, 8195)
    ax.set_xlabel("total tokens in doc")
    ax.set_ylabel("# docs")
    ax.set_title("Context-window utilization")
    ax.legend()
    save_plot_with_meta(
        fig, out / "06_context_fill.png",
        caption=(
            "Every doc fills the 8192-token budget within a handful "
            "of tokens. The small deficit (max=8192, median=8190) "
            "comes from integer-division when allocating distance "
            "statements out of the remaining budget after contacts "
            "are placed."
        ),
    )
    plt.close(fig)


def _plot_think_distribution(df: pd.DataFrame, out: Path) -> None:
    """Three-panel breakdown of think-token cost."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    # Panel 1: total think tokens per doc.
    ax = axes[0]
    ax.hist(df["n_think_total"], bins=range(0, max(df["n_think_total"]) + 2),
            color="#dd8452", edgecolor="white")
    ax.set_xlabel("total <think> tokens per doc")
    ax.set_ylabel("# docs")
    ax.set_title(f"All docs (mean={df['n_think_total'].mean():.1f})")
    # Panel 2: initial-run length conditional on gate firing.
    ax = axes[1]
    fired = df.loc[df["n_think_initial_k1"] > 0, "n_think_initial_k1"]
    ax.hist(fired, bins=range(0, int(fired.max()) + 2),
            color="#4c72b0", edgecolor="white")
    ax.axvline(1 / 0.13, color="black", ls="--", lw=1, label=f"E[k1] = 1/0.13 ≈ 7.7")
    ax.set_xlabel("k1 (initial run length when gate fires)")
    ax.set_title(f"Gate fired on {len(fired)}/{len(df)} ({100*len(fired)/len(df):.0f}%)")
    ax.legend(fontsize=8)
    # Panel 3: # additional runs.
    ax = axes[2]
    counts = df["n_think_additional_runs"].value_counts().sort_index()
    ax.bar(counts.index, counts.values, color="#55a467", edgecolor="white")
    ax.set_xlabel("# additional <think> runs in doc")
    ax.set_title("Inter-statement insertions")
    ax.set_xticks(sorted(counts.index))
    for ax in axes:
        ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    save_plot_with_meta(
        fig, out / "07_think_distribution.png",
        caption=(
            "Three views of think-token cost. Left: total <think> "
            "tokens per doc (mean ~8). Middle: initial-run length "
            "k1 when the P=0.75 gate fires — distribution matches "
            "Geom(0.13). Right: # additional inter-statement runs — "
            "0..3 (matches int(Uniform(-4, 4)) clamped to 0)."
        ),
    )
    plt.close(fig)


def _plot_eligible_vs_total(df: pd.DataFrame, out: Path) -> None:
    """Per-mode: eligible (post-pLDDT filter) vs shown vs total in protein."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), sharey=False)
    for ax, mode in zip(axes, _MODE_ORDER):
        elig = df[f"n_{mode}_eligible"]
        shown = df[f"n_{mode}_shown"]
        ax.scatter(elig, shown, alpha=0.7, color=_MODE_COLORS[mode],
                   edgecolor="white", s=28)
        m = max(elig.max(), shown.max(), 1)
        ax.plot([0, m], [0, m], color="grey", ls=":", lw=1, label="y = x")
        ax.set_xlabel(f"# {_MODE_LABEL[mode]} contacts eligible")
        ax.set_ylabel(f"# {_MODE_LABEL[mode]} contacts shown")
        ax.set_title(f"{_MODE_LABEL[mode]} contacts")
        ax.legend(fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    save_plot_with_meta(
        fig, out / "08_eligible_vs_shown.png",
        caption=(
            "Per-mode: x = eligible contacts in the protein, y = "
            "contacts placed in the doc. Points on y=x mean the doc "
            "captured *every* available contact of that mode. "
            "Medium- and short-range proteins frequently saturate; "
            "long-range proteins routinely have far more eligible "
            "contacts than the per-mode budget can fit."
        ),
    )
    plt.close(fig)


def _plot_eligible_pool_size_vs_length(df: pd.DataFrame, out: Path) -> None:
    """Eligible-contact pool size vs sequence length, per mode (log-log)."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for mode in _MODE_ORDER:
        ax.scatter(
            df["n_residues"], df[f"n_{mode}_eligible"].clip(lower=0.5),
            c=_MODE_COLORS[mode], alpha=0.7, s=24,
            label=f"{_MODE_LABEL[mode]} (CB-CB ≤ 8 Å, sep in range)",
            edgecolor="white",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("sequence length (residues)")
    ax.set_ylabel("# eligible contacts in protein")
    ax.set_title("Eligible-contact pool size by mode (log-log)")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend(fontsize=8)
    save_plot_with_meta(
        fig, out / "09_eligible_pool_vs_length.png",
        caption=(
            "Eligible-contact pool size by mode versus sequence "
            "length. Long-range pools grow super-linearly with "
            "length (~N^1.5 visually on this log-log) while "
            "medium- and short-range pools grow roughly linearly. "
            "This is what makes long-range the bottleneck on "
            "fraction-captured."
        ),
    )
    plt.close(fig)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


_PLOTS = [
    ("01", _plot_length_distribution),
    ("02", _plot_statements_per_doc),
    ("03", _plot_fraction_captured),
    ("04", _plot_token_composition),
    ("05", _plot_statements_vs_length),
    ("06", _plot_context_fill),
    ("07", _plot_think_distribution),
    ("08", _plot_eligible_vs_total),
    ("09", _plot_eligible_pool_size_vs_length),
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="csv", type=Path,
                    default=EXP_ROOT / "data" / "sample_stats.csv")
    ap.add_argument("--out", type=Path,
                    default=EXP_ROOT / "plots")
    args = ap.parse_args(argv)

    df = _load(args.csv)
    args.out.mkdir(parents=True, exist_ok=True)
    for label, fn in _PLOTS:
        fn(df, args.out)
        print(f"  [{label}] {fn.__name__}")
    print(f"wrote {len(_PLOTS)} plots under {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
