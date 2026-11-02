# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""The new default against every baseline on eval2, split natural vs de novo.

The 554-protein benchmark is ~75% designed proteins and is not homology
controlled, so a single number off it answers neither "does this generalize"
nor "what is it good at". eval2 (#226) splits both ways at once: it keeps only
proteins under 40% identity to anything in training, and it carries a
`designed_any` flag. Cutting on that flag is what these figures do.

The split is the result. On natural proteins every predictor is weak and
MarinFold is mid-field; on de novo proteins every predictor is strong and
MarinFold is last. Those are different regimes, and pooling them (which the
554 effectively does, at 75% designed) produces a number that describes
neither.

Inputs, both already in git and both already verified:

* `../exp180_.../data/exp199_cw_p06_cool_step290400_rows.csv.gz` — the
  cooldown's per-protein scores. The same file #180's head-to-head figure
  reads, deliberately: two figures citing one checkpoint must not be able to
  disagree. #238 published it to the bucket; its sha256 is in #180's
  `plot_vs_protenix.py`.
* `../exp226_.../data/eval2_per_protein.csv.gz` — every baseline's per-protein
  scores plus the `designed_any` and `passes_30` flags, from the experiment
  that built the eval set.

**Every run writes the joined per-protein table it actually plotted** to
`data/eval2_per_protein_scores.csv.gz` — one row per (protein, range, cut) with
every predictor's score side by side. That file, not the two inputs, is this
experiment's record of what these figures are made of, and `--per-protein`
redraws from it alone. The inputs are owned by other experiments and will move:
#180's rows file gets re-pointed the next time the accuracy frontier does, at
which point rebuilding from upstream stops reproducing *these* figures.

    uv run python plot_eval2_comparison.py              # rebuild from upstream
    uv run python plot_eval2_comparison.py --per-protein data/eval2_per_protein_scores.csv.gz
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

HERE = Path(__file__).parent
COOLDOWN_ROWS = (HERE / ".." / "exp180_evals_contacts_v1_progress_over_time"
                 / "data" / "exp199_cw_p06_cool_step290400_rows.csv.gz")
EVAL2_ROWS = (HERE / ".." / "exp226_evals_expand_foldbench_eval_set"
              / "data" / "eval2_per_protein.csv.gz")

OURS = "MarinFold #199 cooldown"
PREVIOUS = "MarinFold #199 (1.5B, seq only)"
PREVIOUS_LABEL = "MarinFold #199 p06-aug"
NULL = "seq-KNN k=10 (null)"
BASELINES = ["Protenix-v2 + MSA", "ESMFold2", "ESMFold", "Protenix-v2 single-seq"]

# data-viz reference palette, light mode — the same slots plot_progress.py
# uses, assigned in the same fixed order. Slot 1 carries MarinFold everywhere
# in this repo; slot 2 is the second category when there is one.
OURS_C = "#2a78d6"          # slot 1 — the model this experiment promotes
PREVIOUS_C = "#86b6ef"      # slot 1, lighter step — the checkpoint it displaces
BASELINE_C = "#898781"      # recessive: baselines are context, not series
NULL_C = "#c3c2b7"          # the null is context for the context
NATURAL_C, DENOVO_C = "#2a78d6", "#eb6834"   # slots 1 and 2
TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, SURFACE = "#e1e0d9", "#fcfcfb"

SUBSETS = (("natural", "eval2 natural", 0), ("denovo", "eval2 de novo", 1))


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, alpha=0.9, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#c3c2b7")
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)


def build_joined() -> pd.DataFrame:
    """Join the cooldown onto every baseline, for every (range, cut) in common.

    Kept wider than the figures need — `long` and `AUC` come along with `all`
    and `R` — so a later question about long-range contacts or ranking quality
    does not have to re-derive this from two other experiments' data
    directories.
    """
    cool = pd.read_csv(COOLDOWN_ROWS)
    cool = cool[["dataset", "stem", "range", "cut", "precision"]].rename(
        columns={"precision": OURS})
    everyone = pd.read_csv(EVAL2_ROWS)

    joined = everyone.merge(cool, on=["dataset", "stem", "range", "cut"],
                            how="left", validate="one_to_one")
    missing = joined[OURS].isna().sum()
    if missing:
        raise SystemExit(f"{missing} of {len(joined)} eval2 rows have no cooldown "
                         f"score — the rows file does not cover this eval set")
    return joined


def load(per_protein: Path | None, metric_range: str = "all",
         cut: str = "R") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (the full joined table, the slice the figures use)."""
    joined = pd.read_csv(per_protein) if per_protein else build_joined()
    sliced = joined[(joined["range"] == metric_range) & (joined["cut"] == cut)]
    if sliced.empty:
        raise SystemExit(f"no rows with range={metric_range!r} cut={cut!r}")
    return joined, sliced


def paired(sub: pd.DataFrame, other: str) -> dict:
    """Ours minus theirs over the same proteins, with a 95% interval."""
    delta = (sub[OURS] - sub[other]).dropna()
    interval = stats.t.interval(0.95, len(delta) - 1, loc=delta.mean(),
                                scale=stats.sem(delta))
    return dict(mean_ours=sub[OURS].mean(), mean_other=sub[other].mean(),
                delta=delta.mean(), ci_low=interval[0], ci_high=interval[1],
                ours_higher=float((sub[OURS] > sub[other]).mean()), n=len(delta))


def summarise(data: pd.DataFrame) -> pd.DataFrame:
    """The numbers behind both figures, one row per (subset, predictor)."""
    rows = []
    for key, label, flag in SUBSETS:
        sub = data[data["designed_any"] == flag]
        for predictor in [OURS, PREVIOUS, *BASELINES, NULL]:
            row = dict(subset=key, subset_label=label, n=len(sub),
                       predictor=predictor, mean=sub[predictor].mean())
            if predictor != OURS:
                row.update({f"vs_ours_{k}": v for k, v in paired(sub, predictor).items()})
            rows.append(row)
    return pd.DataFrame(rows)


def plot_bars(data: pd.DataFrame, out: Path) -> None:
    """Both cuts as one ranked bar chart each, in a single shared order.

    The order is the natural cut's, held fixed in the de novo panel rather than
    re-sorted. Re-sorting would hide the finding: the ranking is not stable
    across the two, and MarinFold moves from mid-field to last.
    """
    natural = data[data["designed_any"] == 0]
    order = ([OURS, PREVIOUS, *BASELINES, NULL])
    order = sorted(order, key=lambda p: natural[p].mean(), reverse=True)

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.9), sharex=True)
    for ax, (key, label, flag) in zip(axes, SUBSETS):
        _style(ax)
        sub = data[data["designed_any"] == flag]
        values = [sub[p].mean() for p in order]
        colors = [OURS_C if p == OURS else PREVIOUS_C if p == PREVIOUS
                  else NULL_C if p == NULL else BASELINE_C for p in order]
        y = np.arange(len(order))[::-1]
        ax.barh(y, values, height=0.62, color=colors, zorder=3)
        for yi, value, predictor in zip(y, values, order):
            ax.text(value + 0.012, yi, f"{value:.3f}", va="center", fontsize=9.5,
                    color=TEXT_PRIMARY if predictor == OURS else TEXT_SECONDARY,
                    fontweight="bold" if predictor == OURS else "normal", zorder=4)
        ax.set_yticks(y)
        ax.set_yticklabels(
            [PREVIOUS_LABEL if p == PREVIOUS else p for p in order], fontsize=9.5,
            color=TEXT_SECONDARY)
        for tick, predictor in zip(ax.get_yticklabels(), order):
            if predictor == OURS:
                tick.set_color(TEXT_PRIMARY)
                tick.set_fontweight("bold")
        ax.set_xlim(0, 0.95)
        ax.set_title(f"{label}   (n = {len(sub)})", fontsize=12, loc="left",
                     color=TEXT_PRIMARY, pad=10)

    axes[0].set_xlabel("R-precision, all ranges", fontsize=10.5, color=TEXT_SECONDARY)
    axes[1].set_xlabel("R-precision, all ranges", fontsize=10.5, color=TEXT_SECONDARY)
    fig.suptitle("Contact R-precision on eval2, split by whether the protein was designed",
                 fontsize=13.5, color=TEXT_PRIMARY, x=0.007, ha="left", y=0.985)
    fig.text(0.5, 0.045,
             "eval2 keeps only proteins under 40% sequence identity to anything in training (#226). "
             "Bar order is the natural cut's, held fixed on the right.",
             ha="center", fontsize=8.5, color=TEXT_MUTED)
    fig.text(0.5, 0.012,
             "MarinFold reads sequence alone. So do Protenix-v2 single-seq, ESMFold and ESMFold2; "
             "Protenix-v2 + MSA does not, and designed proteins have almost no MSA to read.",
             ha="center", fontsize=8.5, color=TEXT_MUTED)
    fig.tight_layout(rect=(0, 0.075, 1, 0.965))
    fig.savefig(out, dpi=170, facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


def plot_scatter(data: pd.DataFrame, out: Path) -> None:
    """Per-protein, against the two baselines that also read sequence alone."""
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 6.0))
    for ax, baseline in zip(axes, ["Protenix-v2 single-seq", "ESMFold2"]):
        _style(ax)
        ax.plot([0, 1], [0, 1], color="#52514e", linewidth=1.0, linestyle="--", zorder=2)
        for key, label, flag in SUBSETS:
            sub = data[data["designed_any"] == flag]
            ax.scatter(sub[baseline], sub[OURS], s=26, alpha=0.75, zorder=3,
                       color=NATURAL_C if flag == 0 else DENOVO_C,
                       edgecolor="white", linewidth=0.6,
                       label=f"{label}  (n={len(sub)})")
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel(f"{baseline}  -  R-precision", fontsize=10.5, color=TEXT_SECONDARY)
        ax.set_ylabel("MarinFold #199 cooldown  -  R-precision", fontsize=10.5,
                      color=TEXT_SECONDARY)
        deltas = "   ".join(
            f"{label}: {paired(data[data['designed_any'] == flag], baseline)['delta']:+.3f}"
            for _, label, flag in SUBSETS)
        ax.set_title(f"vs {baseline}\n{deltas}", fontsize=11.5, loc="left",
                     color=TEXT_PRIMARY, pad=10)
        ax.annotate("MarinFold higher", xy=(0.03, 0.97), fontsize=9,
                    color=TEXT_MUTED, va="top")
        ax.annotate("baseline higher", xy=(0.97, 0.03), fontsize=9,
                    color=TEXT_MUTED, ha="right")
    # Figure-level, below the axes: every in-axes corner is either occupied by
    # the "who is higher" annotations or inside the point cloud.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.055),
               ncol=2, frameon=False, fontsize=9.5, labelcolor=TEXT_SECONDARY)
    fig.suptitle("Where the eval2 gap is: one point per protein, both sequence-only baselines",
                 fontsize=13.5, color=TEXT_PRIMARY, x=0.007, ha="left", y=0.985)
    fig.text(0.5, 0.015,
             "Natural proteins sit low for every predictor; de novo proteins sit high for the "
             "baselines and only middling for MarinFold. Same model, opposite verdicts.",
             ha="center", fontsize=8.5, color=TEXT_MUTED)
    fig.tight_layout(rect=(0, 0.105, 1, 0.965))
    fig.savefig(out, dpi=170, facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=HERE / "plots")
    parser.add_argument("--data-dir", type=Path, default=HERE / "data")
    parser.add_argument(
        "--per-protein", type=Path, default=None,
        help="redraw from a previously written joined table instead of "
             "rebuilding it from the two upstream experiments")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)

    joined, data = load(args.per_protein)
    if args.per_protein is None:
        rows_out = args.data_dir / "eval2_per_protein_scores.csv.gz"
        joined.to_csv(rows_out, index=False, compression={"method": "gzip", "mtime": 0})
        print(f"wrote {rows_out} ({len(joined)} rows, "
              f"{joined['stem'].nunique()} proteins)")
    table = summarise(data)
    csv_out = args.data_dir / "eval2_comparison.csv"
    table.to_csv(csv_out, index=False)
    print(f"wrote {csv_out} ({len(table)} rows)")

    plot_bars(data, args.out_dir / "eval2_predictor_comparison.png")
    plot_scatter(data, args.out_dir / "eval2_vs_sequence_only_baselines.png")

    meta = {}
    for key, label, flag in SUBSETS:
        sub = data[data["designed_any"] == flag]
        meta[key] = dict(n=len(sub), ours=float(sub[OURS].mean()),
                         **{p: float(sub[p].mean()) for p in [PREVIOUS, *BASELINES, NULL]},
                         vs_protenix_ss=paired(sub, "Protenix-v2 single-seq"),
                         vs_esmfold2=paired(sub, "ESMFold2"),
                         vs_previous=paired(sub, PREVIOUS))
    (args.out_dir / "eval2_predictor_comparison.png.meta.json").write_text(
        json.dumps(dict(figure="eval2_predictor_comparison",
                        metric="R-precision, range=all", subsets=meta,
                        caption="R-precision on eval2's natural (n=78) and de novo "
                                "(n=229) cuts. The ranking is not stable across them."),
                   indent=2))

    for key, label, flag in SUBSETS:
        sub = data[data["designed_any"] == flag]
        print(f"\n{label}  n={len(sub)}")
        print(f"  {OURS:34s} {sub[OURS].mean():.4f}")
        for predictor in [PREVIOUS, *BASELINES, NULL]:
            p = paired(sub, predictor)
            print(f"  {predictor:34s} {p['mean_other']:.4f}   ours {p['delta']:+.4f} "
                  f"[{p['ci_low']:+.4f}, {p['ci_high']:+.4f}]  higher on {p['ours_higher']:.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
