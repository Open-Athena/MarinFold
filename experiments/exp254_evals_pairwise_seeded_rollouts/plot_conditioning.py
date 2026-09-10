# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot completed conditioning analyses without inference or invented values.

Inputs are analyze_conditioning.py's per-protein scores, paired summaries,
per-replicate prediction changes, and conclusion JSON. The turnover panel first
averages context/sampling replicates within a protein, then bootstraps paired
excess turnover relative to independent iid resampling on the same withheld
universe. Changes in predictions are distinct from improvements in accuracy.
"""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from analyze_conditioning import paired_interval
from build_summary import save_plot_with_meta

FAMILIES = (
    ("true", "True (oracle)", "#15803d"),
    ("false", "False (oracle)", "#b45309"),
    ("pred", "Predicted", "#0369a1"),
)
DOSES = (
    ("small", "10 contacts", "o", 0.14),
    ("large", "floor(L/3) contacts", "D", -0.14),
)


def one_row(frame: pd.DataFrame, **filters: str) -> pd.Series:
    """Require exactly one saved contrast instead of dropping unavailable arms."""
    selected = frame
    for column, value in filters.items():
        selected = selected[selected[column] == value]
    if len(selected) != 1:
        raise ValueError(f"expected one contrast {filters}, found {len(selected)}")
    return selected.iloc[0]


def load_inputs(data: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Load complete matching analysis outputs, failing on stale or missing rows."""
    per_protein = pd.read_csv(data / "conditioning_per_protein.csv")
    summary = pd.read_csv(data / "conditioning_summary.csv")
    changes = pd.read_csv(data / "conditioning_changes.csv")
    conclusion = json.loads((data / "conditioning_conclusion.json").read_text())
    count, repeats = conclusion["n_targets"], conclusion["n_repeats"]
    keys = ["dataset", "stem"]
    proteins = set(map(tuple, per_protein[keys].to_numpy()))
    if len(proteins) != count or not (per_protein.n_repeats == repeats).all():
        raise ValueError(
            "per-protein scores disagree with completed target/replicate counts"
        )
    if per_protein.duplicated(keys + ["scope", "arm"]).any():
        raise ValueError("duplicate per-protein score rows")
    if changes.duplicated(keys + ["repeat", "scope", "arm"]).any():
        raise ValueError("duplicate prediction-change rows")
    if not np.isfinite(summary[["mean_delta", "ci_low", "ci_high"]].to_numpy()).all():
        raise ValueError("paired summaries contain nonfinite values")
    if not (
        (summary.ci_low <= summary.mean_delta) & (summary.mean_delta <= summary.ci_high)
    ).all():
        raise ValueError("paired interval does not bracket its point estimate")
    if not (summary.n_proteins == count).all():
        raise ValueError("summary contrasts have inconsistent protein counts")
    primary = one_row(summary, role="practical_primary")
    if (
        primary.arm != "pred_large"
        or primary.reference != "source_plus_iid200"
        or primary.scope != "full_pipeline"
    ):
        raise ValueError("unexpected practical primary contrast")
    for field in ("mean_delta", "ci_low", "ci_high", "practical_margin"):
        if not np.isclose(
            primary[field], conclusion["primary"][field], rtol=0, atol=1e-12
        ):
            raise ValueError(f"conclusion JSON and summary disagree on {field}")
    # Verify displayed effects against the separately saved protein-level means.
    displayed = [
        ("full_pipeline", f"pred_{dose}", "source_plus_iid200") for dose, *_ in DOSES
    ]
    displayed += [
        ("withheld_continuation", f"{family}_{dose}", "iid")
        for family, *_ in FAMILIES
        for dose, *_ in DOSES
    ]
    for scope, arm, reference in displayed:
        row = one_row(summary, scope=scope, arm=arm, reference=reference)
        scores = per_protein[per_protein.scope == scope].pivot(
            index=keys, columns="arm", values="precision"
        )
        if set(scores.index) != proteins or scores[[arm, reference]].isna().any().any():
            raise ValueError(f"incomplete matched proteins for {scope}/{arm}")
        difference = float((scores[arm] - scores[reference]).mean())
        if not np.isclose(difference, row.mean_delta, rtol=0, atol=1e-12):
            raise ValueError(f"summary and protein scores disagree for {scope}/{arm}")
    return per_protein, summary, changes, conclusion


def turnover_contrasts(
    changes: pd.DataFrame, per_protein: pd.DataFrame, n_repeats: int
) -> pd.DataFrame:
    """Compute protein-bootstrap excess map turnover over iid-repeat sampling."""
    keys = ["dataset", "stem"]
    expected_proteins = set(map(tuple, per_protein[keys].to_numpy()))
    required = ["iid_repeat"] + [
        f"{family}_{dose}" for family, *_ in FAMILIES for dose, *_ in DOSES
    ]
    selected = changes[
        (changes.scope == "withheld_continuation") & changes.arm.isin(required)
    ]
    if not (selected.reference == "iid").all():
        raise ValueError(
            "turnover input must compare each map with the matched iid map"
        )
    if (
        not np.isfinite(selected.top_R_turnover).all()
        or not selected.top_R_turnover.between(0, 1).all()
    ):
        raise ValueError("invalid map-turnover values")
    for arm in required:
        rows = selected[selected.arm == arm]
        if set(map(tuple, rows[keys].to_numpy())) != expected_proteins:
            raise ValueError(f"turnover is missing matched proteins for {arm}")
        for _, group in rows.groupby(keys):
            if set(group["repeat"]) != set(range(n_repeats)) or len(group) != n_repeats:
                raise ValueError(f"turnover is missing context replicates for {arm}")
    averaged = selected.groupby(keys + ["arm"]).top_R_turnover.mean().unstack("arm")
    rows = []
    for arm in required[1:]:
        interval = paired_interval((averaged[arm] - averaged.iid_repeat).to_numpy())
        rows.append(
            dict(
                arm=arm,
                reference="iid_repeat",
                **interval,
                arm_turnover=float(averaged[arm].mean()),
                iid_repeat_turnover=float(averaged.iid_repeat.mean()),
            )
        )
    return pd.DataFrame(rows)


def point(
    ax: plt.Axes,
    row: pd.Series,
    y: float,
    color: str,
    marker: str,
    emphasize: bool = False,
) -> None:
    """Draw a saved or recomputed paired interval in percentage points."""
    mean, lo, hi = row[["mean_delta", "ci_low", "ci_high"]].to_numpy(dtype=float) * 100
    ax.errorbar(
        mean,
        y,
        xerr=[[mean - lo], [hi - mean]],
        fmt=marker,
        markersize=9 if emphasize else 7,
        color=color,
        capsize=4,
        elinewidth=2,
        markeredgecolor="white",
        markeredgewidth=0.8,
        zorder=4,
    )


def style_axis(
    ax: plt.Axes, frame: pd.DataFrame, extra: tuple[float, ...] = ()
) -> None:
    """Choose bounds from actual intervals and retain visible reference lines."""
    lower = min(0.0, float(frame.ci_low.min()) * 100, *extra)
    upper = max(0.0, float(frame.ci_high.max()) * 100, *extra)
    spread = max(upper - lower, 0.5)
    ax.set_xlim(lower - spread * 0.12, upper + spread * 0.12)
    ax.axvline(0, color="#64748b", linewidth=1.1, zorder=1)
    ax.grid(axis="x", color="#e2e8f0", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0, pad=9)
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw(
    summary: pd.DataFrame,
    turnover: pd.DataFrame,
    conclusion: dict,
    draft_label: str | None,
) -> plt.Figure:
    """Create practical accuracy, mechanistic accuracy, and response panels."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 6.4))
    fig.subplots_adjust(left=0.09, right=0.975, bottom=0.25, top=0.72, wspace=0.65)
    practical = summary[
        (summary.scope == "full_pipeline")
        & (summary.reference == "source_plus_iid200")
        & summary.arm.isin(["pred_small", "pred_large"])
    ]
    margin = float(conclusion["primary"]["practical_margin"]) * 100
    for y, (dose, label, marker, _) in zip((1, 0), DOSES):
        row = one_row(practical, arm=f"pred_{dose}")
        point(axes[0], row, y, "#0369a1", marker, emphasize=dose == "large")
        axes[0].text(
            0.5,
            y - 0.22,
            f"{row.mean_delta * 100:+.2f} [{row.ci_low * 100:+.2f}, {row.ci_high * 100:+.2f}] pp",
            transform=axes[0].get_yaxis_transform(),
            ha="center",
            va="top",
            fontsize=9,
            color="#475569",
        )
    axes[0].set_yticks(
        [1, 0], ["Predicted\n10 contacts", "Predicted\nfloor(L/3)\nPRIMARY"]
    )
    axes[0].set_ylim(-0.48, 1.42)
    style_axis(axes[0], practical, extra=(margin,))
    axes[0].axvline(margin, color="#be123c", linestyle="--", linewidth=1.6)
    axes[0].text(
        margin,
        1.3,
        f"+{margin:g} pp target",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#be123c",
        bbox=dict(facecolor="white", edgecolor="none", pad=2),
    )
    axes[0].set_title(
        "A  Practical decoding\nFull map vs source + fresh iid (200)",
        fontsize=12,
        fontweight="bold",
        loc="left",
        pad=16,
    )
    axes[0].set_xlabel("R-precision difference (pp)", labelpad=12)
    mechanistic = summary[
        (summary.scope == "withheld_continuation")
        & (summary.reference == "iid")
        & summary.arm.isin(turnover.arm)
    ]
    for y, (family, label, color) in zip((2, 1, 0), FAMILIES):
        for dose, _, marker, offset in DOSES:
            point(
                axes[1],
                one_row(mechanistic, arm=f"{family}_{dose}"),
                y + offset,
                color,
                marker,
            )
            point(
                axes[2],
                one_row(turnover, arm=f"{family}_{dose}"),
                y + offset,
                color,
                marker,
            )
    for ax in axes[1:]:
        ax.set_yticks([2, 1, 0], [family[1] for family in FAMILIES])
        ax.set_ylim(-0.5, 2.5)
    style_axis(axes[1], mechanistic)
    style_axis(axes[2], turnover)
    axes[1].set_title(
        "B  Conditioning diagnostic\nShared withheld pairs vs fresh iid",
        fontsize=12,
        fontweight="bold",
        loc="left",
        pad=16,
    )
    axes[1].set_xlabel("Remaining-contact R-precision difference (pp)", labelpad=12)
    axes[2].set_title(
        "C  Changed predictions\nExcess turnover vs independent iid repeat",
        fontsize=12,
        fontweight="bold",
        loc="left",
        pad=16,
    )
    axes[2].set_xlabel("Additional top-R map turnover (pp)", labelpad=12)
    handles = [
        Line2D(
            [],
            [],
            color="#475569",
            marker=marker,
            linestyle="None",
            markersize=7,
            label=label,
        )
        for _, label, marker, _ in DOSES
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.62, 0.105),
        ncol=2,
        frameon=False,
        fontsize=10,
    )
    fig.text(
        0.035,
        0.955,
        "Multi-contact conditioning: accuracy and response",
        fontsize=20,
        fontweight="bold",
        va="top",
    )
    fig.text(
        0.035,
        0.875,
        f"{conclusion['n_targets']} eval-val proteins · {conclusion['n_repeats']} replicates averaged per protein · "
        "paired protein-bootstrap 95% intervals",
        fontsize=11,
        color="#475569",
    )
    if draft_label:
        fig.text(
            0.975,
            0.995,
            draft_label,
            ha="right",
            va="top",
            color="#be123c",
            fontsize=12,
            fontweight="bold",
        )
    fig.text(
        0.035,
        0.07,
        "A is the practical test. B uses a common context-excluded universe; true/false contexts use ground truth. "
        "C measures response, not accuracy; zero is iid-repeat turnover.",
        fontsize=9,
        color="#475569",
    )
    fig.text(
        0.035,
        0.035,
        "Intervals condition on the saved first pass and inference draws; exploratory comparisons are unadjusted. "
        "Panel scales differ. Equal rollout counts do not imply equal compute.",
        fontsize=9,
        color="#475569",
    )
    return fig


def main() -> int:
    """Write PNG and standalone vector PDF with exact generation provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--draft-label",
        help="Optional visible label for layout validation or a draft artifact",
    )
    args = parser.parse_args()
    per_protein, summary, changes, conclusion = load_inputs(args.data)
    turnover = turnover_contrasts(changes, per_protein, conclusion["n_repeats"])
    fig = draw(summary, turnover, conclusion, args.draft_label)
    args.out.mkdir(parents=True, exist_ok=True)
    caption = (
        "Multi-contact conditioning on eval-val. Practical full-map effects compare "
        "predicted contexts with the shared archived source plus fresh iid rollout votes. "
        "Mechanistic effects and excess map turnover share the context-excluded universe. "
        "Replicates are averaged within protein before bootstrap; turnover does not measure accuracy."
    )
    for extension in ("png", "pdf"):
        save_plot_with_meta(
            fig,
            args.out / f"conditioning_intervention.{extension}",
            caption=caption,
            dpi=180,
        )
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
