# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot matched ordinary validation CE against the exp232 m2/p06 continuation."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from experiments.exp279_models_exact_soft_contact_targets.build_summary import (  # noqa: E402
    save_plot_with_meta,
)

EXPERIMENT = Path(__file__).resolve().parents[1]
TOKENS_PER_STEP = 128 * 8192
FORK_STEP = 116160  # the sweep stops here; the continuation holds peak LR
PHASE_STEP = 217801  # exp279 base -> recovery


def main() -> None:
    source = EXPERIMENT / "data/exp232_m2_p06_continuation_validation.csv"
    with source.open() as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No comparison rows in {source}")

    step = [int(r["global_step"]) for r in rows]
    soft = [float(r["soft_eval_loss"]) for r in rows]
    one_hot = [float(r["one_hot_eval_loss"]) for r in rows]
    delta = [float(r["soft_minus_one_hot"]) for r in rows]

    fig, (loss_axis, delta_axis) = plt.subplots(
        2, 1, figsize=(10.5, 7.4), height_ratios=(2.2, 1),
        sharex=True, layout="constrained",
    )

    for axis in (loss_axis, delta_axis):
        axis.axvline(FORK_STEP, color="#9CA3AF", linewidth=1, linestyle="--", zorder=1)
        axis.axvline(PHASE_STEP, color="#9CA3AF", linewidth=1, linestyle=":", zorder=1)
        axis.grid(alpha=0.25)

    loss_axis.plot(step, one_hot, color="#D97706", marker="o", markersize=3,
                   linewidth=2, label="exp232 m2/p06 — one-hot (continuation lineage)")
    loss_axis.plot(step, soft, color="#087E8B", marker="o", markersize=3,
                   linewidth=2, label="exp279 — exact soft contact targets")
    loss_axis.set_ylabel("Ordinary validation CE (nats)")
    loss_axis.set_title("Matched decontaminated training exposure, learning-rate matched throughout")
    loss_axis.legend(loc="upper right")
    # Both curves collapse onto each other at this scale, so the separation that
    # matters lives in the inset and in the lower panel, not here.
    for position, label in ((FORK_STEP, "sweep ends /\ncontinuation forks"),
                            (PHASE_STEP, "exp279\nbase→recovery")):
        loss_axis.text(position, 0.17, f" {label}", transform=loss_axis.get_xaxis_transform(),
                       fontsize=8, color="#6B7280", va="top")

    zoom = [i for i, s in enumerate(step) if s >= 100000]
    inset = loss_axis.inset_axes([0.30, 0.30, 0.42, 0.40])
    inset.plot([step[i] for i in zoom], [one_hot[i] for i in zoom],
               color="#D97706", marker="o", markersize=2.5, linewidth=1.5)
    inset.plot([step[i] for i in zoom], [soft[i] for i in zoom],
               color="#087E8B", marker="o", markersize=2.5, linewidth=1.5)
    inset.axvline(PHASE_STEP, color="#9CA3AF", linewidth=1, linestyle=":")
    inset.set_title("zoom: step ≥ 100,000", fontsize=8.5)
    inset.tick_params(labelsize=7.5)
    inset.grid(alpha=0.25)

    colors = ["#087E8B" if v < 0 else "#D97706" for v in delta]
    delta_axis.axhline(0, color="#555555", linewidth=1)
    delta_axis.plot(step, delta, color="#4B5563", linewidth=1.2, zorder=2)
    delta_axis.scatter(step, delta, c=colors, s=18, zorder=3)
    delta_axis.fill_between(step, delta, 0, where=[v < 0 for v in delta],
                            color="#087E8B", alpha=0.12)
    delta_axis.set_xlabel("Global step (update)")
    delta_axis.set_ylabel("Soft − one-hot CE")
    # Two early evaluations swing far enough to flatten everything after them.
    limit = (-0.042, 0.022)
    delta_axis.set_ylim(*limit)
    off = sum(1 for v in delta if not limit[0] <= v <= limit[1])
    if off:
        delta_axis.text(0.01, 0.06, f"{off} early points off scale",
                        transform=delta_axis.transAxes, ha="left", va="bottom",
                        color="#6B7280", fontsize=8)
    delta_axis.text(0.99, 0.06, "below zero favors soft targets",
                    transform=delta_axis.transAxes, ha="right", va="bottom",
                    color="#087E8B", fontsize=9)

    tokens = loss_axis.secondary_xaxis(
        "top",
        functions=(lambda s: s * TOKENS_PER_STEP / 1e9,
                   lambda t: t * 1e9 / TOKENS_PER_STEP),
    )
    tokens.set_xlabel("Nominal training tokens (billions)")

    loss_axis.annotate(
        f"step {step[-1]:,}\nΔCE {delta[-1]:+.4f}",
        xy=(step[-1], soft[-1]), xytext=(-70, 62), textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "#087E8B"},
        fontsize=9, color="#075E66",
    )
    fig.suptitle("Exact soft targets vs the exp232 m2/p06 one-hot continuation")
    save_plot_with_meta(
        fig,
        EXPERIMENT / "plots/exp232_m2_p06_continuation_validation.png",
        caption=(
            f"Ordinary validation CE at {len(rows)} matched steps through "
            f"{step[-1]:,}, against the spliced exp232 continuation lineage at "
            "matched peak LR. Negative favors soft targets."
        ),
        script=str(Path(__file__).resolve().relative_to(EXPERIMENT.parents[1])),
        args=[],
        dpi=180,
    )
    plt.close(fig)
    print(f"Wrote plots/exp232_m2_p06_continuation_validation.png ({len(rows)} points)")


if __name__ == "__main__":
    main()
