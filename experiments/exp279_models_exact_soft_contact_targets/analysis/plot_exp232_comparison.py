"""Plot matched ordinary validation CE for exp279 and exp232 m2/p06."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt

from experiments.exp279_models_exact_soft_contact_targets.build_summary import (
    save_plot_with_meta,
)


EXPERIMENT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Render the committed matched-step comparison and its signed difference."""
    source = EXPERIMENT / "data/exp232_m2_p06_matched_validation.csv"
    with source.open() as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No comparison rows in {source}")

    tokens = [float(row["nominal_tokens"]) / 1e9 for row in rows]
    soft = [float(row["soft_eval_loss"]) for row in rows]
    one_hot = [float(row["one_hot_eval_loss"]) for row in rows]
    difference = [float(row["soft_minus_one_hot"]) for row in rows]

    fig, (loss_axis, delta_axis) = plt.subplots(
        2,
        1,
        figsize=(10, 7),
        height_ratios=(2.2, 1),
        sharex=True,
        layout="constrained",
    )
    loss_axis.plot(
        tokens,
        one_hot,
        color="#D97706",
        marker="o",
        markersize=3.5,
        linewidth=2,
        label="exp232 m2/p06 — one-hot",
    )
    loss_axis.plot(
        tokens,
        soft,
        color="#087E8B",
        marker="o",
        markersize=3.5,
        linewidth=2,
        label="exp279 — exact soft contact targets",
    )
    loss_axis.set_ylabel("Ordinary validation CE (nats)")
    loss_axis.set_title("Matched decontaminated training exposure")
    loss_axis.grid(alpha=0.25)
    loss_axis.legend()

    colors = ["#087E8B" if value < 0 else "#D97706" for value in difference]
    delta_axis.axhline(0, color="#555555", linewidth=1)
    delta_axis.scatter(tokens, difference, c=colors, s=20, zorder=3)
    delta_axis.plot(tokens, difference, color="#4B5563", linewidth=1.2)
    delta_axis.fill_between(
        tokens,
        difference,
        0,
        where=[value < 0 for value in difference],
        color="#087E8B",
        alpha=0.12,
    )
    delta_axis.set_xlabel("Nominal training tokens (billions)")
    delta_axis.set_ylabel("Soft − one-hot CE")
    delta_axis.grid(alpha=0.25)
    delta_axis.text(
        0.99,
        0.08,
        "below zero favors soft targets",
        transform=delta_axis.transAxes,
        ha="right",
        va="bottom",
        color="#087E8B",
        fontsize=9,
    )

    latest_step = int(rows[-1]["global_step"])
    latest_delta = difference[-1]
    loss_axis.annotate(
        f"step {latest_step:,}\nΔCE {latest_delta:+.4f}",
        xy=(tokens[-1], soft[-1]),
        xytext=(-94, 35),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "#087E8B"},
        fontsize=9,
        color="#075E66",
    )
    fig.suptitle("Exact soft targets vs exp232 m2/p06")
    save_plot_with_meta(
        fig,
        EXPERIMENT / "plots/exp232_m2_p06_matched_validation.png",
        caption=(
            "Ordinary one-hot validation CE on the same validation cache at 41 "
            "exactly matched global steps. Both runs use the same decontaminated "
            "native-data mixture and effective LR through this window. The lower "
            "panel is soft minus one-hot, so negative values favor soft targets."
        ),
        script=str(Path(__file__).resolve().relative_to(EXPERIMENT.parents[1])),
        args=[],
        dpi=180,
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
