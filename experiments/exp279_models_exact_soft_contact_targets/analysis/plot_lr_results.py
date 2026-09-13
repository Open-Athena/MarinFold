"""Plot the recorded ordinary validation losses of the matched LR forks."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

from experiments.exp279_models_exact_soft_contact_targets.build_summary import (
    save_plot_with_meta,
)

EXPERIMENT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Render both CE metrics from the committed W&B observations."""
    with (EXPERIMENT / "data/lr_validation.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    catalog = json.loads((EXPERIMENT / "data/lr_sweep_launch.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for axis, metric, title in zip(
        axes,
        ("eval/loss", "contact_eval/loss"),
        ("Document CE: full validation set", "Endpoint CE: fixed 128 packs"),
        strict=True,
    ):
        for trial in catalog["trials"]:
            selected = sorted(
                (
                    row
                    for row in rows
                    if row["trial"] == trial["trial"] and row["metric"] == metric
                ),
                key=lambda row: int(row["step"]),
            )
            if not selected:
                raise ValueError(f"Missing {metric} observations for {trial['trial']}")
            axis.plot(
                [int(row["step"]) + 1 - catalog["start_update"] for row in selected],
                [float(row["value"]) for row in selected],
                marker="o",
                markersize=4,
                label=f"LR {trial['learning_rate']:g}",
            )
        axis.set(title=title, xlabel="Additional optimizer updates", ylabel="CE (nats)")
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle("Exact soft targets: matched learning-rate continuations")
    save_plot_with_meta(
        fig,
        EXPERIMENT / "plots/lr_validation.png",
        caption="Ordinary one-hot validation CE, lower is better. Each panel uses "
        "a fixed evaluation set across the three branches. All start from the same "
        "full-state checkpoint; the control also includes its two-update pilot.",
        script=str(Path(__file__).resolve().relative_to(EXPERIMENT.parents[1])),
        args=[],
        dpi=180,
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
