"""Plot the measured 4B pilot and interim eval-val R-precision points."""

import json
from pathlib import Path

import matplotlib.pyplot as plt

from build_summary import save_plot_with_meta


def main() -> None:
    here = Path(__file__).parent
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    for document_format, label, color in [
        ("contacts_v1", "contacts-v1", "#2166ac"),
        ("prompted", "Prompted", "#b35806"),
    ]:
        pilot = json.loads(
            (here / "data/eval_val_pilot" / document_format / "summary.json").read_text()
        )
        interim = json.loads(
            (
                here
                / "data/eval_val_interim_20261021"
                / document_format
                / "summary.json"
            ).read_text()
        )
        for ax, contact_range in zip(axes, ["all", "long"], strict=True):
            ax.plot(
                [0, interim["tokens"] / 1e9],
                [pilot["r_precision"][contact_range], interim["r_precision"][contact_range]],
                marker="o",
                color=color,
                label=label,
            )
            ax.set_title(f"{contact_range.capitalize()}-range contacts")
            ax.set_xlabel("Additional full-corpus tokens (billions)")
            ax.set_ylabel("Mean per-protein R-precision")
            ax.set_ylim(bottom=0, top=0.18)
            ax.grid(alpha=0.2)
    axes[0].legend(frameon=False)
    fig.suptitle("Qwen3.5-4B: 97 eval-val proteins × 100 completed rollouts")
    fig.tight_layout()
    save_plot_with_meta(
        fig,
        here / "plots/eval_val_progress.png",
        caption=(
            "Measured checkpoints only; zero additional tokens is the corresponding "
            "1B-token AFDB pilot. Interim checkpoints use unequal native-token budgets "
            "and protein exposure. Long-range scores are tied within approximately "
            "0.005 evaluation resolution. Neither continuation is near its 100B target."
        ),
        dpi=160,
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
