"""Plot all committed 4B pilot and continuation eval-val measurements."""

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
        periodic = [
            json.loads(path.read_text())
            for path in (here / "data/eval_val_periodic" / document_format).glob(
                "step-*/summary.json"
            )
        ]
        continuation = sorted([interim, *periodic], key=lambda result: result["tokens"])
        for ax, contact_range in zip(axes, ["all", "long"], strict=True):
            ax.plot(
                [0, *(result["tokens"] / 1e9 for result in continuation)],
                [
                    pilot["r_precision"][contact_range],
                    *(result["r_precision"][contact_range] for result in continuation),
                ],
                marker="o",
                color=color,
                label=label,
            )
            ax.set_title(f"{contact_range.capitalize()}-range contacts")
            ax.set_xlabel("Additional full-corpus tokens (billions)")
            ax.set_ylabel("Mean per-protein R-precision")
            ax.set_ylim(bottom=0, top=0.28)
            ax.grid(alpha=0.2)
    axes[0].legend(frameon=False)
    fig.suptitle("Qwen3.5-4B: 97 eval-val proteins × 100 completed rollouts")
    fig.tight_layout()
    save_plot_with_meta(
        fig,
        here / "plots/eval_val_progress.png",
        caption=(
            "Measured checkpoints only; zero additional tokens is the corresponding "
            "1B-token AFDB pilot. Native-token counts imply different protein exposure "
            "between formats. Differences below approximately 0.005 are evaluator ties. "
            "Neither continuation is near its 100B target."
        ),
        dpi=160,
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
