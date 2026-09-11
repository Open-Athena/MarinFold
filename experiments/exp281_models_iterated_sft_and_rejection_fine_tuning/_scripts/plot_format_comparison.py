"""Compare the pilot and extended warm-up using their saved format-gate audits."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from experiments.exp281_models_iterated_sft_and_rejection_fine_tuning.build_summary import (
    save_plot_with_meta,
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    modes = ["natural", "forced"]
    for offset, slug, label, color in [(-0.18, "trial_s03", "Step 256", "#396ea4"),
                                       (0.18, "format_s02", "Step 2,000", "#b75638")]:
        gate = json.loads((root / f"data/{slug}_format_gate.json").read_text())
        values = [100 * gate["modes"][mode]["valid_fraction"] for mode in modes]
        bars = ax.bar([i + offset for i in range(2)], values, width=0.35, color=color, label=label)
        for bar, mode in zip(bars, modes, strict=True):
            record = gate["modes"][mode]
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                    f"{record['valid']}/{record['completions']}", ha="center", fontsize=10)
    ax.axhline(99, color="#777777", linestyle="--", linewidth=1, label="99% validity gate")
    ax.set(xticks=[0, 1], xticklabels=["Natural finalization", "Forced finalization"],
           ylabel="Valid completions (%)", ylim=(0, 110),
           title="exp281 · format validity after extended warm-up")
    ax.legend(frameon=False, loc="center right")
    save_plot_with_meta(
        fig, root / "plots/format_comparison.png",
        script=("uv run --no-project --with matplotlib python -m "
                "experiments.exp281_models_iterated_sft_and_rejection_fine_tuning._scripts.plot_format_comparison"),
        args=[],
        caption=("Eight unselected completions per mode on the same 25 held-out proteins, "
                 "with the same seeds and forced budgets. Invalid outputs remain in the denominator. "
                 "This is a format gate, not a FoldBench accuracy evaluation."),
        dpi=160,
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
