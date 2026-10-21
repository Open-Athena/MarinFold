"""Plot the frozen FoldBench complex benchmark composition."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt

from build_summary import save_plot_with_meta

HERE = Path(__file__).resolve().parent


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read CSV rows."""
    with path.open() as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    """Render selection flow and development/test composition."""
    audit = read_csv(HERE / "data/foldbench_freeze_audit.csv")
    frozen = read_csv(HERE / "data/foldbench_complex_contact_eval.csv")
    counts = [
        len(audit),
        sum("helico" not in row["freeze_status"] for row in audit),
        sum(row["freeze_status"] == "included" for row in audit),
        len(frozen),
    ]
    labels = [
        "MarinFold\npair-clean",
        "Helico fine-tuning\npair-clean",
        "Natural\nfrozen set",
        "Context-complete\ncontact set",
    ]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    bars = axes[0].bar(
        labels,
        counts,
        color=["#5577aa", "#66a07a", "#d08b49", "#8b5fa8"],
    )
    for bar, count in zip(bars, counts, strict=True):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            count + 0.5,
            str(count),
            ha="center",
            fontweight="bold",
        )
    axes[0].set_ylim(0, 39)
    axes[0].set_ylabel("FoldBench dimers")
    axes[0].set_title("Freeze decisions")
    axes[0].spines[["top", "right"]].set_visible(False)

    colors = {"dev": "#d08b49", "test": "#5577aa"}
    markers = {"homodimer": "o", "heterodimer": "^"}
    for split in ("dev", "test"):
        for complex_type in ("homodimer", "heterodimer"):
            rows = [
                row
                for row in frozen
                if row["split"] == split and row["complex_type"] == complex_type
            ]
            axes[1].scatter(
                [int(row["length"]) for row in rows],
                [int(row["n_gt"]) for row in rows],
                color=colors[split],
                marker=markers[complex_type],
                label=f"{split}, {complex_type}",
                alpha=0.8,
                edgecolor="white",
                linewidth=0.5,
                s=55,
            )
    axes[1].set_xlabel("Total residues")
    axes[1].set_ylabel("Resolved inter-chain contacts (R)")
    axes[1].set_title("Development/test composition")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    save_plot_with_meta(
        figure,
        HERE / "plots/foldbench_eval_freeze.png",
        caption=(
            "The 35 MarinFold pair-clean FoldBench dimers yield 30 natural "
            "structural targets, of which 23 complete all 100 contact rollouts "
            "inside the model's 8,192-token context. The 6/17 contact split is "
            "grouped by chain homology and balances mean total length."
        ),
    )


if __name__ == "__main__":
    main()
