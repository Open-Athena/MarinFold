"""Plot the contact and Helico results for the frozen complex benchmark."""

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
    """Render the primary contact result, budget selection and test comparison."""
    contact = {
        row["split"]: row
        for row in read_csv(HERE / "data/contact_r_precision_summary.csv")
    }
    structural = read_csv(HERE / "data/helico_summary.csv")
    dev = {
        row["arm"]: row
        for row in structural
        if row["split"] == "dev" and row["arm"].startswith("marinfold_all_")
    }
    test = {
        row["arm"]: row for row in structural if row["split"] == "test"
    }

    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.5))
    test_contact = contact["test"]
    values = [
        float(test_contact["mean_random_r_precision"]),
        float(test_contact["mean_r_precision"]),
    ]
    bars = axes[0].bar(
        ["random", "MarinFold"], values, color=["#b8b8b8", "#5577aa"]
    )
    low = float(test_contact["group_bootstrap_95_low"])
    high = float(test_contact["group_bootstrap_95_high"])
    axes[0].errorbar(
        [1],
        [values[1]],
        yerr=[[values[1] - low], [high - values[1]]],
        color="black",
        capsize=4,
        linewidth=1.2,
    )
    for bar, value in zip(bars, values, strict=True):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.001,
            f"{value:.3f}",
            ha="center",
            fontsize=9,
        )
    axes[0].set_title("Test inter-chain contacts")
    axes[0].set_ylabel("R-precision")
    axes[0].set_ylim(0, max(0.055, high * 1.2))

    dev_order = ["marinfold_all_L5", "marinfold_all_L2", "marinfold_all_L"]
    dev_labels = ["L/5", "L/2", "L"]
    dev_values = [float(dev[arm]["mean_dockq"]) for arm in dev_order]
    dev_bars = axes[1].bar(
        dev_labels,
        dev_values,
        color=["#9ab2d1", "#7898c2", "#5577aa"],
    )
    for bar, value in zip(dev_bars, dev_values, strict=True):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.004,
            f"{value:.3f}",
            ha="center",
            fontsize=9,
        )
    axes[1].set_title("Dev contact budget")
    axes[1].set_ylabel("Mean pair-specific DockQ")
    axes[1].set_ylim(0, max(dev_values) * 1.25)

    test_order = ["off", "marinfold_intra_L", "marinfold_all_L", "oracle"]
    test_labels = ["off", "intra L", "all L", "oracle"]
    test_values = [float(test[arm]["mean_dockq"]) for arm in test_order]
    test_low = [float(test[arm]["group_bootstrap_95_low"]) for arm in test_order]
    test_high = [float(test[arm]["group_bootstrap_95_high"]) for arm in test_order]
    test_bars = axes[2].bar(
        test_labels,
        test_values,
        color=["#b8b8b8", "#d08b49", "#5577aa", "#66a07a"],
    )
    axes[2].errorbar(
        range(len(test_order)),
        test_values,
        yerr=[
            [value - low for value, low in zip(test_values, test_low, strict=True)],
            [high - value for value, high in zip(test_values, test_high, strict=True)],
        ],
        fmt="none",
        color="black",
        capsize=3,
        linewidth=1,
    )
    axes[2].axhline(0.23, color="#555", linestyle="--", linewidth=1)
    axes[2].text(3.45, 0.235, "acceptable", ha="right", va="bottom", fontsize=8)
    for bar, arm, value in zip(test_bars, test_order, test_values, strict=True):
        success = float(test[arm]["acceptable_or_better_rate"])
        axes[2].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.008,
            f"{value:.3f}\n{success:.0%}",
            ha="center",
            fontsize=8,
        )
    axes[2].set_title("Blinded structural test")
    axes[2].set_ylabel("Mean pair-specific DockQ")
    axes[2].set_ylim(0, max(0.30, max(test_high) * 1.18))

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    save_plot_with_meta(
        figure,
        HERE / "plots/eval_results.png",
        caption=(
            "The contact model beats the resolved-pair random baseline on the "
            "17-target test split. Helico contact budgets are selected only on "
            "the six-target development split; the test bars use confidence-selected "
            "samples and exact pair-specific, symmetry-aware DockQ. Error bars are "
            "95% homology-group bootstrap intervals."
        ),
    )


if __name__ == "__main__":
    main()
