# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Summarise the local pilot sweep and draw its figures.

Reads the per-run JSONs written by ``train_pilot.py`` and produces:

* ``data/pilot_stage1_learning_rates.csv`` — the per-arm learning-rate search.
* ``data/pilot_results.csv`` — every stage-2 run's final metrics.
* ``data/pilot_summary.csv`` — arm means with the across-seed spread, and each
  arm's gap to the control measured against that spread. An architecture change
  that does not clear the seed noise has not been shown to do anything.
* ``plots/pilot_loss_curves.png`` and ``plots/pilot_arm_comparison.png``.
"""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arms import ARMS  # noqa: E402
from build_summary import save_plot_with_meta  # noqa: E402

CONTROL = "a-rope"
COLORS = {"a-rope": "#4a5568", "b-rope-smear": "#2b6cb0", "c-nope-smear": "#dd6b20", "d-nope": "#e53e3e"}


def load_runs(directory: Path) -> list[dict]:
    return [json.loads(path.read_text()) for path in sorted(directory.glob("*.json"))]


def stage1_table(directory: Path) -> pd.DataFrame:
    rows = [
        {
            "arm": run["arm"]["key"],
            "learning_rate": run["config"]["learning_rate"],
            "val_nll": run["final"]["val_nll"],
            "tokens": run["config"]["tokens"],
        }
        for run in load_runs(directory)
    ]
    return pd.DataFrame(rows).sort_values(["arm", "learning_rate"])


def stage2_table(directory: Path) -> pd.DataFrame:
    rows = []
    for run in load_runs(directory):
        row = {
            "arm": run["arm"]["key"],
            "label": run["arm"]["label"],
            "seed": run["config"]["seed"],
            "learning_rate": run["config"]["learning_rate"],
            "parameters": run["parameters"],
            "tokens": run["config"]["tokens"],
        }
        row.update({key: value for key, value in run["final"].items() if key.endswith(("_nll", "_early"))})
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["arm", "seed"])


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    """Arm means, the across-seed spread, and each arm's gap to the control.

    The gap is reported in units of the pooled seed standard deviation as well
    as in nats: at pilot scale the seed spread *is* the resolution, and a change
    smaller than it has not been demonstrated.
    """
    metrics = [column for column in results.columns if column.endswith(("_nll", "_early"))]
    grouped = results.groupby("arm")[metrics]
    summary = grouped.mean().add_suffix("_mean").join(grouped.std().add_suffix("_sd"))
    summary["seeds"] = results.groupby("arm").size()
    control = summary.loc[CONTROL]
    pooled = float(np.sqrt(np.nanmean(np.square(summary["val_nll_sd"].to_numpy(dtype=float)))))
    summary["val_nll_delta"] = summary["val_nll_mean"] - control["val_nll_mean"]
    summary["structure_nll_delta"] = summary["structure_nll_mean"] - control["structure_nll_mean"]
    summary["sequence_nll_delta"] = summary["sequence_nll_mean"] - control["sequence_nll_mean"]
    summary["val_nll_delta_in_sds"] = summary["val_nll_delta"] / pooled if pooled else np.nan
    summary["pooled_seed_sd"] = pooled
    order = [arm.key for arm in ARMS if arm.key in summary.index]
    return summary.loc[order]


def plot_curves(stage2: Path, plots_dir: Path) -> None:
    figure, (loss, guard) = plt.subplots(1, 2, figsize=(13, 4.6))
    for run in load_runs(stage2):
        key = run["arm"]["key"]
        history = pd.DataFrame(run["history"])
        alpha = 0.35 if run["config"]["seed"] else 1.0
        loss.plot(history.tokens / 1e6, history.val_nll, color=COLORS[key], alpha=alpha,
                  label=run["arm"]["label"] if run["config"]["seed"] == 0 else None)
        guard.plot(history.tokens / 1e6, history.p_end_early, color=COLORS[key], alpha=alpha,
                   label=run["arm"]["label"] if run["config"]["seed"] == 0 else None)
    loss.set_xlabel("training tokens (M)")
    loss.set_ylabel("validation NLL (nats/token)")
    loss.set_title("pilot validation loss (solid = seed 0, faint = replicates)")
    loss.legend(fontsize=8)
    loss.grid(alpha=0.3)
    guard.set_xlabel("training tokens (M)")
    guard.set_ylabel("mean P(<end>) where the document is not over")
    guard.set_yscale("log")
    guard.set_title("over-eagerness to stop — the counting guardrail")
    guard.grid(alpha=0.3)
    figure.tight_layout()
    save_plot_with_meta(
        figure,
        plots_dir / "pilot_loss_curves.png",
        caption=(
            "Local pilot: 15M-parameter twins of the exp232 Qwen3 on real decontaminated "
            "contacts-v1, one line per seed. Right panel is the guardrail Phase 0 asked for — "
            "the probability the model puts on <end> when the document is not finished, which "
            "is the counting job NoPE takes away."
        ),
        dpi=150,
    )
    plt.close(figure)


def plot_comparison(results: pd.DataFrame, summary: pd.DataFrame, plots_dir: Path) -> None:
    figure, (final, sections) = plt.subplots(1, 2, figsize=(13, 4.6))
    order = list(summary.index)
    x = np.arange(len(order))
    for index, arm in enumerate(order):
        seeds = results[results.arm == arm]
        final.scatter([index] * len(seeds), seeds.val_nll, color=COLORS[arm], zorder=3, s=28)
    final.errorbar(
        x, summary["val_nll_mean"], yerr=summary["val_nll_sd"], fmt="_",
        color="black", markersize=22, capsize=5, zorder=2,
    )
    final.set_xticks(x, order, rotation=15, fontsize=9)
    final.set_ylabel("final validation NLL (nats/token)")
    final.set_title("final loss by arm (points = seeds)")
    final.grid(axis="y", alpha=0.3)

    width = 0.38
    sections.bar(x - width / 2, summary["sequence_nll_delta"], width, label="sequence section", color="#63b3ed")
    sections.bar(x + width / 2, summary["structure_nll_delta"], width, label="structure section", color="#2b6cb0")
    sections.axhline(0, color="black", linewidth=0.8)
    pooled = float(summary["pooled_seed_sd"].iloc[0])
    sections.axhspan(-pooled, pooled, color="gray", alpha=0.2, label="pooled seed sd")
    sections.set_xticks(x, order, rotation=15, fontsize=9)
    sections.set_ylabel("Δ NLL vs control (nats/token)")
    sections.set_title("where each arm's change lands, by document section")
    sections.legend(fontsize=8)
    sections.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    save_plot_with_meta(
        figure,
        plots_dir / "pilot_arm_comparison.png",
        caption=(
            "Left: final validation loss per arm, one point per seed, with the across-seed mean "
            "and spread. Right: each arm's gap to the control split by document section, against "
            "the pooled seed spread — the band is the pilot's resolution, and a bar inside it "
            "shows nothing."
        ),
        dpi=150,
    )
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", type=Path, default=Path("data/pilot"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--plots-dir", type=Path, default=Path("plots"))
    arguments = parser.parse_args()
    arguments.plots_dir.mkdir(parents=True, exist_ok=True)

    stage1 = arguments.pilot_dir / "stage1"
    stage2 = arguments.pilot_dir / "stage2"

    if stage1.is_dir() and any(stage1.glob("*.json")):
        table = stage1_table(stage1)
        table.to_csv(arguments.data_dir / "pilot_stage1_learning_rates.csv", index=False)
        print("[analyze] stage 1 — learning-rate search")
        print(table.pivot(index="arm", columns="learning_rate", values="val_nll").to_string())

    if not (stage2.is_dir() and any(stage2.glob("*.json"))):
        print("[analyze] no stage-2 runs yet")
        return

    results = stage2_table(stage2)
    results.to_csv(arguments.data_dir / "pilot_results.csv", index=False)
    summary = summarise(results)
    summary.to_csv(arguments.data_dir / "pilot_summary.csv")
    print("\n[analyze] stage 2 — arms at their own best learning rate")
    columns = [
        "seeds", "val_nll_mean", "val_nll_sd", "val_nll_delta", "val_nll_delta_in_sds",
        "structure_nll_delta", "sequence_nll_delta", "end_nll_mean", "p_end_early_mean",
    ]
    print(summary[columns].to_string(float_format=lambda value: f"{value:9.4f}"))

    plot_curves(stage2, arguments.plots_dir)
    plot_comparison(results, summary, arguments.plots_dir)
    print(f"\n[analyze] wrote plots to {arguments.plots_dir}")


if __name__ == "__main__":
    main()
