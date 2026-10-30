"""Plot the five October 29 contact evaluations against their earlier snapshots."""

import argparse
import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from build_summary import save_plot_with_meta

HERE = Path(__file__).resolve().parent
DATA = HERE / "data/contact_eval"
TRIALS = [
    ("lr5e-4-cooldown", "5e-4: GPU one-epoch final"),
    ("lr5e-4-cooldown-tpu", "5e-4: TPU one-epoch final"),
    ("lr5e-4", "5e-4: two-epoch run, step 282830"),
    ("lr1e-3", "1e-3: two-epoch run, step 314987"),
    ("lr2e-3", "2e-3: two-epoch run, step 207662"),
]


def paired_snapshot_changes() -> None:
    """Keep checkpoint improvements paired on the same proteins and contact ranges."""
    rng = np.random.default_rng(20261029)
    rows = []
    for trial, _ in TRIALS:
        old_path = DATA / "2026-10-21/results" / trial / "contact_precision_all.csv"
        if not old_path.exists():
            continue
        old = pd.read_csv(old_path)
        new = pd.read_csv(DATA / "results" / trial / "contact_precision_all.csv")
        old = old[(old.cut == "R") & old.range.isin(["all", "long"])]
        new = new[(new.cut == "R") & new.range.isin(["all", "long"])]
        joined = old.merge(
            new,
            on=["dataset", "stem", "range", "cut", "eval_set"],
            validate="one_to_one",
            suffixes=("_old", "_new"),
        )
        if len(joined) != 232 or len(old) != 232 or len(new) != 232:
            raise ValueError(
                "Paired comparisons must cover all 116 proteins and both ranges"
            )
        for (eval_set, contact_range), group in joined.groupby(["eval_set", "range"]):
            delta = (group.precision_new - group.precision_old).to_numpy()
            means = delta[rng.integers(len(delta), size=(10000, len(delta)))].mean(
                axis=1
            )
            low, high = np.quantile(means, [0.025, 0.975])
            rows.append(
                {
                    "trial": trial,
                    "eval_set": eval_set,
                    "range": contact_range,
                    "n": len(delta),
                    "delta_oct29_minus_oct21": delta.mean(),
                    "paired_ci_low": low,
                    "paired_ci_high": high,
                }
            )
    pd.DataFrame(rows).to_csv(DATA / "snapshot_changes.csv", index=False)


def main() -> None:
    """Use complete, independently validated reports; never infer missing scores."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    report = json.loads((DATA / "report_manifest.json").read_text())
    if report["completed_checkpoints"] != 5 or report["requested_checkpoints"] != 5:
        raise ValueError("All five checkpoint evaluations must be complete")
    paired_snapshot_changes()
    current = pd.read_csv(DATA / "r_precision.csv")
    previous = pd.read_csv(DATA / "2026-10-21/r_precision.csv")
    references = pd.read_csv(
        HERE.parent
        / "exp277_models_single_mpnn_pilot/data/eval_rollout_v2_epochs/epoch_comparison.csv"
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), sharey=True)
    for axis, eval_set, title in zip(
        axes,
        ("eval-val", "eval-denovo"),
        ("eval-val: 97 natural proteins", "eval-denovo: 19 designs"),
        strict=True,
    ):
        selected = current[
            (current.eval_set == eval_set) & (current.range == "all")
        ].set_index("trial")
        old = previous[
            (previous.eval_set == eval_set) & (previous.range == "all")
        ].set_index("trial")
        for y, (trial, _) in enumerate(TRIALS):
            row = selected.loc[trial]
            axis.errorbar(
                row.r_precision,
                y,
                xerr=np.array(
                    [[row.r_precision - row.ci_low], [row.ci_high - row.r_precision]]
                ),
                fmt="o",
                color="#0072B2",
                capsize=3,
                label="October 29, 95% protein-bootstrap CI" if y == 0 else None,
            )
            if trial in old.index:
                axis.plot(
                    old.loc[trial, "r_precision"],
                    y,
                    "o",
                    markerfacecolor="white",
                    markeredgecolor="#777777",
                    label="October 21 snapshot" if y == 0 else None,
                )
        reference = references[
            (references.subset == eval_set) & (references.range == "all")
        ].iloc[0]
        axis.axvline(
            reference.epoch1,
            color="#222222",
            linestyle="--",
            label="Completed full-data first epoch",
        )
        axis.set(
            title=title,
            xlabel="All-range R-precision (higher is better)",
            xlim=(0.2, 0.8),
        )
        axis.set_yticks(range(len(TRIALS)), [label for _, label in TRIALS])
        axis.grid(axis="x", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=1,
        frameon=False,
    )
    fig.text(
        0.5,
        0.025,
        "100 rollouts/protein; identical CUDA scoring recipe. Training exposure and schedules differ.\n"
        "Error bars describe protein sampling uncertainty; differences below roughly 0.005 are treated as ties.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.83))
    save_plot_with_meta(
        fig,
        HERE / "plots/contact_r_precision.png",
        caption="October 29 all-range R-precision with 95% protein-bootstrap intervals, October 21 snapshots, and the completed full-data first epoch. Natural proteins and designs are separate. Training exposure and schedules differ; all five evaluations use the same CUDA rollout recipe.",
        script="plot_contact_results.py",
        args=[],
        dpi=170,
    )
    fig.savefig(HERE / "plots/contact_r_precision.pdf", bbox_inches="tight")
    plt.close(fig)
    if args.publish:
        plan = json.loads((DATA / "plan.json").read_text())
        for name in (
            "plots/contact_r_precision.png",
            "plots/contact_r_precision.pdf",
            "plots/contact_r_precision.png.meta.json",
            "plot_contact_results.py",
            "data/contact_eval/snapshot_changes.csv",
        ):
            subprocess.run(
                [
                    "hf",
                    "buckets",
                    "cp",
                    str(HERE / name),
                    plan["artifact_root"] + "/reports/" + name,
                ],
                check=True,
            )


if __name__ == "__main__":
    main()
