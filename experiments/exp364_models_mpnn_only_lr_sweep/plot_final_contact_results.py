"""Compare completed redesign-only checkpoints with the full-data default."""

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
TRIALS = (
    ("lr5e-4", "5e-4: two epochs", "#0072B2"),
    ("lr1e-3", "1e-3: two epochs", "#56B4E9"),
    ("lr5e-4-cooldown", "5e-4: one epoch, GPU", "#D55E00"),
    ("lr5e-4-cooldown-tpu", "5e-4: one epoch, TPU", "#E69F00"),
)


def main() -> None:
    """Plot validated final scores and publish only small analysis artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    report = json.loads((DATA / "report_manifest.json").read_text())
    if (report["completed_checkpoints"], report["requested_checkpoints"]) != (2, 2):
        raise ValueError("Both final checkpoint evaluations must be complete")
    current = pd.read_csv(DATA / "r_precision.csv")
    if set(current.step) != {373768}:
        raise ValueError("Expected the completed two-epoch endpoints")
    previous = pd.read_csv(DATA / "2026-10-29/r_precision.csv")
    selected = pd.concat(
        [current, previous[previous.trial.isin([t[0] for t in TRIALS[2:]])]],
        ignore_index=True,
    )
    references = pd.read_csv(
        HERE.parent
        / "exp277_models_single_mpnn_pilot/data/eval_rollout_v2_epochs/epoch_comparison.csv"
    )
    reference_rows = references[references.subset.isin(["eval-val", "eval-denovo"])]
    reference_rows = reference_rows.rename(
        columns={"subset": "eval_set", "epoch1": "r_precision"}
    )[["eval_set", "range", "n", "r_precision"]].assign(
        trial="full-data-default", step=266344
    )
    pd.concat([selected, reference_rows], ignore_index=True).to_csv(
        DATA / "final_comparison.csv", index=False
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for axis, eval_set, title in zip(
        axes,
        ("eval-val", "eval-denovo"),
        ("eval-val: 97 natural proteins", "eval-denovo: 19 designs"),
        strict=True,
    ):
        frame = selected[
            (selected.eval_set == eval_set) & (selected.range == "all")
        ].set_index("trial")
        for y, (trial, _, color) in enumerate(TRIALS):
            row = frame.loc[trial]
            axis.errorbar(
                row.r_precision,
                y,
                xerr=np.array(
                    [[row.r_precision - row.ci_low], [row.ci_high - row.r_precision]]
                ),
                fmt="o",
                color=color,
                capsize=4,
                markersize=7,
            )
            axis.annotate(
                f"{row.r_precision:.4f}",
                (row.r_precision, y),
                xytext=(0, 12),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )
        reference = reference_rows[
            (reference_rows.eval_set == eval_set) & (reference_rows.range == "all")
        ].iloc[0]
        axis.axvline(reference.r_precision, color="#333333", linestyle="--")
        axis.set_title(f"{title}\nFull-data default: {reference.r_precision:.4f}")
        axis.set_xlabel("All-range R-precision (higher is better)")
        axis.set_xlim(
            0.38 if eval_set == "eval-val" else 0.52,
            0.60 if eval_set == "eval-val" else 0.81,
        )
        axis.set_yticks(range(len(TRIALS)), [t[1] for t in TRIALS])
        axis.set_ylim(len(TRIALS) - 0.5, -0.7)
        axis.grid(axis="x", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Completed ProteinMPNN-only checkpoints — November 2, 2026")
    fig.text(
        0.5,
        0.02,
        "100 rollouts/protein; 95% protein-bootstrap intervals. Dashed lines: current full-data default.\n"
        "Training exposure and schedules differ. Designed-protein set has only 19 examples.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.11, 1, 0.93))
    save_plot_with_meta(
        fig,
        HERE / "plots/final_contact_r_precision.png",
        caption="Final two-epoch step 373768 evaluations versus completed one-epoch forks and the exp277 full-data default. Macro all-range R-precision and 95% protein-bootstrap intervals; eval-val and eval-denovo are separate. The default uses its prior published evaluation.",
        script="plot_final_contact_results.py",
        args=[],
        dpi=170,
    )
    fig.savefig(HERE / "plots/final_contact_r_precision.pdf", bbox_inches="tight")
    plt.close(fig)
    if args.publish:
        plan = json.loads((DATA / "plan.json").read_text())
        for name in (
            "plot_final_contact_results.py",
            "data/contact_eval/final_comparison.csv",
            "plots/final_contact_r_precision.png",
            "plots/final_contact_r_precision.pdf",
            "plots/final_contact_r_precision.png.meta.json",
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
