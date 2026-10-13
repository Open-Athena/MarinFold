"""Save full validation curves and compare depth at matched training exposure."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import wandb

from build_summary import save_plot_with_meta

HERE = Path(__file__).resolve().parent
BASELINE_RUN = "contacts-v1-exp277-m2-p06-full-epoch-1.5B"
FINAL_STEP = 266344
RUNS = (
    ("255M / 4 layers", 254941696, "contacts-v1-exp362-depth4-full-epoch"),
    ("498M / 8 layers", 498228224, "contacts-v1-exp362-depth8-full-epoch"),
    ("985M / 16 layers", 984801280, "contacts-v1-exp362-depth16-full-epoch"),
    ("1.47B / 24 layers", 1471374336, BASELINE_RUN),
)


def main() -> None:
    """Fetch complete validation history, preserving unfinished-run status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    api = wandb.Api(timeout=60)
    available = {
        run.id: run for run in api.runs(
            "open-athena/MarinFold", filters={"$or": [
                {"group": "exp362-smaller-model-scaling"}, {"name": BASELINE_RUN},
            ]},
        )
    }
    curves = []
    summaries = []
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, parameters, run_id in RUNS:
        if run_id not in available:
            if args.require_complete:
                raise ValueError(f"Production run not registered: {run_id}")
            continue
        run = available[run_id]
        progress = float(run.summary.get("run_progress", 0))
        rows = [
            {"run_id": run_id, "parameters": parameters, "step": int(row["_step"]), "eval_loss": float(row["eval/loss"])}
            for row in run.scan_history(keys=["_step", "eval/loss"])
        ]
        rows.sort(key=lambda row: row["step"])
        final = [row["eval_loss"] for row in rows if row["step"] == FINAL_STEP]
        if args.require_complete and (progress < 1 or len(final) != 1):
            raise ValueError(f"Incomplete final validation: {run_id}, {progress=}, {final=}")
        curves.extend(rows)
        summaries.append({
            "run_id": run_id, "parameters": parameters, "wandb_state": run.state,
            "run_progress": progress, "final_eval_loss": final[0] if final else "",
        })
        if rows:
            ax.plot([row["step"] for row in rows], [row["eval_loss"] for row in rows], label=label)
    for name, rows, columns in (
        ("validation_curves.csv", curves, ["run_id", "parameters", "step", "eval_loss"]),
        ("training_summary.csv", summaries, ["run_id", "parameters", "wandb_state", "run_progress", "final_eval_loss"]),
    ):
        with (HERE / "data" / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    ax.set(xlabel="Optimizer step (same data exposure)", ylabel="Full validation cross-entropy", title="Default-corpus depth scaling")
    ax.legend()
    fig.tight_layout()
    save_plot_with_meta(fig, HERE / "plots/validation_curves.png", caption="Full validation curves at identical data exposure. Incomplete curves are not final results.")


if __name__ == "__main__":
    main()
