"""Snapshot and plot exp364 validation against the completed full-corpus runs."""

import argparse
import csv
import hashlib
import json
import math
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import wandb
from build_summary import save_plot_with_meta

HERE = Path(__file__).resolve().parent
BASELINE = HERE.parent / "exp277_models_single_mpnn_pilot"
DATA = HERE / "data"
RUNS = {
    "full_epoch1": "contacts-v1-exp277-m2-p06-full-epoch-1.5B",
    "full_continuation": "contacts-v1-exp277-m2-p06-full-epoch2-from213072-1.5B",
    "lr5e-4": "contacts-v1-exp364-mpnn-only-2ep-lr5e-4-1.5B",
    "lr1e-3": "contacts-v1-exp364-mpnn-only-2ep-lr1e-3-1.5B",
    "lr2e-3": "contacts-v1-exp364-mpnn-only-2ep-lr2e-3-1.5B",
    "lr5e-4-cooldown": "contacts-v1-exp364-mpnn-only-1ep-lr5e-4-cooldown-1.5B",
}
REFERENCE_FILES = {
    "full_epoch1": BASELINE / "data/epoch_validation_progress.csv",
    "full_continuation": BASELINE / "data/epoch2_validation_progress.csv",
}
STYLES = {
    "full_epoch1": ("Full data: first epoch", "#20252b", "-"),
    "full_continuation": ("Full data: continuation", "#777777", "--"),
    "lr5e-4": ("MPNN only: LR 5e-4", "#0072b2", "-"),
    "lr1e-3": ("MPNN only: LR 1e-3", "#d55e00", "-"),
    "lr2e-3": ("MPNN only: LR 2e-3", "#009e73", "-"),
    "lr5e-4-cooldown": ("MPNN only: LR 5e-4, one-epoch fork", "#aa4499", "--"),
}


def validation_signature(config: dict[str, Any]) -> dict[str, Any]:
    """Extract fields that establish the same model and full validation task."""
    data = config["data"]
    return {
        "model": config["model"],
        "validation_component": data["components"]["input/validation"],
        "tokenizer": data["tokenizer"],
        "train_seq_len": config["train_seq_len"],
        "block_cross_document_attention": data["block_cross_document_attention"],
        "num_validation_sequences": data["num_validation_sequences"],
        "max_eval_batches": config["trainer"]["max_eval_batches"],
        "mixed_precision": config["trainer"]["mp"],
        "z_loss_weight": config["z_loss_weight"],
        "labeled_eval": config["labeled_eval"],
    }


def snapshot() -> None:
    """Save complete validation curves and their verified comparison contract."""
    api = wandb.Api()
    rows = []
    metadata: dict[str, Any] = {
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "metric": "eval/loss",
        "runs": {},
        "caveats": [
            "Equal steps mean equal packed token slots, not equal real tokens or proteins.",
            "Training duration stretches both LR and amino-acid augmentation schedules.",
            "The full-data continuation restores step 213072, before first-epoch cooldown; it is not an independent seed.",
            "The one-epoch cooldown fork inherits the 5e-4 run through step 139053; only newly measured validation points are plotted for the fork.",
            "This is language-model validation loss, not a contact-accuracy evaluation.",
        ],
    }
    reference = None
    for key, run_id in RUNS.items():
        run = api.run(f"open-athena/MarinFold/{run_id}")
        config = run.config
        signature = validation_signature(config)
        if reference is None:
            reference = signature
            metadata["validation_contract"] = signature
        elif signature != reference:
            differences = [
                field for field in signature if signature[field] != reference[field]
            ]
            raise ValueError(f"Validation mismatch for {run_id}: {differences}")
        trainer = config["trainer"]
        if (
            trainer["train_batch_size"] != 128
            or trainer["max_eval_batches"] is not None
        ):
            raise ValueError(f"Unexpected batch or truncated validation for {run_id}")
        summary = dict(run.summary)
        details = {
            "run_id": run_id,
            "url": run.url,
            "state": run.state,
            "observed_step": summary["_step"],
            "run_progress": summary["run_progress"],
            "num_train_steps": trainer["num_train_steps"],
            "optimizer": config["optimizer"],
            "augmentation_num_train_steps": config["data"][
                "augmentation_num_train_steps"
            ],
            "per_device_eval_parallelism": trainer["per_device_eval_parallelism"],
        }
        if key in REFERENCE_FILES:
            path = REFERENCE_FILES[key]
            with path.open() as source:
                curve = {
                    int(row["global_step"]): float(row["validation_loss"])
                    for row in csv.DictReader(source)
                }
            details["curve_source"] = str(path.relative_to(HERE.parent))
            details["curve_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            if (
                max(curve) != summary["_step"]
                or abs(curve[max(curve)] - summary["eval/loss"]) > 1e-7
            ):
                raise ValueError(
                    f"Saved final validation differs from W&B for {run_id}"
                )
        else:
            first_step = 0
            if key == "lr5e-4-cooldown":
                source = json.loads((DATA / "cooldown_source.json").read_text())
                if trainer["initialize_from"] != source["destination"]:
                    raise ValueError(f"Unexpected fork source for {run_id}")
                first_step = source["step"] + 1
                details["source_checkpoint_step"] = source["step"]
                details["source_run_id"] = RUNS["lr5e-4"]
            history = run.history(
                keys=["_step", "eval/loss"], samples=10000, pandas=False
            )
            curve = {
                int(row["_step"]): float(row["eval/loss"])
                for row in history
                if row["_step"] <= summary["_step"]
            }
            expected = {
                step
                for step in range(
                    trainer["steps_per_eval"],
                    summary["_step"] + 1,
                    trainer["steps_per_eval"],
                )
                if step >= first_step
            }
            if summary["run_progress"] >= 1:
                expected.add(trainer["num_train_steps"] - 1)
            if set(curve) != expected:
                raise ValueError(
                    f"Incomplete validation history for {run_id}: missing {sorted(expected - set(curve))}; unexpected {sorted(set(curve) - expected)}"
                )
            details["curve_source"] = (
                "W&B full sparse eval/loss history; cadence coverage checked"
            )
        details["latest_eval_step"] = max(curve) if curve else None
        details["latest_eval_loss"] = curve[max(curve)] if curve else None
        metadata["runs"][key] = details
        rows.extend(
            {"run": key, "step": step, "validation_loss": loss}
            for step, loss in sorted(curve.items())
        )
    DATA.mkdir(exist_ok=True)
    with (DATA / "validation_comparison.csv").open("w", newline="") as output:
        writer = csv.DictWriter(
            output, fieldnames=["run", "step", "validation_loss"], lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    (DATA / "validation_comparison.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )


def plot() -> None:
    """Plot saved curves without accessing W&B or rerunning any evaluation."""
    curves: dict[str, dict[int, float]] = {key: {} for key in RUNS}
    with (DATA / "validation_comparison.csv").open() as source:
        for row in csv.DictReader(source):
            curves[row["run"]][int(row["step"])] = float(row["validation_loss"])
    metadata = json.loads((DATA / "validation_comparison.json").read_text())
    final = json.loads((DATA / "tpu_final_verification.json").read_text())
    if not final["export_verification"]["all_parameters_finite"]:
        raise ValueError("The TPU endpoint must have a verified finite export")
    tpu_step = final["final_validation"]["step"]
    if tpu_step != final["native_metadata"]["step"]:
        raise ValueError("TPU validation and native checkpoint steps differ")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2))
    for key, curve in curves.items():
        if not curve:
            continue
        label, color, style = STYLES[key]
        steps = sorted(curve)
        for axis in axes:
            axis.plot(
                [step / 1000 for step in steps],
                [curve[step] for step in steps],
                label=label,
                color=color,
                linestyle=style,
                linewidth=2,
            )
            axis.scatter(
                steps[-1] / 1000, curve[steps[-1]], color=color, s=22, zorder=3
            )
    for axis in axes:
        axis.scatter(
            tpu_step / 1000,
            final["final_validation"]["eval/loss"],
            marker="*",
            s=150,
            color="#882255",
            edgecolors="white",
            linewidths=0.5,
            zorder=5,
            label="MPNN only: TPU one-epoch final (verified)",
        )
    sweep_max_step = max(max(curves[key]) for key in ("lr5e-4", "lr1e-3", "lr2e-3"))
    late_peak = max(
        loss
        for curve in curves.values()
        for step, loss in curve.items()
        if step >= 30000
    )
    axes[0].set(
        xlim=(0, math.ceil(sweep_max_step / 25_000) * 25),
        ylim=(2.96, max(4.02, late_peak + 0.05)),
        title="Current sweep at matched optimizer steps",
    )
    axes[1].set(
        xlim=(20, 490), ylim=(2.96, 3.32), title="Full trajectories — zoomed loss range"
    )
    axes[1].axvspan(213.076, 266.345, color="#20252b", alpha=0.055)
    axes[1].axvspan(426.149, 479.418, color="#777777", alpha=0.10)
    axes[1].text(240, 3.305, "First cooldown", ha="center", fontsize=9)
    axes[1].text(
        453, 3.305, "Continuation\ncooldown", ha="center", va="top", fontsize=9
    )
    for axis in axes:
        axis.set_xlabel("Absolute optimizer steps (thousands)")
        axis.set_ylabel("Validation cross-entropy (nats/token; lower is better)")
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.96),
        ncol=3,
        frameon=False,
        fontsize=10,
    )
    fig.suptitle(
        "ProteinMPNN-only training vs. native + ProteinMPNN training",
        y=1.02,
        fontsize=16,
    )
    fig.text(
        0.5,
        0.015,
        f"Snapshot: {metadata['observed_at_utc'][:16]} UTC  |  Same full validation cache, tokenizer, 1.47B model and 128 × 8192 training slots/step.\nSchedules differ in absolute steps. Full-data continuation branches from step 213,072; it is not a separate seed.\n"
        + (
            "One-epoch 5e-4 fork: awaiting its first validation result."
            if not curves["lr5e-4-cooldown"]
            else "One-epoch 5e-4 fork branches from step 139,053; only new validation measurements are shown."
        ),
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.79))
    save_plot_with_meta(
        fig,
        HERE / "plots/validation_comparison.png",
        caption="Exp364 validation against exp277 first epoch and its pre-cooldown continuation. Left: matched-step curves including the high-LR spikes. Right: full training with a zoomed loss range; spikes above the range are visible on the left. Shading marks completed-run cooldowns. The TPU star is the repaired, independently verified final checkpoint from data/tpu_final_verification.json. GPU curves use data/validation_comparison.csv; no smoothing. Schedules and TPU/GPU attention kernels differ.",
        script="compare_validation.py",
        args=[],
        dpi=170,
    )
    fig.savefig(HERE / "plots/validation_comparison.pdf", bbox_inches="tight")
    plt.close(fig)
    baseline = curves["full_epoch1"]
    common = max(
        set.intersection(
            *(set(curves[key]) for key in ["full_epoch1", "lr5e-4", "lr1e-3", "lr2e-3"])
        )
    )
    latest_matched = {}
    for key in ["lr5e-4", "lr1e-3", "lr2e-3"]:
        step = max(set(curves[key]) & set(baseline))
        latest_matched[key] = {
            "step": step,
            "loss": curves[key][step],
            "full_data_loss": baseline[step],
            "gap": curves[key][step] - baseline[step],
        }
    print(
        json.dumps(
            {
                "common_step": common,
                "common_losses": {
                    key: curves[key][common]
                    for key in ["full_epoch1", "lr5e-4", "lr1e-3", "lr2e-3"]
                },
                "latest_matched": latest_matched,
            },
            indent=2,
        )
    )


def publish() -> None:
    """Publish the small comparison artifacts with the authenticated HF CLI."""
    metadata = json.loads((DATA / "validation_comparison.json").read_text())
    date = metadata["observed_at_utc"][:10]
    prefix = f"data/exp364-models-mpnn-only-lr-sweep/validation/{date}"
    for relative in (
        "data/validation_comparison.csv",
        "data/validation_comparison.json",
        "data/cooldown_source.json",
        "data/tpu_transfer_plan.json",
        "data/tpu_input_verification.json",
        "data/tpu_smoke.csv",
        "data/tpu_smoke_verification.json",
        "data/tpu_final_verification.json",
        "data/gpu_cooldown_final_verification.json",
        "verify_tpu_smoke.py",
        "plots/validation_comparison.png",
        "plots/validation_comparison.pdf",
        "plots/validation_comparison.png.meta.json",
        "compare_validation.py",
        "build_summary.py",
        "summary_narrative.md",
        "plots/summary.pdf",
    ):
        subprocess.run(
            [
                "hf",
                "buckets",
                "cp",
                str(HERE / relative),
                f"hf://buckets/open-athena/MarinFold/{prefix}/{relative}",
            ],
            check=True,
        )
    print(f"https://huggingface.co/buckets/open-athena/MarinFold/tree/{prefix}")


def main() -> None:
    """Optionally refresh source data, then reproduce the comparison figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Fetch W&B metadata/current histories and verify validation parity.",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Publish the saved comparison artifacts to the public HF bucket.",
    )
    args = parser.parse_args()
    if args.refresh:
        snapshot()
    plot()
    if args.publish:
        publish()


if __name__ == "__main__":
    main()
