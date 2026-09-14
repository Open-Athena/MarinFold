"""Collect completed LR-trial metrics and verify their saved checkpoint manifests."""

import csv
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import wandb
from rigging.filesystem.storage_path import StoragePath

from experiments.exp279_models_exact_soft_contact_targets.checkpoints import (
    validate_training_restore,
)
from experiments.exp279_models_exact_soft_contact_targets.launch_gpu import (
    OUTPUT,
    configure_local_s3,
)
from experiments.exp279_models_exact_soft_contact_targets.lr_trial import (
    build_trial_config,
)

EXPERIMENT = Path(__file__).resolve().parents[1]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write one complete table with portable line endings."""
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Require complete runs before publishing their final local result tables."""
    catalog = json.loads((EXPERIMENT / "data/lr_sweep_launch.json").read_text())
    contract = json.loads((EXPERIMENT / "data/lr_fork_contract.json").read_text())
    manifest = {
        "source": catalog["source"],
        "inputs": contract["parent_identity"]["manifest"]["inputs"],
    }
    start, stop = catalog["start_update"], catalog["stop_update"]
    api = wandb.Api()
    configure_local_s3()
    validation, diagnostics, completions = [], [], []
    for trial in catalog["trials"]:
        run = api.run("open-athena/MarinFold/" + trial["run_name"])
        if run.state != "finished" or run.summary["lr_trial/progress"] < 1:
            raise ValueError(f"Trial is not complete: {trial['trial']}")
        final = {}
        for metric in ("eval/loss", "contact_eval/loss"):
            rows = list(
                run.scan_history(
                    keys=["global_step", metric],
                    min_step=start,
                    max_step=stop,
                    page_size=stop - start + 1,
                )
            )
            # Resumed W&B histories can include explicit nulls at training-only
            # steps even when scan_history requests an evaluation metric.
            rows = [row for row in rows if row.get(metric) is not None]
            if not rows or rows[-1]["global_step"] != stop - 1:
                raise ValueError(f"Missing final {metric} for {trial['trial']}")
            if not all(math.isfinite(row[metric]) for row in rows):
                raise ValueError(f"Nonfinite {metric} for {trial['trial']}")
            final[metric] = rows[-1][metric]
            validation.extend(
                dict(
                    trial=trial["trial"],
                    metric=metric,
                    step=row["global_step"],
                    value=row[metric],
                )
                for row in rows
            )
        rows = list(
            run.scan_history(
                keys=[
                    "global_step",
                    "train/loss",
                    "grad/clipped",
                    "grad/norm/total",
                    "throughput/duration",
                ],
                min_step=start,
                max_step=stop,
                page_size=1000,
            )
        )
        if [row["global_step"] for row in rows] != list(range(start, stop)):
            raise ValueError(f"Incomplete per-update diagnostics: {trial['trial']}")
        losses = [row["train/loss"] for row in rows]
        diagnostics.append(
            dict(
                trial=trial["trial"],
                updates=len(rows),
                nonfinite_losses=sum(not math.isfinite(x) for x in losses),
                peak_train_loss=max(losses),
                clipped_updates=sum(row["grad/clipped"] for row in rows),
                peak_gradient_norm=max(row["grad/norm/total"] for row in rows),
                last_100_mean_train_loss=statistics.mean(losses[-100:]),
                median_seconds_per_update=statistics.median(
                    row["throughput/duration"] for row in rows
                ),
            )
        )
        checkpoint = trial["checkpoint_root"] + f"/step-{stop - 1}"
        metadata = json.loads(StoragePath(checkpoint + "/metadata.json").read_text())
        if metadata["step"] != stop - 1 or metadata["is_temporary"]:
            raise ValueError(f"Incorrect final checkpoint: {checkpoint}")
        config = build_trial_config(
            manifest,
            trial=trial["trial"],
            run_name=trial["run_name"],
            output=OUTPUT,
            # Build the original trial's abstract state; a completed checkpoint
            # cannot itself start another update within this fixed trial budget.
            resume=contract["parent_checkpoint"],
        )
        validate_training_restore(config, checkpoint)
        hf = trial["checkpoint_root"] + f"/hf/step-{stop - 1}"
        for name in (
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
        ):
            if not StoragePath(hf + "/" + name).exists():
                raise FileNotFoundError(f"Missing final HF artifact: {hf}/{name}")
        completions.append(
            dict(
                trial=trial["trial"],
                wandb_url=trial["wandb_url"],
                native_checkpoint=checkpoint,
                native_metadata=metadata,
                full_state_array_manifest_verified=True,
                hf_model_and_tokenizer_verified=True,
                hf_checkpoint=hf,
                final_metrics=final,
            )
        )
        print(json.dumps(completions[-1]), flush=True)
    write_csv(EXPERIMENT / "data/lr_validation.csv", validation)
    write_csv(EXPERIMENT / "data/lr_training_summary.csv", diagnostics)
    (EXPERIMENT / "data/lr_completion.json").write_text(
        json.dumps(
            dict(
                observed_at_utc=datetime.now(timezone.utc).isoformat(),
                source=catalog["source"],
                trials=completions,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
