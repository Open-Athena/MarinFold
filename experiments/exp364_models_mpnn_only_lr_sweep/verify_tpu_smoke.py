"""Verify ten restored TPU updates and persist their metrics/checkpoint evidence."""

import argparse
import csv
import json
import math
from datetime import UTC, datetime
from pathlib import Path

import fsspec
import tensorstore as ts
import wandb
from levanter.tensorstore_serialization import build_kvstore_spec

from experiments.exp364_models_mpnn_only_lr_sweep.tpu_cooldown import (
    PREFIX,
    RUN_ID,
    SOURCE_STEP,
)


def main() -> None:
    """Require all ten finite updates, their cooldown, and a durable final state."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", type=int, required=True)
    args = parser.parse_args()
    run_id = f"{RUN_ID}-smoke-a{args.attempt:02d}"
    run = wandb.Api().run(f"open-athena/MarinFold/{run_id}")
    first, end = SOURCE_STEP + 1, SOURCE_STEP + 11
    keys = ["global_step", "train/loss", "optim/learning_rate", "throughput/duration"]
    rows = list(run.scan_history(keys=keys))
    if [row["global_step"] for row in rows] != list(range(first, end)):
        raise ValueError(f"Expected exactly steps {first} through {end - 1}: {rows}")
    for row in rows:
        if not all(math.isfinite(row[key]) for key in keys):
            raise ValueError(f"Non-finite smoke metric: {row}")
        expected_lr = 5e-4 * (1 - 0.9 * (row["global_step"] - first) / 10)
        if not math.isclose(row["optim/learning_rate"], expected_lr, rel_tol=1e-5):
            raise ValueError(
                f"Smoke cooldown did not preserve the absolute step: {row}"
            )
    if run.summary.get("run_progress", 0) < 1:
        raise ValueError("Smoke has not reached its stopping target")
    eval_loss = run.summary["eval/loss"]
    if not math.isfinite(eval_loss):
        raise ValueError(f"Non-finite validation loss: {eval_loss}")

    fs = fsspec.filesystem("gcs", cache_timeout=0)
    output = f"{PREFIX}/runs/{run_id}"
    checkpoint = f"{output}/checkpoints/step-{end - 1}"
    with fs.open(f"{checkpoint}/metadata.json") as handle:
        metadata = json.load(handle)
    if metadata["step"] != end - 1:
        raise ValueError(f"Unexpected final checkpoint: {metadata}")
    scalars = {}
    for key in (
        "step",
        "opt_state/count",
        "opt_state/hyperparams_states/learning_rate/count",
        "opt_state/inner_state/1/count",
    ):
        array = ts.open(
            {
                "driver": "zarr3",
                "kvstore": {
                    "driver": "ocdbt",
                    "base": build_kvstore_spec(checkpoint + "/"),
                    "path": key,
                },
            }
        ).result()
        scalars[key] = array.read().result().item()
        if scalars[key] != end:
            raise ValueError(f"Incomplete final optimizer state: {key}={scalars[key]}")
    hf_path = f"{output}/hf/step-{end - 1}"
    files = fs.find(hf_path, detail=True)
    names = {Path(path).name for path in files}
    if not {"config.json", "tokenizer.json", "tokenizer_config.json"} <= names:
        raise ValueError(f"Missing model/tokenizer configuration: {names}")
    weights = [info for path, info in files.items() if path.endswith(".safetensors")]
    if not weights or any(info["size"] <= 0 for info in weights):
        raise ValueError(f"Missing HF weights: {names}")

    data = Path(__file__).with_name("data")
    with (data / "tpu_smoke.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows({key: row[key] for key in keys} for row in rows)
    result = {
        "verified_at_utc": datetime.now(UTC).isoformat(),
        "wandb_url": run.url,
        "run_id": run_id,
        "devices": run.summary["num_devices"],
        "first_step": first,
        "final_step": end - 1,
        "updates": len(rows),
        "run_progress": run.summary["run_progress"],
        "eval_loss": eval_loss,
        "eval_max_batches": run.config["trainer"]["max_eval_batches"],
        "checkpoint": checkpoint,
        "checkpoint_scalars": scalars,
        "hf_export": hf_path,
        "hf_files": {Path(path).name: info["size"] for path, info in files.items()},
    }
    (data / "tpu_smoke_verification.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
