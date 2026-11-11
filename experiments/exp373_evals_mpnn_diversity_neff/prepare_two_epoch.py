"""Freeze the supplied regional exports and the matched follow-up protocol."""

import json
from datetime import UTC, datetime
from pathlib import Path

from prepare_extension import digest

HERE = Path(__file__).resolve().parent


def main() -> None:
    """Pin identities and inputs once before any new checkpoint is sampled."""
    destination = HERE / "data/two_epoch_plan.json"
    if destination.exists():
        raise FileExistsError("Two-epoch plan is already frozen")
    original = json.loads((HERE / "data/plan.json").read_text())
    manifests = json.loads((HERE / "data/two_epoch_export_manifests.json").read_text())
    models = {}
    for rate, manifest in manifests.items():
        checkpoint = manifest["checkpoint"]
        assert checkpoint["step"] == 373768
        assert checkpoint["evaluation_cluster"] == original["cluster"]
        models["mpnn2-" + rate] = {
            "nickname": checkpoint["run_id"],
            "step": checkpoint["step"],
            "uri": checkpoint["hf_export"],
            "wandb_url": checkpoint["wandb_url"],
            "final_lm_loss": checkpoint["latest_eval_loss"],
            "files": manifest["files"],
        }
    plan = {
        **original,
        "models": models,
        "frozen_utc": datetime.now(UTC).isoformat(),
        "artifact_root": original["artifact_root"].removesuffix("v1") + "v3-two-epoch",
        "fixed_t1_batches": [300, 700],
        "expected_conditions": 1392,
        "expected_samples": 440800,
        "input_hashes": {
            name: digest(HERE / name)
            for name in [
                "data/plan.json",
                "data/extension_plan.json",
                "data/two_epoch_export_manifests.json",
                "TWO_EPOCH_PLAN.md",
                "inputs/targets.json",
                "inputs/ground_truth.json",
                "data/metadata.csv",
                "data/raw_inventory.csv",
                "data/raw_inventory_1000.csv",
            ]
        },
    }
    destination.write_text(json.dumps(plan, indent=2) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
