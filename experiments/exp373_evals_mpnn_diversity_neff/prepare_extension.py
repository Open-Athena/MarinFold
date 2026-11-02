"""Freeze the extension and verify all original 300-sample input files."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    """Return a file's SHA256 identity."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Create the immutable plan once, before launching any extension jobs."""
    destination = HERE / "data/extension_plan.json"
    if destination.exists():
        raise FileExistsError("Extension plan already frozen; do not overwrite it")
    inventory = pd.read_csv(HERE / "data/raw_inventory.csv")
    selected = inventory[inventory.temperature.eq(1) & inventory["mode"].eq("fixed")]
    assert len(selected) == 232 and selected.rollouts.eq(300).all()
    for row in selected.itertuples():
        path = HERE / "scratch/raw" / row.model / f"{row.stem}-t1.0-fixed.json.gz"
        assert digest(path) == row.sha256
    selected.to_csv(HERE / "data/extension_prefix_inventory.csv", index=False)
    original = json.loads((HERE / "data/plan.json").read_text())
    plan = {
        "frozen_utc": datetime.now(UTC).isoformat(),
        "original_commit": "e163932d1d93f73d1fab42795d797ba6d4693bd2",
        "original_artifact_root": original["artifact_root"],
        "artifact_root": original["artifact_root"].removesuffix("v1") + "v2-1000",
        "original_plan_sha256": digest(HERE / "data/plan.json"),
        "prefix_inventory_sha256": digest(HERE / "data/extension_prefix_inventory.csv"),
        "protocol_sha256": digest(HERE / "EXTENSION_PLAN.md"),
        "targets_sha256": digest(HERE / "inputs/targets.json"),
        "truth_sha256": digest(HERE / "inputs/ground_truth.json"),
        "rollout_start": 300,
        "rollout_stop": 1000,
        "temperature": 1.0,
        "mode": "fixed",
        "conditions": 232,
        "new_samples": 162400,
        "discovery_k": [1, 10, 25, 50, 100, 200, 300, 400, 500, 600, 750, 900, 1000],
        "oracle_k": "Every integer from 1 to 1000",
        "mean_f1_baseline": "Mean over 1000 individual rollout F1 scores per protein",
        "bootstrap_draws": 10000,
        "bootstrap_seed": 373,
    }
    destination.write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
