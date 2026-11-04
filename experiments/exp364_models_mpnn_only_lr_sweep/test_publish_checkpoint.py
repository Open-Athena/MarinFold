"""Reject publication manifests bound to the wrong checkpoint or finite check."""

import json
from pathlib import Path

import pytest

from experiments.exp364_models_mpnn_only_lr_sweep.publish_checkpoint import (
    validate_source,
)

HERE = Path(__file__).resolve().parent


@pytest.mark.parametrize("prefix", ["publication", "publication_two_epoch"])
def test_verified_sources_resolve_distinct_destinations(prefix: str) -> None:
    source = json.loads((HERE / f"data/{prefix}_source.json").read_text())
    checkpoint = source["checkpoint"]
    destination = validate_source(source, checkpoint["run_id"], checkpoint["step"])
    assert destination.endswith(f"/{checkpoint['run_id']}/hf/step-{checkpoint['step']}")


@pytest.mark.parametrize(
    "fault", ["step", "run", "export", "hash", "finite", "duplicate"]
)
def test_rejects_wrong_or_unverified_publication(fault: str) -> None:
    source = json.loads((HERE / "data/publication_two_epoch_source.json").read_text())
    run_id, step = source["checkpoint"]["run_id"], source["checkpoint"]["step"]
    if fault == "step":
        step = 186884
    elif fault == "run":
        run_id = "another-training-run"
    elif fault == "export":
        source["finite_verification"]["export"] += "/other-export"
    elif fault == "hash":
        source["finite_verification"]["files"]["model-00001-of-00002.safetensors"][
            "sha256"
        ] = "0" * 64
    elif fault == "finite":
        source["finite_verification"]["all_parameters_finite"] = False
    else:
        source["files"].append(source["files"][0])
    with pytest.raises(ValueError):
        validate_source(source, run_id, step)
