"""A scheduler success status must not hide incomplete training."""

import json
from pathlib import Path

import pytest

from experiments.exp363_models_double_channel_width.runtime import require_final_checkpoint


def test_incomplete_checkpoint_is_not_success(tmp_path: Path) -> None:
    path = tmp_path / "checkpoints/step-19"
    path.mkdir(parents=True)
    (path / "manifest.json").write_text("{}")
    with pytest.raises(RuntimeError, match="without its final step-19 checkpoint"):
        require_final_checkpoint(str(tmp_path), 20)


def test_committed_final_checkpoint_is_required(tmp_path: Path) -> None:
    path = tmp_path / "checkpoints/step-19"
    path.mkdir(parents=True)
    metadata = path / "metadata.json"
    metadata.write_text(json.dumps({"step": 18}))
    with pytest.raises(ValueError, match="step mismatch"):
        require_final_checkpoint(str(tmp_path), 20)
    metadata.write_text(json.dumps({"step": 19}))
    require_final_checkpoint(str(tmp_path), 20)
