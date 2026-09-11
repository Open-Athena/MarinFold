"""Small, shared control surface for independently checkpointed scale workers."""

import io
import json
from datetime import datetime, timezone

import fsspec
import numpy as np


def sampling_history(
    fs, prefix: str, length: int, batch_size: int
) -> tuple[list[dict], set[int]]:
    """Recover original timings directly from committed backbone archives."""
    rows = []
    completed = set()
    for path in sorted(fs.glob(prefix + "/batch-*.npz")):
        with fs.open(path, "rb") as handle:
            archive = np.load(io.BytesIO(handle.read()), allow_pickle=False)
            saved = json.loads(str(archive["timings_json"]))
            if len(saved) != batch_size or archive["ca"].shape != (
                batch_size,
                length,
                3,
            ):
                raise ValueError("Saved sampling timings and coordinates disagree")
        index = saved[0]["batch_index"]
        if index in completed or any(row["batch_index"] != index for row in saved):
            raise ValueError(
                "Backbone archive contains duplicate or mixed batch identities"
            )
        if not path.endswith(f"/batch-{index:05d}.npz"):
            raise ValueError("Backbone archive name differs from its batch identity")
        rows.extend(saved)
        completed.add(index)
    return rows, completed


def read_json(uri: str) -> dict:
    with fsspec.open(uri, "rt") as handle:
        return json.load(handle)


def write_json(uri: str, value: dict) -> None:
    with fsspec.open(uri, "wt") as handle:
        json.dump(value, handle)


def paused(control_uri: str) -> bool:
    """Read cooperative pause state at a durable batch boundary."""
    if not control_uri:
        return False
    control = read_json(control_uri)
    deadline = control.get("pause_at_utc")
    return control["pause"] or bool(
        deadline and datetime.now(timezone.utc) >= datetime.fromisoformat(deadline)
    )
