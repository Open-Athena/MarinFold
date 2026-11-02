# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Export the low-MSA-depth vote matrices out of CoreWeave, for the dashboard.

Runs **inside** the cluster: the dense `[L,L]` rollout vote matrices live under
the run's `dense_scores/` prefix, which a workstation cannot read. This pulls the
29 matrices named by the low-MSA-depth set, thins each to its top `3L` pairs
(the dashboard never draws more, and a dense float16 matrix for a 400-residue
protein is 320 KB against ~14 KB sparse), and publishes one JSON to the public
bucket.

Submitted by ``submit_coreweave.py --export-low-depth``; needs `HF_TOKEN` in the
job environment, same as the driver's publish step.

    python export_low_depth_maps.py --run-id v1-01 --stems-b64 <base64 csv>
"""

import argparse
import base64
import io
import json
import shutil
import subprocess
import sys
from pathlib import Path

import fsspec
import numpy as np
import pandas as pd
from checkpoint_specs import PUBLISH_BUCKET, PUBLISH_PREFIX, run_root

#: Pairs kept per protein, as a multiple of length. R-precision cuts at the
#: number of true contacts (~2L here) and the dashboard's densest view is top-L,
#: so 3L leaves headroom without shipping the matrix.
TOP_K_MULTIPLE = 3
CHECKPOINT_LABEL = "exp232_decontam_train_m2_p06_step363000"


def top_pairs(matrix: np.ndarray, keep: int) -> list[list[float]]:
    """Return ``[i, j, votes]`` for the ``keep`` highest-voted pairs, i < j."""

    upper = np.triu(matrix, k=1)
    flat = upper.ravel()
    keep = min(keep, int((flat > 0).sum()))
    if keep == 0:
        return []
    indices = np.argpartition(flat, -keep)[-keep:]
    indices = indices[np.argsort(-flat[indices])]
    rows, columns = np.unravel_index(indices, upper.shape)
    return [
        [int(row), int(column), float(flat[index])]
        for row, column, index in zip(rows, columns, indices, strict=True)
    ]


def hf_binary() -> str:
    """Return an ``hf`` CLI with the ``buckets`` subcommand."""

    candidate = Path(sys.executable).with_name("hf")
    if candidate.exists():
        return str(candidate)
    found = shutil.which("hf")
    if found is None:
        raise RuntimeError("no `hf` CLI available")
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--stems-b64",
        required=True,
        help="base64 of a CSV with dataset,stem columns — the set to export.",
    )
    args = parser.parse_args()

    wanted = pd.read_csv(io.BytesIO(base64.b64decode(args.stems_b64)))
    root = run_root(args.run_id)
    filesystem, dense_root = fsspec.core.url_to_fs(
        f"{root}/dense_scores/{CHECKPOINT_LABEL}"
    )

    exported: dict[str, dict] = {}
    for record in wanted.itertuples(index=False):
        key = f"{record.dataset}__{record.stem}"
        path = f"{dense_root.rstrip('/')}/{key}.npz"
        if not filesystem.exists(path):
            raise FileNotFoundError(f"no vote matrix at {path}")
        with filesystem.open(path, "rb") as handle:
            matrix = np.load(io.BytesIO(handle.read()))["score"].astype(np.float32)
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"{key}: vote matrix is not square: {matrix.shape}")
        exported[key] = {
            "dataset": record.dataset,
            "stem": record.stem,
            "L": int(matrix.shape[0]),
            "n_rollouts": 100,
            "top_pairs": top_pairs(matrix, TOP_K_MULTIPLE * matrix.shape[0]),
        }
        print(f"[export] {key}: L={matrix.shape[0]} pairs={len(exported[key]['top_pairs'])}",
              flush=True)

    if len(exported) != len(wanted):
        raise ValueError(f"exported {len(exported)} of {len(wanted)} matrices")

    payload = {
        "checkpoint": CHECKPOINT_LABEL,
        "run_root": root,
        "top_k_multiple": TOP_K_MULTIPLE,
        "proteins": exported,
    }
    local = Path("/tmp/marinfold_low_depth_contacts.json")
    local.write_text(json.dumps(payload, sort_keys=True))
    destination = (
        f"hf://buckets/{PUBLISH_BUCKET}/{PUBLISH_PREFIX}/{args.run_id}"
        "/analysis/marinfold_low_depth_contacts.json"
    )
    subprocess.run([hf_binary(), "buckets", "cp", str(local), destination], check=True)
    print(
        json.dumps(
            {
                "event": "exported",
                "proteins": len(exported),
                "bytes": local.stat().st_size,
                "destination": destination,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
