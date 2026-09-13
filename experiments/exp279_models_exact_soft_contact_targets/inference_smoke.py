# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Check an accelerator smoke export through MarinFold's ordinary backend.

Run in marinfold's own transformers<5 environment, from the repository root.
No Levanter dependency or custom model implementation is imported here.
"""

import argparse
import csv
import platform
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "marinfold"))
from marinfold.inference._transformers import TransformersBackend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-dir", type=Path, required=True)
    parser.add_argument("--arm", choices=("ce", "soft"), required=True)
    parser.add_argument("--timings", type=Path, required=True)
    args = parser.parse_args()
    reference = np.load(args.smoke_dir / "expected.npz")
    tokens = reference["tokens"].tolist()
    model_path = (
        args.smoke_dir / "checkpoints" / f"exp279-{args.arm}-smoke" / "hf" / "step-2"
    )
    started = time.perf_counter()
    backend = TransformersBackend(model_path, dtype="float32", device="cpu")
    loaded = time.perf_counter()
    target_ids = [143, 144, 145, 10]
    # Use the public prefix-cache scoring path at the first contact marker.
    contact_index = tokens.index(5)
    observed = backend.next_token_probs(tokens[:contact_index], [[5]], target_ids)
    completed = time.perf_counter()
    logits = reference["logits"][contact_index].astype(np.float64)
    probabilities = np.exp(logits - logits.max())
    probabilities /= probabilities.sum()
    np.testing.assert_allclose(
        observed[0], probabilities[target_ids], rtol=2e-4, atol=2e-7
    )
    row = dict(
        stem=f"exp279-{args.arm}-synthetic",
        n_residues=3,
        n_pairs=2,
        mode="next_token_probs",
        elapsed_seconds=completed - loaded,
        model_load_seconds=loaded - started,
        total_seconds=completed - started,
        model_nickname=f"exp279-{args.arm}-smoke-step-2",
        runner_tag="local",
        gpu_name="CPU",
        gpu_total_memory_gb=0,
        gpu_compute_capability="",
        hostname=socket.gethostname(),
        platform=platform.platform(),
        torch_version=torch.__version__,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )
    args.timings.parent.mkdir(parents=True, exist_ok=True)
    with args.timings.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row), lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)
    print(
        f"{args.arm}: ordinary MarinFold prefix-cache inference matches the exported Levanter model"
    )


if __name__ == "__main__":
    main()
