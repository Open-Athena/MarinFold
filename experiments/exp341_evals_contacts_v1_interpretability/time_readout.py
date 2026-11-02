# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Harness check 5: one timed clean readout pass over eval-val and eval-denovo.

The plan's compute estimates (section 4.8) assume 50,000 forward tokens/s per
H100; this pass replaces that assumption with a measured number before any
sweep is sized. It also produces the clean seed-averaged P(contact) matrices
that B2's matched-degree recall reuses.

Writes to ``--output-dir``:

- ``timings.csv``: one row per protein in the AGENTS.md predictor-timing
  schema, plus forward-token counts, R-precision and the tail batch size.
- ``readout_summary.json``: totals and measured tokens/s.
- ``pcontact_clean.npz``: one ``[L, L]`` float32 matrix per stem.

Forward tokens per seed are counted from the readout's actual calls:
``fwd_matrix`` runs the prefix twice (once per ``next_token_probs`` call),
one ``<contact>`` tail and L two-token tails, so ``2 * P + 1 + 2 * L`` where
P is the tokenized prefix length.

    uv run python time_readout.py --device cuda --output-dir out/readout
"""

import argparse
import json
import platform
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from marinfold.document_structures.contacts_v1.inference import prefix_and_positions
from marinfold.registry import resolve_model_entry

import harness


def forward_tokens(backend, protein: harness.EvalProtein, seeds: int) -> int:
    total = 0
    for k in range(seeds):
        entry_id = protein.stem if seeds == 1 else f"{protein.stem}#cv1ens{k}"
        prefix, _, length = prefix_and_positions(protein.structure, entry_id=entry_id)
        prefix_length = len(backend.tokenizer.encode(prefix, add_special_tokens=False))
        total += 2 * prefix_length + 1 + 2 * length
    return total


def gpu_columns(device: str) -> dict:
    if not device.startswith("cuda"):
        return {"gpu_name": "", "gpu_total_memory_gb": None, "gpu_compute_capability": ""}
    props = torch.cuda.get_device_properties(0)
    return {
        "gpu_name": props.name,
        "gpu_total_memory_gb": round(props.total_memory / 2**30, 1),
        "gpu_compute_capability": f"{props.major}.{props.minor}",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=None, help="MODELS.yaml nickname (default: registry default)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seeds", type=int, default=harness.DEFAULT_SEEDS)
    parser.add_argument("--tail-batch-size", type=int, default=256)
    parser.add_argument("--runner-tag", default="local")
    parser.add_argument("--limit", type=int, default=None, help="score only the first N proteins (smoke test)")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    entry = resolve_model_entry(args.model)
    proteins = harness.load_eval_proteins()[: args.limit]
    load_start = time.perf_counter()
    backend = harness.load_model(
        args.model, device=args.device, dtype=args.dtype, tail_batch_size=args.tail_batch_size
    )
    model_load_seconds = time.perf_counter() - load_start
    static = {
        "mode": f"pairwise-{args.seeds}seed",
        "model_load_seconds": round(model_load_seconds, 2),
        "model_nickname": entry.nickname,
        "runner_tag": args.runner_tag,
        **gpu_columns(args.device),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "n_seeds": args.seeds,
        "tail_batch_size": args.tail_batch_size,
    }

    # One untimed warm-up readout so CUDA context setup and kernel autotuning
    # do not land on the first protein's row.
    harness.pcontact(backend, proteins[0], seeds=1)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()

    rows = []
    matrices = {}
    pass_start = time.perf_counter()
    for protein in proteins:
        start = time.perf_counter()
        score = harness.pcontact(backend, protein, seeds=args.seeds)
        elapsed = time.perf_counter() - start
        tokens = forward_tokens(backend, protein, args.seeds)
        matrices[protein.stem] = score.astype(np.float32)
        rows.append({
            "stem": protein.stem,
            "eval_set": protein.eval_set,
            "n_residues": protein.length,
            "n_pairs": protein.length * (protein.length - 1) // 2,
            "elapsed_seconds": round(elapsed, 3),
            "total_seconds": round(time.perf_counter() - start, 3),
            "forward_tokens": tokens,
            "tokens_per_second": round(tokens / elapsed, 1),
            "r_precision": harness.r_precision(score, protein),
            "r_precision_long": harness.r_precision(
                score, protein, min_separation=harness.LONG_RANGE_SEPARATION
            ),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **static,
        })
        print(f"{protein.stem} L={protein.length} {elapsed:.2f}s {tokens / elapsed:,.0f} tok/s", flush=True)
    pass_seconds = time.perf_counter() - pass_start

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timings = pd.DataFrame(rows)
    timings.to_csv(args.output_dir / "timings.csv", index=False)
    np.savez_compressed(args.output_dir / "pcontact_clean.npz", **matrices)
    summary = {
        "model": entry.nickname,
        "n_proteins": len(rows),
        "seeds": args.seeds,
        "tail_batch_size": args.tail_batch_size,
        "model_load_seconds": round(model_load_seconds, 1),
        "pass_seconds": round(pass_seconds, 1),
        "forward_tokens": int(timings["forward_tokens"].sum()),
        "tokens_per_second": round(timings["forward_tokens"].sum() / timings["elapsed_seconds"].sum(), 1),
        "plan_assumed_tokens_per_second": 50_000,
        "mean_r_precision_by_set": timings.groupby("eval_set")["r_precision"].mean().round(4).to_dict(),
        **gpu_columns(args.device),
    }
    (args.output_dir / "readout_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
