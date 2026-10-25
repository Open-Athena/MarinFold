# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Harness checks 1 and 2 from the exp341 plan (section 4.6), on a real checkpoint.

Check 1: the readout with residual-capture hooks on every layer reproduces
the unhooked production readout, and the rope-repair negative control (same
weights, raw exported config) visibly diverges from it.

Check 2: mean ablation with no layer selected reproduces the clean
R-precision exactly.

Runs on the shortest eval-val proteins so a CPU-only workstation can do it.
Writes ``data/harness_checks.json`` (numbers plus worker metadata) and exits
non-zero if a check fails.

    uv run python check_harness.py                      # registry default (exp277)
    uv run python check_harness.py --device cuda --seeds 4
"""

import argparse
import gc
import json
import platform
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from marinfold.registry import resolve_model_entry

import harness

OUTPUT = Path(__file__).resolve().parent / "data" / "harness_checks.json"
# Check 1's divergence criterion for the unrepaired control: mean absolute
# difference in log P(contact) over resolved candidate pairs. A wrong rope
# base cost 0.25-1.34 nats/token of document NLL on the #117 checkpoint
# (marinfold/inference/_config.py), so 0.1 nats is a deliberately low bar
# that a working repair path still has to clear.
DIVERGENCE_FLOOR_NATS = 0.1
# Hooks that only read should leave the readout bit-identical; any nonzero
# difference means a hook is perturbing the forward pass.
HOOKED_TOLERANCE = 0.0


def candidate_log_scores(score: np.ndarray, protein: harness.EvalProtein) -> np.ndarray:
    a, b = np.triu_indices(len(protein.resolved), k=1)
    rows, cols = protein.resolved[a], protein.resolved[b]
    keep = cols - rows >= harness.MIN_SEQ_SEPARATION
    return np.log(score[rows[keep], cols[keep]])


def worker_metadata(device: str) -> dict:
    meta = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "device": device,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if device.startswith("cuda"):
        props = torch.cuda.get_device_properties(0)
        meta["gpu_name"] = props.name
        meta["gpu_total_memory_gb"] = round(props.total_memory / 2**30, 1)
    else:
        meta["cpu_threads"] = torch.get_num_threads()
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=None, help="MODELS.yaml nickname (default: registry default)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--n-proteins", type=int, default=3)
    parser.add_argument("--seeds", type=int, default=1, help="document seeds per readout (plan default 4)")
    args = parser.parse_args()

    entry = resolve_model_entry(args.model)
    proteins = sorted(harness.load_eval_proteins(["eval-val"]), key=lambda p: p.length)[: args.n_proteins]
    print(f"model {entry.nickname}; proteins {[(p.stem, p.length) for p in proteins]}", flush=True)

    load_start = time.perf_counter()
    backend = harness.load_model(args.model, device=args.device, dtype=args.dtype)
    load_seconds = time.perf_counter() - load_start
    n_boundaries = len(harness.decoder_layers(backend.model)) + 1

    rows = []
    clean_scores = {}
    for protein in proteins:
        start = time.perf_counter()
        clean = harness.pcontact(backend, protein, seeds=args.seeds)
        clean_seconds = time.perf_counter() - start
        with harness.capture_residuals(backend.model) as captured:
            hooked = harness.pcontact(backend, protein, seeds=args.seeds)
        with harness.mean_ablation(backend.model, {}):
            empty_ablation = harness.pcontact(backend, protein, seeds=args.seeds)
        clean_scores[protein.stem] = clean
        rows.append({
            "stem": protein.stem,
            "n_residues": protein.length,
            "clean_seconds": round(clean_seconds, 2),
            "r_precision_clean": harness.r_precision(clean, protein),
            "r_precision_long_clean": harness.r_precision(clean, protein, min_separation=harness.LONG_RANGE_SEPARATION),
            "hooked_max_abs_diff": float(np.abs(hooked - clean).max()),
            "captured_boundaries": sum(1 for calls in captured if calls),
            "r_precision_empty_ablation": harness.r_precision(empty_ablation, protein),
        })
        print(rows[-1], flush=True)

    # Both models are resident briefly (~3 GB each in bf16); the repaired
    # one is dropped as soon as the control has taken its tokenizer.
    control = harness.load_unrepaired_model(args.model, backend)
    control_theta = control.model.config.rope_theta
    del backend
    gc.collect()
    for row, protein in zip(rows, proteins, strict=True):
        unrepaired = harness.pcontact(control, protein, seeds=args.seeds)
        clean = clean_scores[protein.stem]
        row["r_precision_unrepaired"] = harness.r_precision(unrepaired, protein)
        row["unrepaired_mean_abs_log_diff"] = float(
            np.abs(candidate_log_scores(unrepaired, protein) - candidate_log_scores(clean, protein)).mean()
        )
        print(row, flush=True)

    check1_hooked = all(r["hooked_max_abs_diff"] <= HOOKED_TOLERANCE for r in rows) and all(
        r["captured_boundaries"] == n_boundaries for r in rows
    )
    check1_control = control_theta != harness.TRAINED_ROPE_THETA and all(
        r["unrepaired_mean_abs_log_diff"] > DIVERGENCE_FLOOR_NATS for r in rows
    )
    check2 = all(r["r_precision_empty_ablation"] == r["r_precision_clean"] for r in rows)
    result = {
        "model": entry.nickname,
        "dtype": args.dtype,
        "seeds": args.seeds,
        "model_load_seconds": round(load_seconds, 1),
        "unrepaired_rope_theta": control_theta,
        "checks": {
            "check1_hooked_matches_clean": check1_hooked,
            "check1_unrepaired_diverges": check1_control,
            "check2_empty_ablation_is_clean": check2,
        },
        "criteria": {
            "hooked_tolerance": HOOKED_TOLERANCE,
            "divergence_floor_nats": DIVERGENCE_FLOOR_NATS,
            "layer_boundaries": n_boundaries,
        },
        "proteins": rows,
        "worker": worker_metadata(args.device),
    }
    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["checks"], indent=2))
    return 0 if all(result["checks"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
