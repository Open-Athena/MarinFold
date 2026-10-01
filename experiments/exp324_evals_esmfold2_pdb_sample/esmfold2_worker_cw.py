"""Run ESMFold2 predictions for one manifest shard on CoreWeave.

This script is intended to run inside an Iris CoreWeave GPU task after the
bootstrap installs Biohub's `esm` package. It reads a CSV manifest with columns
`stem` and `sequence`, folds rows assigned to this shard, and writes mmCIF,
provenance, and timing artifacts to an S3 prefix via fsspec.
"""

import argparse
import csv
import json
import math
import platform
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fsspec
import torch
from esm.models.esmfold2 import ESMFold2InputBuilder, EsmFold2Model, ProteinInput, StructurePredictionInput

HF_MODEL_ID = "biohub/ESMFold2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--n-samples", type=int, default=1)
    parser.add_argument("--num-loops", type=int, default=20)
    parser.add_argument("--num-sampling-steps", type=int, default=100)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true", help="Recompute rows even when structure.cif already exists.")
    return parser.parse_args()


def score_confidence(result: Any) -> float:
    for attr in ("ptm", "mean_plddt", "plddt", "confidence"):
        value = getattr(result, attr, None)
        if value is None and hasattr(result, "complex"):
            value = getattr(result.complex, attr, None)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            try:
                import numpy as np

                return float(np.asarray(value, dtype=float).mean())
            except Exception:  # noqa: BLE001
                continue
    return math.nan


def write_text(uri: str, text: str) -> None:
    with fsspec.open(uri, "wt") as handle:
        handle.write(text)


def load_rows(path: Path, shard_index: int, num_shards: int, limit: int | None) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for i, row in enumerate(rows) if i % num_shards == shard_index]
    if limit is not None:
        rows = rows[:limit]
    return rows


def worker_meta() -> dict[str, Any]:
    meta: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "model": HF_MODEL_ID,
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        meta.update(
            {
                "gpu_name": props.name,
                "gpu_total_memory_gb": round(props.total_memory / 1e9, 2),
                "gpu_compute_capability": f"{props.major}.{props.minor}",
            }
        )
    return meta


def main() -> None:
    args = parse_args()
    rows = load_rows(args.manifest, args.shard_index, args.num_shards, args.limit)
    print(
        f"[esmfold2] shard {args.shard_index}/{args.num_shards}; rows={len(rows)}; "
        f"n_samples={args.n_samples}",
        flush=True,
    )
    print(f"[esmfold2] torch cuda={torch.cuda.is_available()}", flush=True)
    model_load_start = time.time()
    model = EsmFold2Model.from_pretrained(HF_MODEL_ID).cuda().eval()
    model_load_seconds = time.time() - model_load_start
    builder = ESMFold2InputBuilder()
    meta = worker_meta()
    print(f"[esmfold2] meta={meta}; model_load_seconds={model_load_seconds:.2f}", flush=True)

    summary_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        stem = row["stem"]
        sequence = row["sequence"]
        print(f"[esmfold2] {idx}/{len(rows)} {stem} L={len(sequence)}", flush=True)
        prefix = f"{args.out_prefix.rstrip('/')}/structures/{stem}"
        if not args.force and fsspec.filesystem("s3").exists(f"{prefix}/structure.cif"):
            print(f"[esmfold2] skip existing {stem}", flush=True)
            summary_rows.append({"stem": stem, "status": "skipped_existing"})
            continue
        t0 = time.time()
        try:
            best: tuple[float, int, str] | None = None
            sample_scores: list[dict[str, Any]] = []
            spi = StructurePredictionInput(sequences=[ProteinInput(id="A", sequence=sequence)])
            for seed in range(args.n_samples):
                result = builder.fold(
                    model,
                    spi,
                    num_loops=args.num_loops,
                    num_sampling_steps=args.num_sampling_steps,
                    num_diffusion_samples=1,
                    seed=seed,
                )
                confidence = score_confidence(result)
                mmcif = result.complex.to_mmcif()
                sample_scores.append({"seed": seed, "confidence": confidence})
                if best is None or (confidence == confidence and confidence > best[0]):
                    best = (confidence, seed, mmcif)
            assert best is not None
            elapsed = time.time() - t0
            provenance = {
                "stem": stem,
                "entry_id": row.get("entry_id", stem),
                "pdb_id": row.get("pdb_id", ""),
                "chain_id": row.get("chain_id", ""),
                "sequence_length": len(sequence),
                "chosen_seed": best[1],
                "chosen_confidence": best[0],
                "n_samples": args.n_samples,
                "num_loops": args.num_loops,
                "num_sampling_steps": args.num_sampling_steps,
                "sample_scores": sample_scores,
            }
            timing = {
                "stem": stem,
                "n_residues": len(sequence),
                "elapsed_seconds": round(elapsed, 4),
                "model_load_seconds": round(model_load_seconds, 4),
                "total_seconds": round(elapsed, 4),
                "n_samples": args.n_samples,
                "model_nickname": "esmfold2",
                "runner_tag": "iris-cw-rno2a",
                "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                **meta,
            }
            write_text(f"{prefix}/structure.cif", best[2])
            write_text(f"{prefix}/provenance.json", json.dumps(provenance, indent=2, sort_keys=True))
            write_text(f"{prefix}/timings.json", json.dumps(timing, indent=2, sort_keys=True))
            summary_rows.append({**provenance, "elapsed_seconds": round(elapsed, 4), "status": "ok"})
            print(f"[esmfold2] done {stem}: conf={best[0]} elapsed={elapsed:.1f}s", flush=True)
        except Exception as error:  # noqa: BLE001 - record row-level failures and continue the shard.
            elapsed = time.time() - t0
            failure = {
                "stem": stem,
                "entry_id": row.get("entry_id", stem),
                "pdb_id": row.get("pdb_id", ""),
                "chain_id": row.get("chain_id", ""),
                "sequence_length": len(sequence),
                "elapsed_seconds": round(elapsed, 4),
                "status": "failed",
                "error": repr(error),
            }
            write_text(f"{prefix}/failure.json", json.dumps(failure, indent=2, sort_keys=True))
            summary_rows.append(failure)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[esmfold2] FAILED {stem}: {error!r}", flush=True)

    summary_uri = f"{args.out_prefix.rstrip('/')}/shard_summaries/shard-{args.shard_index:03d}-of-{args.num_shards:03d}.json"
    write_text(summary_uri, json.dumps(summary_rows, indent=2, sort_keys=True))
    print(f"[esmfold2] wrote {summary_uri}", flush=True)


if __name__ == "__main__":
    main()
