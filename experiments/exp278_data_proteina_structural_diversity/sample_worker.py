"""Generate timed C-alpha batches with pinned Proteina checkpoints."""

import argparse
import csv
import io
import json
import os
import platform
import random
import socket
import time
from datetime import datetime, timezone
from pathlib import Path

import fsspec
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from proteinfoundation.proteinflow.proteina import Proteina

from prepare_assets import CHECKPOINTS, SOURCE_SHA
from scale_common import paused, sampling_history, write_json


def relocate_config(config: DictConfig, data_root: Path) -> DictConfig:
    """Copy checkpoint configuration and relocate its CATH vocabulary reference."""
    plain = OmegaConf.to_container(config, resolve=True)

    def relocate(value: object) -> object:
        if isinstance(value, dict):
            return {
                k: str(data_root / "pdb_raw") if k == "cath_code_dir" else relocate(v)
                for k, v in value.items()
            }
        if isinstance(value, list):
            return [relocate(v) for v in value]
        return value

    return OmegaConf.create(relocate(plain))


def put_bytes(uri: str, content: bytes) -> None:
    """Persist one completed artifact through the injected S3 filesystem."""
    with fsspec.open(uri, "wb") as handle:
        handle.write(content)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=CHECKPOINTS, default="short")
    parser.add_argument("--length", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--batches", type=int, default=1)
    parser.add_argument("--seed", type=int, default=278)
    parser.add_argument("--cath", default="unconditional")
    parser.add_argument("--noise", type=float, default=0.45)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--control", default="")
    args = parser.parse_args()
    if not 60 <= args.length <= 500 or args.batch_size < 1 or args.batches < 1:
        parser.error("Require 60–500 residues and positive batch/count")
    run_case(args)


def run_case(args: argparse.Namespace, model: Proteina | None = None) -> Proteina:
    """Sample a fixed-length case, reusing owned model instances across cases."""
    began = time.perf_counter()
    print(json.dumps({"event": "starting", "args": vars(args)}), flush=True)
    output_fs, output_path = fsspec.core.url_to_fs(args.output)
    if output_fs.exists(output_path + "/started.json"):
        with output_fs.open(output_path + "/started.json", "rt") as handle:
            if json.load(handle) != vars(args):
                raise ValueError("Sampling output prefix has a different configuration")
    else:
        put_bytes(args.output + "/started.json", json.dumps(vars(args)).encode())
    data_root = Path(os.environ["DATA_PATH"])
    load_seconds = 0.0
    if model is None:
        checkpoint = data_root / CHECKPOINTS[args.model][1]
        loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
        config = relocate_config(loaded["hyper_parameters"]["cfg_exp"], data_root)
        model = Proteina(config, store_dir="/tmp/proteina-worker")
        model.load_state_dict(loaded["state_dict"], strict=True)
        del loaded
        model = model.cuda().eval()
        if args.compile:
            # Compile this owned instance without replacing imported attributes.
            model.nn.compile(dynamic=False)
        torch.cuda.synchronize()
        load_seconds = time.perf_counter() - began
    props = torch.cuda.get_device_properties(0)
    meta = {
        "model_nickname": f"proteina-{args.model}",
        "runner_tag": "iris",
        "gpu_name": props.name,
        "gpu_total_memory_gb": props.total_memory / 1e9,
        "gpu_compute_capability": f"{props.major}.{props.minor}",
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha": SOURCE_SHA,
        "model_load_seconds": load_seconds,
        "batch_size": args.batch_size,
        "length": args.length,
        "cath": args.cath,
        "noise": args.noise,
        "compiled": args.compile,
    }
    put_bytes(args.output + "/assets.json", (data_root / "assets.json").read_bytes())
    put_bytes(
        args.output + "/numerics.json",
        json.dumps(
            {"float32_matmul_precision": torch.get_float32_matmul_precision()}
        ).encode(),
    )
    print(json.dumps({"event": "loaded", **meta}), flush=True)
    # A batch archive is the atomic durability unit: coordinates and their
    # original inference timings survive even if the process dies before a
    # progress marker or aggregate CSV is written. Scale runs use new prefixes.
    rows, completed_batches = sampling_history(
        output_fs, output_path, args.length, args.batch_size
    )
    for batch_index in range(args.batches):
        if paused(args.control):
            print("Paused at durable sampling batch boundary", flush=True)
            return model
        if batch_index in completed_batches:
            print(f"Already complete: batch {batch_index}", flush=True)
            continue
        batch_seed = args.seed + batch_index
        torch.manual_seed(batch_seed)
        np.random.seed(batch_seed)
        random.seed(batch_seed)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            sample = model.generate(
                nsamples=args.batch_size,
                n=args.length,
                dt=0.0025,
                self_cond=True,
                cath_code=None
                if args.cath == "unconditional"
                else [[args.cath]] * args.batch_size,
                dtype=torch.float32,
                schedule_mode="log",
                schedule_p=2.0,
                sampling_mode="sc",
                sc_scale_noise=args.noise,
                sc_scale_score=1.0,
                gt_mode="1/t",
                gt_p=1.0,
                gt_clamp_val=None,
                guidance_weight=1.0,
                autoguidance_ratio=0.0,
            )
            coordinates = model.samples_to_atom37(sample)[:, :, 1, :].cpu().numpy()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if (
            coordinates.shape != (args.batch_size, args.length, 3)
            or not np.isfinite(coordinates).all()
        ):
            raise ValueError("Generator produced invalid coordinate array")
        first_row = len(rows)
        for index in range(args.batch_size):
            rows.append(
                {
                    **meta,
                    "stem": f"batch-{batch_index:05d}-{index:04d}",
                    "n_residues": args.length,
                    "n_pairs": args.length * (args.length - 1) // 2,
                    "mode": "backbone_sampling",
                    "seed": batch_seed,
                    "sample_in_batch": index,
                    "batch_index": batch_index,
                    "batch_elapsed_seconds": elapsed,
                    "elapsed_seconds": elapsed / args.batch_size,
                    "peak_gpu_memory_gb": torch.cuda.max_memory_allocated() / 1e9,
                    "total_seconds": time.perf_counter() - began,
                }
            )
        buffer = io.BytesIO()
        np.savez_compressed(
            buffer, ca=coordinates, timings_json=json.dumps(rows[first_row:])
        )
        put_bytes(f"{args.output}/batch-{batch_index:05d}.npz", buffer.getvalue())
        write_json(
            args.output + "/progress.json",
            {
                "count": len(rows),
                "batch_index": batch_index,
                "elapsed_seconds": sum(float(row["elapsed_seconds"]) for row in rows),
                "updated_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        print(
            json.dumps(
                {
                    "event": "batch_done",
                    "batch": batch_index,
                    "count": len(rows),
                    "seconds": elapsed,
                }
            ),
            flush=True,
        )
    text = io.StringIO()
    writer = csv.DictWriter(text, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    put_bytes(args.output + "/timings.csv", text.getvalue().encode())
    put_bytes(
        args.output + "/complete.json",
        json.dumps({"count": len(rows), **meta}).encode(),
    )
    return model


if __name__ == "__main__":
    main()
