"""Run AF3 on native dimers in the existing local AlphaFold3 Docker image.

Use the official runner API, with a subclass timing the synchronized forward
pass. Structure selection is AF3's own ranking_score, never experimental truth.
"""

import argparse
import csv
import hashlib
import json
import platform
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import jax
import run_alphafold as af3
from alphafold3.common import folding_input


class TimedRunner(af3.ModelRunner):
    """Measure inference including compilation, excluding feature generation."""

    def run_inference(self, featurised_example: dict, rng_key: object) -> dict:
        start = time.monotonic()
        result = super().run_inference(featurised_example, rng_key)
        self.elapsed_seconds = time.monotonic() - start
        return result


def main() -> None:
    """Persist structures, per-input timing CSVs, and input hashes atomically."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--models", type=Path, default=Path("/models"))
    parser.add_argument(
        "--mode", choices=("single_sequence", "colabfold_msa"), required=True
    )
    args = parser.parse_args()
    jax.config.update("jax_compilation_cache_dir", str(args.outputs / "jax_cache"))
    paths = sorted(
        args.inputs.glob("*.json"),
        key=lambda p: sum(
            len(x["protein"]["sequence"])
            for x in json.loads(p.read_text())["sequences"]
        ),
    )
    start = time.monotonic()
    runner = TimedRunner(
        config=af3.make_model_config(num_diffusion_samples=5, num_recycles=10),
        device=jax.local_devices()[0],
        model_dir=args.models,
    )
    _ = runner.model_params
    load_seconds = time.monotonic() - start
    gpu = (
        subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,compute_cap",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        .strip()
        .split(", ")
    )
    for path in paths:
        dest = args.outputs / path.stem
        if (dest / "timings.csv").exists():
            continue
        start = time.monotonic()
        raw = path.read_text()
        inputs = folding_input.Input.from_json(raw)
        af3.process_fold_input(
            inputs,
            data_pipeline_config=None,
            model_runner=runner,
            output_dir=dest,
            buckets=[256, 384, 512, 640, 896],
            force_output_dir=True,
        )
        elapsed = time.monotonic() - start
        data = json.loads(raw)
        length = sum(len(x["protein"]["sequence"]) for x in data["sequences"])
        timing = {
            "stem": path.stem,
            "n_residues": length,
            "n_pairs": length * (length - 1) // 2,
            "mode": args.mode,
            "elapsed_seconds": runner.elapsed_seconds,
            "model_load_seconds": load_seconds,
            "total_seconds": load_seconds + elapsed,
            "model_nickname": "AlphaFold3",
            "runner_tag": "local-docker",
            "gpu_name": gpu[0],
            "gpu_total_memory_gb": float(gpu[1]) / 1024,
            "gpu_compute_capability": gpu[2],
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "torch_version": "not_used",
            "jax_version": jax.__version__,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "n_seeds": 1,
            "n_samples_per_seed": 5,
            "input_sha256": hashlib.sha256(raw.encode()).hexdigest(),
            "num_recycles": 10,
        }
        with (dest / "timings.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(timing))
            writer.writeheader()
            writer.writerow(timing)
        print("COMPLETE", path.stem, timing, flush=True)


if __name__ == "__main__":
    main()
