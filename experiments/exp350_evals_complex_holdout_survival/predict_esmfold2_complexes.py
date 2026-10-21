"""Run native ESMFold2 dimers using the cached exp78 image and weights."""

import json
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
IMAGE_ID = "im-kwWMSghQITLwEPo7jpK2eR"
REVISION = "8fc3ff471022fdce52c77030685eb775de0c00a3"
image = modal.Image.from_id(IMAGE_ID).env({"HF_HUB_OFFLINE": "1"})
weights = modal.Volume.from_name("esmfold2-weights")
outputs = modal.Volume.from_name("exp350-four-model-structures", create_if_missing=True)
app = modal.App("exp350-esmfold2-complexes-v1", image=image)


@app.cls(
    gpu="H100",
    volumes={"/weights": weights, "/outputs": outputs},
    timeout=3600,
    max_containers=4,
    scaledown_window=60,
)
@modal.concurrent(max_inputs=1)
class Predictor:
    """Keep ESMC and ESMFold2 loaded while processing independent complexes."""

    @modal.enter()
    def setup(self) -> None:
        import importlib.metadata
        import platform
        import socket
        import time

        import torch
        from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

        start = time.monotonic()
        self.model = (
            ESMFold2Model.from_pretrained(
                "biohub/ESMFold2", revision=REVISION, local_files_only=True
            )
            .cuda()
            .eval()
        )
        torch.cuda.synchronize()
        self.load_seconds = time.monotonic() - start
        props = torch.cuda.get_device_properties(0)
        self.metadata = {
            "model_nickname": "ESMFold2",
            "runner_tag": "modal",
            "gpu_name": props.name,
            "gpu_total_memory_gb": props.total_memory / 2**30,
            "gpu_compute_capability": f"{props.major}.{props.minor}",
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "torch_version": str(torch.__version__),
            "esm_version": importlib.metadata.version("esm"),
            "transformers_version": importlib.metadata.version("transformers"),
            "model_revision": REVISION,
            "image_id": IMAGE_ID,
        }

    @modal.method()
    def predict(self, target: dict) -> dict:
        import csv
        import time
        from datetime import datetime, timezone

        import torch
        from esm.models.esmfold2 import (
            ESMFold2InputBuilder,
            ProteinInput,
            StructurePredictionInput,
        )

        directory = Path("/outputs/esmfold2") / target["stem"]
        if (directory / "provenance.json").exists():
            return json.loads((directory / "provenance.json").read_text())
        directory.mkdir(parents=True, exist_ok=True)
        builder = ESMFold2InputBuilder()
        spi = StructurePredictionInput(
            sequences=[
                ProteinInput(id=cid, sequence=seq)
                for cid, seq in zip(("A", "B"), target["chain_sequences"], strict=True)
            ]
        )
        samples = []
        total_start = time.monotonic()
        for seed in range(5):
            start = time.monotonic()
            with torch.inference_mode():
                result = builder.fold(
                    self.model,
                    spi,
                    num_loops=20,
                    num_sampling_steps=100,
                    num_diffusion_samples=1,
                    seed=seed,
                )
            torch.cuda.synchronize()
            elapsed = time.monotonic() - start
            ranking = 0.8 * float(result.iptm) + 0.2 * float(result.ptm)
            (directory / f"sample_{seed}.cif").write_text(result.complex.to_mmcif())
            sample = {
                "seed": seed,
                "iptm": float(result.iptm),
                "ptm": float(result.ptm),
                "ranking_score": ranking,
                "elapsed_seconds": elapsed,
            }
            samples.append(sample)
            timing = dict(
                stem=target["stem"],
                n_residues=target["L"],
                n_pairs=target["L"] * (target["L"] - 1) // 2,
                mode="native_dimer_single_sequence",
                elapsed_seconds=elapsed,
                model_load_seconds=self.load_seconds,
                total_seconds=self.load_seconds + time.monotonic() - total_start,
                seed=seed,
                n_samples=5,
                num_loops=20,
                num_sampling_steps=100,
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                **self.metadata,
            )
            with (directory / f"timing_{seed}.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(timing))
                writer.writeheader()
                writer.writerow(timing)
            outputs.commit()
            print(target["stem"], sample, flush=True)
        best = max(samples, key=lambda s: s["ranking_score"])
        (directory / "structure.cif").write_bytes(
            (directory / f"sample_{best['seed']}.cif").read_bytes()
        )
        provenance = dict(
            stem=target["stem"],
            samples=samples,
            selected_seed=best["seed"],
            selection="max(0.8*ipTM+0.2*pTM)",
            input=target,
            model_load_seconds=self.load_seconds,
            **self.metadata,
        )
        (directory / "provenance.json").write_text(
            json.dumps(provenance, indent=2) + "\n"
        )
        outputs.commit()
        return provenance


@app.local_entrypoint()
def main(limit: int = 0) -> None:
    """Predict every frozen target, recording failures without changing the cohort."""
    targets = json.loads(
        (HERE / "data/four_model_v1/predictor_inputs.json").read_text()
    )
    targets.sort(key=lambda x: x["L"])
    if limit:
        targets = targets[:limit]
    completed = {
        Path(entry.path).parent.name
        for entry in outputs.iterdir("/", recursive=True)
        if entry.path.startswith("esmfold2/") and entry.path.endswith("/provenance.json")
    }
    targets = [target for target in targets if target["stem"] not in completed]
    print(f"Saved complete targets: {len(completed)}; pending: {len(targets)}", flush=True)
    if not targets:
        return
    worker = Predictor()
    failed = []
    for target, result in zip(
        targets, worker.predict.map(targets, return_exceptions=True), strict=True
    ):
        if isinstance(result, Exception):
            print("FAILED", target["stem"], repr(result), flush=True)
            failed.append(target["stem"])
        else:
            print("COMPLETE", result["stem"], result["selected_seed"], flush=True)
    if failed:
        raise RuntimeError(f"Failed complexes: {failed}")
