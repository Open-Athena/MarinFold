"""Retain 100 seeded ESMFold2 predictions for each of five low-depth targets.

Uses the previously staged exp78 weights, with the official native ESM
implementation pinned below. No confidence or accuracy filtering is applied.
Run with ``uv run --project generation modal run generation/run_esmfold2_decoys.py``.
"""

import csv
import datetime as dt
import hashlib
import io
import importlib.metadata
import json
import platform
import socket
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
SOURCE_SHA = "43b4548b86762edfa747b07d5f440aad3c33acee"
MODEL_REVISION = "1ebf0e3481a5184eb6171d40615c79e384b48796"
MODEL_DIR = f"/weights/hf/models--biohub--ESMFold2/snapshots/{MODEL_REVISION}"
ESMC_REVISION = "45b0fa5d7fb06faefbd5e3b89bdcef35d564e79a"
ESMC_DIR = f"/weights/hf/hub/models--biohub--ESMC-6B/snapshots/{ESMC_REVISION}"
PROTOCOL = {
    "model": "biohub/ESMFold2", "model_revision": MODEL_REVISION,
    "esm_source_sha": SOURCE_SHA, "num_loops": 20, "num_sampling_steps": 100,
    "num_diffusion_samples": 1, "lm_dropout": 0.3, "seeds": list(range(100)),
    "msa": False, "templates": False, "selection": "retain every sample",
    "targets": "all five natural FoldBench monomers with archived query-inclusive MSA depth <10",
    "sequence": "frozen exp14 resolved-residue query, shared with exp325 AF2/AF3/Boltz2",
}
PROTOCOL_SHA = hashlib.sha256(json.dumps(PROTOCOL, sort_keys=True).encode()).hexdigest()
weights = modal.Volume.from_name("esmfold2-weights")
outputs = modal.Volume.from_name("marinfold-exp325-structured-decoys", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12").apt_install("git")
         .pip_install(f"esm @ git+https://github.com/Biohub/esm.git@{SOURCE_SHA}", "numpy==2.2.6")
         .env({"HF_HOME": "/weights/hf", "HF_HUB_OFFLINE": "1",
               "ESMCFOLD_CCD_PATH": f"{MODEL_DIR}/ccd.pkl"}))
app = modal.App("marinfold-exp325-esmfold2-decoys", image=image)


@app.cls(gpu="H100", cpu=8, memory=65536, region="us-east", max_containers=5,
         timeout=3600, volumes={"/weights": weights, "/outputs": outputs})
class Predictor:
    """Keep weights resident across a fixed block of diffusion seeds."""

    @modal.enter()
    def setup(self) -> None:
        import torch
        from esm.models.esmfold2 import ESMFold2InputBuilder, EsmFold2Model

        started = time.monotonic()
        self.model = EsmFold2Model.from_pretrained(MODEL_DIR, device="cuda", load_esmc=False).eval()
        self.model.load_esmc(ESMC_DIR, precision="bf16")
        self.model.set_chunk_size(None)
        self.builder = ESMFold2InputBuilder()
        props = torch.cuda.get_device_properties(0)
        self.metadata = {
            "model_load_seconds": time.monotonic() - started,
            "gpu_name": props.name, "gpu_total_memory_gb": props.total_memory / 1e9,
            "gpu_compute_capability": f"{props.major}.{props.minor}",
            "hostname": socket.gethostname(), "platform": platform.platform(),
            "torch_version": str(torch.__version__), "model_nickname": "esmfold2",
            "runner_tag": "modal",
        }
        print(f"Loaded ESMFold2: {self.metadata}", flush=True)

    @modal.method()
    def predict(self, task: dict) -> dict:
        import torch
        from esm.models.esmfold2 import ProteinInput, StructurePredictionInput

        target, seeds = task["target"], task["seeds"]
        stem, sequence = target["target_id"], target["input_seq"]
        spi = StructurePredictionInput(sequences=[ProteinInput(id="A", sequence=sequence)])
        rows = []
        for seed in seeds:
            directory = Path("/outputs/esmfold2") / stem / str(seed)
            completed = directory / "result.json"
            if completed.exists():
                result = json.loads(completed.read_text())
                if result["protocol_sha256"] != PROTOCOL_SHA or result["sequence"] != sequence:
                    raise ValueError(f"Stale result: {completed}")
                rows.append(result)
                continue
            total_start = time.monotonic()
            torch.cuda.synchronize()
            started = time.monotonic()
            result = self.builder.fold(self.model, spi, num_loops=20, num_sampling_steps=100,
                                       num_diffusion_samples=1, seed=seed, lm_dropout=0.3)
            torch.cuda.synchronize()
            elapsed = time.monotonic() - started
            cif = result.complex.to_mmcif()
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "structure.cif").write_text(cif)
            row = {
                "stem": stem, "seed": seed, "sequence": sequence,
                "n_residues": len(sequence), "n_pairs": 0, "mode": "single_sequence",
                "elapsed_seconds": elapsed, "total_seconds": time.monotonic() - total_start + self.metadata["model_load_seconds"],
                "ptm": float(result.ptm), "mean_plddt": float(result.plddt.mean()),
                "structure_sha256": hashlib.sha256(cif.encode()).hexdigest(),
                "protocol_sha256": PROTOCOL_SHA, "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                **self.metadata,
            }
            completed.write_text(json.dumps(row, indent=2))
            outputs.commit()
            rows.append(row)
            print(f"{stem} seed={seed} {elapsed:.1f}s", flush=True)
        return {"stem": stem, "rows": rows}


@app.function(image=modal.Image.debian_slim(python_version="3.12"), cpu=4,
              region="us-east", volumes={"/outputs": outputs}, timeout=900)
def collect(samples: list[tuple[str, int]]) -> bytes:
    """Archive the generated structures and their inference records."""
    outputs.reload()
    paths = [Path("esmfold2") / stem / str(seed) / filename
             for stem, seed in samples for filename in ("structure.cif", "result.json")]
    def read(path: Path) -> tuple[Path, bytes]:
        return path, (Path("/outputs") / path).read_bytes()

    buffer = io.BytesIO()
    with ThreadPoolExecutor(max_workers=24) as pool, tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path, content in pool.map(read, paths):
            info = tarfile.TarInfo(str(path))
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


@app.function(image=image, region="us-east", cpu=4, timeout=900,
              volumes={"/weights": weights})
def runtime_provenance() -> dict:
    """Record the cached image's package versions and verify staged weight bytes."""
    files = {}
    for name in ("model.safetensors", "config.json", "ccd.pkl"):
        path = Path(MODEL_DIR) / name
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        files[name] = {"sha256": digest, "bytes": path.stat().st_size}
    backbone = Path(ESMC_DIR)
    cached_ref = (backbone.parent.parent / "refs/main").read_text().strip()
    if cached_ref != ESMC_REVISION or len(list(backbone.parent.iterdir())) != 1:
        raise ValueError("Cached ESMC identity differs from the generation run")
    backbone_files = {}
    for path in sorted(backbone.iterdir()):
        if path.suffix not in (".json", ".safetensors"):
            continue
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        backbone_files[path.name] = {"sha256": digest, "bytes": path.stat().st_size}
    return {"packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
            "files": files, "esm_source_sha": SOURCE_SHA, "model_revision": MODEL_REVISION,
            "esmc_revision": ESMC_REVISION, "esmc_files": backbone_files,
            "model_dtype": "folding parameters fp32, ESMC backbone bf16; official internal autocast policy",
            "kernel_backend": "reference PyTorch, no Transformer Engine or flash-attn",
            "modal_run_url": "https://modal.com/apps/open-athena/main/ap-qU0fdFnYiy2KVeR0N93zBl",
            "smoke_run_url": "https://modal.com/apps/open-athena/main/ap-CxHmgE1w8yccsXJ9LAYyNP"}


@app.local_entrypoint()
def run(smoke: bool = False, collect_only: bool = False, provenance_only: bool = False) -> None:
    """Generate all seeds, or perform a two-seed API/throughput smoke test."""
    if provenance_only:
        (ROOT / "data/esmfold2_decoy_run.json").write_text(json.dumps(runtime_provenance.remote(), indent=2) + "\n")
        return
    low = {r["stem"] for r in csv.DictReader((ROOT / "data/confidence_targets.csv").open()) if int(r["msa_depth"]) < 10}
    targets = [r for r in csv.DictReader((ROOT / "scratch/helico/confidence/targets.csv").open()) if r["target_id"] in low]
    if len(targets) != 5:
        raise ValueError("Expected exactly five low-depth natural targets")
    if smoke:
        targets = sorted(targets, key=lambda r: len(r["input_seq"]))[:1]
    tasks = [{"target": target, "seeds": list(range(start, min(start + 20, 100)))}
             for start in range(0, 100, 20) for target in targets]
    if smoke:
        tasks = [{"target": targets[0], "seeds": [0, 1]}]
    if not collect_only:
        failures = []
        for result in Predictor().predict.map(tasks, order_outputs=False, return_exceptions=True):
            if isinstance(result, Exception):
                failures.append(repr(result))
            else:
                print(f"Completed {result['stem']}: {len(result['rows'])} seeds")
        if failures:
            raise RuntimeError(f"Prediction tasks failed: {failures}")
    scratch = ROOT / "scratch/structured_decoys"
    if smoke:
        scratch = scratch / "smoke"
    scratch.mkdir(parents=True, exist_ok=True)
    payload = collect.remote([(task["target"]["target_id"], seed) for task in tasks for seed in task["seeds"]])
    (scratch / "esmfold2_predictions.tar.gz").write_bytes(payload)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        archive.extractall(scratch, filter="data")
    records = [json.loads(p.read_text()) for p in sorted((scratch / "esmfold2").glob("*/*/result.json"))]
    if not smoke and len(records) != 500:
        raise ValueError(f"Expected 500 predictions, got {len(records)}")
    if not smoke:
        with (ROOT / "data/esmfold2_decoy_timings.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        (ROOT / "data/esmfold2_decoy_protocol.json").write_text(json.dumps({**PROTOCOL, "protocol_sha256": PROTOCOL_SHA}, indent=2) + "\n")
    print(f"Collected {len(records)} structures ({len(payload) / 1e6:.1f} MB)")
