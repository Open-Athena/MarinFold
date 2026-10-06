"""Fresh full AF3 runs for the five low-depth proteins; one sample per seed.

AF_VARIANT=af3 uv run --project generation modal run generation/run_af3_sampling.py
Pass --budget 1000 to extend the same frozen seed prefix. Completed seeds resume.
Only prediction outputs are exported from the private weights volume.
"""

import concurrent.futures
import datetime as dt
import hashlib
import io
import json
import platform
import socket
import subprocess
import tarfile
import time
from pathlib import Path

import modal

from run_af_baselines import AF3_SHA, IMAGE, ROOT, VARIANT, VOLUME, read_export_file

if VARIANT != "af3":
    raise ValueError("Set AF_VARIANT=af3 before importing this runner")
PROTOCOL_PATH = ROOT / "data/af3_sampling_protocol.json"
IMAGE = (IMAGE.add_local_file(PROTOCOL_PATH, "/root/af3_sampling_protocol.json")
         .add_local_file(ROOT / "generation/run_af_baselines.py", "/root/run_af_baselines.py"))
app = modal.App("marinfold-exp325-af3-sampling", image=IMAGE)
REMOTE = Path("/work/af3_sampling/results")


@app.cls(gpu="H100", cpu=4, memory=32768, max_containers=20, timeout=7200,
         region="us-east", volumes={"/work": VOLUME})
class Sampler:
    """Resident model; each seed independently featurises and runs the full model."""

    @modal.enter()
    def setup(self) -> None:
        import jax
        from alphafold_worker import AF3

        VOLUME.reload()
        payload = Path("/root/af3_sampling_protocol.json").read_bytes()
        self.protocol = json.loads(payload)
        self.protocol_sha = hashlib.sha256(payload).hexdigest()
        if self.protocol["af3_source_sha"] != AF3_SHA:
            raise ValueError("AF3 source mismatch")
        worker_sha = hashlib.sha256(Path("/root/alphafold_worker.py").read_bytes()).hexdigest()
        if worker_sha != self.protocol["worker_sha256"]:
            raise ValueError("AF3 adapter changed after protocol freeze")
        weights = Path("/work/weights/af3")
        with (weights / "af3.bin.zst").open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != self.protocol["weights_sha256"]:
                raise ValueError("AF3 weights mismatch")
        started = time.monotonic()
        self.model = AF3(weights, diffusion_samples=1, recycles=10)
        load = time.monotonic() - started
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader,nounits"],
            text=True).strip().split(", ")
        self.metadata = dict(model_load_seconds=load, gpu_name=gpu[0],
                             gpu_total_memory_gb=float(gpu[1]) / 1024, gpu_compute_capability=gpu[2],
                             hostname=socket.gethostname(), platform=platform.platform(),
                             torch_version="not_used", jax_version=jax.__version__, runner_tag="modal-us-east",
                             timing_scope="Inference includes cold-shape JIT; excludes features and writes. "
                             "Model setup is shared across seeds; total_seconds includes one setup allocation.")

    @modal.method()
    def predict(self, request: dict) -> list[dict]:
        """Commit each bounded seed chunk, with immutable input checks on resume."""
        from alphafold_worker import read_msa

        record = next(r for r in self.protocol["targets"] if r["stem"] == request["stem"])
        msa_path = Path("/work/inputs") / f"{record['stem']}.a3m.gz"
        if hashlib.sha256(msa_path.read_bytes()).hexdigest() != record["msa_sha256"]:
            raise ValueError("MSA digest mismatch")
        msa = read_msa(msa_path)
        reports = []
        for seed in request["seeds"]:
            if not self.protocol["seed_start"] <= seed < self.protocol["seed_start"] + self.protocol["max_budget"]:
                raise ValueError(f"Seed outside frozen budget: {seed}")
            output = REMOTE / record["stem"] / f"seed-{seed}"
            marker = output / "complete.json"
            if marker.exists():
                report = json.loads(marker.read_text())
                if report["protocol_sha256"] != self.protocol_sha or report["seed"] != seed:
                    raise ValueError(f"Incompatible completed seed: {marker}")
                reports.append(report)
                continue
            started = time.monotonic()
            output.mkdir(parents=True, exist_ok=True)
            prediction = self.model.predict(record, msa, output, seeds=(seed,))
            if prediction["n_samples"] != 1:
                raise ValueError("Expected exactly one full-model sample per seed")
            report = {**record, **prediction, **self.metadata, "seed": seed,
                      "protocol_sha256": self.protocol_sha, "mode": "msa_no_templates",
                      "n_pairs": 0, "model_nickname": "af3", "n_samples_per_seed": 1,
                      "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                      "total_seconds": self.metadata["model_load_seconds"] + time.monotonic() - started}
            marker.write_text(json.dumps(report, indent=2) + "\n")
            reports.append(report)
        VOLUME.commit()
        print(f"Completed {record['stem']}: seeds {request['seeds'][0]}–{request['seeds'][-1]}", flush=True)
        return reports


@app.function(volumes={"/work": VOLUME},
              cpu=4, memory=16384, timeout=3600, region="us-east")
def collect(stem: str) -> bytes:
    """Export completed seeds for one protein, never private weights."""
    VOLUME.reload()
    directories = sorted(p.parent for p in (REMOTE / stem).glob("*/complete.json"))
    if not directories:
        raise ValueError(f"No completed seeds: {stem}")
    # Keep the official candidate CIF, its summary, metadata and output notices;
    # omit duplicate selected CIFs and embedded MSA copies in official input JSON.
    files = []
    for directory in directories:
        files.append(directory / "complete.json")
        files.extend(directory.glob("seed-*_sample-0/*_model.cif"))
        files.extend(directory.glob("seed-*_sample-0/*_summary_confidences.json"))
        files.extend(directory.glob("TERMS_OF_USE.md"))
    buffer = io.BytesIO()
    with concurrent.futures.ThreadPoolExecutor(max_workers=24) as pool:
        with tarfile.open(fileobj=buffer, mode="w:gz", compresslevel=3) as archive:
            for start in range(0, len(files), 64):
                for path, payload, mtime in pool.map(read_export_file, files[start:start + 64]):
                    info = tarfile.TarInfo(str(path.relative_to(REMOTE)))
                    info.size, info.mtime, info.mode = len(payload), mtime, 0o644
                    archive.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


@app.local_entrypoint()
def main(budget: int = 100, chunk_size: int = 10, collect_only: bool = False) -> None:
    """Smoke the longest target, run a fixed prefix, then recover every output."""
    protocol = json.loads(PROTOCOL_PATH.read_text())
    if budget not in {1, 100, 1000} or chunk_size < 1:
        raise ValueError("Use budgets 1 (smoke), 100 or 1000 and a positive chunk size")
    targets = sorted(protocol["targets"], key=lambda r: (-r["n_residues"], r["stem"]))
    start = protocol["seed_start"]
    scratch = ROOT / "scratch/af3_sampling"
    scratch.mkdir(parents=True, exist_ok=True)
    errors = []
    if not collect_only:
        sampler = Sampler()
        sampler.predict.remote({"stem": targets[0]["stem"], "seeds": [start]})
        requests = [{"stem": r["stem"], "seeds": list(range(first, min(first + chunk_size, start + budget)))}
                    for first in range(start, start + budget, chunk_size) for r in targets]
        for result in sampler.predict.map(requests, return_exceptions=True):
            if isinstance(result, Exception):
                errors.append(str(result))
                print(f"FAILED: {result}", flush=True)
        (scratch / f"run_{budget}.json").write_text(json.dumps({"budget": budget, "errors": errors}, indent=2) + "\n")
    for record in targets:
        payload = collect.remote(record["stem"])
        path = scratch / f"{record['stem']}.tar.gz"
        path.write_bytes(payload)
        with tarfile.open(path) as archive:
            if any(m.name.split("/")[0] != record["stem"] for m in archive.getmembers()):
                raise ValueError("Unexpected path in prediction export")
            archive.extractall(scratch / "results", filter="data")
        print(f"Collected {path.name}: {len(payload):,} bytes", flush=True)
    if errors:
        raise RuntimeError(f"{len(errors)} chunks failed; recovered partial results. Rerun to resume.")
