"""Run the frozen AlphaFold baselines; AF_VARIANT selects af2 or af3.

Run prepare_alphafold.py and stage_alphafold.py first. --build-only builds the
environment without a GPU; --limit 1 gates each full run on one real protein.
The volume contains private weights. collect() exports predictions only.
"""

import datetime as dt
import concurrent.futures
import hashlib
import io
import json
import os
import platform
import socket
import subprocess
import tarfile
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
VARIANT = os.environ.get("AF_VARIANT", "af2")
if VARIANT not in {"af2", "af3"}:
    raise ValueError(VARIANT)
AF3_SHA = "3c89cc7b89aa7042b72885af9016a45b262da008"
VOLUME = modal.Volume.from_name("marinfold-exp325-alphafold", create_if_missing=True)
base = modal.Image.debian_slim(python_version="3.12").apt_install("git", "gcc", "g++", "make", "zlib1g-dev", "zstd")
if VARIANT == "af2":
    IMAGE = base.pip_install("colabfold[alphafold]==1.6.2", "alphafold-colabfold==2.3.18",
                             "jax[cuda12]==0.10.2", "numpy==2.4.6")
else:
    IMAGE = (base.pip_install("pip", "setuptools", "wheel")
             .run_commands("git clone https://github.com/google-deepmind/alphafold3.git /opt/alphafold3",
                           f"git -C /opt/alphafold3 checkout {AF3_SHA}",
                           "python -m pip install --no-cache-dir -r /opt/alphafold3/requirements.txt",
                           "python -m pip install --no-cache-dir --no-deps /opt/alphafold3",
                           "build_data"))
IMAGE = (IMAGE.env({"AF_VARIANT": VARIANT, "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                   "XLA_CLIENT_MEM_FRACTION": "0.90", "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false"})
         .add_local_file(ROOT / "generation/alphafold_worker.py", "/root/alphafold_worker.py"))
app = modal.App(f"marinfold-exp325-{VARIANT}", image=IMAGE)


@app.cls(gpu="H100", cpu=4, memory=32768, max_containers=8, timeout=7200,
         region="us-east", volumes={"/work": VOLUME})
class Predictor:
    """One resident predictor per GPU with resumable per-protein output."""

    @modal.enter()
    def setup(self) -> None:
        import jax
        from alphafold_worker import AF2, AF3

        VOLUME.reload()
        protocol = json.loads(Path("/work/protocol.json").read_text())
        if VARIANT == "af3" and protocol["af3"]["source_sha"] != AF3_SHA:
            raise ValueError("Staged AF3 source revision differs from this runner")
        for name, pin in protocol["weights"].items():
            if name.startswith(VARIANT + "/"):
                with (Path("/work/weights") / name).open("rb") as stream:
                    actual = hashlib.file_digest(stream, "sha256").hexdigest()
                if actual != pin["sha256"]:
                    raise ValueError(f"Weight digest mismatch: {name}")
        started = time.monotonic()
        self.model = (AF2 if VARIANT == "af2" else AF3)(Path("/work/weights") / VARIANT)
        load = time.monotonic() - started
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader,nounits"], text=True,
        ).strip().split(", ")
        self.metadata = dict(model_load_seconds=load, gpu_name=gpu[0], gpu_total_memory_gb=float(gpu[1]) / 1024,
                             gpu_compute_capability=gpu[2], hostname=socket.gethostname(), platform=platform.platform(),
                             torch_version="not_used", jax_version=jax.__version__, runner_tag="modal-us-east",
                             timing_scope="Inference calls include cold-shape JIT; exclude feature preparation and file writes")
        self.protocol = protocol

    @modal.method()
    def predict(self, record: dict) -> dict:
        """Persist every candidate, selected output and timing before returning."""
        from alphafold_worker import read_msa

        output = Path("/work/results") / VARIANT / record["stem"]
        marker = output / "complete.json"
        if marker.exists():
            return json.loads(marker.read_text())
        started = time.monotonic()
        output.mkdir(parents=True, exist_ok=True)
        msa_path = Path("/work/inputs") / f"{record['stem']}.a3m.gz"
        if hashlib.sha256(msa_path.read_bytes()).hexdigest() != record["msa_sha256"]:
            raise ValueError("Alignment digest mismatch")
        prediction = self.model.predict(record, read_msa(msa_path), output)
        (output / "input.json").write_text(json.dumps(record, indent=2) + "\n")
        report = {**record, **prediction, **self.metadata, "mode": "msa_no_templates", "n_pairs": 0,
                  "model_nickname": VARIANT, "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                  "total_seconds": self.metadata["model_load_seconds"] + time.monotonic() - started}
        marker.write_text(json.dumps(report, indent=2) + "\n")
        VOLUME.commit()
        print(f"{VARIANT} {record['stem']}: {prediction['elapsed_seconds']:.1f}s", flush=True)
        return report


@app.function(image=modal.Image.debian_slim(python_version="3.12"),
              volumes={"/work": VOLUME}, cpu=4, memory=8192, timeout=3600, region="us-east")
def collect(variant: str) -> bytes:
    """Export this predictor's outputs; private parameter files are excluded."""
    VOLUME.reload()
    if variant not in {"af2", "af3"}:
        raise ValueError(variant)
    root = Path("/work/results") / variant
    directories = sorted(p.parent for p in root.glob("*/complete.json"))
    if not directories:
        raise ValueError("No completed predictions to export")
    files = sorted(p for directory in directories for p in directory.rglob("*") if p.is_file())
    buffer = io.BytesIO()
    # Each volume file is backed by object storage. Overlap read latency while
    # keeping only a bounded batch in memory and writing a single valid archive.
    with concurrent.futures.ThreadPoolExecutor(max_workers=24) as pool:
        with tarfile.open(fileobj=buffer, mode="w:gz", compresslevel=3) as archive:
            for start in range(0, len(files), 64):
                for path, payload, mtime in pool.map(read_export_file, files[start:start + 64]):
                    info = tarfile.TarInfo(str(path.relative_to(root.parent)))
                    info.size, info.mtime, info.mode = len(payload), mtime, 0o644
                    archive.addfile(info, io.BytesIO(payload))
    print(f"Archived {len(directories)} {variant} predictions, {len(files)} files", flush=True)
    return buffer.getvalue()


def read_export_file(path: Path) -> tuple[Path, bytes, int]:
    """Read one immutable completed output for the bounded archive batch."""
    return path, path.read_bytes(), int(path.stat().st_mtime)


@app.local_entrypoint()
def main(limit: int = 0, build_only: bool = False, collect_only: bool = False) -> None:
    """Build, smoke, run or collect the frozen baseline without changing settings."""
    if build_only:
        print(f"Built {VARIANT} environment; no predictions submitted")
        return
    records = json.loads((ROOT / "scratch/alphafold/inputs/targets.json").read_text())
    if limit:
        records = sorted(records, key=lambda row: (row["n_residues"], row["stem"]))[:limit]
    # Round-robin scheduling over descending lengths spreads long proteins;
    # padded shapes are reused inside each resident worker.
    records = sorted(records, key=lambda row: (-row["n_residues"], row["stem"]))
    reports, errors = [], []
    if not collect_only:
        for result in Predictor().predict.map(records, return_exceptions=True):
            if isinstance(result, Exception):
                errors.append(str(result))
                print(f"FAILED: {result}", flush=True)
            else:
                reports.append(result)
        (ROOT / "scratch/alphafold" / f"{VARIANT}_run.json").write_text(json.dumps({
            "variant": VARIANT, "reports": reports, "errors": errors,
        }, indent=2) + "\n")
    payload = collect.remote(VARIANT)
    path = ROOT / "scratch/alphafold" / f"{VARIANT}_predictions.tar.gz"
    path.write_bytes(payload)
    with tarfile.open(path) as archive:
        if any(m.name.split("/")[0] != VARIANT for m in archive.getmembers()):
            raise ValueError("Export contains the wrong predictor's results")
        archive.extractall(ROOT / "scratch/alphafold/results", filter="data")
    if errors:
        raise RuntimeError(f"{len(errors)} targets failed; partial outputs recovered. Rerun to resume.")
    print(f"Collected {VARIANT}: {path}")
