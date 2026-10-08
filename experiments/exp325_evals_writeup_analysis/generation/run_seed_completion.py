"""Complete frozen sparse-oracle prompts on five H100s using the cached model."""

import io
import os
import socket
import subprocess
import sys
import tarfile
from pathlib import Path

import modal

from run_contacts import IMAGE, REGION, volume

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
IMAGE = (IMAGE.add_local_file(str(HERE / "run_contacts.py"), "/root/run_contacts.py")
    .add_local_file(str(HERE / "seed_completion_worker.py"), "/root/seed_completion_worker.py")
    .add_local_file(str(HERE / "seed_completion.py"), "/root/seed_completion.py")
    .add_local_dir(str(ROOT / "scratch/seed_completion/inputs"), "/root/inputs"))
app = modal.App("marinfold-exp325-seed-completion", image=IMAGE)


@app.function(gpu="H100", cpu=8, memory=65536, timeout=7200,
              max_containers=5, region=REGION, volumes={"/data": volume})
def infer(stem: str) -> str:
    """Keep the model resident across all seven conditions for this protein."""
    volume.reload()
    os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
    with socket.socket() as sock:
        sock.bind(("", 0))
        os.environ["VLLM_PORT"] = str(sock.getsockname()[1])
    try:
        subprocess.run([sys.executable, "/root/seed_completion_worker.py", "--stem", stem], check=True)
    finally:
        volume.commit()
    return stem


@app.function(region=REGION, volumes={"/data": volume}, timeout=1200)
def collect() -> bytes:
    """Retrieve reproducibility artifacts without transferring model weights."""
    volume.reload()
    result = io.BytesIO()
    with tarfile.open(fileobj=result, mode="w:gz") as archive:
        archive.add("/data/seed-completion-v1", arcname="results")
    return result.getvalue()


@app.local_entrypoint()
def run(collect_only: bool = False) -> None:
    """Dispatch the complete frozen cohort and collect raw prompts and continuations."""
    if not collect_only:
        failures = []
        for result in infer.map([p.stem for p in sorted((ROOT / "scratch/seed_completion/inputs").glob("*.json"))], return_exceptions=True):
            print(result)
            if isinstance(result, Exception):
                failures.append(repr(result))
        if failures:
            raise RuntimeError(f"Seed-completion failures: {failures}")
    raw = collect.remote()
    destination = ROOT / "scratch/seed_completion"
    (destination / "rollouts.tar.gz").write_bytes(raw)
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        archive.extractall(destination, filter="data")
    print(f"Retrieved {len(raw) / 1e6:.1f} MB of raw seeded rollouts")
