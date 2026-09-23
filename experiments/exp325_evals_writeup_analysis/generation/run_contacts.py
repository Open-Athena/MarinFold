"""Run the fixed exp277 checkpoint on the publication test split on Modal.

Stage the 5.9 GB public checkpoint once into a region-pinned volume, verify every
SHA256, smoke-test one protein, then run eight independent H100 shards. The
checkpoint and all outputs persist across interruption. No training or tuning.
"""

import hashlib
import io
import json
import os
import socket
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TAG = "exp277-step266344-test-v1"
REGION = "us-east"
MODEL_URL = "https://huggingface.co/buckets/open-athena/MarinFold/resolve/checkpoints/contacts-v1-exp277-m2-p06-full-epoch-1.5B/hf/step-266344"
DATA_URL = "https://huggingface.co/buckets/open-athena/MarinFold/resolve/data/contacts-v1-foldbench-monomers-exp245"
INPUTS = {
    "per_protein.csv.gz": "b9670c3547212bf1b23de3247c1b2cba881e5e5abdc37cd905c70a96145449f2",
    "eval_targets_foldbench_monomers.parquet": "2eb4f1fee148fe2d6601bd171ef6e9431b96f38c82eaed1ad119a069a13f1fb8",
    "gt_universe_scored.jsonl": "f30c23e3d2fbab245755fc01548388b41730ddfa45da87325539698cadb153e5",
    "eval_sets.csv": "b13d060a091240921bc8466acecc9fa6ccbb45a56de4efda02cd903f6abf9861",
}
IMAGE = (
    modal.Image.from_registry("vllm/vllm-openai:v0.9.2", setup_dockerfile_commands=["RUN ln -s /usr/bin/python3 /usr/bin/python"])
    .entrypoint([])
    .pip_install("fsspec", "pyarrow", "pandas")
    .pip_install("marinfold @ git+https://github.com/Open-Athena/MarinFold.git@d1bea417a64cc042ad931422200c3edeb873f2e0#subdirectory=marinfold", extra_options="--no-deps")
    .add_local_file(str(HERE / "score_rollout_worker.py"), "/root/score_rollout_worker.py")
    .add_local_file(str(HERE / "checkpoint_manifest.json"), "/root/checkpoint_manifest.json")
)
app = modal.App("marinfold-exp325-contacts", image=IMAGE)
volume = modal.Volume.from_name("marinfold-exp325", create_if_missing=True)


def digest(path: Path) -> str:
    """Stream a SHA256 without loading weights into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(url: str, path: Path, expected: str) -> None:
    """Cache an immutable public file and reject altered bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(url, temporary)
        if digest(temporary) != expected:
            raise ValueError(f"Digest mismatch downloading {url}")
        temporary.replace(path)
    if digest(path) != expected:
        raise ValueError(f"Cached file digest mismatch: {path}")


@app.function(volumes={"/data": volume}, region=REGION, timeout=1800)
def stage() -> dict:
    """Transfer and verify one published checkpoint and three small inputs."""
    import pandas as pd

    manifest = json.loads(Path("/root/checkpoint_manifest.json").read_text())
    for name, record in manifest["files"].items():
        download(f"{MODEL_URL}/{name}", Path("/data/model") / name, record["sha256"])
    for name, expected in INPUTS.items():
        download(f"{DATA_URL}/{name}", Path("/data/inputs") / name, expected)
    targets = pd.read_parquet("/data/inputs/eval_targets_foldbench_monomers.parquet")
    sets = pd.read_csv("/data/inputs/eval_sets.csv")
    wanted = set(sets.loc[(sets.eval_set == "eval-test") & (sets.scorable == 1), "stem"])
    targets = targets[targets.stem.isin(wanted)]
    if len(targets) != 217 or set(targets.stem) != wanted:
        raise ValueError("Publication test target set differs from frozen 217 units")
    targets.to_parquet("/data/inputs/test.parquet", index=False)
    volume.commit()
    return {"targets": len(targets), "model_bytes": sum(r["size"] for r in manifest["files"].values())}


@app.function(gpu="H100", cpu=8, memory=65536, timeout=7200,
              max_containers=8, region=REGION, volumes={"/data": volume})
def infer(shard: int, smoke: bool = False) -> str:
    """Run one shard; outputs and completion markers are durable after each chunk."""
    volume.reload()
    os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
    with socket.socket() as sock:
        sock.bind(("", 0))
        os.environ["VLLM_PORT"] = str(sock.getsockname()[1])
    label = "smoke" if smoke else TAG
    out = Path("/data/results") / label
    for directory in ("scores", "timings", "complete", "failures", "rollouts"):
        (out / directory).mkdir(parents=True, exist_ok=True)
    args = [sys.executable, "/root/score_rollout_worker.py", "--model", "/data/model",
            "--model-manifest-b64", "e30=", "--targets", "/data/inputs/test.parquet",
            "--out", "/data/results", "--label", label, "--shard", f"{shard}/8",
            "--n-rollouts", "100", "--temperature", "1", "--top-p", "0.95",
            "--top-k", "-1", "--contact-mult", "6", "--accept-unfinished",
            "--seed", "0", "--chunk", "1", "--runner-tag", "modal"]
    if smoke:
        args += ["--limit", "1"]
    try:
        subprocess.run(args, check=True)
    finally:
        volume.commit()
    return f"completed {label} shard {shard}/8"


@app.function(region=REGION, cpu=4, volumes={"/data": volume}, timeout=1200)
def collect() -> bytes:
    """Return small raw evaluation artifacts; leave model weights on the volume."""
    volume.reload()
    result = io.BytesIO()
    with tarfile.open(fileobj=result, mode="w:gz") as archive:
        archive.add(Path("/data/results") / TAG, arcname="results")
        archive.add("/data/inputs", arcname="inputs")
    return result.getvalue()


@app.local_entrypoint()
def run(collect_only: bool = False) -> None:
    """Smoke-test, finish all 217 targets, then retrieve raw artifacts for preprocessing."""
    if not collect_only:
        print(stage.remote())
        print(infer.remote(0, True))
        failures = []
        for result in infer.map(range(8), return_exceptions=True):
            print(result)
            if isinstance(result, Exception):
                failures.append(str(result))
        if failures:
            raise RuntimeError(f"Contact shards failed; completed chunks persist: {failures}")
    destination = ROOT / "scratch" / "contacts"
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(collect.remote()), mode="r:gz") as archive:
        archive.extractall(destination, filter="data")
    print(f"Retrieved evaluation to {destination}")
