"""Matched baseline predictions on Modal; retain every single-seed structure.

BASELINE_MODEL=esmfold2 uv run modal run --detach run_baselines.py --smoke
BASELINE_MODEL=af3 uv run modal run --detach run_baselines.py --smoke
Drop --smoke after checking outputs. --collect-only recovers completed files.
"""

import datetime as dt
import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import socket
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import modal

from baseline_config import AF3_SHA, ESM_REVISION, ESM_SHA

HERE = Path(__file__).resolve().parent
VARIANT = os.environ.get("BASELINE_MODEL", "esmfold2")
if VARIANT not in {"esmfold2", "af3"}:
    raise ValueError(VARIANT)
OUTPUT = modal.Volume.from_name("marinfold-exp381-results")
WEIGHTS = modal.Volume.from_name(
    "esmfold2-weights" if VARIANT == "esmfold2" else "marinfold-exp325-alphafold"
)
base = modal.Image.debian_slim(python_version="3.12").apt_install(
    "git", "gcc", "g++", "make", "zlib1g-dev", "zstd"
)
if VARIANT == "af3":
    image = (
        base.pip_install("pip", "setuptools", "wheel")
        .run_commands(
            "git clone https://github.com/google-deepmind/alphafold3.git /opt/alphafold3",
            f"git -C /opt/alphafold3 checkout {AF3_SHA}",
            "python -m pip install --no-cache-dir -r /opt/alphafold3/requirements.txt",
            "python -m pip install --no-cache-dir --no-deps /opt/alphafold3",
            "build_data",
        )
        .env(
            {
                "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                "XLA_CLIENT_MEM_FRACTION": "0.90",
                "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false",
            }
        )
    )
else:
    image = base.pip_install(
        f"esm @ git+https://github.com/Biohub/esm.git@{ESM_SHA}"
    ).env(
        {
            "HF_HOME": "/weights/hf",
            "HF_HUB_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
image = (
    image.env({"BASELINE_MODEL": VARIANT})
    .add_local_dir(HERE / "inputs/baselines", "/root/baseline_inputs")
    .add_local_file(HERE / "data/baseline_plan.json", "/root/baseline_plan.json")
    .add_local_file(HERE / "baseline_config.py", "/root/baseline_config.py")
)
app = modal.App(f"marinfold-exp381-baseline-{VARIANT}", image=image)
REMOTE = Path("/results/baselines-v2")


def sha256(path: Path) -> str:
    """Hash files without buffering model weights."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@app.cls(
    gpu="H100",
    cpu=4,
    memory=65536,
    max_containers=4,
    timeout=7200,
    region="us-east",
    volumes={"/weights": WEIGHTS, "/results": OUTPUT},
)
class Predictor:
    """Resident official models, with one independently seeded structure per input."""

    @modal.enter()
    def setup(self) -> None:
        self.plan = json.loads(Path("/root/baseline_plan.json").read_text())
        self.plan_sha = sha256(Path("/root/baseline_plan.json"))
        started = time.monotonic()
        if VARIANT == "af3":
            import jax

            sys.path.insert(0, "/opt/alphafold3")
            import run_alphafold

            weights = Path("/weights/weights/af3")
            self.weights = {"af3.bin.zst": sha256(weights / "af3.bin.zst")}
            if self.weights["af3.bin.zst"] != self.plan["af3"]["weights_sha256"]:
                raise ValueError("AF3 weight digest mismatch")
            self.api = run_alphafold
            self.model = run_alphafold.ModelRunner(
                config=run_alphafold.make_model_config(
                    num_diffusion_samples=1, num_recycles=10
                ),
                device=jax.local_devices(backend="gpu")[0],
                model_dir=weights,
            )
            if not self.model.model_params:
                raise ValueError("No AF3 parameters")
            versions = {"jax": jax.__version__, "alphafold3_source": AF3_SHA}
        else:
            import torch
            from esm.models.esmfold2 import ESMFold2InputBuilder, EsmFold2Model

            weights = (
                Path("/weights/hf/models--biohub--ESMFold2/snapshots") / ESM_REVISION
            )
            self.weights = {
                name: sha256(weights / name)
                for name in ("model.safetensors", "config.json", "ccd.pkl")
            }
            self.model = EsmFold2Model.from_pretrained(
                str(weights), device="cuda", esmc_precision="bf16"
            ).eval()
            self.builder = ESMFold2InputBuilder(ccd_cache=weights)
            torch.cuda.synchronize()
            versions = {
                p: importlib.metadata.version(p)
                for p in ("torch", "esm", "numpy", "transformers", "biotite")
            }
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
        self.meta = dict(
            model_load_seconds=time.monotonic() - started,
            gpu_name=gpu[0],
            gpu_total_memory_gb=float(gpu[1]) / 1024,
            gpu_compute_capability=gpu[2],
            hostname=socket.gethostname(),
            platform=platform.platform(),
            torch_version=versions.get("torch", "not_used"),
            runner_tag="modal-us-east",
            worker_id=os.environ["MODAL_TASK_ID"],
            versions=versions,
            weights_sha256=self.weights,
        )
        print(f"Loaded {VARIANT}; {self.meta}", flush=True)

    def predict_af3(self, task: dict, folder: Path) -> dict:
        """Featurize and fold a full independent AF3 seed using public interfaces."""
        import jax
        from alphafold3.common import folding_input
        from alphafold3.constants import chemical_components
        from alphafold3.data import featurisation

        msa = ""
        if task["model"] == "af3_msa":
            path = Path("/root/baseline_inputs") / f"{task['stem']}.a3m"
            if sha256(path) != task["msa_sha256"]:
                raise ValueError("MSA checksum mismatch")
            msa = path.read_text()
        payload = dict(
            name=task["stem"].lower(),
            modelSeeds=[task["seed"]],
            dialect="alphafold3",
            version=1,
            sequences=[
                {
                    "protein": {
                        "id": "A",
                        "sequence": task["sequence"],
                        "unpairedMsa": msa,
                        "pairedMsa": "",
                        "templates": [],
                    }
                }
            ],
        )
        (folder / "input.json").write_text(json.dumps(payload) + "\n")
        started = time.monotonic()
        fold_input = folding_input.Input.from_json(json.dumps(payload))
        ccd = chemical_components.Ccd(user_ccd=fold_input.user_ccd)
        examples = featurisation.featurise_input(
            fold_input=fold_input, buckets=(256, 384), ccd=ccd, verbose=False
        )
        if len(examples) != 1:
            raise ValueError("Expected one seed")
        feature_time = time.monotonic() - started
        started = time.monotonic()
        prediction = self.model.run_inference(
            examples[0], jax.random.PRNGKey(task["seed"])
        )
        elapsed = time.monotonic() - started
        results = list(
            self.model.extract_inference_results(
                examples[0], prediction, task["stem"].lower()
            )
        )
        if len(results) != 1:
            raise ValueError("Expected one AF3 diffusion sample")
        result = results[0]
        # The official writer retains its output notices. Unrounded selector
        # values come directly from metadata, not the two-decimal summary JSON.
        self.api.write_outputs(
            [
                self.api.ResultsForSeed(
                    seed=task["seed"],
                    inference_results=results,
                    full_fold_input=fold_input,
                )
            ],
            output_dir=folder,
            job_name=task["stem"].lower(),
        )
        (folder / "prediction.cif").write_bytes(
            (folder / f"{task['stem'].lower()}_model.cif").read_bytes()
        )
        return dict(
            elapsed_seconds=elapsed,
            feature_seconds=feature_time,
            ranking_score=float(result.metadata["ranking_score"]),
            ptm=float(result.metadata["ptm"]),
            confidence_selector="ranking_score",
            n_recycles=10,
        )

    def predict_esm(self, task: dict, folder: Path) -> dict:
        """Run one full ESMFold2 draw, retaining native full-precision pTM."""
        import torch
        from esm.models.esmfold2 import ProteinInput, StructurePredictionInput

        spi = StructurePredictionInput(
            sequences=[ProteinInput(id="A", sequence=task["sequence"])]
        )
        torch.cuda.synchronize()
        started = time.monotonic()
        result = self.builder.fold(
            self.model,
            spi,
            num_loops=20,
            num_sampling_steps=200,
            num_diffusion_samples=1,
            seed=task["seed"],
            lm_dropout=0.3,
        )
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        (folder / "prediction.cif").write_text(result.complex.to_mmcif())
        return dict(
            elapsed_seconds=elapsed,
            feature_seconds=None,
            ranking_score=float(result.ptm),
            ptm=float(result.ptm),
            mean_plddt=float(result.plddt.float().mean().cpu()),
            confidence_selector="pTM",
            n_recycles=20,
        )

    @modal.method()
    def predict(self, tasks: list[dict], runner_sha: str) -> list[dict]:
        """Commit every prediction before starting the next; resume by immutable identity."""
        rows = []
        for task in tasks:
            folder = REMOTE / task["task_id"]
            marker = folder / "result.json"
            if marker.exists():
                result = json.loads(marker.read_text())
                if (
                    result["plan_sha256"] != self.plan_sha
                    or result["seed"] != task["seed"]
                ):
                    raise ValueError("Incompatible completed prediction")
                rows.append(result)
                continue
            started = time.monotonic()
            folder.mkdir(parents=True, exist_ok=True)
            prediction = (
                self.predict_af3(task, folder)
                if VARIANT == "af3"
                else self.predict_esm(task, folder)
            )
            if not math.isfinite(prediction["ranking_score"]):
                raise ValueError("Nonfinite confidence selector")
            row = dict(
                **task,
                **prediction,
                **self.meta,
                n_pairs=0,
                mode=task["model"],
                model_nickname=task["model"],
                n_seeds=1,
                n_samples_per_seed=1,
                total_seconds=time.monotonic() - started,
                plan_sha256=self.plan_sha,
                runner_sha256=runner_sha,
                prediction_sha256=sha256(folder / "prediction.cif"),
                timestamp_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            )
            marker.write_text(json.dumps(row, indent=2) + "\n")
            OUTPUT.commit()
            print(
                f"DONE {task['task_id']} {prediction['elapsed_seconds']:.2f}s",
                flush=True,
            )
            rows.append(row)
        return rows


@app.function(
    image=modal.Image.debian_slim(python_version="3.12")
    .env({"BASELINE_MODEL": VARIANT})
    .add_local_file(HERE / "baseline_config.py", "/root/baseline_config.py"),
    volumes={"/results": OUTPUT},
    timeout=1800,
    region="us-east",
)
def collect(model: str) -> bytes:
    """Export only this experiment's prediction outputs, never model parameters."""
    OUTPUT.reload()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for marker in sorted(REMOTE.glob(f"*__{model}__r*/result.json")):
            folder = marker.parent
            for name in ("result.json", "prediction.cif", "TERMS_OF_USE.md"):
                if (folder / name).exists():
                    archive.add(folder / name, arcname=f"{folder.name}/{name}")
    return buffer.getvalue()


@app.local_entrypoint()
def main(smoke: bool = False, collect_only: bool = False) -> None:
    """Run the fixed seeds, retaining the smoke predictions in the final budget."""
    tasks = json.loads((HERE / "inputs/baselines/tasks.json").read_text())
    models = ["esmfold2"] if VARIANT == "esmfold2" else ["af3_single", "af3_msa"]
    tasks = [t for t in tasks if t["model"] in models]
    if smoke:
        tasks = [t for t in tasks if t["stem"] == "8bux_A" and t["rollout"] == 0]
    groups = [
        [t for t in tasks if t["model"] == model and t["stem"] == stem]
        for model in models
        for stem in sorted({t["stem"] for t in tasks})
    ]
    errors = []
    if not collect_only:
        worker = Predictor()
        for result in worker.predict.starmap(
            [(g, sha256(Path(__file__))) for g in groups], return_exceptions=True
        ):
            if isinstance(result, Exception):
                errors.append(repr(result))
                print(f"FAILED {result}", flush=True)
    for model in models:
        payload = collect.remote(model)
        root = HERE / "scratch/baselines"
        root.mkdir(exist_ok=True)
        path = root / f"{model}.tar.gz"
        path.write_bytes(payload)
        with tarfile.open(path) as archive:
            archive.extractall(root / "results", filter="data")
        print(f"Recovered {model}: {len(payload)} bytes", flush=True)
    if errors:
        raise RuntimeError(errors)
