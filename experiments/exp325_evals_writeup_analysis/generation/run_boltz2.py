"""Frozen Boltz-2 benchmark: stage public weights, smoke, run, and export.

Reuses the archived AF/Protenix MSA input volume. Writes to a separate Boltz-2
volume; exports only completed per-target results, never model parameters.
"""

import concurrent.futures
import datetime as dt
import hashlib
import importlib.metadata
import io
import json
import platform
import shutil
import socket
import subprocess
import tarfile
import time
import urllib.request
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
SOURCE_SHA = "b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc"
HF_REVISION = "6fdef46d763fee7fbb83ca5501ccceff43b85607"
WEIGHTS = {
    "boltz2_conf.ckpt": "090e82ac8c92f5e943fa1b39e7410a44027bea7243c0bbb3caa67a77fc1428e1",
    "mols.tar": "39e076d96dbec6b4e86982bbda16f3a53a2a60c9bdc17828d88f6f9a0c7d1fd7",
}
CANONICAL_CCD = "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL UNK".split()
PROTOCOL = dict(model="Boltz-2", source_sha=SOURCE_SHA, weights_revision=HF_REVISION,
                weights=WEIGHTS, seed=42, diffusion_samples=25, recycling_steps=10,
                sampling_steps=200, max_parallel_samples=5, step_scale=1.5,
                max_msa_seqs=8192, subsample_msa=False, templates=False,
                constraints=False, affinity=False, use_potentials=False,
                selection="maximum upstream confidence_score", precision="bf16-mixed",
                msa_source="data/alphafold_inputs.json; identical archived query/alignment",
                recipe_source=f"https://github.com/jwohlwend/boltz/blob/{SOURCE_SHA}/docs/prediction.md")
VOLUME = modal.Volume.from_name("marinfold-exp325-boltz2", create_if_missing=True)
INPUTS = modal.Volume.from_name("marinfold-exp325-alphafold")
CPU = modal.Image.debian_slim(python_version="3.12")
IMAGE = (modal.Image.from_registry("nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04", add_python="3.12")
         .apt_install("git", "gcc", "g++", "libxrender1", "libxext6")
         .pip_install("torch==2.7.1", "cuequivariance_ops_cu12==0.6.1",
                      "cuequivariance_ops_torch_cu12==0.6.1", "cuequivariance_torch==0.6.1",
                      f"boltz @ git+https://github.com/jwohlwend/boltz.git@{SOURCE_SHA}")
         .env({"CUEQ_DEFAULT_CONFIG": "1", "CUEQ_DISABLE_AOT_TUNING": "1"})
         .add_local_file(ROOT / "generation/boltz2_worker.py", "/root/boltz2_worker.py"))
app = modal.App("marinfold-exp325-boltz2", image=IMAGE)


@app.function(image=CPU, volumes={"/work": VOLUME}, cpu=4, memory=8192, timeout=3600, region="us-east")
def stage_weights() -> None:
    """Download pinned public parameters directly on the prediction service."""
    cache = Path("/work/cache")
    cache.mkdir(parents=True, exist_ok=True)
    for name, expected in WEIGHTS.items():
        path = cache / name
        if not path.exists():
            temporary = path.with_suffix(path.suffix + ".partial")
            url = f"https://huggingface.co/boltz-community/boltz-2/resolve/{HF_REVISION}/{name}"
            urllib.request.urlretrieve(url, temporary)
            temporary.replace(path)
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"Parameter digest mismatch: {name}")
    # The archived queries contain only canonical amino acids and X. Avoid
    # tens of thousands of unrelated ligand files on the object-backed volume.
    # Scan a local copy: small random tar-header reads on a volume are costly.
    local_tar = Path("/tmp/boltz2-mols.tar")
    shutil.copyfile(cache / "mols.tar", local_tar)
    required = {f"mols/{name}.pkl" for name in CANONICAL_CCD}
    with tarfile.open(local_tar) as archive:
        members = [m for m in archive.getmembers() if m.name in required]
        if {m.name for m in members} != required:
            raise ValueError("Pinned CCD archive is missing a canonical residue")
        archive.extractall(cache, members=members, filter="data")
    (Path("/work") / "protocol.json").write_text(json.dumps(PROTOCOL, indent=2) + "\n")
    VOLUME.commit()
    print("Pinned Boltz-2 parameters staged and verified", flush=True)


@app.cls(gpu="H100", cpu=4, memory=32768, max_containers=16, timeout=7200,
         region="us-east", volumes={"/work": VOLUME, "/shared": INPUTS})
class Predictor:
    """Resident GPU predictor, one target at a time and resumable by input hash."""

    @modal.enter()
    def setup(self) -> None:
        from boltz2_worker import Boltz2Predictor

        VOLUME.reload()
        INPUTS.reload()
        if json.loads(Path("/work/protocol.json").read_text()) != PROTOCOL:
            raise ValueError("Staged protocol differs from the runner")
        with Path("/work/cache/boltz2_conf.ckpt").open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != WEIGHTS["boltz2_conf.ckpt"]:
                raise ValueError("Model checkpoint hash differs")
        started = time.monotonic()
        self.model = Boltz2Predictor(Path("/work/cache"))
        load = time.monotonic() - started
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader,nounits"], text=True,
        ).strip().split(", ")
        self.metadata = dict(model_load_seconds=load, gpu_name=gpu[0], gpu_total_memory_gb=float(gpu[1]) / 1024,
                             gpu_compute_capability=gpu[2], hostname=socket.gethostname(), platform=platform.platform(),
                             torch_version=importlib.metadata.version("torch"), runner_tag="modal-us-east",
                             packages={name: importlib.metadata.version(name) for name in
                                       ("boltz", "torch", "numpy", "pytorch-lightning", "cuequivariance_torch")},
                             timing_scope="Synchronized prediction batch; excludes input preparation and output writing")

    @modal.method()
    def predict(self, record: dict) -> dict:
        """Retain all samples and persist timing metadata before marking complete."""
        output = Path("/work/results/boltz2") / record["stem"]
        marker = output / "complete.json"
        protocol_hash = hashlib.sha256(json.dumps(PROTOCOL, sort_keys=True).encode()).hexdigest()
        if marker.exists():
            report = json.loads(marker.read_text())
            if report["protocol_sha256"] != protocol_hash or any(report[k] != v for k, v in record.items()):
                raise ValueError("Completed prediction uses different inputs or settings")
            return report
        started = time.monotonic()
        output.mkdir(parents=True, exist_ok=True)
        prediction = self.model.predict(record, Path("/shared/inputs") / f"{record['stem']}.a3m.gz", output)
        (output / "input.json").write_text(json.dumps(record, indent=2) + "\n")
        report = {**record, **prediction, **self.metadata, "mode": "msa_no_templates", "n_pairs": 0,
                  "model_nickname": "boltz2", "protocol_sha256": protocol_hash,
                  "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                  "total_seconds": self.metadata["model_load_seconds"] + time.monotonic() - started}
        marker.write_text(json.dumps(report, indent=2) + "\n")
        VOLUME.commit()
        print(f"boltz2 {record['stem']}: {prediction['elapsed_seconds']:.1f}s", flush=True)
        return report


def read_export_file(path: Path) -> tuple[Path, bytes, int]:
    """Read one immutable output within a bounded parallel archive batch."""
    return path, path.read_bytes(), int(path.stat().st_mtime)


@app.function(image=CPU, volumes={"/work": VOLUME}, cpu=4, memory=8192, timeout=3600, region="us-east")
def collect() -> bytes:
    """Export coordinates, scalar confidence, pLDDT and timings used in this analysis."""
    VOLUME.reload()
    root = Path("/work/results/boltz2")
    directories = sorted(p.parent for p in root.glob("*/complete.json"))
    if not directories:
        raise ValueError("No completed predictions")
    # The output schema is fixed. Enumerating feature directories and stat'ing
    # every excluded PAE/PDE matrix adds thousands of object-storage round trips.
    files = []
    for directory in directories:
        stem = directory.name
        files.extend(directory / name for name in
                     ("complete.json", "input.json", "candidates.json", "selected.cif", f"{stem}.yaml"))
        predictions = directory / "predictions" / stem
        for rank in range(25):
            tag = f"{stem}_model_{rank}"
            files.extend(predictions / name for name in
                         (f"{tag}.cif", f"confidence_{tag}.json", f"plddt_{tag}.npz"))
    files.sort()
    print(f"Exporting {len(directories)} targets, {len(files)} required files", flush=True)
    buffer = io.BytesIO()
    with concurrent.futures.ThreadPoolExecutor(max_workers=24) as pool:
        with tarfile.open(fileobj=buffer, mode="w:gz", compresslevel=3) as archive:
            for start in range(0, len(files), 64):
                for path, payload, mtime in pool.map(read_export_file, files[start:start + 64]):
                    info = tarfile.TarInfo(str(path.relative_to(root.parent)))
                    info.size, info.mtime, info.mode = len(payload), mtime, 0o644
                    archive.addfile(info, io.BytesIO(payload))
    print(f"Archived {len(directories)} Boltz-2 targets, {len(files)} files", flush=True)
    return buffer.getvalue()


@app.local_entrypoint()
def main(stage_only: bool = False, smoke: bool = False, collect_only: bool = False) -> None:
    """Stage, smoke unknown/long inputs, run the fixed set, or recover outputs."""
    scratch = ROOT / "scratch/boltz2"
    scratch.mkdir(parents=True, exist_ok=True)
    if stage_only:
        stage_weights.remote()
        (ROOT / "data/boltz2_inputs.json").write_text(json.dumps(PROTOCOL, indent=2) + "\n")
        return
    records = json.loads((ROOT / "scratch/alphafold/inputs/targets.json").read_text())
    if smoke:
        records = [r for r in records if r["stem"] in {"5sbj_A", "8wnj_A"}]
    records = sorted(records, key=lambda r: (-r["n_residues"], r["stem"]))
    reports, errors = [], []
    if not collect_only:
        for result in Predictor().predict.map(records, return_exceptions=True):
            if isinstance(result, Exception):
                errors.append(str(result))
                print(f"FAILED: {result}", flush=True)
            else:
                reports.append(result)
        (scratch / "boltz2_run.json").write_text(json.dumps(dict(reports=reports, errors=errors), indent=2) + "\n")
    payload = collect.remote()
    path = scratch / "boltz2_predictions.tar.gz"
    path.write_bytes(payload)
    with tarfile.open(path) as archive:
        if any(m.name.split("/")[0] != "boltz2" for m in archive.getmembers()):
            raise ValueError("Unexpected export path")
        archive.extractall(scratch / "results", filter="data")
    if errors:
        raise RuntimeError(f"{len(errors)} targets failed; outputs recovered, rerun to resume")
    print(f"Collected Boltz-2: {path}")
