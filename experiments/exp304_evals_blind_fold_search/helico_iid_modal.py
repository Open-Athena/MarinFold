"""Fold every exp304 iid contact map individually with Helico on Modal.

Run from the Helico environment so the local ``modal`` client is available::

    HELICO_REPO=/home/bizon/git/helico \
    /home/bizon/git/helico/.venv/bin/modal run helico_iid_modal.py \
      --checkpoint /ckpts/contacts-msafree-01/final.pt --out-tag iid1000

Results and one gzipped PDB per target are written incrementally under
``_cache/helico_iid/results``. Reusing an output tag resumes missing targets.
"""

import csv
import json
import os
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
TARGETS_DIR = HERE / "_cache" / "helico_iid" / "data"
HELICO_REPO = Path(os.environ.get("HELICO_REPO", "/home/bizon/git/helico"))
N_WORKERS = int(os.environ.get("HELICO_IID_WORKERS", "128"))
GPU_TYPE = os.environ.get("HELICO_IID_GPU", "H100")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("wget", "curl")
    .pip_install(
        "torch>=2.7", "cuequivariance-torch>=0.8,<0.9",
        "cuequivariance-ops-torch-cu12>=0.8,<0.9", "biopython>=1.80",
        "numpy>=2.0", "scipy", "pyyaml>=6.0", "huggingface_hub>=0.20",
        "requests", "tqdm", "pyconfind>=0.6", "tmtools", "pyarrow",
    )
    .run_commands(
        "python -c 'from pyconfind import cached_rotamer_library; print(cached_rotamer_library())'"
    )
    .add_local_dir(str(TARGETS_DIR), remote_path="/root/exp304-helico-iid")
    .add_local_dir(str(HELICO_REPO / "src"), remote_path="/root/helico/src")
    .add_local_file(str(HELICO_REPO / "pyproject.toml"), remote_path="/root/helico/pyproject.toml")
    .add_local_file(str(HELICO_REPO / "README.md"), remote_path="/root/helico/README.md")
)

app = modal.App("marinfold-exp304-helico-iid", image=image)
data_volume = modal.Volume.from_name("helico-bench-data", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("helico-checkpoints", create_if_missing=True)
DATA_CACHE = "/cache/helico-data"


@app.cls(
    image=image, gpu=GPU_TYPE, timeout=3600, max_containers=N_WORKERS,
    volumes={DATA_CACHE: data_volume, "/ckpts": checkpoint_volume},
    secrets=[modal.Secret.from_name("helico-hf-modal")],
)
class Predictor:
    checkpoint_path: str = modal.parameter(default="")

    @modal.enter()
    def setup(self) -> None:
        import platform
        import socket
        import sys
        import time

        import pyarrow.parquet as pq

        os.environ["HELICO_DATA_DIR"] = DATA_CACHE
        os.makedirs(DATA_CACHE, exist_ok=True)
        sys.path.insert(0, "/root/helico/src")
        table = pq.read_table(
            "/root/exp304-helico-iid/inputs.parquet",
            columns=["target_id", "pair_id", "contacts"],
        )
        self.inputs = {
            row["target_id"]: (row["pair_id"], row["contacts"])
            for row in table.to_pylist()
        }

        import torch
        from helico.data import parse_ccd
        from helico.model import Helico, HelicoConfig
        from huggingface_hub import snapshot_download

        for attempt in range(5):
            try:
                snapshot_download(
                    "timodonnell/helico-data", repo_type="dataset", local_dir=DATA_CACHE,
                    allow_patterns=["processed/ccd_cache.pkl"], max_workers=8, etag_timeout=30,
                )
                break
            except Exception as error:  # noqa: BLE001
                print(f"CCD download attempt {attempt + 1} failed: {error}", flush=True)
        else:
            raise RuntimeError("ccd_cache.pkl download failed after 5 attempts")
        data_volume.commit()
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        saved = checkpoint.get("config") or {}
        config = HelicoConfig(**{key: value for key, value in saved.items()
                                 if hasattr(HelicoConfig, key)})
        load_started = time.monotonic()
        self.model = Helico(config).cuda().to(torch.bfloat16).eval()
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.ccd = parse_ccd()
        self.model_load_seconds = time.monotonic() - load_started
        properties = torch.cuda.get_device_properties(0)
        self.run_meta = {
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_step": checkpoint.get("step"),
            "use_contacts": bool(config.use_contacts), "use_msa": bool(config.use_msa),
            "dtype": "bfloat16", "gpu_name": str(properties.name),
            "gpu_total_memory_gb": round(properties.total_memory / 1e9, 2),
            "gpu_compute_capability": f"{properties.major}.{properties.minor}",
            "hostname": socket.gethostname(), "platform": platform.platform(),
            "python_version": platform.python_version(), "torch_version": str(torch.__version__),
            "model_load_seconds": round(self.model_load_seconds, 3),
        }

    @modal.method()
    def predict(self, target_id: str, n_cycles: int = 6, max_tokens: int = 2048) -> dict:
        import gzip
        import logging
        import time

        import torch
        from helico.bench import match_atoms, predict_target, score_monomer, structure_to_chains
        from helico.data import parse_mmcif
        from helico.train import coords_to_pdb

        row = {"target_id": target_id, "status": "error", "n_samples": 1,
               "n_cycles": n_cycles, "seed": 42, **self.run_meta}
        started = time.monotonic()
        try:
            pair_id, contact_pairs = self.inputs[target_id]
            ground_truth = parse_mmcif(
                Path("/root/exp304-helico-iid/gt") / f"{pair_id}.cif.gz",
                max_resolution=float("inf"),
            )
            if ground_truth is None:
                raise ValueError(f"failed to parse input chain for {pair_id}")
            chains = structure_to_chains(ground_truth)
            proteins = [chain for chain in chains if chain["type"] == "protein"]
            if len(proteins) != 1:
                raise ValueError(f"{pair_id}: expected one protein chain")
            chain_id = proteins[0]["id"]
            pairs = [(chain_id, int(i), chain_id, int(j)) for i, j in contact_pairs]
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            predict_started = time.monotonic()
            prediction = predict_target(
                self.model, chains, self.ccd, target_name=target_id,
                n_samples=1, max_tokens=max_tokens, msa_server_url=None,
                single_sequence=True, n_cycles=n_cycles, contact_pairs=pairs or None,
            )
            if prediction is None:
                row["status"] = "too_large"
                row["elapsed_seconds"] = round(time.monotonic() - started, 3)
                return row
            tokenized, result = prediction
            row["predict_seconds"] = round(time.monotonic() - predict_started, 3)
            row["n_tokens"] = int(tokenized.n_tokens)
            coords = result["coords"][0].cpu().float().numpy()
            matched = match_atoms(tokenized, coords, ground_truth)
            if len(matched.pred_coords) == 0:
                row["status"] = "no_match"
                return row
            row.update(score_monomer(matched))
            row["n_matched_atoms"] = len(matched.pred_coords)
            row["n_contacts"] = len(pairs)
            row["mean_plddt"] = float(result["plddt"][0].mean())
            pdb = coords_to_pdb(result["coords"][0], result["plddt"][0], tokenized)
            row["pdb_gz"] = gzip.compress(pdb.encode())
            row["status"] = "ok"
            row["elapsed_seconds"] = round(time.monotonic() - started, 3)
            return row
        except Exception as error:  # noqa: BLE001
            logging.exception("%s failed", target_id)
            row["error"] = f"{type(error).__name__}: {error}"
            row["elapsed_seconds"] = round(time.monotonic() - started, 3)
            return row


def read_progress(path: Path, structures: Path) -> dict[str, dict]:
    """Load the latest successful record per target for restart safety."""
    if not path.exists():
        return {}
    rows = {}
    for line in path.read_text().splitlines():
        record = json.loads(line)
        if record.get("status") == "ok" and (structures / f"{record['target_id']}.pdb.gz").exists():
            rows[record["target_id"]] = record
    return rows


@app.local_entrypoint()
def run(checkpoint: str, out_tag: str, n_cycles: int = 6, max_tokens: int = 2048,
        limit: int = 0, longest_first: bool = True) -> None:
    with (TARGETS_DIR / "targets.csv").open() as handle:
        metadata = list(csv.DictReader(handle))
    if longest_first:
        metadata.sort(key=lambda row: (-int(row["L"]), row["target_id"]))
    if limit:
        metadata = metadata[:limit]
    out = TARGETS_DIR.parent / "results"
    structures = out / "predictions" / out_tag
    out.mkdir(parents=True, exist_ok=True)
    structures.mkdir(parents=True, exist_ok=True)
    progress_path = out / f"{out_tag}.progress.jsonl"
    complete = read_progress(progress_path, structures)
    target_ids = [row["target_id"] for row in metadata]
    wanted = [target_id for target_id in target_ids if target_id not in complete]
    print(f"{len(metadata):,} targets; {len(complete):,} complete; {len(wanted):,} remaining")
    predictor = Predictor(checkpoint_path=checkpoint)
    with progress_path.open("a") as progress:
        for index, result in enumerate(predictor.predict.map(
            wanted, kwargs={"n_cycles": n_cycles, "max_tokens": max_tokens},
            order_outputs=False, return_exceptions=True,
        ), 1):
            if isinstance(result, Exception):
                print(f"remote exception: {result}", flush=True)
                continue
            blob = result.pop("pdb_gz", None)
            if blob is not None:
                (structures / f"{result['target_id']}.pdb.gz").write_bytes(blob)
            progress.write(json.dumps(result) + "\n")
            progress.flush()
            if index % 100 == 0 or index == len(wanted):
                print(f"received {index:,}/{len(wanted):,}", flush=True)

    latest = {}
    for line in progress_path.read_text().splitlines():
        record = json.loads(line)
        latest[record["target_id"]] = record
    rows = [latest[target_id] for target_id in target_ids if target_id in latest]
    fields = sorted({key for row in rows for key in row})
    with (out / f"{out_tag}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "tag": out_tag, "n_requested": len(metadata), "n_results": len(rows),
        "n_ok": sum(row.get("status") == "ok" for row in rows),
        "n_structures": len(list(structures.glob("*.pdb.gz"))),
        "sampling": {"n_diffusion_samples": 1, "n_trunk_recycles": n_cycles,
                     "seed": 42, "single_sequence": True, "msa": False},
        "workers": N_WORKERS, "gpu_type": GPU_TYPE,
    }
    (out / f"{out_tag}.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
