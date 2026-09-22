"""Fold every 0, 10, ..., L exp277 contact cut with Helico on Modal.

The input directory is built by prepare.py. Set HELICO_REPO to a checkout of
Open-Athena/helico at HELICO_SHA, then run ``HELICO_DRY_RUN=1 uv run python
run_sweep.py`` before ``uv run python run_sweep.py``. Each target stays in one
GPU worker across all of its cuts so model setup is paid only once.
"""

import csv
import dataclasses
import datetime as dt
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

import modal

from sweep_common import cuts


HERE = Path(__file__).resolve().parent
TARGETS = Path("/root/sweep/data") if Path("/root/sweep/data/targets.csv").exists() else HERE / "scratch" / "targets"
RESULTS = HERE / "scratch" / "results"
HELICO_REPO = Path("/root/helico") if Path("/root/helico/src").exists() else Path(os.environ.get("HELICO_REPO", "/home/bizon/git/helico"))
HELICO_SHA = "b10385d736673c81b10e70d1099962af6f2573c0"
CHECKPOINT = "/ckpts/contacts-msafree-01/final.pt"
N_SAMPLES = 3
N_CYCLES = 6
SEED = 42
MAX_TOKENS = 2048
N_WORKERS = 8
TAG = "exp311-exp277-step266344-top0-to-L-10s-v1"


def source_sha() -> str:
    """Confirm the Helico source revision before packaging it into Modal."""
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HELICO_REPO, text=True).strip()
    if sha != HELICO_SHA:
        raise RuntimeError(f"Helico checkout is {sha}, expected {HELICO_SHA}")
    if subprocess.check_output(["git", "status", "--porcelain", "--", "src", "pyproject.toml"], cwd=HELICO_REPO):
        raise RuntimeError("Helico src or pyproject.toml has uncommitted changes")
    return sha


def estimate() -> tuple[int, float]:
    """Apply Helico's dry-run cost gate to the complete sweep."""
    with (TARGETS / "targets.csv").open() as stream:
        targets = list(csv.DictReader(stream))
    if len(targets) != 115:
        raise ValueError(f"expected 115 verified targets, found {len(targets)}")
    limit = int(os.environ.get("SWEEP_TARGET_LIMIT", "0"))
    if limit:
        targets = targets[:limit]
    expected_cuts = sum(len(cuts(int(t["L_exp245"]))) for t in targets)
    estimated_gpu_hours = expected_cuts * 11.0 * 1.7 / 3600 + N_WORKERS * 0.1
    estimated_usd = estimated_gpu_hours * 3.95
    print(f"{len(targets)} targets, {expected_cuts} cuts, {expected_cuts * N_SAMPLES} samples")
    print(f"conservative estimate: {estimated_gpu_hours:.1f} H100-hours, ${estimated_usd:.2f}")
    if estimated_usd >= 100:
        raise RuntimeError("estimate reaches Helico's $100 cost gate; obtain explicit approval")
    return expected_cuts, estimated_usd


if not (TARGETS / "targets.csv").exists():
    raise RuntimeError("run prepare.py first")

IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("wget", "curl")
    .pip_install(
        "torch>=2.7", "cuequivariance-torch>=0.8,<0.9",
        "cuequivariance-ops-torch-cu12>=0.8,<0.9", "biopython>=1.80",
        "numpy>=2.0", "scipy", "pyyaml>=6.0", "huggingface_hub>=0.20",
        "requests", "tqdm", "pyconfind>=0.6", "tmtools",
    )
    .run_commands(
        "python -c 'from pyconfind import cached_rotamer_library;"
        " print(cached_rotamer_library())'"
    )
    .add_local_dir(str(TARGETS), remote_path="/root/sweep/data")
    .add_local_file(str(HERE / "sweep_common.py"), remote_path="/root/sweep_common.py")
    .add_local_dir(str(HELICO_REPO / "src"), remote_path="/root/helico/src")
    .add_local_file(str(HELICO_REPO / "pyproject.toml"), remote_path="/root/helico/pyproject.toml")
)

app = modal.App("marinfold-exp311-helico-contact-count-sweep", image=IMAGE)
ckpt_volume = modal.Volume.from_name("helico-checkpoints", create_if_missing=False)
ccd_volume = modal.Volume.from_name("helico-bench-data", create_if_missing=False)
results_volume = modal.Volume.from_name("marinfold-exp311-results", create_if_missing=True)


@app.cls(
    image=IMAGE, gpu="H100", timeout=7200, max_containers=N_WORKERS,
    volumes={
        "/ckpts": ckpt_volume,
        "/cache/helico-data": ccd_volume,
        "/results": results_volume,
    },
    secrets=[modal.Secret.from_name("helico-hf-modal")],
)
class Predictor:
    """Keep one Helico checkpoint resident while sweeping a target's cuts."""

    @modal.enter()
    def setup(self) -> None:
        os.environ["HELICO_DATA_DIR"] = "/cache/helico-data"
        sys.path.insert(0, "/root/helico/src")

        import torch
        from huggingface_hub import snapshot_download
        from helico.data import parse_ccd
        from helico.model import Helico, HelicoConfig

        snapshot_download(
            "timodonnell/helico-data", repo_type="dataset", local_dir="/cache/helico-data",
            allow_patterns=["processed/ccd_cache.pkl"], max_workers=8,
        )
        ccd_volume.commit()

        started = time.monotonic()
        checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        if "model_state_dict" not in checkpoint or checkpoint.get("step") != 6000:
            raise ValueError(f"unexpected Helico checkpoint at {CHECKPOINT}")
        saved = checkpoint.get("config")
        if not isinstance(saved, dict):
            raise ValueError("Helico checkpoint is missing its configuration dictionary")
        fields = {field.name for field in dataclasses.fields(HelicoConfig)}
        unknown = set(saved) - fields
        if unknown:
            raise ValueError(f"checkpoint has unknown HelicoConfig keys: {sorted(unknown)}")
        config = HelicoConfig(**saved)
        if not config.use_contacts or config.use_msa:
            raise ValueError("expected contact-conditioned, MSA-free Helico checkpoint")
        self.model = Helico(config).cuda().to(torch.bfloat16).eval()
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.ccd = parse_ccd()
        self.rankings = json.loads(Path("/root/sweep/data/ranked_pairs.json").read_text())
        properties = torch.cuda.get_device_properties(0)
        self.worker_meta = {
            "model_load_seconds": round(time.monotonic() - started, 3),
            "gpu_name": str(properties.name),
            "gpu_total_memory_gb": round(properties.total_memory / 1e9, 2),
            "gpu_compute_capability": f"{properties.major}.{properties.minor}",
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "torch_version": str(torch.__version__),
        }

    @modal.method()
    def predict(self, target: dict) -> dict:
        import torch
        from helico.bench import match_atoms, predict_target, score_monomer, structure_to_chains
        from helico.data import CONTACT_PRESENT, parse_mmcif
        from helico.inference import contacts_from_pairs

        stem = target["target_id"]
        gt_path = Path("/root/sweep/data/gt") / f"{stem}.cif.gz"
        gt = parse_mmcif(gt_path, max_resolution=float("inf"))
        if gt is None:
            raise ValueError(f"failed to parse {gt_path}")
        chains = structure_to_chains(gt)
        proteins = [chain for chain in chains if chain["type"] == "protein"]
        if len(proteins) != 1:
            raise ValueError(f"{stem}: expected one protein chain, found {len(proteins)}")
        chain_id = proteins[0]["id"]
        ranked = self.rankings[stem]
        length = int(target["L_exp245"])
        if len(ranked) != length:
            raise ValueError(f"{stem}: expected {length} ranked contacts, found {len(ranked)}")
        sample_rows = []
        timing_rows = []
        for k in cuts(length):
            selected = ranked[:k]
            pairs = [(chain_id, int(i), chain_id, int(j)) for i, j in selected]
            torch.manual_seed(SEED)
            torch.cuda.manual_seed_all(SEED)
            started = time.monotonic()
            result = predict_target(
                self.model, chains, self.ccd, target_name=stem,
                n_samples=N_SAMPLES, n_cycles=N_CYCLES, max_tokens=MAX_TOKENS,
                msa_server_url=None, single_sequence=True,
                contact_pairs=pairs if k else None,
            )
            if result is None:
                raise ValueError(f"{stem}: predict_target returned None at k={k}")
            predict_seconds = time.monotonic() - started
            tokenized, prediction = result
            if int(tokenized.n_tokens) > MAX_TOKENS:
                raise ValueError(f"{stem}: {tokenized.n_tokens} tokens exceed {MAX_TOKENS}")
            contact_state = contacts_from_pairs(pairs, tokenized=tokenized) if k else None
            effective_contacts = (
                int(torch.triu(contact_state == CONTACT_PRESENT, diagonal=1).sum().item())
                if contact_state is not None else 0
            )
            ranks = prediction["all_ranking_score"][0].cpu().float().tolist()
            ptms = prediction["all_ptm"][0].cpu().float().tolist()
            iptms = prediction["all_iptm"][0].cpu().float().tolist()
            if len(ranks) != N_SAMPLES:
                raise ValueError(f"{stem} k={k}: {len(ranks)} samples, expected {N_SAMPLES}")
            for si in range(N_SAMPLES):
                coords = prediction["all_coords"][0, si].cpu().float().numpy()
                matched = match_atoms(tokenized, coords, gt)
                if len(matched.pred_coords) == 0:
                    raise ValueError(f"{stem} k={k} sample={si}: no matched atoms")
                sample_rows.append({
                    "stem": stem, "eval_set": target["eval_set"],
                    "n_residues": int(target["L_helico"]), "L": length,
                    "n_contacts": k, "effective_contacts": effective_contacts,
                    "sample_idx": si,
                    "ranking_score": float(ranks[si]), "ptm": float(ptms[si]),
                    "iptm": float(iptms[si]), "n_matched_atoms": len(matched.pred_coords),
                    **score_monomer(matched),
                })
            n_residues = int(target["L_helico"])
            timing_rows.append({
                "stem": stem, "eval_set": target["eval_set"],
                "n_residues": n_residues,
                "n_pairs": n_residues * (n_residues - 1) // 2,
                "n_contacts": k, "n_effective_contacts": effective_contacts,
                "mode": f"top-{k}", "elapsed_seconds": round(predict_seconds, 3),
                "model_load_seconds": self.worker_meta["model_load_seconds"],
                "total_seconds": round(
                    self.worker_meta["model_load_seconds"] + time.monotonic() - started, 3
                ),
                "model_nickname": "helico-contacts-msafree-01-step-6000",
                "contact_model": "contacts-v1-exp277-m2-p06-full-epoch-1.5B-step-266344",
                "runner_tag": "modal", "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                **{key: value for key, value in self.worker_meta.items() if key != "model_load_seconds"},
            })
            durable = Path("/results") / TAG / stem
            durable.mkdir(parents=True, exist_ok=True)
            payload = {"samples": sample_rows[-N_SAMPLES:], "timing": timing_rows[-1]}
            result_path = durable / f"top-{k}.json"
            result_path.write_text(json.dumps(payload, separators=(",", ":")))
            timing_rows[-1]["total_seconds"] = round(
                self.worker_meta["model_load_seconds"] + time.monotonic() - started, 3
            )
            result_path.write_text(json.dumps(payload, separators=(",", ":")))
            results_volume.commit()
        return {"stem": stem, "samples": sample_rows, "timings": timing_rows}


@app.local_entrypoint()
def run() -> None:
    """Dispatch one GPU task per target and save complete per-sample tables."""
    with (TARGETS / "targets.csv").open() as stream:
        targets = list(csv.DictReader(stream))
    limit = int(os.environ.get("SWEEP_TARGET_LIMIT", "0"))
    if limit:
        targets = targets[:limit]
    expected_cuts, _ = estimate()
    source_sha()
    output_dir = RESULTS / "smoke" if limit else RESULTS
    partial_dir = output_dir / f"partial-{TAG}"
    partial_dir.mkdir(parents=True, exist_ok=True)
    expected_stems = {target["target_id"] for target in targets}
    completed_stems = {path.stem for path in partial_dir.glob("*.json")}
    unexpected = completed_stems - expected_stems
    if unexpected:
        raise ValueError(f"partial result directory contains unexpected targets: {sorted(unexpected)}")
    pending = [target for target in targets if target["target_id"] not in completed_stems]
    failures = []
    for result in Predictor().predict.map(pending, order_outputs=False, return_exceptions=True):
        if isinstance(result, Exception):
            failures.append(repr(result))
            continue
        stem = result["stem"]
        if stem not in expected_stems or (partial_dir / f"{stem}.json").exists():
            raise ValueError(f"unexpected or duplicate target result: {stem}")
        temp = partial_dir / f".{stem}.json.tmp"
        temp.write_text(json.dumps(result, separators=(",", ":")))
        temp.replace(partial_dir / f"{stem}.json")
    if failures:
        raise RuntimeError(f"{len(failures)} Modal targets failed; partial results retained: {failures}")
    results = [json.loads((partial_dir / f"{stem}.json").read_text()) for stem in sorted(expected_stems)]
    if {result["stem"] for result in results} != {target["target_id"] for target in targets}:
        raise ValueError("incomplete or duplicate target results")
    samples = [row for result in results for row in result["samples"]]
    timings = [row for result in results for row in result["timings"]]
    if len(samples) != expected_cuts * N_SAMPLES or len(timings) != expected_cuts:
        raise ValueError(f"incomplete sweep: {len(samples)} samples, {len(timings)} cuts")
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, rows in (("samples.csv", samples), ("timings.csv", timings)):
        with (output_dir / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    manifest = {
        "tag": TAG, "helico_source_sha": HELICO_SHA,
        "helico_checkpoint": CHECKPOINT, "helico_checkpoint_step": 6000,
        "contact_checkpoint": "contacts-v1-exp277-m2-p06-full-epoch-1.5B-step-266344",
        "n_targets": len(targets), "n_cuts": expected_cuts, "n_samples": len(samples),
        "n_diffusion_samples_per_cut": N_SAMPLES, "n_trunk_recycles": N_CYCLES,
        "seed_per_cut": SEED, "max_tokens": MAX_TOKENS,
        "single_sequence": True, "msa": False,
        "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if not limit:
        committed = HERE / "data"
        committed.mkdir(exist_ok=True)
        for source, destination in (
            (output_dir / "samples.csv", committed / "per_sample_metrics.csv"),
            (output_dir / "timings.csv", committed / "timings.csv"),
            (output_dir / "run_manifest.json", committed / "run_manifest.json"),
        ):
            destination.write_bytes(source.read_bytes())
    print(f"wrote complete results to {output_dir}")


if __name__ == "__main__":
    estimate()
    if os.environ.get("HELICO_DRY_RUN") == "1":
        sys.exit(0)
    source_sha()
    subprocess.run(["modal", "run", str(__file__)], check=True)
