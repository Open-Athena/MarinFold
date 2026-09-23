"""Fold fixed top-L contacts and matched oracle/random controls on Modal.

Adapted from exp311 run_sweep.py at MarinFold dc3a1969. The model, source,
diffusion budget and selection rule are unchanged. No cut-count tuning.

The input directory is built by prepare.py. Set HELICO_REPO to a checkout of
Open-Athena/helico at HELICO_SHA, then run ``HELICO_DRY_RUN=1 uv run python
run_sweep.py`` before ``uv run python run_sweep.py``. Each target stays in one
GPU worker across all of its cuts so model setup is paid only once.
"""

import csv
import dataclasses
import datetime as dt
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

import modal
import numpy as np

from plan_missing import randomized_map


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PHASE = os.environ.get("WRITEUP_PHASE", "confidence")
if PHASE not in ("confidence", "folding"):
    raise ValueError(PHASE)
TARGETS = Path("/root/sweep/data") if Path("/root/sweep/data/targets.csv").exists() else ROOT / "scratch" / "helico" / PHASE
RESULTS = ROOT / "scratch" / "helico_results" / PHASE
HELICO_REPO = Path("/root/helico") if Path("/root/helico/src").exists() else Path(os.environ.get("HELICO_REPO", "/home/bizon/git/helico"))
HELICO_SHA = "b10385d736673c81b10e70d1099962af6f2573c0"
CHECKPOINT = "/ckpts/contacts-msafree-01/contacts-msafree-01-step-6000.pt"
CHECKPOINT_SHA256 = "779d540e5bb45cd26bb970188da2f1937ed3079942923a1356c29d06f20fe644"
N_SAMPLES = 3
N_CYCLES = 6
SEED = 42
MAX_TOKENS = 2048
N_WORKERS = 8
TAG = f"exp325-exp277-step266344-{PHASE}-v1"


def source_sha() -> str:
    """Confirm the Helico source revision before packaging it into Modal."""
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HELICO_REPO, text=True).strip()
    if sha != HELICO_SHA:
        raise RuntimeError(f"Helico checkout is {sha}, expected {HELICO_SHA}")
    if subprocess.check_output(["git", "status", "--porcelain", "--", "src", "pyproject.toml"], cwd=HELICO_REPO):
        raise RuntimeError("Helico src or pyproject.toml has uncommitted changes")
    return sha


def sha256(path: str | Path) -> str:
    """Hash a checkpoint before loading so its path cannot silently drift."""
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def estimate() -> tuple[int, float]:
    """Apply Helico's dry-run cost gate to the complete sweep."""
    with (TARGETS / "targets.csv").open() as stream:
        targets = list(csv.DictReader(stream))
    if len(targets) != (20 if PHASE == "confidence" else 211):
        raise ValueError(f"unexpected target count: {len(targets)}")
    limit = int(os.environ.get("SWEEP_TARGET_LIMIT", "0"))
    if limit:
        targets = targets[:limit]
    expected_cuts = len(targets) * (11 if PHASE == "confidence" else 2)
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
        "torch==2.13.0", "cuequivariance-torch>=0.8,<0.9",
        "cuequivariance-ops-torch-cu12>=0.8,<0.9", "biopython>=1.80",
        "numpy>=2.0", "scipy", "pyyaml>=6.0", "huggingface_hub>=0.20",
        "requests", "tqdm", "pyconfind==0.6.0", "tmtools", "pandas",
    )
    .run_commands(
        "python -c 'from pyconfind import cached_rotamer_library;"
        " print(cached_rotamer_library())'"
    )
    .env({"WRITEUP_PHASE": PHASE})
    .add_local_dir(str(TARGETS), remote_path="/root/sweep/data")
    .add_local_file(str(ROOT / "plan_missing.py"), remote_path="/root/plan_missing.py")
    .add_local_dir(str(HELICO_REPO / "src"), remote_path="/root/helico/src")
    .add_local_file(str(HELICO_REPO / "pyproject.toml"), remote_path="/root/helico/pyproject.toml")
)

app = modal.App("marinfold-exp325-helico", image=IMAGE)
ckpt_volume = modal.Volume.from_name("helico-checkpoints", create_if_missing=False)
ccd_volume = modal.Volume.from_name("helico-bench-data", create_if_missing=False)
results_volume = modal.Volume.from_name("marinfold-exp325-helico-results", create_if_missing=True)


@app.cls(
    image=IMAGE, gpu="H100", timeout=7200, max_containers=N_WORKERS, region="us-east",
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
        from helico.train import TrainConfig

        snapshot_download(
            "timodonnell/helico-data", repo_type="dataset", local_dir="/cache/helico-data",
            allow_patterns=["processed/ccd_cache.pkl"], max_workers=8,
        )
        ccd_volume.commit()

        started = time.monotonic()
        actual_digest = sha256(CHECKPOINT)
        if actual_digest != CHECKPOINT_SHA256:
            raise ValueError(f"Helico checkpoint digest {actual_digest} != {CHECKPOINT_SHA256}")
        checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        if "model_state_dict" not in checkpoint or checkpoint.get("step") != 6000:
            raise ValueError(f"unexpected Helico checkpoint at {CHECKPOINT}")
        saved = checkpoint.get("config")
        if not isinstance(saved, dict):
            raise ValueError("Helico checkpoint is missing its configuration dictionary")
        fields = {field.name for field in dataclasses.fields(HelicoConfig)}
        unknown = set(saved) - fields - {field.name for field in dataclasses.fields(TrainConfig)}
        if unknown:
            raise ValueError(f"checkpoint has unknown HelicoConfig keys: {sorted(unknown)}")
        config = HelicoConfig(**{key: value for key, value in saved.items() if key in fields})
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
        from helico.bench import match_atoms, oracle_contact_state, score_monomer, single_sequence_msa, structure_to_chains
        from helico.contacts import load_rotamer_library
        from helico.data import CONTACT_UNKNOWN, CONTACT_ABSENT, CONTACT_PRESENT, parse_mmcif, tokenize_sequences
        from helico.inference import contacts_from_pairs
        from helico.train import run_inference

        if (CONTACT_UNKNOWN, CONTACT_ABSENT, CONTACT_PRESENT) != (0, 1, 2):
            raise ValueError("Contact state encoding changed")
        stem = target["target_id"]
        gt = parse_mmcif(Path("/root/sweep/data/gt") / f"{stem}.cif.gz", max_resolution=float("inf"))
        if gt is None:
            raise ValueError(f"Cannot parse ground truth for {stem}")
        chains = structure_to_chains(gt)
        proteins = [c for c in chains if c["type"] == "protein"]
        if len(proteins) != 1:
            raise ValueError(f"{stem}: expected one protein chain")
        if proteins[0]["sequence"] != target["input_seq"]:
            raise ValueError(f"{stem}: ground-truth input sequence differs from frozen mapping source")
        tokenized = tokenize_sequences(chains, self.ccd)
        if tokenized.n_tokens > MAX_TOKENS:
            raise ValueError(f"{stem}: too many tokens")
        length = int(target["L_exp245"])
        maps = []
        if PHASE == "confidence":
            oracle = oracle_contact_state(gt, tokenized, load_rotamer_library())
            if oracle is None:
                raise ValueError(f"{stem}: missing oracle contact state")
            state = oracle.cpu().numpy()
            maps.append(("oracle", 0, state))
            for separation in (False, True):
                arm = "separation_matched" if separation else "uniform"
                for seed in range(1, 6):
                    maps.append((arm, seed, randomized_map(state, seed, separation, tokenized.to_features()["res_indices"].numpy())))
        else:
            chain = proteins[0]["id"]
            ranked = self.rankings[stem]
            if len(ranked) != length:
                raise ValueError(f"{stem}: not exactly top-L pairs")
            pairs = [(chain, int(i), chain, int(j)) for i, j in ranked]
            top_l = contacts_from_pairs(pairs, tokenized=tokenized, strict=False).numpy()
            expected = sum(abs(int(i) - int(j)) >= 6 for i, j in ranked)
            if int(np.triu(top_l == 2, 1).sum()) != expected:
                raise ValueError(f"{stem}: unexpected loss of mapped contacts")
            maps = [("top_0", 0, np.zeros_like(top_l)), ("top_L", 0, top_l)]
        sample_rows, timing_rows = [], []
        durable = Path("/results") / TAG / stem
        durable.mkdir(parents=True, exist_ok=True)
        for arm, map_seed, state in maps:
            key = f"{arm}-{map_seed}"
            result_path = durable / f"{key}.json"
            if result_path.exists():
                saved = json.loads(result_path.read_text())
                sample_rows.extend(saved["samples"])
                timing_rows.append(saved["timing"])
                continue
            # Same feature construction as bench.predict_target; the public
            # run_inference API accepts our explicit three-state control matrix.
            features = tokenized.to_features()
            features["contact_state"] = torch.from_numpy(state)
            batch = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v for k, v in features.items()}
            for field in ("n_tokens", "n_atoms"):
                if not isinstance(batch[field], torch.Tensor):
                    batch[field] = torch.tensor([batch[field]])
            batch["token_mask"] = torch.ones(1, tokenized.n_tokens, dtype=torch.bool)
            batch["atom_mask"] = torch.ones(1, features["n_atoms"], dtype=torch.bool)
            batch.update(single_sequence_msa(batch["restype"]))
            torch.manual_seed(SEED)
            torch.cuda.manual_seed_all(SEED)
            torch.cuda.synchronize()
            started = time.monotonic()
            prediction = run_inference(self.model, batch, n_samples=N_SAMPLES, device="cuda",
                                       dtype=torch.bfloat16, n_cycles=N_CYCLES)
            torch.cuda.synchronize()
            elapsed = time.monotonic() - started
            ranks = prediction["all_ranking_score"][0].cpu().float().tolist()
            ptms = prediction["all_ptm"][0].cpu().float().tolist()
            iptms = prediction["all_iptm"][0].cpu().float().tolist()
            n_present = int(np.triu(state == 2, 1).sum())
            n_absent = int(np.triu(state == 1, 1).sum())
            n_unknown = int(np.triu(state == 0, 1).sum())
            if len(ranks) != N_SAMPLES:
                raise ValueError(f"{stem}: wrong diffusion sample count")
            for si in range(N_SAMPLES):
                coords = prediction["all_coords"][0, si].cpu().float().numpy()
                matched = match_atoms(tokenized, coords, gt)
                if not len(matched.pred_coords):
                    raise ValueError(f"{stem}: no matched atoms")
                sample_rows.append({
                    "stem": stem, "eval_set": target["eval_set"], "arm": arm, "map_seed": map_seed,
                    "sample_idx": si, "L": length, "n_present": n_present,
                    "requested_contacts": length if arm == "top_L" else n_present,
                    "dropped_short_after_mapping": length - n_present if arm == "top_L" else 0,
                    "n_absent": n_absent, "n_unknown": n_unknown,
                    "ranking_score": float(ranks[si]), "ptm": float(ptms[si]), "iptm": float(iptms[si]),
                    "mean_plddt": float(prediction["plddt"][0].cpu().float().mean()) if si == int(np.argmax(ranks)) else None,
                    "has_clash": float(prediction["all_has_clash"][0, si].cpu()),
                    "n_matched_atoms": len(matched.pred_coords), **score_monomer(matched),
                })
            timing = {
                "stem": stem, "eval_set": target["eval_set"], "n_residues": int(target["L_helico"]),
                "n_pairs": n_present + n_absent + n_unknown, "mode": key,
                "elapsed_seconds": elapsed, "total_seconds": self.worker_meta["model_load_seconds"] + time.monotonic() - started,
                "model_nickname": "helico-contacts-msafree-01-step-6000", "runner_tag": "modal",
                "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "n_samples": N_SAMPLES, "n_cycles": N_CYCLES, "seed": SEED, **self.worker_meta,
            }
            timing_rows.append(timing)
            np.savez_compressed(durable / f"{key}.npz", coords=prediction["all_coords"][0].cpu().float().numpy(),
                                contact_state=state)
            result_path.write_text(json.dumps({"samples": sample_rows[-N_SAMPLES:], "timing": timing}))
            results_volume.commit()
            print(f"[{PHASE}] {stem} {key}: {elapsed:.1f}s", flush=True)
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
        "helico_checkpoint_sha256": CHECKPOINT_SHA256,
        "contact_checkpoint": "contacts-v1-exp277-m2-p06-full-epoch-1.5B-step-266344" if PHASE == "folding" else None,
        "conditioning_source": "MarinFold votes" if PHASE == "folding" else "ground-truth oracle and randomized three-state maps",
        "n_targets": len(targets), "n_cuts": expected_cuts, "n_samples": len(samples),
        "n_diffusion_samples_per_cut": N_SAMPLES, "n_trunk_recycles": N_CYCLES,
        "seed_per_map": SEED, "phase": PHASE, "selection": "highest ranking_score among three diffusion samples per map", "max_tokens": MAX_TOKENS,
        "single_sequence": True, "msa": False,
        "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if not limit:
        committed = ROOT / "data"
        committed.mkdir(exist_ok=True)
        for source, destination in (
            (output_dir / "samples.csv", committed / f"helico_{PHASE}_samples.csv"),
            (output_dir / "timings.csv", committed / f"helico_{PHASE}_timings.csv"),
            (output_dir / "run_manifest.json", committed / f"helico_{PHASE}_run.json"),
        ):
            destination.write_bytes(source.read_bytes())
    print(f"wrote complete results to {output_dir}")


if __name__ == "__main__":
    estimate()
    if os.environ.get("HELICO_DRY_RUN") == "1":
        sys.exit(0)
    source_sha()
    subprocess.run(["modal", "run", str(__file__)], check=True)
