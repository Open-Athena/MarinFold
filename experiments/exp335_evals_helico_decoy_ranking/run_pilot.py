"""Run the preregistered Helico confidence pilot on Modal H100s.

Run ``HELICO_DRY_RUN=1 uv run python run_pilot.py`` first. A bounded smoke run
can be selected with ``PILOT_TARGET_LIMIT`` and ``PILOT_CANDIDATE_LIMIT``.
Each candidate is a separate task so the same target-derived random seed can be
reset before diffusion; this keeps stochastic variation paired across a
target's candidate structures.
"""

import csv
import dataclasses
import datetime as dt
import gzip
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

HERE = Path(__file__).resolve().parent
MAPS = HERE / "scratch" / "pilot_contact_maps.jsonl.gz"
RESULTS = HERE / "scratch" / "results"
HELICO_REPO = (
    Path("/root/helico")
    if os.geteuid() == 0
    else Path(os.environ.get("HELICO_REPO", "/home/bizon/git/helico"))
)
HELICO_SHA = "b10385d736673c81b10e70d1099962af6f2573c0"
CHECKPOINT = "/ckpts/contacts-msafree-01/contacts-msafree-01-step-6000.pt"
CHECKPOINT_SHA256 = "779d540e5bb45cd26bb970188da2f1937ed3079942923a1356c29d06f20fe644"
N_SAMPLES = 3
N_CYCLES = 6
MAX_TOKENS = 2048
N_WORKERS = 8
H100_USD_PER_HOUR = 3.95
SECONDS_PER_CANDIDATE_PRIOR = 16.2  # measured in exp311, including three samples
TAG = "exp335-helico-af2rank-decoy-pilot-v1"


def sha256_file(path: str | Path) -> str:
    """Hash a file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_sha() -> str:
    """Require the preregistered clean Helico source revision."""
    sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=HELICO_REPO, text=True
    ).strip()
    if sha != HELICO_SHA:
        raise RuntimeError(f"Helico checkout is {sha}, expected {HELICO_SHA}")
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--", "src", "pyproject.toml"],
        cwd=HELICO_REPO,
        text=True,
    )
    if status:
        raise RuntimeError("Helico src or pyproject.toml has uncommitted changes")
    return sha


def load_maps() -> list[dict]:
    """Load the locally prepared, index-validated pilot contact maps."""
    if not MAPS.is_file():
        raise FileNotFoundError(f"run prepare_pilot.py first: {MAPS}")
    with gzip.open(MAPS, "rt") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if len(rows) != 225:
        raise ValueError(f"expected 225 pilot candidates, found {len(rows)}")
    keys = [(row["target"], row["decoy_id"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("pilot maps contain duplicate candidate identifiers")
    return rows


def selected_maps() -> list[dict]:
    """Apply deterministic smoke-test limits from the environment."""
    rows = load_maps()
    target_limit = int(os.environ.get("PILOT_TARGET_LIMIT", "0"))
    candidate_limit = int(os.environ.get("PILOT_CANDIDATE_LIMIT", "0"))
    if target_limit:
        targets = sorted({row["target"] for row in rows})[:target_limit]
        rows = [row for row in rows if row["target"] in targets]
    if candidate_limit:
        rows = rows[:candidate_limit]
    return rows


def estimate() -> tuple[int, float]:
    """Print a deliberately conservative prior cost estimate."""
    count = len(selected_maps())
    gpu_hours = count * SECONDS_PER_CANDIDATE_PRIOR / 3600
    compute_usd = gpu_hours * H100_USD_PER_HOUR
    # Up to eight containers may each pay a model-load/startup tail. This is an
    # upper allowance, not expected active GPU compute.
    startup_allowance_usd = min(count, N_WORKERS) * 0.1 * H100_USD_PER_HOUR
    upper_usd = compute_usd + startup_allowance_usd
    print(f"{count} candidates x {N_SAMPLES} diffusion samples")
    print(
        f"prior estimate: {gpu_hours:.2f} H100-hours / ${compute_usd:.2f} compute; "
        f"${upper_usd:.2f} including conservative startup allowance"
    )
    if upper_usd >= 100:
        raise RuntimeError("pilot estimate reaches the $100 cost gate")
    return count, upper_usd


IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("wget", "curl")
    .pip_install(
        "torch>=2.7",
        "cuequivariance-torch>=0.8,<0.9",
        "cuequivariance-ops-torch-cu12>=0.8,<0.9",
        "biopython>=1.80",
        "numpy>=2.0",
        "scipy",
        "pyyaml>=6.0",
        "huggingface_hub>=0.20",
        "requests",
        "tqdm",
        "pyconfind>=0.6",
        "tmtools",
    )
    .add_local_dir(str(HELICO_REPO / "src"), remote_path="/root/helico/src")
    .add_local_file(
        str(HELICO_REPO / "pyproject.toml"), remote_path="/root/helico/pyproject.toml"
    )
)

app = modal.App("marinfold-exp335-helico-decoy-ranking", image=IMAGE)
checkpoint_volume = modal.Volume.from_name(
    "helico-checkpoints", create_if_missing=False
)
ccd_volume = modal.Volume.from_name("helico-bench-data", create_if_missing=False)
results_volume = modal.Volume.from_name(
    "marinfold-exp335-helico-decoy-ranking-results", create_if_missing=True
)


@app.cls(
    image=IMAGE,
    gpu="H100",
    timeout=1800,
    max_containers=N_WORKERS,
    volumes={
        "/ckpts": checkpoint_volume,
        "/cache/helico-data": ccd_volume,
        "/results": results_volume,
    },
    secrets=[modal.Secret.from_name("helico-hf-modal")],
)
class Predictor:
    """Keep one pinned Helico checkpoint resident across candidate tasks."""

    @modal.enter()
    def setup(self) -> None:
        os.environ["HELICO_DATA_DIR"] = "/cache/helico-data"
        sys.path.insert(0, "/root/helico/src")

        import torch
        from helico.data import parse_ccd
        from helico.model import Helico, HelicoConfig
        from huggingface_hub import snapshot_download

        snapshot_download(
            "timodonnell/helico-data",
            repo_type="dataset",
            local_dir="/cache/helico-data",
            allow_patterns=["processed/ccd_cache.pkl"],
            max_workers=8,
        )
        ccd_volume.commit()

        started = time.monotonic()
        checkpoint_digest = sha256_file(CHECKPOINT)
        if checkpoint_digest != CHECKPOINT_SHA256:
            raise ValueError(
                f"Helico checkpoint digest {checkpoint_digest} != {CHECKPOINT_SHA256}"
            )
        checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        if "model_state_dict" not in checkpoint or checkpoint.get("step") != 6000:
            raise ValueError(f"unexpected Helico checkpoint at {CHECKPOINT}")
        saved = checkpoint.get("config")
        if not isinstance(saved, dict):
            raise TypeError("Helico checkpoint is missing its configuration dictionary")
        fields = {field.name for field in dataclasses.fields(HelicoConfig)}
        # save_checkpoint stores TrainConfig, whose model-relevant fields are
        # intentionally mirrored into HelicoConfig. This is the same filtered
        # reconstruction used by Helico's pinned load_model/infer_main APIs.
        config = HelicoConfig(
            **{key: value for key, value in saved.items() if key in fields}
        )
        if not config.use_contacts or config.use_msa:
            raise ValueError("expected contact-conditioned, MSA-free Helico checkpoint")
        self.model = Helico(config).cuda().to(torch.bfloat16).eval()
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.ccd = parse_ccd()
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
    def predict(self, candidate: dict, map_mode: str = "full") -> dict:
        """Predict one candidate with a paired target-level diffusion seed."""
        import numpy as np
        import torch
        from helico.bench import compute_tm_score, single_sequence_msa
        from helico.data import (
            CONTACT_ABSENT,
            CONTACT_PRESENT,
            CONTACT_UNKNOWN,
            tokenize_sequences,
        )
        from helico.train import run_inference

        if map_mode not in {"full", "present-only"}:
            raise ValueError(f"unknown contact-map mode: {map_mode}")
        target = candidate["target"]
        decoy_id = candidate["decoy_id"]
        sequence = candidate["sequence"]
        tokenized = tokenize_sequences(
            [{"type": "protein", "id": "A", "sequence": sequence}], self.ccd
        )
        if tokenized.n_tokens != len(sequence):
            raise ValueError(
                f"{target}/{decoy_id}: {tokenized.n_tokens} tokens for {len(sequence)} residues"
            )
        if tokenized.n_tokens > MAX_TOKENS:
            raise ValueError(f"{target}/{decoy_id}: exceeds {MAX_TOKENS} tokens")

        features = tokenized.to_features()
        state = torch.full(
            (tokenized.n_tokens, tokenized.n_tokens),
            CONTACT_UNKNOWN,
            dtype=torch.uint8,
        )
        if map_mode == "full":
            for left in range(tokenized.n_tokens):
                state[left, left + 6 :] = CONTACT_ABSENT
                state[left + 6 :, left] = CONTACT_ABSENT
        for left, right in candidate["present_pairs"]:
            if right - left < 6:
                raise ValueError(f"unexpected short-range pair {(left, right)}")
            state[left, right] = CONTACT_PRESENT
            state[right, left] = CONTACT_PRESENT
        features["contact_state"] = state

        batch = {
            key: value.unsqueeze(0) if isinstance(value, torch.Tensor) else value
            for key, value in features.items()
        }
        for key in ("n_tokens", "n_atoms"):
            if not isinstance(batch[key], torch.Tensor):
                batch[key] = torch.tensor([batch[key]])
        batch["token_mask"] = torch.ones(1, features["n_tokens"], dtype=torch.bool)
        batch["atom_mask"] = torch.ones(1, features["n_atoms"], dtype=torch.bool)
        batch.update(single_sequence_msa(batch["restype"]))

        seed = int(hashlib.sha256(target.encode()).hexdigest()[:8], 16)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        started = time.monotonic()
        results = run_inference(
            self.model,
            batch,
            n_samples=N_SAMPLES,
            device="cuda",
            dtype=torch.bfloat16,
            n_cycles=N_CYCLES,
        )
        elapsed_seconds = time.monotonic() - started

        ptms = results["all_ptm"][0].cpu().float().tolist()
        rankings = results["all_ranking_score"][0].cpu().float().tolist()
        if len(ptms) != N_SAMPLES or any(not np.isfinite(value) for value in ptms):
            raise ValueError(f"{target}/{decoy_id}: invalid pTM values {ptms}")
        selected_sample = max(range(N_SAMPLES), key=rankings.__getitem__)
        if selected_sample != max(range(N_SAMPLES), key=ptms.__getitem__):
            raise ValueError("monomer ranking score did not select maximum pTM")

        rep_atom_idx = features["rep_atom_idx"].cpu().numpy()
        candidate_ca = np.asarray(candidate["candidate_ca"], dtype=np.float64)
        if len(rep_atom_idx) != len(candidate_ca):
            raise ValueError("candidate and predicted C-alpha counts differ")
        sample_rows = []
        for sample_index in range(N_SAMPLES):
            predicted_ca = (
                results["all_coords"][0, sample_index, rep_atom_idx]
                .cpu()
                .float()
                .numpy()
            )
            candidate_output_tm = compute_tm_score(predicted_ca, candidate_ca)
            sample_rows.append(
                {
                    "sample_idx": sample_index,
                    "ptm": float(ptms[sample_index]),
                    "ranking_score": float(rankings[sample_index]),
                    "candidate_output_tm": candidate_output_tm,
                    "helico_composite": float(ptms[sample_index]) * candidate_output_tm,
                    "selected_by_helico": int(sample_index == selected_sample),
                }
            )

        selected_plddt = results["plddt"][0, rep_atom_idx].cpu().float().numpy()
        if not np.isfinite(selected_plddt).all():
            raise ValueError(f"{target}/{decoy_id}: nonfinite pLDDT")
        primary = {
            "target": target,
            "decoy_id": decoy_id,
            "map_mode": map_mode,
            "n_residues": len(sequence),
            "n_pairs": len(sequence) * (len(sequence) - 1) // 2,
            "n_present_contacts": len(candidate["present_pairs"]),
            "contact_map_sha256": candidate["contact_map_sha256"],
            "seed": seed,
            "selected_sample_idx": selected_sample,
            "max_ptm": float(ptms[selected_sample]),
            "mean_sample_ptm": float(np.mean(ptms)),
            "mean_ca_plddt": float(np.mean(selected_plddt)),
            "candidate_output_tm": sample_rows[selected_sample]["candidate_output_tm"],
            "helico_composite": sample_rows[selected_sample]["helico_composite"],
        }
        timing = {
            "stem": f"{target}/{decoy_id}",
            "n_residues": len(sequence),
            "n_pairs": len(sequence) * (len(sequence) - 1) // 2,
            "mode": map_mode,
            "elapsed_seconds": round(elapsed_seconds, 3),
            "model_load_seconds": self.worker_meta["model_load_seconds"],
            "total_seconds": round(
                self.worker_meta["model_load_seconds"] + elapsed_seconds, 3
            ),
            "model_nickname": "helico-contacts-msafree-01-step-6000",
            "runner_tag": "modal",
            "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
            **{
                key: value
                for key, value in self.worker_meta.items()
                if key != "model_load_seconds"
            },
        }
        payload = {"candidate": primary, "samples": sample_rows, "timing": timing}
        key = hashlib.sha256(f"{map_mode}\t{target}\t{decoy_id}".encode()).hexdigest()
        durable = Path("/results") / TAG / map_mode / target
        durable.mkdir(parents=True, exist_ok=True)
        (durable / f"{key}.json").write_text(json.dumps(payload, separators=(",", ":")))
        results_volume.commit()
        return payload


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write dictionaries to CSV with stable columns."""
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


@app.local_entrypoint()
def run(map_mode: str = "full") -> None:
    """Dispatch candidate tasks and assemble resumable local result tables."""
    candidates = selected_maps()
    expected_count, _ = estimate()
    source_sha()
    limited = bool(
        int(os.environ.get("PILOT_TARGET_LIMIT", "0"))
        or int(os.environ.get("PILOT_CANDIDATE_LIMIT", "0"))
    )
    output_dir = RESULTS / "smoke" if limited else RESULTS / "pilot"
    partial_dir = output_dir / f"partial-{map_mode}"
    partial_dir.mkdir(parents=True, exist_ok=True)

    def result_key(candidate: dict) -> str:
        return hashlib.sha256(
            f"{map_mode}\t{candidate['target']}\t{candidate['decoy_id']}".encode()
        ).hexdigest()

    expected_keys = {result_key(candidate) for candidate in candidates}
    completed_keys = {path.stem for path in partial_dir.glob("*.json")}
    unexpected = completed_keys - expected_keys
    if unexpected:
        raise ValueError(
            f"partial directory contains unexpected results: {sorted(unexpected)}"
        )
    pending = [
        candidate
        for candidate in candidates
        if result_key(candidate) not in completed_keys
    ]

    failures = []
    predictor = Predictor()
    inputs = [(candidate, map_mode) for candidate in pending]
    for result in predictor.predict.starmap(
        inputs,
        order_outputs=False,
        return_exceptions=True,
        wrap_returned_exceptions=False,
    ):
        if isinstance(result, Exception):
            failures.append(repr(result))
            continue
        key = hashlib.sha256(
            f"{map_mode}\t{result['candidate']['target']}\t"
            f"{result['candidate']['decoy_id']}".encode()
        ).hexdigest()
        if key not in expected_keys or (partial_dir / f"{key}.json").exists():
            raise ValueError(f"unexpected or duplicate candidate result: {key}")
        temporary = partial_dir / f".{key}.json.tmp"
        temporary.write_text(json.dumps(result, separators=(",", ":")))
        temporary.replace(partial_dir / f"{key}.json")
    if failures:
        raise RuntimeError(f"{len(failures)} Modal candidates failed: {failures}")

    results = [
        json.loads((partial_dir / f"{key}.json").read_text())
        for key in sorted(expected_keys)
    ]
    if len(results) != expected_count:
        raise ValueError(f"expected {expected_count} results, found {len(results)}")
    candidate_rows = [result["candidate"] for result in results]
    sample_rows = [
        {
            "target": result["candidate"]["target"],
            "decoy_id": result["candidate"]["decoy_id"],
            "map_mode": result["candidate"]["map_mode"],
            **sample,
        }
        for result in results
        for sample in result["samples"]
    ]
    timing_rows = [result["timing"] for result in results]
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / f"candidate_metrics_{map_mode}.csv", candidate_rows)
    write_csv(output_dir / f"sample_metrics_{map_mode}.csv", sample_rows)
    write_csv(output_dir / f"timings_{map_mode}.csv", timing_rows)
    manifest = {
        "tag": TAG,
        "map_mode": map_mode,
        "helico_source_sha": HELICO_SHA,
        "helico_checkpoint": CHECKPOINT,
        "helico_checkpoint_step": 6000,
        "helico_checkpoint_sha256": CHECKPOINT_SHA256,
        "input_maps_sha256": sha256_file(MAPS),
        "n_candidates": len(results),
        "n_diffusion_samples_per_candidate": N_SAMPLES,
        "n_trunk_recycles": N_CYCLES,
        "seed": "first 32 bits of SHA-256(target); reset per candidate",
        "single_sequence": True,
        "msa": False,
        "completed_at_utc": dt.datetime.now(dt.UTC).isoformat(),
    }
    (output_dir / f"run_manifest_{map_mode}.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(f"wrote {len(results)} complete candidate results to {output_dir}")


if __name__ == "__main__":
    estimate()
    if os.environ.get("HELICO_DRY_RUN") == "1":
        sys.exit(0)
    source_sha()
    subprocess.run(["modal", "run", str(__file__), "--map-mode", "full"], check=True)
