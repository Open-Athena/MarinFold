"""Run resumable Helico decoy-ranking parts on one CoreWeave H100.

This file is base64-inlined by ``dispatch_full_cw.py``. Each worker computes a
deterministic subset of candidate parts, resets the same target-derived seed for
every candidate (matching the pilot), and writes fingerprinted gzip JSON plus a
timing companion to co-located CoreWeave S3.
"""

import argparse
import dataclasses
import datetime as dt
import gzip
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import socket
import sys
import tarfile
import time
from pathlib import Path

HELICO_SHA = "b10385d736673c81b10e70d1099962af6f2573c0"
CHECKPOINT_SHA256 = "779d540e5bb45cd26bb970188da2f1937ed3079942923a1356c29d06f20fe644"
SOURCE_ARCHIVE_SHA256 = (
    "28db5ab75a21424c3ea977eb61152e60d33758d04cf798f49c1bce3ba1a276f2"
)
CCD_SHA256 = "8531c4b72693d3afddfe4242c56a257be578445a32b25d260d2deb676715da04"
TORCH_VERSION = "2.13.0+cu130"
CUEQUIVARIANCE_VERSION = "0.8.1"
N_SAMPLES = 3
N_CYCLES = 6
PART_SIZE = 32
MAP_MODE = "full"
ASSET_ROOT = "s3://marin-us-east-02a/MarinFold/exp335/assets"
INPUT_ROOT = "s3://marin-us-east-02a/MarinFold/exp335/full-v1/inputs"
OUTPUT_ROOT = "s3://marin-us-east-02a/MarinFold/exp335/full-v1/results"
CHECKPOINT_URI = f"{ASSET_ROOT}/contacts-msafree-01-step-6000.pt"
SOURCE_URI = f"{ASSET_ROOT}/helico-{HELICO_SHA[:12]}-src.tar.gz"
CCD_URI = f"{ASSET_ROOT}/ccd_cache.pkl"
MANIFEST_URI = f"{INPUT_ROOT}/manifest.json"
WORK_DIR = Path("/tmp/exp335_helico_full")
WORKER_SCRIPT_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

CONFIG = {
    "schema_version": 1,
    "worker_script_sha256": WORKER_SCRIPT_SHA256,
    "helico_source_sha": HELICO_SHA,
    "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
    "checkpoint_sha256": CHECKPOINT_SHA256,
    "ccd_sha256": CCD_SHA256,
    "torch_version": TORCH_VERSION,
    "cuequivariance_version": CUEQUIVARIANCE_VERSION,
    "n_diffusion_samples": N_SAMPLES,
    "n_trunk_recycles": N_CYCLES,
    "map_mode": MAP_MODE,
    "part_size": PART_SIZE,
    "seed": "first 32 bits of SHA-256(target), reset per candidate",
    "single_sequence": True,
    "msa": False,
}
RUN_FINGERPRINT = hashlib.sha256(
    json.dumps(CONFIG, separators=(",", ":"), sort_keys=True).encode()
).hexdigest()


def sha256_bytes(payload: bytes) -> str:
    """Hash bytes."""
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(uri: str) -> dict:
    """Read JSON through fsspec."""
    import fsspec

    with fsspec.open(uri, "rt") as stream:
        return json.load(stream)


def download(uri: str, destination: Path, expected_sha256: str | None = None) -> None:
    """Download an S3 object and verify its digest."""
    import fsspec

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and (
        expected_sha256 is None or sha256_file(destination) == expected_sha256
    ):
        return
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with fsspec.open(uri, "rb") as source, temporary.open("wb") as output:
        shutil.copyfileobj(source, output, length=8 << 20)
    if expected_sha256 is not None:
        observed = sha256_file(temporary)
        if observed != expected_sha256:
            raise ValueError(f"{uri}: digest {observed} != {expected_sha256}")
    temporary.replace(destination)


def assign_tasks(manifest: dict, num_shards: int) -> list[list[dict]]:
    """Greedily balance fixed candidate parts by length-squared work."""
    tasks = []
    for target in manifest["targets"]:
        for start in range(0, target["n_candidates"], PART_SIZE):
            end = min(start + PART_SIZE, target["n_candidates"])
            tasks.append(
                {
                    "target": target["target"],
                    "start": start,
                    "end": end,
                    "n_residues": target["n_residues"],
                    "input_sha256": target["input_sha256"],
                    "file_sha256": target["file_sha256"],
                    "relative_path": target["relative_path"],
                    "weight": (end - start) * target["n_residues"] ** 2,
                }
            )
    shards: list[list[dict]] = [[] for _ in range(num_shards)]
    loads = [0] * num_shards
    for task in sorted(
        tasks, key=lambda item: (-item["weight"], item["target"], item["start"])
    ):
        shard = min(range(num_shards), key=lambda index: (loads[index], index))
        shards[shard].append(task)
        loads[shard] += task["weight"]
    for tasks_for_shard in shards:
        tasks_for_shard.sort(key=lambda item: (item["target"], item["start"]))
    return shards


def load_target(task: dict) -> dict:
    """Download and validate one prepared target payload."""
    local = WORK_DIR / "inputs" / task["relative_path"]
    download(f"{INPUT_ROOT}/{task['relative_path']}", local, task["file_sha256"])
    with gzip.open(local, "rb") as stream:
        encoded = stream.read()
    if sha256_bytes(encoded) != task["input_sha256"]:
        raise ValueError(f"{task['target']}: uncompressed input digest mismatch")
    payload = json.loads(encoded)
    if payload["target"] != task["target"]:
        raise ValueError(
            f"target payload mismatch: {payload['target']} != {task['target']}"
        )
    return payload


def output_uris(task: dict) -> tuple[str, str]:
    """Return the metrics and timing URIs for a task part."""
    stem = f"part-{task['start']:05d}-{task['end']:05d}"
    base = f"{OUTPUT_ROOT}/{RUN_FINGERPRINT}/{task['target']}"
    return f"{base}/{stem}.json.gz", f"{base}/{stem}.timing.json"


def existing_part_valid(task: dict, candidate_ids: list[str]) -> bool:
    """Validate both durable objects before treating a part as complete."""
    import fsspec

    metrics_uri, timing_uri = output_uris(task)
    fs = fsspec.filesystem("s3")
    metrics_path = metrics_uri.removeprefix("s3://")
    timing_path = timing_uri.removeprefix("s3://")
    if not fs.exists(metrics_path) or not fs.exists(timing_path):
        return False
    try:
        compressed = fs.cat_file(metrics_path)
        payload = json.loads(gzip.decompress(compressed))
        timing = json.loads(fs.cat_file(timing_path))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    expected = {
        "run_fingerprint": RUN_FINGERPRINT,
        "target_input_sha256": task["input_sha256"],
        "candidate_ids": candidate_ids,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        return False
    if any(timing.get(key) != value for key, value in expected.items()):
        return False
    if timing.get("metrics_sha256") != sha256_bytes(compressed):
        return False
    return len(payload.get("results", [])) == len(candidate_ids)


def setup_model() -> tuple[object, object, dict, float]:
    """Download pinned assets and initialize the Helico model once."""
    setup_started = time.monotonic()
    source = WORK_DIR / "assets" / Path(SOURCE_URI).name
    checkpoint_path = WORK_DIR / "assets" / "checkpoint.pt"
    ccd_path = WORK_DIR / "helico-data" / "processed" / "ccd_cache.pkl"
    download(SOURCE_URI, source, SOURCE_ARCHIVE_SHA256)
    download(CHECKPOINT_URI, checkpoint_path, CHECKPOINT_SHA256)
    download(CCD_URI, ccd_path, CCD_SHA256)
    source_dir = WORK_DIR / "source"
    if not (source_dir / "helico" / "src").is_dir():
        source_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(source, "r:gz") as archive:
            archive.extractall(source_dir, filter="data")
    os.environ["HELICO_DATA_DIR"] = str(WORK_DIR / "helico-data")
    sys.path.insert(0, str(source_dir / "helico" / "src"))

    import torch
    from helico.data import parse_ccd
    from helico.model import Helico, HelicoConfig

    if torch.__version__ != TORCH_VERSION:
        raise ValueError(f"torch {torch.__version__} != pinned {TORCH_VERSION}")
    cue_version = importlib.metadata.version("cuequivariance-torch")
    if cue_version != CUEQUIVARIANCE_VERSION:
        raise ValueError(
            f"cuequivariance-torch {cue_version} != pinned {CUEQUIVARIANCE_VERSION}"
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" not in checkpoint or checkpoint.get("step") != 6000:
        raise ValueError("unexpected Helico checkpoint payload")
    saved = checkpoint.get("config")
    if not isinstance(saved, dict):
        raise TypeError("checkpoint is missing its configuration dictionary")
    fields = {field.name for field in dataclasses.fields(HelicoConfig)}
    config = HelicoConfig(
        **{key: value for key, value in saved.items() if key in fields}
    )
    if not config.use_contacts or config.use_msa:
        raise ValueError("expected contact-conditioned, MSA-free Helico checkpoint")
    model = Helico(config).cuda().to(torch.bfloat16).eval()
    model.load_state_dict(checkpoint["model_state_dict"])
    ccd = parse_ccd()
    properties = torch.cuda.get_device_properties(0)
    worker_metadata = {
        "gpu_name": str(properties.name),
        "gpu_total_memory_gb": round(properties.total_memory / 1e9, 2),
        "gpu_compute_capability": f"{properties.major}.{properties.minor}",
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "torch_version": str(torch.__version__),
        "cuequivariance_torch_version": cue_version,
    }
    return model, ccd, worker_metadata, time.monotonic() - setup_started


def target_features(sequence: str, ccd: object) -> tuple[dict, object]:
    """Build reusable single-sequence features for one target."""
    import torch
    from helico.bench import single_sequence_msa
    from helico.data import tokenize_sequences

    tokenized = tokenize_sequences(
        [{"type": "protein", "id": "A", "sequence": sequence}], ccd
    )
    if tokenized.n_tokens != len(sequence):
        raise ValueError(f"{tokenized.n_tokens} tokens for {len(sequence)} residues")
    features = tokenized.to_features()
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
    return batch, tokenized


def predict_candidate(
    model: object,
    base_batch: dict,
    tokenized: object,
    candidate: dict,
    target: str,
) -> dict:
    """Run one candidate with the exact paired-seed pilot protocol."""
    import numpy as np
    import torch
    from helico.bench import compute_tm_score
    from helico.data import CONTACT_ABSENT, CONTACT_PRESENT, CONTACT_UNKNOWN
    from helico.train import run_inference

    total_started = time.monotonic()
    n_tokens = tokenized.n_tokens
    state = torch.full((n_tokens, n_tokens), CONTACT_UNKNOWN, dtype=torch.uint8)
    for left in range(n_tokens):
        state[left, left + 6 :] = CONTACT_ABSENT
        state[left + 6 :, left] = CONTACT_ABSENT
    for left, right in candidate["present_pairs"]:
        if right - left < 6:
            raise ValueError(f"unexpected short-range pair {(left, right)}")
        state[left, right] = CONTACT_PRESENT
        state[right, left] = CONTACT_PRESENT
    batch = dict(base_batch)
    batch["contact_state"] = state.unsqueeze(0)

    seed = int(hashlib.sha256(target.encode()).hexdigest()[:8], 16)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    inference_started = time.monotonic()
    results = run_inference(
        model,
        batch,
        n_samples=N_SAMPLES,
        device="cuda",
        dtype=torch.bfloat16,
        n_cycles=N_CYCLES,
    )
    torch.cuda.synchronize()
    inference_seconds = time.monotonic() - inference_started

    ptms = results["all_ptm"][0].cpu().float().tolist()
    rankings = results["all_ranking_score"][0].cpu().float().tolist()
    if len(ptms) != N_SAMPLES or any(not np.isfinite(value) for value in ptms):
        raise ValueError(f"{target}/{candidate['decoy_id']}: invalid pTM values {ptms}")
    selected_sample = max(range(N_SAMPLES), key=rankings.__getitem__)
    if selected_sample != max(range(N_SAMPLES), key=ptms.__getitem__):
        raise ValueError("monomer ranking score did not select maximum pTM")

    rep_atom_idx = base_batch["rep_atom_idx"][0].cpu().numpy()
    candidate_ca = np.asarray(candidate["candidate_ca"], dtype=np.float64)
    if len(rep_atom_idx) != len(candidate_ca):
        raise ValueError(
            f"{target}/{candidate['decoy_id']}: {len(rep_atom_idx)} predicted "
            f"representative atoms for {len(candidate_ca)} candidate C-alpha atoms"
        )
    sample_rows = []
    for sample_index in range(N_SAMPLES):
        predicted_ca = (
            results["all_coords"][0, sample_index, rep_atom_idx].cpu().float().numpy()
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
        raise ValueError(f"{target}/{candidate['decoy_id']}: nonfinite pLDDT")
    return {
        "target": target,
        "decoy_id": candidate["decoy_id"],
        "candidate_kind": "native" if candidate["decoy_id"] == "native" else "decoy",
        "map_mode": MAP_MODE,
        "n_residues": n_tokens,
        "n_pairs": n_tokens * (n_tokens - 1) // 2,
        "n_present_contacts": len(candidate["present_pairs"]),
        "contact_map_sha256": candidate["contact_map_sha256"],
        "seed": seed,
        "selected_sample_idx": selected_sample,
        "max_ptm": float(ptms[selected_sample]),
        "mean_sample_ptm": float(np.mean(ptms)),
        "mean_ca_plddt": float(np.mean(selected_plddt)),
        "candidate_output_tm": sample_rows[selected_sample]["candidate_output_tm"],
        "helico_composite": sample_rows[selected_sample]["helico_composite"],
        "samples": sample_rows,
        "timing": {
            "elapsed_seconds": round(inference_seconds, 6),
            "preoutput_seconds": round(time.monotonic() - total_started, 6),
        },
    }


def write_part(
    task: dict,
    results: list[dict],
    *,
    worker_metadata: dict,
    model_load_seconds: float,
    model_load_share_seconds: float,
    runner_setup_share_seconds: float,
    target_setup_share_seconds: float,
    part_started: float,
) -> None:
    """Write durable metrics, then a timing companion measured through upload."""
    import fsspec

    metrics_uri, timing_uri = output_uris(task)
    candidate_ids = [result["decoy_id"] for result in results]
    payload = {
        "schema_version": 1,
        "run_fingerprint": RUN_FINGERPRINT,
        "config": CONFIG,
        "target_input_sha256": task["input_sha256"],
        "candidate_ids": candidate_ids,
        "results": results,
    }
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    compressed = gzip.compress(encoded, compresslevel=6, mtime=0)
    serialization_done = time.monotonic()
    with fsspec.open(metrics_uri, "wb") as stream:
        stream.write(compressed)
    output_done = time.monotonic()
    output_seconds = output_done - serialization_done
    serialization_seconds = max(
        0.0,
        serialization_done
        - part_started
        - sum(result["timing"]["preoutput_seconds"] for result in results),
    )
    output_share = (serialization_seconds + output_seconds) / len(results)
    timing_rows = []
    for result in results:
        timing = result["timing"]
        timing_rows.append(
            {
                "stem": f"{result['target']}/{result['decoy_id']}",
                "n_residues": result["n_residues"],
                "n_pairs": result["n_pairs"],
                "mode": MAP_MODE,
                "elapsed_seconds": timing["elapsed_seconds"],
                "model_load_seconds": round(model_load_seconds, 6),
                "model_load_share_seconds": round(model_load_share_seconds, 6),
                "runner_setup_share_seconds": round(runner_setup_share_seconds, 6),
                "target_setup_share_seconds": round(target_setup_share_seconds, 6),
                "output_share_seconds": round(output_share, 6),
                "total_seconds": round(
                    timing["preoutput_seconds"]
                    + model_load_share_seconds
                    + runner_setup_share_seconds
                    + target_setup_share_seconds
                    + output_share,
                    6,
                ),
                "model_nickname": "helico-contacts-msafree-01-step-6000",
                "runner_tag": "iris-cw-rno2a",
                "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
                **worker_metadata,
            }
        )
    timing_payload = {
        "schema_version": 1,
        "run_fingerprint": RUN_FINGERPRINT,
        "target_input_sha256": task["input_sha256"],
        "candidate_ids": candidate_ids,
        "metrics_uri": metrics_uri,
        "metrics_sha256": sha256_bytes(compressed),
        "metrics_bytes": len(compressed),
        "part_total_seconds_through_metrics_upload": round(
            output_done - part_started, 6
        ),
        "serialization_seconds": round(serialization_seconds, 6),
        "metrics_upload_seconds": round(output_seconds, 6),
        "timings": timing_rows,
    }
    with fsspec.open(timing_uri, "wt") as stream:
        json.dump(timing_payload, stream, separators=(",", ":"), sort_keys=True)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--limit-parts", type=int, default=0)
    parser.add_argument("--target", default=None)
    return parser.parse_args()


def main() -> None:
    """Run or resume this worker's balanced subset of candidate parts."""
    args = parse_args()
    bootstrap_started = float(os.environ.get("EXP335_BOOTSTRAP_STARTED", time.time()))
    if not 0 <= args.shard < args.num_shards:
        raise ValueError(f"invalid shard {args.shard}/{args.num_shards}")
    manifest = read_json(MANIFEST_URI)
    if manifest.get("schema_version") != 2:
        raise ValueError(
            f"expected prepared-input schema 2, found {manifest.get('schema_version')}"
        )
    if manifest.get("n_targets") != 133 or manifest.get("n_candidates") != 180_212:
        raise ValueError(
            "unexpected benchmark coverage: "
            f"{manifest.get('n_targets')} targets / "
            f"{manifest.get('n_candidates')} candidates"
        )
    shards = assign_tasks(manifest, args.num_shards)
    tasks = shards[args.shard]
    if args.target is not None:
        tasks = [task for task in tasks if task["target"] == args.target]
    if args.limit_parts:
        tasks = tasks[: args.limit_parts]
    assigned_candidates = sum(task["end"] - task["start"] for task in tasks)
    print(
        f"[exp335] shard={args.shard}/{args.num_shards} parts={len(tasks)} "
        f"candidates={assigned_candidates} fingerprint={RUN_FINGERPRINT}",
        flush=True,
    )
    if not tasks:
        return

    target_payloads = {}
    pending = []
    completed_candidates = 0
    for task in tasks:
        if task["target"] not in target_payloads:
            target_payloads[task["target"]] = load_target(task)
        payload = target_payloads[task["target"]]
        candidates = payload["candidates"][task["start"] : task["end"]]
        candidate_ids = [candidate["decoy_id"] for candidate in candidates]
        if existing_part_valid(task, candidate_ids):
            completed_candidates += len(candidates)
        else:
            pending.append(task)
    if not pending:
        print(
            f"[exp335] SUCCEEDED shard={args.shard}/{args.num_shards}; "
            f"all {completed_candidates} candidates already complete",
            flush=True,
        )
        return

    model, ccd, worker_metadata, model_load_seconds = setup_model()
    worker_setup_seconds = time.time() - bootstrap_started
    pending_candidates = sum(task["end"] - task["start"] for task in pending)
    model_load_share = model_load_seconds / pending_candidates
    runner_setup_share = (
        max(0.0, worker_setup_seconds - model_load_seconds) / pending_candidates
    )
    pending_by_target = {}
    for task in pending:
        pending_by_target[task["target"]] = pending_by_target.get(task["target"], 0) + (
            task["end"] - task["start"]
        )
    cached_target = None
    cached_features = None
    cached_tokenized = None
    target_setup_share = 0.0
    for part_index, task in enumerate(pending, 1):
        cached_payload = target_payloads[task["target"]]
        if task["target"] != cached_target:
            target_setup_started = time.monotonic()
            cached_features, cached_tokenized = target_features(
                cached_payload["sequence"], ccd
            )
            target_setup_share = (
                time.monotonic() - target_setup_started
            ) / pending_by_target[task["target"]]
            cached_target = task["target"]
        candidates = cached_payload["candidates"][task["start"] : task["end"]]
        part_started = time.monotonic()
        results = [
            predict_candidate(
                model,
                cached_features,
                cached_tokenized,
                candidate,
                task["target"],
            )
            for candidate in candidates
        ]
        write_part(
            task,
            results,
            worker_metadata=worker_metadata,
            model_load_seconds=model_load_seconds,
            model_load_share_seconds=model_load_share,
            runner_setup_share_seconds=runner_setup_share,
            target_setup_share_seconds=target_setup_share,
            part_started=part_started,
        )
        completed_candidates += len(candidates)
        print(
            f"[exp335] {part_index}/{len(pending)} wrote {task['target']} "
            f"{task['start']}:{task['end']} ({completed_candidates}/{assigned_candidates})",
            flush=True,
        )

    print(
        f"[exp335] SUCCEEDED shard={args.shard}/{args.num_shards} "
        f"candidates={completed_candidates}",
        flush=True,
    )


if __name__ == "__main__":
    main()
