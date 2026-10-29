"""Pin and mirror the verified TPU export once for the common CUDA evaluator.

Preparation reads metadata only. Copying runs on a CPU worker in us-east5 and
streams fewer than 10 GB to CoreWeave, without moving optimizer state. The GPU
worker verifies the recorded SHA-256 digests after its region-local download.
"""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import fsspec

HERE = Path(__file__).resolve().parent
PLAN = HERE / "data/contact_eval/plan.json"
TRIAL = "lr5e-4-cooldown-tpu"
BLOCK_BYTES = 8 * 1024**2


def prepare(plan: dict, verification: dict) -> dict:
    """Pin immutable GCS generations and the previously verified weight digests."""
    source = verification["export_verification"]["export"]
    if not verification["export_verification"]["all_parameters_finite"]:
        raise ValueError("The TPU export is not verified finite")
    if set(verification["scalar_counters"].values()) != {186885}:
        raise ValueError("The TPU endpoint counters are not exactly one epoch")
    if any(c["trial"] == TRIAL for c in plan["checkpoints"]):
        raise ValueError("The TPU checkpoint is already in the plan")
    fs = fsspec.filesystem("gcs")
    files = []
    for entry in sorted(fs.ls(source, detail=True), key=lambda item: item["name"]):
        if entry["type"] != "file":
            raise ValueError("Unexpected directory in TPU export")
        name = entry["name"].rsplit("/", 1)[-1]
        if name in verification["export_verification"]["files"]:
            digest = verification["export_verification"]["files"][name]["sha256"]
        else:
            if entry["size"] > 1024**2:
                raise ValueError("Unexpected large non-weight export file")
            digest = hashlib.sha256(fs.cat_file(entry["name"])).hexdigest()
        files.append(
            {
                "name": name,
                "size": entry["size"],
                "sha256": digest,
                "generation": entry["generation"],
            }
        )
    total = sum(obj["size"] for obj in files)
    if total >= 10_000_000_000:
        raise ValueError("The one-time mirror exceeds the transfer budget")
    run_id = verification["root"].rsplit("/", 1)[-1]
    checkpoint = {
        "trial": TRIAL,
        "run_id": run_id,
        "step": 186884,
        "label": f"{run_id}-step-186884",
        "source_format": "huggingface",
        "source": source,
        "source_files": files,
        "source_bytes": total,
        "hf_export": f"{plan['root']}/models/{run_id}/hf/step-186884",
        "wandb_url": f"https://wandb.ai/open-athena/MarinFold/runs/{run_id}",
        "latest_eval_loss": verification["final_validation"]["eval/loss"],
        "model_egress_bytes": total,
        "source_region": "us-east5",
        "evaluation_region": "cw-us-east-02a",
    }
    return {**plan, "checkpoints": [*plan["checkpoints"], checkpoint]}


def copy_export(plan: dict) -> None:
    """Stream the pinned export and publish its identity only after verification."""
    checkpoint = next(c for c in plan["checkpoints"] if c["trial"] == TRIAL)
    if checkpoint["source_bytes"] >= 10_000_000_000:
        raise ValueError("The mirror exceeds the transfer budget")
    source_fs = fsspec.filesystem("gcs", cache_timeout=0)
    destination_fs = fsspec.filesystem("s3", use_listings_cache=False)
    for obj in checkpoint["source_files"]:
        source = checkpoint["source"] + "/" + obj["name"]
        destination = checkpoint["hf_export"] + "/" + obj["name"]
        if source_fs.info(source)["generation"] != obj["generation"]:
            raise ValueError(f"GCS source generation changed: {source}")
        if destination_fs.exists(destination):
            info = destination_fs.info(destination)
            metadata = destination_fs.getxattr(destination, "source_sha256")
            if info["size"] == obj["size"] and metadata == obj["sha256"]:
                print("REUSED", obj["name"], flush=True)
                continue
            raise ValueError(f"An unverified destination already exists: {destination}")
        digest = hashlib.sha256()
        size = 0
        with (
            source_fs.open(
                source, "rb", block_size=BLOCK_BYTES, cache_type="none"
            ) as reader,
            destination_fs.open(destination, "wb", block_size=BLOCK_BYTES) as writer,
        ):
            while block := reader.read(BLOCK_BYTES):
                digest.update(block)
                size += len(block)
                writer.write(block)
        source_fs.invalidate_cache(source)
        if (size, digest.hexdigest()) != (obj["size"], obj["sha256"]):
            raise ValueError(
                f"Transferred content differs from verified export: {source}"
            )
        if source_fs.info(source)["generation"] != obj["generation"]:
            raise ValueError(f"GCS source changed during transfer: {source}")
        if destination_fs.info(destination)["size"] != size:
            raise ValueError(f"Incomplete destination: {destination}")
        destination_fs.setxattr(destination, source_sha256=obj["sha256"])
        print("COPIED", obj["name"], size, flush=True)
    manifest = {
        "checkpoint": checkpoint,
        "files": [
            {k: obj[k] for k in ("name", "size", "sha256")}
            for obj in checkpoint["source_files"]
        ],
        "dtype": "float32",
        "inference_dtype": "bfloat16",
        "transferred_at": datetime.now(UTC).isoformat(),
        "transfer_bytes": checkpoint["source_bytes"],
    }
    with destination_fs.open(
        f"{plan['root']}/manifests/{checkpoint['label']}.json", "wt"
    ) as stream:
        json.dump(manifest, stream, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--copy", action="store_true")
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    if args.copy:
        copy_export(plan)
        return
    verification = json.loads((HERE / "data/tpu_final_verification.json").read_text())
    plan = prepare(plan, verification)
    PLAN.write_text(json.dumps(plan, indent=2) + "\n")
    print("Pinned TPU mirror:", plan["checkpoints"][-1]["source_bytes"], "bytes")


if __name__ == "__main__":
    main()
