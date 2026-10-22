"""Freeze and export one pinned checkpoint entirely inside CoreWeave."""

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import fsspec
from levanter.main.export_lm_to_hf import ConvertLmConfig
from levanter.main.export_lm_to_hf import main as export_model
from levanter.trainer import TrainerConfig

from experiments.exp232_sweep_cv1_decontam.training_contract import MODEL_CONFIG
from experiments.exp277_models_single_mpnn_pilot.config import TOKENIZER


def freeze_source(fs: fsspec.AbstractFileSystem, checkpoint: dict) -> None:
    """Make an idempotent same-bucket copy before training garbage-collects it."""
    source, destination = checkpoint["source"], checkpoint["frozen"]
    marker = destination + "/verified.json"
    if fs.exists(marker):
        with fs.open(marker) as handle:
            previous = json.load(handle)
        if previous != checkpoint["objects"]:
            raise ValueError("Frozen checkpoint inventory changed")
        for obj in previous:
            if fs.info(destination + "/" + obj["path"])["size"] != obj["size"]:
                raise ValueError(f"Frozen checkpoint object changed: {obj['path']}")
        return

    def copy_object(obj: dict) -> None:
        src, dst = source + "/" + obj["path"], destination + "/" + obj["path"]
        info = fs.info(src)
        if (info["size"], info["ETag"]) != (obj["size"], obj["etag"]):
            raise ValueError(f"Source object changed: {src}")
        fs.copy(src, dst)
        if fs.info(dst)["size"] != obj["size"]:
            raise ValueError(f"Incomplete frozen object: {dst}")

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(copy_object, checkpoint["objects"]))
    with fs.open(destination + "/metadata.json") as handle:
        metadata = json.load(handle)
    if metadata["step"] != checkpoint["step"]:
        raise ValueError("Frozen checkpoint step changed")
    with fs.open(marker, "wt") as handle:
        json.dump(checkpoint["objects"], handle)


def stage_native_local(
    fs: fsspec.AbstractFileSystem, checkpoint: dict, directory: Path
) -> dict:
    """Pin native bytes on worker disk when the storage account rejects writes."""
    native = directory / checkpoint["run_id"] / f"step-{checkpoint['step']}"
    native.mkdir(parents=True, exist_ok=True)

    def download(obj: dict) -> None:
        source = checkpoint["source"] + "/" + obj["path"]
        destination = native / obj["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        before = fs.info(source)
        if (before["size"], before["ETag"]) != (obj["size"], obj["etag"]):
            raise ValueError(f"Source changed before download: {source}")
        fs.get_file(source, str(destination))
        after = fs.info(source)
        if (after["size"], after["ETag"]) != (obj["size"], obj["etag"]):
            raise ValueError(f"Source changed during download: {source}")
        if destination.stat().st_size != obj["size"]:
            raise ValueError(f"Incomplete local checkpoint object: {source}")

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(download, checkpoint["objects"]))
    metadata = json.loads((native / "metadata.json").read_text())
    if metadata["step"] != checkpoint["step"]:
        raise ValueError("Local native checkpoint has the wrong step")
    return {
        **checkpoint,
        "frozen": str(native),
        "hf_export": str(
            directory / checkpoint["run_id"] / "hf" / f"step-{checkpoint['step']}"
        ),
    }


def main() -> None:
    """Convert full-precision model weights and save the tokenizer beside them."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--trial", required=True)
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument(
        "--scratch", type=Path, help="Export to worker disk without any S3 writes"
    )
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    checkpoint = next(c for c in plan["checkpoints"] if c["trial"] == args.trial)
    fs = fsspec.filesystem("s3", use_listings_cache=False)
    if args.scratch:
        checkpoint = stage_native_local(fs, checkpoint, args.scratch)
        fs = fsspec.filesystem("file")
        manifest_uri = str(args.scratch / "export.json")
    else:
        freeze_source(fs, checkpoint)
        manifest_uri = f"{plan['root']}/manifests/{checkpoint['label']}.json"
    if args.freeze_only:
        return
    if fs.exists(manifest_uri):
        with fs.open(manifest_uri) as handle:
            manifest = json.load(handle)
        for obj in manifest["files"]:
            if (
                fs.info(checkpoint["hf_export"] + "/" + obj["name"])["size"]
                != obj["size"]
            ):
                raise ValueError(f"Export object changed: {obj['name']}")
        print(f"Already exported and verified {checkpoint['label']}", flush=True)
        return
    for key in (
        "hidden_dim",
        "intermediate_dim",
        "num_heads",
        "num_kv_heads",
        "num_layers",
        "max_seq_len",
        "use_qk_norm",
        "layer_norm_epsilon",
        "tie_word_embeddings",
        "activation_function",
        "use_bias",
    ):
        if checkpoint["model_config"][key] != getattr(MODEL_CONFIG, key):
            raise ValueError(f"Export architecture mismatch at {key}")
    if checkpoint["model_config"]["rope"] != asdict(MODEL_CONFIG.rope):
        raise ValueError("Export rotary-embedding configuration mismatch")
    export_model(
        ConvertLmConfig(
            trainer=TrainerConfig(require_accelerator=False),
            checkpoint_path=checkpoint["frozen"],
            checkpoint_subpath="model",
            output_dir=checkpoint["hf_export"],
            model=MODEL_CONFIG,
            tokenizer=TOKENIZER,
            override_vocab_size=2845,
            save_tokenizer=True,
            max_shard_size=5_000_000_000,
            use_cpu=True,
        )
    )
    files = []
    for info in fs.ls(checkpoint["hf_export"], detail=True):
        if info["type"] != "file":
            continue
        digest = hashlib.sha256()
        with fs.open(info["name"], "rb") as handle:
            while block := handle.read(8 * 1024**2):
                digest.update(block)
        files.append(
            {
                "name": Path(info["name"]).name,
                "size": info["size"],
                "sha256": digest.hexdigest(),
            }
        )
    names = {entry["name"] for entry in files}
    required = {
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
    }
    if not required <= names:
        raise ValueError(f"Incomplete export: {names}")
    with fs.open(checkpoint["hf_export"] + "/model.safetensors.index.json") as handle:
        index = json.load(handle)
    if not set(index["weight_map"].values()) <= names:
        raise ValueError("An indexed weight shard is missing")
    with fs.open(manifest_uri, "wt") as handle:
        json.dump(
            {
                "checkpoint": checkpoint,
                "files": files,
                "dtype": "float32",
                "inference_dtype": "bfloat16",
            },
            handle,
            indent=2,
        )
    print(f"EXPORTED {checkpoint['label']}", flush=True)


if __name__ == "__main__":
    main()
