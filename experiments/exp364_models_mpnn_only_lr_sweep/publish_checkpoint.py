"""Publish the selected, evaluated one-epoch GPU checkpoint without changing bytes."""

import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import fsspec
import hf_xet
from huggingface_hub import HfFileSystem

HERE = Path(__file__).resolve().parent
RUN = "contacts-v1-exp364-mpnn-only-1ep-lr5e-4-cooldown-1.5B"
DESTINATION = f"buckets/open-athena/MarinFold/checkpoints/{RUN}/hf/step-186884"


def sha256(path: Path) -> str:
    """Hash a model file without loading it into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    """Verify the entire source before publishing and check every public Xet hash."""
    source = json.loads((HERE / "data/publication_source.json").read_text())
    checkpoint = source["checkpoint"]
    if (checkpoint["run_id"], checkpoint["step"]) != (RUN, 186884):
        raise ValueError("Publication source is not the selected checkpoint")
    if not source["finite_verification"]["all_parameters_finite"]:
        raise ValueError("The source must have a complete finite-weight verification")
    required = {
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    }
    if {entry["name"] for entry in source["files"]} != required:
        raise ValueError("Unexpected publication file inventory")
    fs = fsspec.filesystem("s3", use_listings_cache=False)
    hub = HfFileSystem(token=os.environ["HF_TOKEN"])
    if hub.exists(DESTINATION):
        raise ValueError("Destination already exists; inspect it before any retry")
    manifest = {
        "published_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": os.environ["GIT_COMMIT"],
        "source": source,
        "published_uri": "hf://" + DESTINATION,
        "publication_job": os.environ["PUBLICATION_JOB"],
        "source_region": "cw-us-east-02a",
        "source_bytes": sum(entry["size"] for entry in source["files"]),
        "weights_dtype": "float32",
        "recommended_inference_dtype": "bfloat16",
        "all_source_files_byte_identical": True,
        "files": {},
    }
    with tempfile.TemporaryDirectory(prefix="exp364-publish-") as temporary:
        directory = Path(temporary)
        for entry in source["files"]:
            path = directory / entry["name"]
            fs.get_file(checkpoint["hf_export"] + "/" + entry["name"], str(path))
            if path.stat().st_size != entry["size"] or sha256(path) != entry["sha256"]:
                raise ValueError(f"Source identity mismatch: {entry['name']}")
            print("VERIFIED_SOURCE", entry["name"], flush=True)
        config = json.loads((directory / "config.json").read_text())
        tokenizer = json.loads((directory / "tokenizer.json").read_text())
        tokenizer_config = json.loads((directory / "tokenizer_config.json").read_text())
        if (
            config["model_type"] != "qwen3"
            or config["vocab_size"] != 2845
            or config["rope_theta"] != 500000
            or config["rope_parameters"]["rope_theta"] != 500000
            or config["rope_scaling"]["rope_type"] != "llama3"
            or config["rope_scaling"]["factor"] != 8.0
            or len(tokenizer["model"]["vocab"]) != 2845
            or tokenizer["model"]["vocab"]["<contact>"] != 5
            or tokenizer_config["tokenizer_class"] != "PreTrainedTokenizerFast"
        ):
            raise ValueError("Unexpected model configuration or tokenizer")
        index = json.loads((directory / "model.safetensors.index.json").read_text())
        if set(index["weight_map"].values()) != {
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        }:
            raise ValueError("Weight index does not cover the verified shards")
        (directory / "README.md").write_bytes(
            (HERE / "publication_model_card.md").read_bytes()
        )
        for path in sorted(directory.iterdir()):
            digest = sha256(path)
            local_xet = hf_xet.hash_files([str(path)])[0].hash
            hub.put_file(str(path), DESTINATION + "/" + path.name)
            public = HfFileSystem(token=False).info(DESTINATION + "/" + path.name)
            if public["size"] != path.stat().st_size or public["xet_hash"] != local_xet:
                raise ValueError(f"Published content differs: {path.name}")
            manifest["files"][path.name] = {
                "size": path.stat().st_size,
                "sha256": digest,
                "xet_hash": local_xet,
                "source_sha256": next(
                    (
                        entry["sha256"]
                        for entry in source["files"]
                        if entry["name"] == path.name
                    ),
                    None,
                ),
            }
            print("VERIFIED_PUBLIC", path.name, path.stat().st_size, digest, flush=True)
        with hub.open(DESTINATION + "/publication_manifest.json", "wt") as stream:
            json.dump(manifest, stream, indent=2)
            stream.write("\n")
    print("COMPLETE", "hf://" + DESTINATION, flush=True)


if __name__ == "__main__":
    main()
