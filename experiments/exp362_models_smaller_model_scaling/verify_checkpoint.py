"""Check native completion metadata and HF export structure without reading weights."""

import argparse
import configparser
import json
import math
import re
from datetime import UTC, datetime
from pathlib import Path

import fsspec

PREFIX = "marin-us-east-02a/MarinFold/exp362_models_smaller_model_scaling/runs"
PARAMETERS = {4: 254941696, 8: 498228224, 16: 984801280}


def main() -> None:
    """Record export identities and validate shapes, tokenizer and checkpoint step."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--workstation", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    match = re.fullmatch(r"contacts-v1-exp362-depth(4|8|16)-full-epoch(?:-smoke-[\w-]+)?", args.run_id)
    if match is None:
        raise ValueError(f"Invalid exp362 run identity: {args.run_id}")
    layers = int(match[1])
    if args.workstation:
        credentials = configparser.ConfigParser()
        credentials.read(Path.home() / ".aws/credentials")
        cw = credentials["cw"]
        fs = fsspec.filesystem(
            "s3", key=cw["aws_access_key_id"], secret=cw["aws_secret_access_key"],
            endpoint_url="https://cwobject.com",
            config_kwargs={"s3": {"addressing_style": "virtual"}},
        )
    else:
        fs = fsspec.filesystem("s3")
    root = f"{PREFIX}/{args.run_id}"
    native = f"{root}/checkpoints/step-{args.step}"
    metadata = json.loads(fs.cat(f"{native}/metadata.json"))
    if metadata["step"] != args.step or metadata["is_temporary"]:
        raise ValueError(f"Native checkpoint is not a completed step {args.step}")
    for name in ("manifest.json", "manifest.ocdbt"):
        if fs.info(f"{native}/{name}")["size"] <= 0:
            raise ValueError(f"Empty native checkpoint manifest: {name}")
    if not fs.ls(f"{native}/d"):
        raise ValueError("Native checkpoint has no OCDBT data")
    export = f"{root}/hf/step-{args.step}"
    files = []
    for info in fs.ls(export, detail=True):
        if info["type"] == "directory":
            continue
        files.append({
            "name": info["name"].rsplit("/", 1)[-1], "size": info["size"],
            "digest": info["ETag"].strip('"'), "digest_kind": "s3-etag",
        })
    files.sort(key=lambda entry: entry["name"])
    names = {entry["name"] for entry in files}
    required = {"config.json", "tokenizer.json", "tokenizer_config.json"}
    if required - names:
        raise ValueError(f"Missing export files: {sorted(required - names)}")
    config = json.loads(fs.cat(f"{export}/config.json"))
    for key, expected in {"num_hidden_layers": layers, "hidden_size": 2048,
                          "intermediate_size": 8192, "num_attention_heads": 32,
                          "num_key_value_heads": 8, "vocab_size": 2845}.items():
        if config[key] != expected:
            raise ValueError(f"Unexpected exported {key}: {config[key]}")
    vocabulary = json.loads(fs.cat(f"{export}/tokenizer.json"))["model"]["vocab"]
    if len(vocabulary) != 2845 or vocabulary["<contact>"] != 5:
        raise ValueError("Unexpected contacts-v1 tokenizer")
    parameter_count = 0
    weights = [entry for entry in files if entry["name"].endswith(".safetensors")]
    for entry in weights:
        with fs.open(f"{export}/{entry['name']}", "rb", block_size=65536) as reader:
            header_size = int.from_bytes(reader.read(8), "little")
            if not 0 < header_size < 16_000_000:
                raise ValueError("Invalid safetensors header length")
            header = json.loads(reader.read(header_size))
        tensors = [value for key, value in header.items() if key != "__metadata__"]
        if 8 + header_size + max(t["data_offsets"][1] for t in tensors) != entry["size"]:
            raise ValueError(f"Truncated or malformed weights: {entry['name']}")
        parameter_count += sum(math.prod(t["shape"]) for t in tensors)
    if parameter_count != PARAMETERS[layers]:
        raise ValueError(f"Export contains {parameter_count} parameters; expected {PARAMETERS[layers]}")
    result = {
        "label": f"{args.run_id}-step-{args.step}", "run_name": args.run_id,
        "step": args.step, "coreweave_uri": f"s3://{export}",
        "native_uri": f"s3://{native}", "native_metadata": metadata,
        "parameter_count": parameter_count, "checkpoint_files": files,
        "total_bytes": sum(entry["size"] for entry in files),
        "verified_at": datetime.now(UTC).isoformat(),
        "verification_scope": "Native metadata and manifests; HF configuration, tokenizer, tensor headers and object sizes. Weight payload bytes were not downloaded or hashed.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
