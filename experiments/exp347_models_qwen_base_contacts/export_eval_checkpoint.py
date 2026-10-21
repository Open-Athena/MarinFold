"""Export a preserved FP32 checkpoint to BF16 for eval-val scoring."""

import argparse
import json
import tempfile
from pathlib import Path

import fsspec
import torch
from safetensors.torch import load_file, save_file

from common import ROOT


def convert_shard(source: Path, destination: Path) -> int:
    """Convert floating tensors only, preserving tensor names and shapes."""
    original = load_file(source)
    converted = {
        name: value.to(torch.bfloat16) if value.is_floating_point() else value
        for name, value in original.items()
    }
    save_file(converted, destination, metadata={"format": "pt"})
    return sum(value.numel() * value.element_size() for value in converted.values())


def export_snapshot(snapshot: str) -> dict:
    """Publish a complete inference export and return its evaluation request."""
    fs, source = fsspec.core.url_to_fs(snapshot)
    marker = json.loads(fs.cat_file(source + "/snapshot.json"))
    root = f"{ROOT}/checkpoints/{marker['run_id']}"
    destination = f"{root}/hf/step-{marker['step']}"
    request = {
        "run_id": marker["run_id"],
        "step": marker["step"],
        "tokens": marker["tokens"],
        "format": marker["format"],
        "checkpoint": destination,
        "final": False,
    }
    files = fs.find(source)
    index = json.loads(fs.cat_file(source + "/model.safetensors.index.json"))
    required = set(index["weight_map"].values()) | {
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    if not required.issubset({p.rsplit("/", 1)[1] for p in files}):
        raise ValueError("Snapshot is missing indexed weights or tokenizer files")
    total_size = 0
    with tempfile.TemporaryDirectory(prefix="exp347-export-") as directory:
        local = Path(directory)
        for filename in sorted(set(index["weight_map"].values())):
            fs.get_file(source + "/" + filename, str(local / "source.safetensors"))
            total_size += convert_shard(
                local / "source.safetensors", local / "export.safetensors"
            )
            fs.put_file(str(local / "export.safetensors"), destination + "/" + filename)
            print(json.dumps({"exported": destination + "/" + filename}), flush=True)
    for path in files:
        filename = path.rsplit("/", 1)[1]
        if filename.endswith(".safetensors") or filename in {
            "model.safetensors.index.json",
            "snapshot.json",
        }:
            continue
        payload = fs.cat_file(path)
        if filename == "config.json":
            config = json.loads(payload)
            config["dtype"] = "bfloat16"
            payload = json.dumps(config).encode()
        fs.pipe_file(destination + "/" + filename, payload)
    index["metadata"]["total_size"] = total_size
    fs.pipe_file(
        destination + "/model.safetensors.index.json", json.dumps(index).encode()
    )
    fs.pipe_file(
        destination + "/export.json",
        json.dumps(
            {"source": marker, "dtype": "bfloat16", "request": request}
        ).encode(),
    )
    return request


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True, nargs="+")
    args = parser.parse_args()
    torch.set_num_threads(2)
    for snapshot in args.snapshot:
        print(json.dumps(export_snapshot(snapshot)), flush=True)


if __name__ == "__main__":
    main()
