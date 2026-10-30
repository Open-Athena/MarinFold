"""Stream every exported parameter in-region before accepting a final model."""

import hashlib
import json
import struct
from datetime import UTC, datetime

import fsspec
import numpy as np


def verify_export(export_path: str) -> dict:
    """Check all float32 HF weights, tensor inventory and co-located tokenizer."""
    fs, root = fsspec.core.url_to_fs(export_path)
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        with fs.open(root + "/" + name, "rb") as stream:
            json.load(stream)
    with fs.open(root + "/model.safetensors.index.json", "rb") as stream:
        index = json.load(stream)
    total_bytes = 0
    files = {}
    for shard in sorted(set(index["weight_map"].values())):
        digest = hashlib.sha256()
        with fs.open(root + "/" + shard, "rb", block_size=8 * 1024**2) as stream:
            size_bytes = stream.read(8)
            header_bytes = stream.read(struct.unpack("<Q", size_bytes)[0])
            digest.update(size_bytes + header_bytes)
            header = json.loads(header_bytes)
            tensors = {k: v for k, v in header.items() if k != "__metadata__"}
            expected = {k for k, v in index["weight_map"].items() if v == shard}
            if set(tensors) != expected:
                raise ValueError(f"Tensor inventory differs from index: {shard}")
            ordered = sorted(
                tensors.items(), key=lambda item: item[1]["data_offsets"][0]
            )
            end = 0
            for name, tensor in ordered:
                start, end_offset = tensor["data_offsets"]
                nbytes = int(np.prod(tensor["shape"])) * 4
                if (
                    tensor["dtype"] != "F32"
                    or start != end
                    or end_offset - start != nbytes
                ):
                    raise ValueError(f"Unexpected export layout: {name}")
                remaining = nbytes
                while remaining:
                    chunk = stream.read(min(8 * 1024**2, remaining))
                    if not chunk or len(chunk) % 4:
                        raise ValueError(f"Truncated exported tensor: {name}")
                    if not np.isfinite(np.frombuffer(chunk, dtype="<f4")).all():
                        raise ValueError(f"Nonfinite exported tensor: {name}")
                    digest.update(chunk)
                    remaining -= len(chunk)
                end = end_offset
            if stream.read(1):
                raise ValueError(f"Unexpected trailing weight bytes: {shard}")
        total_bytes += end
        files[shard] = {"parameter_bytes": end, "sha256": digest.hexdigest()}
    if total_bytes != index["metadata"]["total_size"]:
        raise ValueError("Exported parameter count differs from index metadata")
    return {
        "verified_at": datetime.now(UTC).isoformat(),
        "export": export_path,
        "all_parameters_finite": True,
        "parameter_bytes": total_bytes,
        "files": files,
    }
