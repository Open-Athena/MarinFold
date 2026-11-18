"""Export TPU parameters to the original HF coordinates for regional evaluation."""

import json
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
from flax import nnx
from maxtext.checkpoint_conversion.utils.param_mapping import (
    QWEN3_5_MAXTEXT_TO_HF_PARAM_HOOK_FN,
)
from safetensors.torch import save_file

from tpu_checkpoint import ROOT, as_nnx, read_json
from tpu_model import parameter_mapping
from tpu_state import parameter_shapes, serialized_parameter_names


def hf_parameters(
    params: dict, template: nnx.State, source: dict, config: Any
) -> Iterator[tuple[str, np.ndarray]]:
    """Invert every parameter permutation and require exact source-model coverage."""
    mapping, _ = parameter_mapping(source, config)
    hooks = QWEN3_5_MAXTEXT_TO_HF_PARAM_HOOK_FN(
        source, config, scan_layers=False, saving_to_hf=True
    )
    shapes = parameter_shapes(source)
    seen = set()
    for path, variable in nnx.to_flat_state(as_nnx(template, params)):
        key = "params-" + "-".join(map(str, path))
        names = mapping[key]
        value = np.asarray(variable.get_value())
        target_shape = shapes[names] if isinstance(names, str) else None
        if key in hooks:
            value = hooks[key](value, target_shape)
        pairs = (
            ((names, value),)
            if isinstance(names, str)
            else zip(names, value, strict=True)
        )
        for name, array in pairs:
            if name in seen or array.shape != shapes[name] or array.dtype != np.float32:
                raise ValueError(f"Invalid inverse parameter mapping: {name}")
            if not np.isfinite(array).all():
                raise ValueError(f"Nonfinite exported parameter: {name}")
            seen.add(name)
            yield name, array
    if seen != set(shapes):
        raise ValueError("Export did not cover every original HF parameter")


def write_shard(fs: Any, destination: str, shard: dict, number: int) -> dict:
    """Upload one independent shard and return its index entries."""
    filename = f"model-{number:05d}.safetensors"
    with tempfile.TemporaryDirectory(prefix="exp347-tpu-export-") as folder:
        local = str(Path(folder) / filename)
        save_file(shard, local, metadata={"format": "pt"})
        fs.put_file(local, destination + "/" + filename)
    return {name: filename for name in shard}


def export_checkpoint(
    fs: Any,
    root: str,
    params: dict,
    template: nnx.State,
    source: dict,
    config: Any,
    metadata: dict,
    tokenizer_source: str,
    run_name: str,
) -> dict:
    """Publish BF16 weights, index, tokenizer, and an idempotent eval request."""
    if root != f"{ROOT}/checkpoints/{run_name}":
        raise ValueError("Export is outside the owning regional run")
    if not tokenizer_source.startswith(ROOT + "/checkpoints/"):
        raise ValueError("Tokenizer source is outside the regional experiment")
    destination = f"{root}/hf/step-{metadata['step']}"
    request = {
        "run_id": run_name,
        "step": metadata["step"],
        "tokens": metadata["tokens"],
        "format": metadata["format"],
        "checkpoint": destination,
        "final": metadata["tokens"] >= metadata["target_tokens"],
    }
    if fs.exists(destination + "/export.json"):
        if read_json(fs, destination + "/export.json")["request"] != request:
            raise ValueError("Existing inference export has different provenance")
    else:
        names = {
            value: key
            for key, value in serialized_parameter_names(
                parameter_shapes(source)
            ).items()
        }
        index: dict[str, Any] = {"metadata": {"total_size": 0}, "weight_map": {}}
        shard, size, number = {}, 0, 1
        for name, array in hf_parameters(params, template, source, config):
            tensor = torch.from_numpy(np.array(array, copy=True, order="C")).to(
                torch.bfloat16
            )
            if shard and size + tensor.numel() * 2 > 1_000_000_000:
                index["weight_map"].update(write_shard(fs, destination, shard, number))
                index["metadata"]["total_size"] += size
                shard, size, number = {}, 0, number + 1
            shard[names[name]] = tensor
            size += tensor.numel() * 2
        if shard:
            index["weight_map"].update(write_shard(fs, destination, shard, number))
            index["metadata"]["total_size"] += size
        for path in fs.glob(tokenizer_source + "/*"):
            filename = path.rsplit("/", 1)[-1]
            if (
                filename.endswith(".safetensors")
                or filename == "model.safetensors.index.json"
            ):
                continue
            payload = fs.cat_file(path)
            if filename == "config.json":
                model_config = json.loads(payload)
                model_config["dtype"] = "bfloat16"
                payload = json.dumps(model_config).encode()
            fs.pipe_file(destination + "/" + filename, payload)
        fs.pipe_file(
            destination + "/model.safetensors.index.json", json.dumps(index).encode()
        )
        fs.pipe_file(
            destination + "/export.json",
            json.dumps({"request": request, "dtype": "bfloat16"}).encode(),
        )
    fs.pipe_file(
        f"{ROOT}/eval_requests/{run_name}/step-{metadata['step']}.json",
        json.dumps(request).encode(),
    )
    return request
