"""Regional full-state conversion and atomic TPU checkpoint publication."""

import gc
import hashlib
import io
import json
import urllib.request
from typing import Any

import jax
import numpy as np
import orbax.checkpoint as ocp
import torch
from flax import nnx
from jax.experimental import multihost_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from safetensors.numpy import load as load_safetensors

from tpu_model import convert_parameters
from tpu_state import (
    decode_rank,
    parameter_shapes,
    serialized_parameter_names,
    zero_partitions,
)

ROOT = "gs://marin-us-east5/protein-structure/MarinFold/exp347_qwen_base_contacts"
LEDGER = ROOT + "/transfers/20261102-v1"


def assert_region() -> str:
    """Refuse all model/data I/O unless compute is physically inside us-east5."""
    request = urllib.request.Request(
        "http://metadata.google.internal/computeMetadata/v1/instance/zone",
        headers={"Metadata-Flavor": "Google"},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        zone = response.read().decode().split("/")[-1]
    if zone not in {"us-east5-a", "us-east5-b", "us-east5-c"}:
        raise ValueError(f"TPU fork must remain in us-east5, found {zone}")
    return zone


def read_json(fs: Any, uri: str) -> dict:
    """Read a JSON object from the regional experiment prefix."""
    if not uri.startswith(ROOT + "/"):
        raise ValueError(f"Nonregional experiment path: {uri}")
    return json.loads(fs.cat_file(uri))


def frozen_source(snapshot: dict) -> str:
    """Resolve the approved immutable parent to its verified regional copy."""
    return f"{ROOT}/checkpoints/{snapshot['fork_run']}/source/step-{snapshot['parent']['step']}"


def verify_snapshot(fs: Any, snapshot: dict) -> str:
    """Require every parent object to have the same generation as its copy receipt."""
    source = frozen_source(snapshot)
    if read_json(fs, source + "/snapshot.json") != snapshot:
        raise ValueError("Regional snapshot metadata differs from the frozen parent")
    for name, item in snapshot["objects"].items():
        identity = hashlib.sha256(item["uri"].encode()).hexdigest()
        receipt = read_json(fs, f"{LEDGER}/objects/{identity}.json")
        if receipt["uri"] != source + "/" + name:
            raise ValueError("Snapshot copy receipt points to another destination")
        current = fs.info(receipt["uri"])
        if any(current.get(k) != v for k, v in receipt["metadata"].items()):
            raise ValueError(f"Frozen parent object changed: {name}")
    return source


def replicated_adam_step(step: int, mesh: Mesh) -> jax.Array:
    """Place the optimizer counter on every device owning the model state."""
    return jax.device_put(
        np.asarray(step, dtype=np.int32), NamedSharding(mesh, PartitionSpec())
    )


def load_parent(
    fs: Any,
    snapshot: dict,
    source_config: dict,
    config: Any,
    template: nnx.State,
    mesh: Mesh,
) -> tuple[dict, dict]:
    """Decode all FP32 parameters, optimizer moments, and eight logical cursors."""
    source = verify_snapshot(fs, snapshot)
    actual_config = read_json(fs, source + "/model/config.json")
    if actual_config != source_config:
        raise ValueError("Source model configuration changed")
    shapes = parameter_shapes(source_config)
    saved_names = serialized_parameter_names(shapes)
    index = read_json(fs, source + "/model/model.safetensors.index.json")
    if set(index["weight_map"]) != set(saved_names):
        raise ValueError("Source safetensors do not cover all unique HF parameters")
    weights = {}
    for filename in sorted(set(index["weight_map"].values())):
        with fs.open(source + "/model/" + filename, "rb") as handle:
            shard = load_safetensors(handle.read())
        if set(shard) != {
            name for name, file in index["weight_map"].items() if file == filename
        }:
            raise ValueError("Weight shard differs from its safetensors index")
        weights.update({saved_names[name]: value for name, value in shard.items()})
    for name, value in weights.items():
        if (
            tuple(value.shape) != shapes[name]
            or value.dtype != np.float32
            or not np.isfinite(value).all()
        ):
            raise ValueError(f"Invalid FP32 source weight: {name}")
    params = convert_parameters(weights, source_config, config, template)
    jax.block_until_ready(params)
    del weights, shard
    gc.collect()
    mu, nu, provenance, counts = {}, {}, [], set()
    partitions = zero_partitions(shapes)
    for rank, names in enumerate(partitions):
        with fs.open(source + f"/resume/rank-{rank}.pt", "rb") as handle:
            state = torch.load(handle, map_location="cpu", weights_only=False)
        first, second, count, origin = decode_rank(
            state, names, shapes, snapshot["parent"], rank
        )
        mu.update(first)
        nu.update(second)
        counts.add(count)
        provenance.append(origin)
        print(
            f"Loaded optimizer rank {rank}: {len(names)} parameters, Adam step {count}",
            flush=True,
        )
    if len(counts) != 1 or set(mu) != set(shapes) or set(nu) != set(shapes):
        raise ValueError("Incomplete or inconsistent full optimizer state")
    adam_step = counts.pop()
    first_moment = convert_parameters(mu, source_config, config, template)
    jax.block_until_ready(first_moment)
    del mu
    second_moment = convert_parameters(nu, source_config, config, template)
    jax.block_until_ready(second_moment)
    del nu, state, first, second
    gc.collect()
    converted = {
        "params": params.to_pure_dict(),
        "mu": first_moment.to_pure_dict(),
        "nu": second_moment.to_pure_dict(),
        "adam_step": replicated_adam_step(adam_step, mesh),
    }
    # Preserve source RNG streams as provenance; the no-dropout TPU model does
    # not claim to interchange CUDA and JAX hardware RNG bit for bit.
    if jax.process_index() == 0:
        buffer = io.BytesIO()
        torch.save(provenance, buffer)
        fs.pipe_file(source + "/gpu_rng_provenance.pt", buffer.getvalue())
    metadata = {
        "step": snapshot["parent"]["step"],
        "tokens": snapshot["parent"]["tokens"],
        "adam_step": adam_step,
        "cursors": [p["stream"] for p in provenance],
        "source_snapshot": source,
        "source_run": snapshot["parent_run"],
        "format": snapshot["format"],
        "world": 8,
        "rng_provenance": source + "/gpu_rng_provenance.pt",
        "target_tokens": 100_000_000_000,
    }
    return converted, metadata


def as_nnx(template: nnx.State, values: dict) -> nnx.State:
    """Attach existing variable metadata to restored numerical arrays."""
    result = nnx.clone(template)
    result.replace_by_pure_dict(values)
    return result


def save_state(
    fs: Any, uri: str, state: dict, metadata: dict, tokenizer_source: str
) -> None:
    """Write all arrays and tokenizer files before publishing a completion marker."""
    if not uri.startswith(ROOT + "/checkpoints/"):
        raise ValueError("TPU checkpoint is outside the regional experiment")
    if fs.exists(uri + "/complete.json"):
        raise ValueError("Refusing to overwrite a committed TPU checkpoint")
    # A single job owns this prefix. A failed save may leave an incomplete state;
    # retrying it is safe only because no committed marker exists above.
    save_arrays(uri + "/state", state)
    if jax.process_index() == 0:
        for path in fs.glob(tokenizer_source + "/*"):
            name = path.rsplit("/", 1)[-1]
            if name.endswith(".safetensors") or name == "model.safetensors.index.json":
                continue
            fs.copy(path, uri + "/model/" + name)
        fs.pipe_file(
            uri + "/complete.json",
            (json.dumps(metadata, sort_keys=True) + "\n").encode(),
        )
    multihost_utils.sync_global_devices("exp347-checkpoint-complete")


def restore_state(fs: Any, uri: str, template: dict) -> tuple[dict, dict]:
    """Restore a committed state into the current TPU mesh without losing moments."""
    metadata = read_json(fs, uri + "/complete.json")
    state = restore_arrays(uri + "/state", template)
    if int(state["adam_step"]) != metadata["adam_step"]:
        raise ValueError("Restored optimizer and checkpoint metadata disagree")
    return state, metadata


def save_arrays(uri: str, state: dict) -> None:
    """Write numerical state to an exclusively owned, uncommitted checkpoint path."""
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(uri, args=ocp.args.PyTreeSave(state), force=True)


def restore_arrays(uri: str, template: dict) -> dict:
    """Restore arrays with explicit shapes and shardings, also used by CPU probes."""
    arguments = jax.tree.map(
        lambda a: ocp.ArrayRestoreArgs(
            sharding=a.sharding, global_shape=a.shape, dtype=a.dtype
        ),
        template,
    )
    with ocp.PyTreeCheckpointer() as checkpointer:
        return checkpointer.restore(
            uri,
            args=ocp.args.PyTreeRestore(item=template, restore_args=arguments),
        )
