"""Bounded full-state TPU validation; never launches a production continuation.

Convert the immutable parent, preserve all eight logical data ranks, take two
global 32-document updates, and require exact Orbax recovery of every parameter,
Adam moment, counter, and cursor. Validation output has a separate run identity.
Only us-east5 compute may read the full input. The initial converted state remains
available for the later production fork, without consuming its data cursors.
"""

import argparse
import copy
import json
import math
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import fsspec
import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import NamedSharding, PartitionSpec
from transformers import AutoTokenizer

import wandb
from common import FORMATS, MAX_LENGTH
from mirrored_corpus import MirroredCorpusStream
from tpu_checkpoint import (
    ROOT,
    assert_region,
    frozen_source,
    load_parent,
    read_json,
    restore_state,
    save_state,
)
from tpu_model import abstract_model, configuration, initial_rest
from tpu_optimizer import learning_rate
from tpu_training import document_batch, training_functions


def verify_roundtrip(before: dict, after: dict) -> None:
    """Compare all numerical arrays on device, including optimizer moments."""
    same = jax.jit(
        lambda a, b: jnp.all(
            jnp.stack(jax.tree.leaves(jax.tree.map(jnp.array_equal, a, b)))
        )
    )(before, after)
    if not bool(same):
        raise ValueError("TPU checkpoint round-trip changed numerical state")


def main() -> None:
    """Run the bounded hardware gate for one explicit frozen document format."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=FORMATS, required=True)
    parser.add_argument("--attempt", type=int, required=True)
    args = parser.parse_args()
    zone = assert_region()
    jax.distributed.initialize()
    if jax.default_backend() != "tpu" or jax.process_count() != 1:
        raise ValueError("First validation requires a single-host TPU slice")
    print(f"TPU devices: {jax.devices()}; zone={zone}", flush=True)
    fs = fsspec.filesystem("gcs")
    snapshot = json.loads(Path(f"data/tpu_forks/{args.format}_parent.json").read_text())
    source = frozen_source(snapshot)
    source_config = read_json(fs, source + "/model/config.json")
    # One full-length document per chip reduces activation memory while retaining
    # the accumulated gradient. The global update still uses 32 documents.
    microbatch_documents = 4
    config = configuration(
        source_config,
        length=MAX_LENGTH,
        batch=microbatch_documents,
        attention="flash",
        tiles=128,
    )
    mesh, model = abstract_model(config)
    graph, template, rest = nnx.split(model, nnx.Param, ...)
    rest = initial_rest(rest)
    began = time.perf_counter()
    print("Loading frozen FP32 parameters and eight optimizer shards", flush=True)
    state, metadata = load_parent(fs, snapshot, source_config, config, template, mesh)
    load_seconds = time.perf_counter() - began
    print(f"Converted full state in {load_seconds:.1f}s", flush=True)
    converted = (
        f"{ROOT}/checkpoints/{snapshot['fork_run']}/converted/step-{metadata['step']}"
    )
    if fs.exists(converted + "/complete.json"):
        restored, restored_metadata = restore_state(fs, converted, state)
        verify_roundtrip(state, restored)
        if metadata != restored_metadata:
            raise ValueError("Existing converted checkpoint has different provenance")
        del restored
    else:
        save_state(fs, converted, state, metadata, source + "/model")

    name = f"debug-{snapshot['fork_run']}-a{args.attempt:02d}"
    run = wandb.init(
        project="MarinFold",
        entity="open-athena",
        id=name,
        name=name,
        resume="never",
        group="exp347-tpu-validation",
        tags=["exp347", "tpu", "us-east5", "validation"],
        config={
            "region": "us-east5",
            "zone": zone,
            "format": args.format,
            "parent": snapshot["parent"],
            "converted_checkpoint": converted,
            "global_batch_documents": 32,
            "microbatch_documents": microbatch_documents,
            "accumulation_steps": 32 // microbatch_documents,
            "target_tokens": metadata["target_tokens"],
            "devices": [str(device) for device in jax.devices()],
        },
    )
    print(json.dumps({"wandb_url": run.url, "wandb_name": name}), flush=True)
    gradient_fn, accumulate_fn, update_fn = training_functions(
        graph, template, rest, config
    )
    data = ROOT + "/data/corpus235m-v1"
    manifest, mirror = [
        read_json(fs, data + f"/{n}.json") for n in ("manifest", "mirror")
    ]
    if not mirror["root"].startswith(ROOT + "/data/"):
        raise ValueError("Corpus mirror points outside the regional experiment")
    batch_sharding = NamedSharding(mesh, PartitionSpec("fsdp", None))
    observations = []
    with tempfile.TemporaryDirectory(prefix="exp347-tokenizer-") as folder:
        for path in fs.glob(source + "/model/*"):
            filename = path.rsplit("/", 1)[-1]
            if (
                filename.endswith(".safetensors")
                or filename == "model.safetensors.index.json"
            ):
                continue
            fs.get_file(path, str(Path(folder) / filename))
        tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True)
        streams = [
            MirroredCorpusStream(data, manifest, tokenizer, rank, 8, mirror)
            for rank in range(8)
        ]
        for stream, cursor in zip(streams, metadata["cursors"], strict=True):
            stream.state = cursor
        for _ in range(2):
            began = time.perf_counter()
            rows = [stream.next() for _ in range(4) for stream in streams]
            count = sum(len(row[args.format]) - 1 for row in rows)
            batches = [
                rows[i : i + microbatch_documents]
                for i in range(0, len(rows), microbatch_documents)
            ]
            data_seconds = time.perf_counter() - began
            print(
                f"Step {metadata['step'] + 1}: {count} tokens; compiling/training",
                flush=True,
            )
            total_loss = 0.0
            accumulated = None
            for microbatch, batch in enumerate(batches):
                inputs = {
                    key: jax.device_put(value, batch_sharding)
                    for key, value in document_batch(
                        batch, args.format, MAX_LENGTH
                    ).items()
                }
                loss, gradient = gradient_fn(state["params"], rest, inputs, count)
                total_loss += float(loss)
                accumulated = (
                    gradient
                    if accumulated is None
                    else accumulate_fn(accumulated, gradient)
                )
                print(
                    f"Completed microbatch {microbatch + 1}/{len(batches)}", flush=True
                )
            lr = learning_rate(metadata["tokens"], metadata["target_tokens"])
            state, norm = update_fn(state, accumulated, lr)
            jax.block_until_ready(state)
            if not math.isfinite(total_loss) or not math.isfinite(float(norm)):
                raise ValueError("Nonfinite training loss or gradient")
            metadata = {
                **metadata,
                "step": metadata["step"] + 1,
                "tokens": metadata["tokens"] + count,
                "adam_step": int(state["adam_step"]),
                "cursors": [copy.deepcopy(stream.state) for stream in streams],
            }
            seconds = time.perf_counter() - began
            metrics = {
                "train/step": metadata["step"],
                "train/tokens": metadata["tokens"],
                "train/loss": total_loss,
                "train/grad_norm": float(norm),
                "train/lr": lr,
                "train/tokens_per_second": count / seconds,
                "train/step_seconds": seconds,
                "train/data_seconds": data_seconds,
                "run_progress": metadata["tokens"] / metadata["target_tokens"],
            }
            run.log(metrics)
            observations.append(metrics)
            print(json.dumps(metrics), flush=True)
        # Restore all data ranks and demand identical next eligible documents,
        # including admission accounting across rejected rows.
        for stream, cursor in zip(streams, metadata["cursors"], strict=True):
            expected = stream.next()
            expected_cursor = copy.deepcopy(stream.state)
            stream.state = cursor
            if stream.next() != expected or stream.state != expected_cursor:
                raise ValueError("Regional data cursor did not replay exactly")
            stream.close()

    checkpoint = f"{ROOT}/checkpoints/{name}/step-{metadata['step']}"
    save_state(fs, checkpoint, state, metadata, source + "/model")
    restored, recovered = restore_state(fs, checkpoint, state)
    verify_roundtrip(state, restored)
    if metadata != recovered:
        raise ValueError("Checkpoint round-trip changed counters or cursors")
    report = {
        "status": "passed",
        "completed_utc": datetime.now(UTC).isoformat(),
        "zone": zone,
        "wandb_url": run.url,
        "source": source,
        "initial_checkpoint": converted,
        "validation_checkpoint": checkpoint,
        "load_seconds": load_seconds,
        "observations": observations,
        "memory": [device.memory_stats() for device in jax.local_devices()],
    }
    report_uri = f"{ROOT}/data/tpu-validation/{name}.json"
    fs.pipe_file(report_uri, (json.dumps(report, sort_keys=True) + "\n").encode())
    run.summary.update({"validation/status": "passed", "validation/report": report_uri})
    run.finish()
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
