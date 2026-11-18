"""Continue a validated regional fork with full-state recovery and eval exports.

The hardware gate must already have passed for this exact frozen parent. Each
update retains the original 32 documents, eight logical cursors, 16K admission
limit, FP32 Adam state, and absolute-token schedule. Completed checkpoints and
inference exports remain in us-east5. Evaluation requests are durable work items;
publishing one does not imply that an evaluator has completed it.
"""

import argparse
import copy
import json
import math
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import fsspec
import jax
from flax import nnx
from jax.sharding import NamedSharding, PartitionSpec
from transformers import AutoTokenizer

import wandb
from common import FORMATS, MAX_LENGTH
from mirrored_corpus import MirroredCorpusStream
from tpu_checkpoint import (
    ROOT,
    abstract_state,
    assert_region,
    committed_checkpoints,
    frozen_source,
    read_json,
    restore_state,
    save_state,
    validate_provenance,
    verify_snapshot,
)
from tpu_export import export_checkpoint
from tpu_model import abstract_model, configuration, initial_rest
from tpu_optimizer import learning_rate
from tpu_training import document_batch, training_functions


class DocumentStream(Protocol):
    """The cursor-bearing part of a logical training-data rank."""

    state: dict

    def next(self) -> dict:
        """Return the next admitted document and advance the cursor."""
        ...


def update_once(
    state: dict,
    metadata: dict,
    streams: Sequence[DocumentStream],
    rest: nnx.State,
    functions: tuple,
    batch_sharding: NamedSharding,
    length: int = MAX_LENGTH,
) -> tuple[dict, dict, dict]:
    """Consume the original logical batch and return one committed optimizer step."""
    began = time.perf_counter()
    rows = [stream.next() for _ in range(4) for stream in streams]
    if len(rows) != 32:
        raise ValueError("The fork requires all eight original data ranks")
    count = sum(len(row[metadata["format"]]) - 1 for row in rows)
    data_seconds = time.perf_counter() - began
    gradient, accumulate, update = functions
    accumulated = None
    loss_sum = 0.0
    for offset in range(0, 32, 4):
        inputs = jax.tree.map(
            lambda value: jax.device_put(value, batch_sharding),
            document_batch(rows[offset : offset + 4], metadata["format"], length),
        )
        if accumulated is None:
            loss, accumulated = gradient(state["params"], rest, inputs, count)
        else:
            loss, accumulated = accumulate(
                state["params"], rest, inputs, count, accumulated
            )
        loss_sum += float(loss)
    lr = learning_rate(metadata["tokens"], metadata["target_tokens"])
    state, norm = update(state, accumulated, lr)
    jax.block_until_ready(state)
    if not math.isfinite(loss_sum) or not math.isfinite(float(norm)):
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
        "train/loss": loss_sum,
        "train/grad_norm": float(norm),
        "train/lr": lr,
        "train/tokens_per_second": count / seconds,
        "train/step_seconds": seconds,
        "train/data_seconds": data_seconds,
        "train/documents_this_step": 32,
        "run_progress": metadata["tokens"] / metadata["target_tokens"],
    }
    return state, metadata, metrics


def publish_checkpoint(
    fs: Any,
    root: str,
    state: dict,
    metadata: dict,
    tokenizer_source: str,
) -> str:
    """Commit the new state before pruning all but the two latest recovery points."""
    uri = f"{root}/step-{metadata['step']}"
    save_state(fs, uri, state, metadata, tokenizer_source)
    # Only direct step directories owned by this production run are eligible.
    # The frozen source, converted parent, HF exports, and validation checkpoints
    # are outside this list. Two committed generations survive a failed save.
    for old in committed_checkpoints(fs, root)[:-2]:
        fs.rm(old, recursive=True)
    return uri


def main() -> None:
    """Restore the latest complete regional state and continue the approved run."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=FORMATS, required=True)
    parser.add_argument("--validation-report", required=True)
    parser.add_argument("--checkpoint-seconds", type=float, default=600)
    parser.add_argument("--eval-every-tokens", type=int, default=1_000_000_000)
    args = parser.parse_args()
    if args.checkpoint_seconds <= 0 or args.eval_every_tokens <= 0:
        raise ValueError("Checkpoint and evaluation intervals must be positive")
    zone = assert_region()
    jax.distributed.initialize()
    if (
        jax.default_backend() != "tpu"
        or jax.process_count() != 1
        or jax.device_count() != 4
    ):
        raise ValueError(
            "Production requires the validated single-host four-chip TPU layout"
        )
    fs = fsspec.filesystem("gcs")
    snapshot = json.loads(Path(f"data/tpu_forks/{args.format}_parent.json").read_text())
    report = read_json(fs, args.validation_report)
    if report["status"] != "passed" or report["source"] != frozen_source(snapshot):
        raise ValueError("This frozen parent has not passed the full-state TPU gate")
    if len(report["observations"]) < 2 or report["zone"] != zone:
        raise ValueError("Hardware validation does not cover this production target")
    source = verify_snapshot(fs, snapshot)
    source_config = read_json(fs, source + "/model/config.json")
    config = configuration(
        source_config, length=MAX_LENGTH, batch=4, attention="flash", tiles=128
    )
    mesh, model = abstract_model(config)
    graph, template, rest = nnx.split(model, nnx.Param, ...)
    rest = initial_rest(rest)
    name = snapshot["fork_run"]
    root = f"{ROOT}/checkpoints/{name}"
    checkpoints = committed_checkpoints(fs, root)
    converted = f"{root}/converted/step-{snapshot['parent']['step']}"
    if report["initial_checkpoint"] != converted:
        raise ValueError("Validation report refers to another converted parent")
    resume = checkpoints[-1] if checkpoints else converted
    state, metadata = restore_state(fs, resume, abstract_state(template, mesh))
    validate_provenance(metadata, snapshot)
    print(
        json.dumps(
            {"resume": resume, "step": metadata["step"], "tokens": metadata["tokens"]}
        ),
        flush=True,
    )
    functions = training_functions(graph, template, rest, config)
    batch_sharding = NamedSharding(mesh, PartitionSpec("fsdp", None))
    data = ROOT + "/data/corpus235m-v1"
    manifest, mirror = [
        read_json(fs, data + f"/{n}.json") for n in ("manifest", "mirror")
    ]
    if not mirror["root"].startswith(ROOT + "/data/"):
        raise ValueError("Corpus mirror points outside the regional experiment")
    run = wandb.init(
        project="MarinFold",
        entity="open-athena",
        id=name,
        name=name,
        resume="allow",
        group="exp347-tpu-continuations",
        tags=["exp347", "tpu", "us-east5", args.format],
        config={
            "region": "us-east5",
            "zone": zone,
            "format": args.format,
            "parent": snapshot["parent"],
            "target_tokens": metadata["target_tokens"],
            "global_batch_documents": 32,
            "microbatch_documents": 4,
            "max_length": MAX_LENGTH,
            "checkpoint_root": root,
            "validation_report": args.validation_report,
            "eval_every_tokens": args.eval_every_tokens,
        },
    )
    print(json.dumps({"wandb_url": run.url, "wandb_name": name}), flush=True)
    with tempfile.TemporaryDirectory(prefix="exp347-tpu-tokenizer-") as folder:
        for path in fs.glob(converted + "/model/*"):
            fs.get_file(path, str(Path(folder) / path.rsplit("/", 1)[-1]))
        tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True)
        streams = [
            MirroredCorpusStream(data, manifest, tokenizer, rank, 8, mirror)
            for rank in range(8)
        ]
        for stream, cursor in zip(streams, metadata["cursors"], strict=True):
            stream.state = copy.deepcopy(cursor)
        # Re-publish the request after a crash between checkpoint commit and HF
        # export. It uses the recovered model, never a newer step's weights.
        if metadata.get("evaluation_export"):
            export_checkpoint(
                fs,
                root,
                state["params"],
                template,
                source_config,
                config,
                metadata,
                source + "/model",
                name,
            )
        next_evaluation = (
            metadata["tokens"] // args.eval_every_tokens + 1
        ) * args.eval_every_tokens
        last_checkpoint = time.monotonic()
        first_update = True
        while metadata["tokens"] < metadata["target_tokens"]:
            state, metadata, metrics = update_once(
                state, metadata, streams, rest, functions, batch_sharding
            )
            run.log(metrics)
            print(json.dumps(metrics), flush=True)
            final = metadata["tokens"] >= metadata["target_tokens"]
            evaluation_due = metadata["tokens"] >= next_evaluation or final
            if (
                first_update
                or evaluation_due
                or time.monotonic() - last_checkpoint >= args.checkpoint_seconds
            ):
                metadata = {**metadata, "evaluation_export": evaluation_due}
                uri = publish_checkpoint(fs, root, state, metadata, source + "/model")
                run.summary["checkpoint"] = uri
                last_checkpoint = time.monotonic()
                if evaluation_due:
                    request = export_checkpoint(
                        fs,
                        root,
                        state["params"],
                        template,
                        source_config,
                        config,
                        metadata,
                        source + "/model",
                        name,
                    )
                    run.summary["evaluation/request"] = request
                    next_evaluation = (
                        metadata["tokens"] // args.eval_every_tokens + 1
                    ) * args.eval_every_tokens
                print(
                    json.dumps(
                        {"checkpoint": uri, "evaluation_export": evaluation_due}
                    ),
                    flush=True,
                )
            first_update = False
        for stream in streams:
            stream.close()
    run.finish()


if __name__ == "__main__":
    main()
