"""Full-weight Qwen training: DDP, sharded FP32 Adam states, fused vocabulary loss.

Each microbatch is one complete document, so neither attention nor DeltaNet state
crosses protein boundaries. The loss is normalized over all target tokens in the
global accumulated batch. Resume checkpoints retain every rank's optimizer, RNG,
and next-document cursor; HF exports always include the pretrained tokenizer.
"""

import argparse
import contextlib
import csv
import json
import math
import os
import platform
import random
import shutil
import socket
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import fsspec
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
import wandb
from common import FORMATS, MAX_LENGTH, MODELS, ROOT, SEED, TOKEN_BUDGET, run_name
from liger_kernel.transformers.functional import liger_fused_linear_cross_entropy
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoTokenizer, PreTrainedTokenizerBase, Qwen3_5ForCausalLM
from transformers.models.qwen3_5.modeling_qwen3_5 import is_fast_path_available


def copy_tree_to_remote(local: Path, uri: str) -> None:
    """Upload a complete directory before its separate completion marker."""
    fs, remote = fsspec.core.url_to_fs(uri)
    for path in sorted(local.rglob("*")):
        if path.is_file():
            fs.put_file(str(path), remote + "/" + str(path.relative_to(local)))


def copy_tree_from_remote(uri: str, local: Path) -> None:
    """Restore only files belonging to one committed checkpoint."""
    fs, remote = fsspec.core.url_to_fs(uri)
    for path in fs.find(remote):
        target = local / path.removeprefix(remote + "/")
        target.parent.mkdir(parents=True, exist_ok=True)
        fs.get_file(path, str(target))


class DocumentStream:
    """Deterministic rank-sharded stream with a serializable next-row cursor."""

    def __init__(
        self, prefix: str, document_format: str, rank: int, world: int
    ) -> None:
        self.fs, path = fsspec.core.url_to_fs(prefix)
        self.files = sorted(self.fs.glob(path + "/*.parquet"))
        if len(self.files) < world:
            raise ValueError("Need at least one data shard per rank")
        self.document_format = document_format
        self.rank, self.world = rank, world
        self.state = {"epoch": 0, "shard": 0, "row": 0}
        self.loaded: tuple[int, int] | None = None
        self.rows: list[dict] = []

    def next(self) -> dict:
        """Read one row and advance the checkpoint cursor."""
        while True:
            files = self.files.copy()
            random.Random(SEED + self.state["epoch"]).shuffle(files)
            files = files[self.rank :: self.world]
            if self.state["shard"] >= len(files):
                self.state = {"epoch": self.state["epoch"] + 1, "shard": 0, "row": 0}
                continue
            key = (self.state["epoch"], self.state["shard"])
            if self.loaded != key:
                with self.fs.open(files[self.state["shard"]], "rb") as handle:
                    self.rows = pq.read_table(handle).to_pylist()
                random.Random(
                    SEED
                    + self.state["epoch"] * 100000
                    + self.state["shard"] * self.world
                    + self.rank
                ).shuffle(self.rows)
                self.loaded = key
            if self.state["row"] >= len(self.rows):
                self.state["shard"] += 1
                self.state["row"] = 0
                continue
            result = self.rows[self.state["row"]]
            self.state["row"] += 1
            return result


class ContactLM(torch.nn.Module):
    """Use Qwen's public text backbone with a memory-bounded fused loss."""

    def __init__(self, model: Qwen3_5ForCausalLM) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, input_ids: torch.Tensor, completion_start: int = 1
    ) -> torch.Tensor:
        hidden = self.model.model(
            input_ids=input_ids, use_cache=False
        ).last_hidden_state
        # Selecting the loss suffix preserves conditioning on the complete prefix.
        hidden = hidden[:, completion_start - 1 : -1].reshape(-1, hidden.shape[-1])
        target = input_ids[:, completion_start:].reshape(-1)
        return liger_fused_linear_cross_entropy(
            hidden,
            self.model.lm_head.weight
            if torch.is_grad_enabled()
            else self.model.lm_head.weight.detach(),
            target,
            reduction="sum",
            accum_dtype=torch.float32,
        )


def cpu_tree(value: Any) -> Any:
    """Copy optimizer tensors to CPU without changing live optimizer state."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value


def inference_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Cast weights on CPU while preserving shared embedding/output tensors."""
    converted: dict[tuple, torch.Tensor] = {}
    result = {}
    for name, value in model.state_dict().items():
        key = (
            str(value.device),
            value.dtype,
            value.data_ptr(),
            tuple(value.shape),
            tuple(value.stride()),
        )
        if key not in converted:
            dtype = torch.bfloat16 if value.is_floating_point() else value.dtype
            converted[key] = value.detach().to(device="cpu", dtype=dtype)
        result[name] = converted[key]
    return result


def checkpoint(
    model: Qwen3_5ForCausalLM,
    optimizer: ZeroRedundancyOptimizer,
    tokenizer: PreTrainedTokenizerBase,
    stream: DocumentStream,
    root: str,
    step: int,
    tokens: int,
    rank: int,
    world: int,
    final: bool,
) -> str:
    """Commit all resume shards and an HF export, then move the resume pointer."""
    uri = f"{root}/step-{step}"
    local = Path(f"/tmp/exp347-checkpoint-rank{rank}")
    local.mkdir(exist_ok=True)
    state = {
        "optimizer": cpu_tree(optimizer.optim.state_dict()),
        "stream": stream.state,
        "rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state(),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "step": step,
        "tokens": tokens,
        "world": world,
    }
    torch.save(state, local / f"rank-{rank}.pt")
    fs, path = fsspec.core.url_to_fs(uri)
    fs.put_file(str(local / f"rank-{rank}.pt"), f"{path}/resume/rank-{rank}.pt")
    if rank == 0:
        # Keep FP32 weights for exact continuation; publish the final inference
        # export in BF16 separately without mutating the live training model.
        model.save_pretrained(local / "model", max_shard_size="4GB")
        tokenizer.save_pretrained(local / "model")
        copy_tree_to_remote(local / "model", uri + "/model")
        if final:
            weights = inference_state_dict(model)
            model.save_pretrained(
                local / "hf", state_dict=weights, max_shard_size="4GB"
            )
            tokenizer.save_pretrained(local / "hf")
            config_path = local / "hf" / "config.json"
            export_config = json.loads(config_path.read_text())
            export_config["dtype"] = "bfloat16"
            config_path.write_text(json.dumps(export_config, indent=2) + "\n")
            copy_tree_to_remote(local / "hf", uri + "/hf")
    dist.barrier()
    if rank == 0:
        marker = {
            "step": step,
            "tokens": tokens,
            "world": world,
            "uri": uri,
            "final": final,
        }
        with fsspec.open(uri + "/complete.json", "w") as handle:
            json.dump(marker, handle)
        pointer = root + "/resume.json"
        previous = None
        if fs.exists(pointer):
            with fsspec.open(pointer) as handle:
                previous = json.load(handle)["uri"]
        with fsspec.open(pointer, "w") as handle:
            json.dump(marker, handle)
        # Retain one committed resumable checkpoint per trial. Never delete
        # until its successor and pointer have both been durably written.
        if previous and previous != uri:
            fs.rm(previous, recursive=True)
    dist.barrier()
    shutil.rmtree(local)
    return uri


def evaluate(
    model: ContactLM,
    rows: list[dict],
    document_format: str,
    rank: int,
    step: int,
    model_load_seconds: float,
    nickname: str,
    output: str,
) -> dict:
    """Measure contact continuation likelihood and record each input's timing."""
    model.eval()
    totals = torch.zeros(3, device="cuda", dtype=torch.float64)
    timings = []
    props = torch.cuda.get_device_properties(rank)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for row in rows:
            total_start = time.perf_counter()
            ids = torch.tensor([row[document_format]], device="cuda")
            start = row[document_format + "_start"]
            torch.cuda.synchronize()
            began = time.perf_counter()
            loss = model(ids, start)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - began
            totals += torch.tensor(
                [loss.item(), ids.numel() - start, row["n_contacts"]], device="cuda"
            )
            timings.append(
                {
                    "stem": row["entry_id"],
                    "n_residues": len(row["sequence"]),
                    "n_pairs": row["n_contacts"],
                    "mode": f"{document_format}-teacher-forced-step-{step}",
                    "elapsed_seconds": elapsed,
                    "model_load_seconds": model_load_seconds,
                    "total_seconds": time.perf_counter() - total_start,
                    "model_nickname": nickname,
                    "runner_tag": "iris",
                    "gpu_name": props.name,
                    "gpu_total_memory_gb": props.total_memory / 1e9,
                    "gpu_compute_capability": f"{props.major}.{props.minor}",
                    "hostname": socket.gethostname(),
                    "platform": platform.platform(),
                    "torch_version": str(torch.__version__),
                    "timestamp_utc": datetime.now(UTC).isoformat(),
                    "rank": rank,
                }
            )
    with fsspec.open(f"{output}/timings/step-{step}-rank-{rank}.csv", "w") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(timings[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(timings)
    dist.all_reduce(totals)
    model.train()
    return {
        "validation/contact_token_nll": (totals[0] / totals[1]).item(),
        "validation/nll_per_contact": (totals[0] / totals[2]).item(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=MODELS, required=True)
    parser.add_argument("--format", choices=FORMATS, required=True)
    parser.add_argument("--data", default=ROOT + "/data/v1")
    parser.add_argument("--tokens", type=int, default=TOKEN_BUDGET)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--checkpoint-seconds", type=float, default=120)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume-check", action="store_true")
    parser.add_argument("--eval-documents", type=int, default=64)
    args = parser.parse_args()
    if args.checkpoint_seconds <= 0:
        raise ValueError("Checkpoint interval must be positive")
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    torch.set_num_threads(4)
    random.seed(SEED + rank)
    np.random.seed(SEED + rank)
    torch.manual_seed(SEED + rank)
    if not is_fast_path_available:
        raise RuntimeError(
            "Qwen DeltaNet fast kernels are required; install causal-conv1d and flash-linear-attention"
        )
    name = run_name(args.size, args.format, args.smoke)
    run_id = name
    root = f"{ROOT}/checkpoints/{name}"
    fs, root_path = fsspec.core.url_to_fs(root)
    resume = None
    if fs.exists(root_path + "/resume.json"):
        with fsspec.open(root + "/resume.json") as handle:
            resume = json.load(handle)
        if resume["world"] != world:
            raise ValueError(
                "This optimizer checkpoint requires the original world size"
            )
    data_manifest = None
    if not args.smoke:
        with fsspec.open(args.data + "/manifest.json") as handle:
            data_manifest = json.load(handle)
        if data_manifest["max_length"] != MAX_LENGTH or data_manifest[
            "tokenizer"
        ] != list(MODELS["0.8B"]):
            raise ValueError(
                "Token cache does not match the pinned experiment contract"
            )
    load_start = time.perf_counter()
    local_model = Path("/tmp/exp347-model")
    source = resume["uri"] + "/model" if resume else f"{ROOT}/base/{args.size}"
    if rank == 0:
        if local_model.exists():
            shutil.rmtree(local_model)
        copy_tree_from_remote(source, local_model)
    dist.barrier()
    if not resume:
        staged = json.loads((local_model / "staged.json").read_text())
        if (staged["repo"], staged["revision"]) != MODELS[args.size]:
            raise ValueError("Staged base model does not match its pinned revision")
    model, info = Qwen3_5ForCausalLM.from_pretrained(
        local_model,
        dtype=torch.float32,
        attn_implementation="sdpa",
        output_loading_info=True,
    )
    if info["missing_keys"] or info.get("mismatched_keys"):
        raise ValueError(f"Incomplete pretrained initialization: {info}")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(local_model)
    wrapped = ContactLM(model)
    ddp = DistributedDataParallel(
        wrapped, device_ids=[rank], gradient_as_bucket_view=True
    )
    optimizer = ZeroRedundancyOptimizer(
        ddp.parameters(),
        optimizer_class=torch.optim.AdamW,
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
        foreach=False,
    )
    stream = DocumentStream(args.data + "/train", args.format, rank, world)
    step, tokens = 0, 0
    if resume:
        state_path = Path(f"/tmp/exp347-rank-{rank}.pt")
        fs.get_file(resume["uri"] + f"/resume/rank-{rank}.pt", str(state_path))
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        optimizer.optim.load_state_dict(state["optimizer"])
        stream.state = state["stream"]
        step, tokens = state["step"], state["tokens"]
        torch.set_rng_state(state["rng"])
        torch.cuda.set_rng_state(state["cuda_rng"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        del state
    if args.resume_check and (not args.smoke or resume is None):
        raise ValueError("Resume checking requires an existing smoke checkpoint")
    resumed_step = step
    model_load_seconds = time.perf_counter() - load_start
    val_stream = DocumentStream(args.data + "/validation", args.format, rank, world)
    val_rows = [val_stream.next() for _ in range(max(1, args.eval_documents // world))]
    run = None
    if rank == 0:
        run = wandb.init(
            entity="open-athena",
            project="MarinFold",
            id=run_id,
            name=name,
            group="exp347-qwen-base-contacts",
            resume="allow",
            config={
                **vars(args),
                "base_model": MODELS[args.size],
                "trainable_parameters": sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                ),
                "world": world,
                "cluster": os.environ["EXP347_CLUSTER"],
                "gpu": "H100",
                "nodes": 1,
                "max_length": MAX_LENGTH,
                "checkpoint_root": root,
                "loss": "all next tokens, global token mean",
                "seed": SEED,
                "data_manifest": data_manifest,
            },
        )
        print(f"WANDB_RUN {run.url}", flush=True)
        with fsspec.open(f"{ROOT}/runs/{name}.json", "w") as handle:
            json.dump(
                {
                    "wandb_url": run.url,
                    "wandb_id": run.id,
                    "name": name,
                    "iris_job_id": os.environ.get("IRIS_JOB_ID"),
                    "checkpoint_root": root,
                },
                handle,
            )
    # Production resumes prioritize saving new progress. Repeating validation
    # at the restored step costs much of a short preemptible allocation; smoke
    # recovery checks still evaluate it to verify exact restoration.
    result = (
        evaluate(
            wrapped,
            val_rows,
            args.format,
            rank,
            step,
            model_load_seconds,
            name,
            f"{ROOT}/runs/{name}",
        )
        if resume is None or args.smoke
        else {}
    )
    if run:
        run.log(
            {
                **result,
                "train/step": step,
                "train/tokens": tokens,
                "run_progress": tokens / args.tokens,
            }
        )
    last_checkpoint = time.monotonic()
    while tokens < args.tokens or (args.resume_check and step == resumed_step):
        began = time.perf_counter()
        batch = [stream.next() for _ in range(args.accumulation)]
        counts = torch.tensor(
            [
                sum(len(r[args.format]) - 1 for r in batch),
                len(batch),
                sum(r["n_contacts"] for r in batch),
            ],
            device="cuda",
            dtype=torch.int64,
        )
        dist.all_reduce(counts)
        ratio = min(tokens / args.tokens, 1.0)
        lr_factor = (
            max(0.01, ratio / 0.01)
            if ratio < 0.01
            else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (ratio - 0.01) / 0.99))
        )
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate * lr_factor
        # Preserve DDP bucket views across accumulation steps. Clearing them to
        # None creates another full set of gradients during no_sync(), which
        # exhausts an 80 GB H100 for the 4B model after its first update.
        optimizer.zero_grad(set_to_none=False)
        total_loss = torch.zeros((), device="cuda")
        for i, row in enumerate(batch):
            ids = torch.tensor([row[args.format]], device="cuda")
            sync = ddp.no_sync() if i + 1 < len(batch) else contextlib.nullcontext()
            with sync, torch.autocast("cuda", dtype=torch.bfloat16):
                loss = ddp(ids)
                scaled_loss = loss * world / counts[0]
            scaled_loss.backward()
            total_loss += loss.detach()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            ddp.parameters(), 1.0, error_if_nonfinite=True
        )
        optimizer.step()
        tokens += int(counts[0])
        step += 1
        dist.all_reduce(total_loss)
        elapsed = time.perf_counter() - began
        metrics = {
            "train/loss": (total_loss / counts[0]).item(),
            "train/step": step,
            "train/tokens": tokens,
            "train/documents_this_step": int(counts[1]),
            "train/contacts_this_step": int(counts[2]),
            "train/lr": args.learning_rate * lr_factor,
            "train/grad_norm": grad_norm.item(),
            "train/tokens_per_second": int(counts[0]) / elapsed,
            "run_progress": tokens / args.tokens,
            "train/peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9,
        }
        if rank == 0:
            print(json.dumps(metrics), flush=True)
        if run:
            run.log(metrics)
        final = tokens >= args.tokens
        save = torch.tensor(
            [
                int(
                    final
                    or (not args.smoke and step == resumed_step + 1)
                    or time.monotonic() - last_checkpoint >= args.checkpoint_seconds
                )
            ],
            device="cuda",
        )
        dist.broadcast(save, src=0)
        if save.item():
            uri = checkpoint(
                model,
                optimizer,
                tokenizer,
                stream,
                root,
                step,
                tokens,
                rank,
                world,
                final,
            )
            last_checkpoint = time.monotonic()
            if run:
                run.summary["checkpoint"] = uri
        if step % 250 == 0 or final:
            metrics = evaluate(
                wrapped,
                val_rows,
                args.format,
                rank,
                step,
                model_load_seconds,
                name,
                f"{ROOT}/runs/{name}",
            )
            if run:
                run.log(
                    {
                        **metrics,
                        "train/step": step,
                        "train/tokens": tokens,
                        "run_progress": tokens / args.tokens,
                    }
                )
    if run:
        run.finish()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
