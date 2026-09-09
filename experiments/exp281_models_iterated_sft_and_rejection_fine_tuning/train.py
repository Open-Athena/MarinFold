"""Single-node DDP SFT with exact global weighted-token normalization.

Each document is an independent attention sequence. Gradients accumulate sums
over all microbatches, divided by one all-reduced supervised-weight denominator
for the optimizer step. This preserves the objective under uneven final lengths,
padding, gradient accumulation, and DDP's gradient averaging. No runtime patches
to transformers, torch, or Levanter are used.
"""

import argparse
import itertools
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Iterator

import fsspec
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
import wandb
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import (
    EXPERIMENT,
    code_identity,
    identity,
    publish_directory,
    read_json,
    seed_for,
    stage_model,
    write_json,
)


class ParquetStream:
    """Deterministic rank-disjoint row groups with resumable row offsets."""

    def __init__(self, paths: list[str], rank: int, world: int, seed: int, *, allow_empty: bool = False):
        groups = []
        for path in paths:
            with fsspec.open(path, "rb") as handle:
                meta = pq.ParquetFile(handle).metadata
                groups.extend((path, i, meta.row_group(i).num_rows) for i in range(meta.num_row_groups))
        self.groups = groups[rank::world]
        if not allow_empty and (not self.groups or sum(g[2] for g in self.groups) == 0):
            raise ValueError("each rank needs at least one nonempty parquet row group")
        self.seed = seed

    def iterate(self, skip: int = 0, repeat: bool = True) -> Iterator[dict]:
        """Skip complete groups by metadata; re-read at most one group on resume."""
        per_epoch = sum(g[2] for g in self.groups)
        if per_epoch == 0:
            if repeat:
                raise ValueError("cannot repeat an empty stream")
            return
        epoch, skip = divmod(skip, per_epoch)
        while True:
            ordered = list(self.groups)
            random.Random(seed_for(self.seed, epoch)).shuffle(ordered)
            for path, group, size in ordered:
                if skip >= size:
                    skip -= size
                    continue
                with fsspec.open(path, "rb") as handle:
                    records = pq.ParquetFile(handle).read_row_group(
                        group, columns=["input_ids", "loss_weights"]
                    ).to_pylist()
                random.Random(seed_for(self.seed, epoch, path, group)).shuffle(records)
                yield from records[skip:]
                skip = 0
            if not repeat:
                return
            epoch += 1


def collate(records: list[dict], pad: int, context: int) -> dict[str, torch.Tensor]:
    """Right-pad independent documents and mask all padding targets."""
    length = max(len(r["input_ids"]) for r in records)
    if length > context or length < 2:
        raise ValueError("document length outside training context")
    ids = torch.full((len(records), length), pad, dtype=torch.long)
    weights = torch.zeros((len(records), length), dtype=torch.float32)
    attention = torch.zeros_like(ids)
    for i, record in enumerate(records):
        n = len(record["input_ids"])
        if len(record["loss_weights"]) != n:
            raise ValueError("loss weights must align with target token ids")
        ids[i, :n] = torch.tensor(record["input_ids"])
        weights[i, :n] = torch.tensor(record["loss_weights"])
        attention[i, :n] = 1
    if not torch.isfinite(weights).all() or (weights < 0).any() or weights[:, 1:].sum() <= 0:
        raise ValueError("nonfinite, negative, or empty supervised weights")
    return {"input_ids": ids, "attention_mask": attention, "weights": weights}


def weighted_loss_sum(logits: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Sum causal cross entropy; weight at i supervises token i, not token i+1."""
    loss = torch.nn.functional.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                                             ids[:, 1:].reshape(-1), reduction="none")
    return (loss * weights[:, 1:].reshape(-1)).sum()


def learning_rate(step: int, steps: int, peak: float, warmup: int) -> float:
    """Linear warm-up followed by cosine decay to ten percent of peak."""
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    fraction = (step - warmup) / max(1, steps - warmup - 1)
    return peak * (0.1 + 0.9 * (1 + math.cos(math.pi * fraction)) / 2)


def all_sum(value: torch.Tensor, world: int) -> torch.Tensor:
    """Sum a detached scalar over the training gang."""
    result = value.detach().clone()
    if world > 1:
        dist.all_reduce(result)
    return result


def evaluate(model: torch.nn.Module, stream: ParquetStream, pad: int, context: int,
             device: torch.device, world: int, limit: int) -> float:
    """Read held-out documents once, reporting globally weighted teacher-forced loss."""
    model.eval()
    totals = torch.zeros(2, device=device, dtype=torch.float64)
    with torch.no_grad():
        for row in itertools.islice(stream.iterate(repeat=False), limit):
            batch = {k: v.to(device) for k, v in collate([row], pad, context).items()}
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            totals[0] += weighted_loss_sum(output.logits, batch["input_ids"], batch["weights"])
            totals[1] += batch["weights"][:, 1:].sum()
    totals = all_sum(totals, world)
    model.train()
    return (totals[0] / totals[1]).item()


def record_history(run: wandb.sdk.wandb_run.Run, output: str) -> None:
    """Create or restore one run history, append this job, and persist it."""
    root = Path(__file__).resolve().parents[2]
    command = ["uv", "run", "--no-project", sys.executable, str(root / "scripts/history.py")]
    matches = [p for p in (root / "history/runs").glob("*.md") if run.url in p.read_text()]
    if not matches:
        fs, prefix = fsspec.core.url_to_fs(f"{output}/history")
        for remote in fs.glob(prefix + "/*.md"):
            with fs.open(remote, "r") as handle:
                content = handle.read()
            if run.url in content:
                local = root / "history/runs" / remote.rsplit("/", 1)[-1]
                local.write_text(content)
                matches.append(local)
    if not matches:
        args = ["new", "--wandb-url", run.url, "--wandb-name", run.name,
                "--experiment", EXPERIMENT, "--kind", "models", "--short", "exp281 weighted contact synthesis SFT"]
        source = root / "source_revision.json"
        if source.exists():
            args.extend(["--git-sha", read_json(str(source))["git_sha"]])
        subprocess.run(command + args, check=True, cwd=root)
        matches = [p for p in (root / "history/runs").glob("*.md") if run.url in p.read_text()]
    if len(matches) != 1:
        raise ValueError("expected exactly one W&B run history")
    if os.environ.get("IRIS_JOB_ID"):
        subprocess.run(command + ["add-iris-job", str(matches[0]), os.environ["IRIS_JOB_ID"]], check=True, cwd=root)
    subprocess.run(command + ["update-index"], check=True, cwd=root)
    with matches[0].open("rb") as src, fsspec.open(f"{output}/history/{matches[0].name}", "wb", auto_mkdir=True) as dst:
        shutil.copyfileobj(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--output", required=True, help="durable experiment prefix")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--global-batch", type=int, default=128)
    parser.add_argument("--microbatch", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.2)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-documents", type=int, default=128)
    parser.add_argument("--seed", type=int, default=281)
    parser.add_argument("--resume", help="explicit complete checkpoint URI, including step")
    parser.add_argument("--work", type=Path, default=Path("/tmp/exp281-train"))
    parser.add_argument("--no-wandb", action="store_true", help="local smoke tests only")
    args = parser.parse_args()
    rank, world = int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if min(args.steps, args.global_batch, args.microbatch, args.save_every, args.eval_every) < 1:
        raise ValueError("step and batch sizes must be positive")
    if args.global_batch % (world * args.microbatch):
        raise ValueError("global batch must be divisible by world size * microbatch")
    if not 0 <= args.warmup <= args.steps or args.lr <= 0:
        raise ValueError("invalid learning-rate schedule")
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    torch.manual_seed(args.seed)
    train_manifest, val_manifest = read_json(args.train_manifest), read_json(args.validation_manifest)
    if set(train_manifest["shards"]) & set(val_manifest["shards"]):
        raise ValueError("training and validation shard lists overlap")
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k not in ("resume", "work")}
    config.update(world_size=world, train_hash=identity(train_manifest), validation_hash=identity(val_manifest),
                  code_hash=code_identity())
    signature = identity(config)
    # This trainer uses one node: stage once, then release every rank to the
    # shared files. Eight optimizer-bearing copies would exceed worker disk.
    staged = [str(stage_model(args.resume or args.model, args.work / "model",
                              training_state=bool(args.resume))) if rank == 0 else None]
    if world > 1:
        dist.broadcast_object_list(staged, src=0)
    local = Path(staged[0])
    if args.resume and not (local / "_SUCCESS.json").exists():
        raise ValueError("resume requires a completed checkpoint manifest")
    tokenizer = AutoTokenizer.from_pretrained(local)
    if identity(tokenizer.get_vocab()) != train_manifest["tokenizer_hash"] or train_manifest["tokenizer_hash"] != val_manifest["tokenizer_hash"]:
        raise ValueError("training/validation/checkpoint tokenizers differ")
    model = AutoModelForCausalLM.from_pretrained(local, dtype=torch.float32, attn_implementation="sdpa")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start, consumed = 0, 0
    if args.resume:
        state = torch.load(local / "trainer.pt", map_location="cpu", weights_only=False, mmap=True)
        if state["signature"] != signature:
            raise ValueError("resume configuration/data/world size differ from checkpoint")
        optimizer.load_state_dict(state["optimizer"])
        start = state["step"]
        consumed = state["ranks"][rank]["consumed"]
    module = model
    if world > 1:
        model = DistributedDataParallel(module, device_ids=[local_rank] if device.type == "cuda" else None)
    torch.manual_seed(args.seed + rank)
    if args.resume:
        torch.set_rng_state(state["ranks"][rank]["rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(state["ranks"][rank]["cuda_rng"], device)
        del state
    stream = ParquetStream(train_manifest["shards"], rank, world, args.seed)
    validation = ParquetStream(val_manifest["shards"], rank, world, args.seed, allow_empty=True)
    iterator = stream.iterate(skip=consumed)
    accumulation = args.global_batch // (world * args.microbatch)
    pad = tokenizer.pad_token_id
    if pad is None:
        raise ValueError("tokenizer needs an explicit pad token")
    run = None
    if rank == 0 and not args.no_wandb:
        run = wandb.init(entity="open-athena", project="MarinFold", name=args.run_name,
                         id=args.run_name, resume="allow", config=config)
        record_history(run, args.output)
        with tempfile.TemporaryDirectory(prefix="exp281-inputs-") as directory:
            artifact = wandb.Artifact(f"{args.run_name}-inputs", type="dataset-manifest")
            for split, manifest in (("train", train_manifest), ("validation", val_manifest)):
                path = Path(directory) / f"{split}.json"
                write_json(str(path), manifest)
                artifact.add_file(str(path))
                run.summary[f"data/{split}"] = manifest.get("totals", {})
            run.log_artifact(artifact).wait()
    module.train()
    for step in range(start, args.steps):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        step_started = time.monotonic()
        batches = [collate(list(itertools.islice(iterator, args.microbatch)), pad, args.context)
                   for _ in range(accumulation)]
        denominator = all_sum(sum(b["weights"][:, 1:].sum() for b in batches).to(device), world)
        numerator = torch.zeros((), device=device)
        optimizer.zero_grad(set_to_none=True)
        lr = learning_rate(step, args.steps, args.lr, args.warmup)
        for group in optimizer.param_groups:
            group["lr"] = lr
        for index, cpu_batch in enumerate(batches):
            batch = {k: v.to(device) for k, v in cpu_batch.items()}
            sync = model.no_sync() if world > 1 and index + 1 < accumulation else nullcontext()
            with sync:
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
                loss_sum = weighted_loss_sum(logits, batch["input_ids"], batch["weights"])
                # DDP averages gradients; multiply by world to obtain the global sum.
                (loss_sum * world / denominator).backward()
                numerator += loss_sum.detach()
            consumed += args.microbatch
        grad_norm = torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step_seconds = time.monotonic() - step_started
        metrics = {"train/loss": (all_sum(numerator, world) / denominator).item(),
                   "train/supervised_weight": denominator.item(), "train/lr": lr,
                   "train/grad_norm": grad_norm.item(), "train/step": step + 1}
        metrics["train/step_seconds"] = step_seconds
        tokens = all_sum(sum(b["attention_mask"].sum() for b in batches).to(device), world).item()
        metrics["train/tokens_per_second"] = tokens / step_seconds
        metrics["train/documents"] = (step + 1) * args.global_batch
        if device.type == "cuda":
            peak = torch.tensor(torch.cuda.max_memory_allocated(device) / 1e9, device=device)
            if world > 1:
                dist.all_reduce(peak, op=dist.ReduceOp.MAX)
            metrics["train/peak_memory_gb"] = peak.item()
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            metrics["validation/loss"] = evaluate(module, validation, pad, args.context, device, world, args.eval_documents)
        if rank == 0:
            print(metrics, flush=True)
            if run:
                run.log(metrics, step=step + 1)
        if (step + 1) % args.save_every == 0 or step + 1 == args.steps:
            rank_state = {"consumed": consumed, "rng": torch.get_rng_state(),
                          "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}
            states = [None] * world
            if world > 1:
                dist.all_gather_object(states, rank_state)
            else:
                states = [rank_state]
            if rank == 0:
                args.work.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(prefix=f"step-{step + 1}-", dir=args.work) as directory:
                    checkpoint = Path(directory)
                    module.save_pretrained(checkpoint)
                    tokenizer.save_pretrained(checkpoint)
                    torch.save({"optimizer": optimizer.state_dict(), "step": step + 1, "ranks": states,
                                "signature": signature}, checkpoint / "trainer.pt")
                    write_json(str(checkpoint / "training.json"), {"config": config, "metrics": metrics})
                    publish_directory(checkpoint, f"{args.output}/checkpoints/{args.run_name}/step-{step + 1}")
            if world > 1:
                dist.barrier()
    if run:
        run.finish()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
