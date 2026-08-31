# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Local single-GPU pilot for the issue #262 ablation.

Trains a ~15M-parameter scaled-down twin of the exp232 Qwen3 on real
decontaminated contacts-v1 documents, one run per (arm, learning rate). The
point is to buy a signal on whether the smear and NoPE help *this document
format* before committing multi-node CoreWeave time to the 1.5B version.

What it is not: a substitute for Phase 1. It is 100x smaller than production and
runs at a 4096-token sequence rather than 8192, so it cannot settle the
scaling question — #169 already showed that matched loss does not imply matched
accuracy across sizes. It can, cheaply, tell us whether an arm is broken, and
which arms are worth production tokens.

Beyond validation loss it logs the two diagnostics the architecture question
turns on:

``structure_nll`` / ``sequence_nll``
    The loss split by document section. The smear's job is intra-statement, so
    it should show up in the structure section if anywhere.
``end_nll`` / ``p_end_early``
    How well the model knows when a document is finished: the loss on the true
    ``<end>`` token, and the probability mass it puts on ``<end>`` at
    structure-section positions where the document is *not* over. Phase 0 found
    the model uses position as a counter, and this is the thing NoPE threatens.
"""

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from arms import ARMS_BY_KEY, build_config, build_model
from marinfold.inference._tokenizer import load_tokenizer, model_source_path
from marinfold.registry import resolve_model


def load_tokens(path: Path) -> np.ndarray:
    return np.memmap(path / "tokens.u16", dtype=np.uint16, mode="r")


def batches(stream: np.ndarray, *, seq_len: int, batch_size: int, seed: int, device: str):
    """Endless stream of random fixed-length windows.

    Random windows rather than a shuffled partition: at pilot scale we make a
    single pass over a fraction of the corpus anyway, and independent windows
    keep the arms from sharing an accidental ordering effect.
    """
    generator = np.random.default_rng(seed)
    limit = stream.size - seq_len - 1
    while True:
        starts = generator.integers(0, limit, size=batch_size)
        block = np.stack([np.asarray(stream[start : start + seq_len + 1], dtype=np.int64) for start in starts])
        chunk = torch.from_numpy(block).to(device, non_blocking=True)
        yield chunk[:, :-1], chunk[:, 1:]


def evaluation_windows(stream: np.ndarray, *, seq_len: int, batch_size: int, batches_wanted: int, device: str):
    """Deterministic, evenly spaced validation windows — identical across arms."""
    limit = stream.size - seq_len - 1
    starts = np.linspace(0, limit, num=batch_size * batches_wanted, dtype=np.int64)
    for index in range(batches_wanted):
        selected = starts[index * batch_size : (index + 1) * batch_size]
        block = np.stack([np.asarray(stream[start : start + seq_len + 1], dtype=np.int64) for start in selected])
        chunk = torch.from_numpy(block).to(device)
        yield chunk[:, :-1], chunk[:, 1:]


@torch.no_grad()
def evaluate(model, stream, *, seq_len, batch_size, batches_wanted, device, token_ids) -> dict:
    """Validation loss, split by section, plus the document-termination diagnostics."""
    model.eval()
    totals = {key: 0.0 for key in ("loss", "sequence", "structure", "end")}
    counts = {key: 0 for key in totals}
    early_mass, early_count = 0.0, 0
    for inputs, targets in evaluation_windows(
        stream, seq_len=seq_len, batch_size=batch_size, batches_wanted=batches_wanted, device=device
    ):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(inputs).logits
        logits = logits.float()
        losses = functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none"
        ).reshape(targets.shape)
        totals["loss"] += losses.sum().item()
        counts["loss"] += losses.numel()

        # Which section is each target in? A position is in the structure
        # section once <begin_statements> has appeared more recently than the
        # <contacts-v1> that opened the current document.
        began = (inputs == token_ids["begin_statements"]).cumsum(dim=1)
        opened = (inputs == token_ids["doc_type"]).cumsum(dim=1)
        in_structure = began >= opened
        is_end = targets == token_ids["end"]

        totals["structure"] += losses[in_structure].sum().item()
        counts["structure"] += int(in_structure.sum())
        totals["sequence"] += losses[~in_structure].sum().item()
        counts["sequence"] += int((~in_structure).sum())
        if is_end.any():
            totals["end"] += losses[is_end].sum().item()
            counts["end"] += int(is_end.sum())

        # Over-eagerness to stop: P(<end>) where the document is not over.
        not_over = in_structure & ~is_end
        if not_over.any():
            probabilities = torch.softmax(logits, dim=-1)[..., token_ids["end"]]
            early_mass += probabilities[not_over].sum().item()
            early_count += int(not_over.sum())

    model.train()
    metrics = {
        f"{name}_nll": (totals[key] / counts[key] if counts[key] else float("nan"))
        for name, key in (("val", "loss"), ("sequence", "sequence"), ("structure", "structure"), ("end", "end"))
    }
    metrics["p_end_early"] = early_mass / early_count if early_count else float("nan")
    return metrics


def learning_rate_at(step: int, *, total: int, peak: float, warmup: float, min_ratio: float) -> float:
    """exp232's WSD shape: linear warmup, hold, linear decay to ``min_ratio``."""
    warmup_steps = max(1, int(total * warmup))
    if step < warmup_steps:
        return peak * (step + 1) / warmup_steps
    decay_start = int(total * 0.8)
    if step < decay_start:
        return peak
    progress = (step - decay_start) / max(1, total - decay_start)
    return peak * (1.0 - (1.0 - min_ratio) * progress)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=sorted(ARMS_BY_KEY))
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--packed", type=Path, default=Path("/data/tim/exp262_pilot/packed"))
    parser.add_argument("--out", type=Path, default=Path("data/pilot"))
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=150_000_000)
    parser.add_argument("--hidden", type=int, default=384)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--intermediate", type=int, default=1536)
    parser.add_argument("--weight-decay", type=float, default=0.2)
    parser.add_argument("--warmup", type=float, default=0.1)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-batches", type=int, default=16)
    arguments = parser.parse_args()

    device = "cuda"
    arm = ARMS_BY_KEY[arguments.arm]
    manifest = json.loads((arguments.packed / "train" / "manifest.json").read_text())
    vocab_size = manifest["vocab_size"]

    tokenizer = load_tokenizer(Path(model_source_path(Path(resolve_model(None)))))
    token_ids = {
        "doc_type": tokenizer.convert_tokens_to_ids("<contacts-v1>"),
        "begin_statements": tokenizer.convert_tokens_to_ids("<begin_statements>"),
        "end": tokenizer.convert_tokens_to_ids("<end>"),
    }
    if any(value is None or value < 0 for value in token_ids.values()):
        raise ValueError(f"tokenizer is missing a contacts-v1 marker: {token_ids}")

    torch.manual_seed(arguments.seed)
    config = build_config(
        vocab_size=vocab_size,
        hidden=arguments.hidden,
        layers=arguments.layers,
        heads=arguments.heads,
        kv_heads=arguments.kv_heads,
        intermediate=arguments.intermediate,
        max_seq_len=arguments.seq_len,
    )
    model = build_model(config, arm).to(device)
    parameters = sum(p.numel() for p in model.parameters())

    tokens_per_step = arguments.seq_len * arguments.batch_size
    total_steps = arguments.tokens // tokens_per_step
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=arguments.learning_rate, weight_decay=arguments.weight_decay, betas=(0.9, 0.95)
    )

    run = f"{arm.key}-lr{arguments.learning_rate:g}-s{arguments.seed}"
    print(
        f"[pilot] {run}: {parameters / 1e6:.2f}M params, {total_steps} steps x "
        f"{tokens_per_step} tokens = {total_steps * tokens_per_step / 1e6:.0f}M",
        flush=True,
    )

    train_stream = load_tokens(arguments.packed / "train")
    val_stream = load_tokens(arguments.packed / "val")
    stream = batches(
        train_stream, seq_len=arguments.seq_len, batch_size=arguments.batch_size,
        seed=arguments.seed, device=device,
    )

    history = []
    started = time.monotonic()
    for step in range(total_steps):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate_at(
                step, total=total_steps, peak=arguments.learning_rate,
                warmup=arguments.warmup, min_ratio=arguments.min_lr_ratio,
            )
        inputs, targets = next(stream)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(inputs).logits
        loss = functional.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1)
        )
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if not math.isfinite(loss.item()):
            raise RuntimeError(f"{run}: loss became {loss.item()} at step {step}")

        if step % arguments.eval_every == 0 or step == total_steps - 1:
            metrics = evaluate(
                model, val_stream, seq_len=arguments.seq_len, batch_size=arguments.batch_size,
                batches_wanted=arguments.eval_batches, device=device, token_ids=token_ids,
            )
            row = {
                "run": run, "arm": arm.key, "learning_rate": arguments.learning_rate,
                "seed": arguments.seed, "step": step,
                "tokens": (step + 1) * tokens_per_step,
                "train_loss": loss.item(), "grad_norm": float(grad_norm),
                "elapsed_s": time.monotonic() - started, **metrics,
            }
            history.append(row)
            print(
                f"[pilot] {run} step {step:5d} train {row['train_loss']:.4f} "
                f"val {row['val_nll']:.4f} (seq {row['sequence_nll']:.4f} struct {row['structure_nll']:.4f}) "
                f"end {row['end_nll']:.4f} p_end_early {row['p_end_early']:.2e} "
                f"[{row['elapsed_s'] / 60:.1f} min]",
                flush=True,
            )

    arguments.out.mkdir(parents=True, exist_ok=True)
    result = {
        "run": run, "arm": asdict(arm), "parameters": parameters,
        "config": {
            "hidden": arguments.hidden, "layers": arguments.layers, "heads": arguments.heads,
            "kv_heads": arguments.kv_heads, "intermediate": arguments.intermediate,
            "seq_len": arguments.seq_len, "batch_size": arguments.batch_size,
            "tokens": total_steps * tokens_per_step, "steps": total_steps,
            "learning_rate": arguments.learning_rate, "weight_decay": arguments.weight_decay,
            "seed": arguments.seed,
        },
        "history": history,
        "final": history[-1],
    }
    (arguments.out / f"{run}.json").write_text(json.dumps(result, indent=2))
    print(f"[pilot] wrote {arguments.out / f'{run}.json'}")


if __name__ == "__main__":
    main()
