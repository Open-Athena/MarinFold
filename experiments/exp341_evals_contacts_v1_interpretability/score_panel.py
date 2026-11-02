# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""A3/A4 readouts: score the held-out panel with each model.

Runs the clean seed-averaged pairwise readout (``harness.pcontact``) on every
structure of ``data/panel.parquet`` and records R-precision over all
separations >= 6 and long range (>= 24). A3 reads exp232's scores against the
kept-relative bins; A4 takes the difference-in-differences between exp199
and exp232 on removed versus matched kept structures.

Writes ``<output-dir>/<model>/scores.csv``: one row per structure with the
panel columns (``set``, ``bin``, ``pair``, ``n_relatives``), both
R-precisions and the AGENTS.md predictor-timing schema. Each model's
P(contact) matrices go to ``pcontact.npz`` beside it for the A4 Jacobian
subset. Proteins are scored in panel order; a model whose ``scores.csv``
already exists is skipped, so a rerun resumes per model.

Tail batch size adapts to each protein. ``next_token_probs`` replicates the
prefix KV cache once per tail in a batch, so a batch of b tails over a
P-token prefix holds b * P * (layers * 2 * kv_heads * head_dim * 2 bytes);
for this architecture 49,152 bytes per token. The panel's longest prefix
(3,109 tokens at L = 1,551) needs 36 GiB at b = 256, which ran a 44 GiB L40S
out of memory. Each protein gets the largest power of two up to
``--tail-batch-size`` whose replicated cache fits ``--kv-budget-gib``, through
``TransformersBackend.from_model`` wrappers around the one loaded model. The
batch size changes only how tails are grouped, not the scores.

    uv run python score_panel.py --device cuda --output-dir out/panel \
        --models contacts-v1-exp232-m2-p06-train-1.5B contacts-v1-exp199-cooldown-1.5B
"""

import argparse
import platform
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from marinfold.document_structures.contacts_v1.inference import prefix_and_positions
from marinfold.registry import resolve_model_entry

import harness
from time_readout import forward_tokens, gpu_columns

DATA = Path(__file__).resolve().parent / "data"
MODELS = ("contacts-v1-exp232-m2-p06-train-1.5B", "contacts-v1-exp199-cooldown-1.5B")


def kv_bytes_per_token(model: torch.nn.Module) -> int:
    config = model.config
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    element = torch.finfo(model.dtype).bits // 8
    return config.num_hidden_layers * 2 * config.num_key_value_heads * head_dim * element


def max_prefix_tokens(backend, protein: harness.EvalProtein, seeds: int) -> int:
    lengths = []
    for k in range(seeds):
        entry_id = protein.stem if seeds == 1 else f"{protein.stem}#cv1ens{k}"
        prefix, _, _ = prefix_and_positions(protein.structure, entry_id=entry_id)
        lengths.append(len(backend.tokenizer.encode(prefix, add_special_tokens=False)))
    return max(lengths)


def tail_batch_for(prefix_tokens: int, per_token: int, args) -> int:
    fits = int(args.kv_budget_gib * 2**30 // (prefix_tokens * per_token))
    batch = 1
    while batch * 2 <= min(fits, args.tail_batch_size):
        batch *= 2
    return batch


class BatchSizedBackends:
    """``from_model`` wrappers around one loaded backend, one per tail batch size.

    ``args`` supplies ``seeds``, ``tail_batch_size`` and ``kv_budget_gib``.
    """

    def __init__(self, backend, args):
        self.backend = backend
        self.args = args
        self.per_token = kv_bytes_per_token(backend.model)
        self.wrappers = {args.tail_batch_size: backend}

    def for_protein(self, protein: harness.EvalProtein) -> tuple[object, int]:
        prefix_tokens = max_prefix_tokens(self.backend, protein, self.args.seeds)
        batch = tail_batch_for(prefix_tokens, self.per_token, self.args)
        if batch not in self.wrappers:
            self.wrappers[batch] = type(self.backend).from_model(
                self.backend.model, self.backend.tokenizer, tail_batch_size=batch
            )
        return self.wrappers[batch], batch


def score_model(nickname: str, proteins: list[harness.EvalProtein], panel: pd.DataFrame, args) -> None:
    out_dir = args.output_dir / nickname
    if (out_dir / "scores.csv").exists():
        print(f"{nickname}: scores.csv exists, skipping", flush=True)
        return
    entry = resolve_model_entry(nickname)
    load_start = time.perf_counter()
    backend = harness.load_model(nickname, device=args.device, tail_batch_size=args.tail_batch_size)
    model_load_seconds = time.perf_counter() - load_start
    static = {
        "mode": f"pairwise-{args.seeds}seed",
        "model_load_seconds": round(model_load_seconds, 2),
        "model_nickname": entry.nickname,
        "runner_tag": args.runner_tag,
        **gpu_columns(args.device),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "n_seeds": args.seeds,
    }
    sized = BatchSizedBackends(backend, args)
    harness.pcontact(sized.for_protein(proteins[0])[0], proteins[0], seeds=1)  # untimed warm-up
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()

    rows, matrices = [], {}
    for done, (protein, meta) in enumerate(zip(proteins, panel.itertuples(index=False)), start=1):
        scorer, batch = sized.for_protein(protein)
        start = time.perf_counter()
        score = harness.pcontact(scorer, protein, seeds=args.seeds)
        elapsed = time.perf_counter() - start
        matrices[protein.stem] = score.astype(np.float16)
        rows.append({
            "stem": protein.stem,
            "set": meta.set,
            "bin": meta.bin,
            "pair": meta.pair,
            "n_relatives": meta.n_relatives,
            "global_plddt": meta.global_plddt,
            "n_residues": protein.length,
            "n_pairs": protein.length * (protein.length - 1) // 2,
            "r_precision": harness.r_precision(score, protein),
            "r_precision_long": harness.r_precision(score, protein, min_separation=harness.LONG_RANGE_SEPARATION),
            "elapsed_seconds": round(elapsed, 3),
            "total_seconds": round(time.perf_counter() - start, 3),
            "forward_tokens": forward_tokens(backend, protein, args.seeds),
            "tail_batch_size": batch,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **static,
        })
        if done % 50 == 0 or done == len(proteins):
            print(f"{nickname}: {done}/{len(proteins)}", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "pcontact.npz", **matrices)
    scores = pd.DataFrame(rows)
    scores.to_csv(out_dir / "scores.csv", index=False)
    print(scores.groupby(["bin", "set"])[["r_precision", "r_precision_long"]].mean().round(4), flush=True)
    del backend, sized
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=list(MODELS), help="MODELS.yaml nicknames")
    parser.add_argument("--panel", type=Path, default=DATA / "panel.parquet")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", type=int, default=harness.DEFAULT_SEEDS)
    parser.add_argument("--tail-batch-size", type=int, default=256, help="upper bound on tails per forward")
    parser.add_argument("--kv-budget-gib", type=float, default=16.0, help="cap on the replicated prefix KV cache")
    parser.add_argument("--runner-tag", default="local")
    parser.add_argument("--limit", type=int, default=None, help="score only the first N structures (smoke test)")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    panel = pd.read_parquet(args.panel).iloc[: args.limit]
    proteins = harness.load_panel_proteins(args.panel)[: args.limit]
    for nickname in args.models:
        score_model(nickname, proteins, panel, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
