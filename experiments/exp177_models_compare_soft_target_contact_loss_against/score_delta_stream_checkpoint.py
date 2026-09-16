# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Score eval-set contacts with a delta-stream Levanter checkpoint.

This is an intentionally simple first-pass inference rule for the compact
``AA DELTA* STOP`` format: build the no-contact delta-stream document for the
input sequence, run one causal forward pass per protein, and use the next-token
log-probability of each signed-delta token immediately after each residue's AA
marker as a directed contact score. Pair scores are the symmetric mean of the
``i -> j`` and ``j -> i`` directed log-probabilities.
"""

import argparse
import io
import json
from pathlib import Path
from typing import Any

import equinox as eqx
import fsspec
import jax
import jax.numpy as jnp
import jmp
import numpy as np

import haliax as hax
from haliax import Axis
from haliax.partitioning import round_axis_for_partitioning

import levanter
from levanter.checkpoint import latest_checkpoint_path, load_checkpoint
from levanter.layers.attention import AttentionBackend, AttentionMask
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.models.qwen import Qwen3Config
from levanter.tracker import NoopConfig
from levanter.trainer import TrainerConfig
from levanter.utils.jax_utils import use_cpu_device
from levanter.utils.tree_utils import inference_mode

STOP_TOKEN_ID = 0
AA_BASE_TOKEN_ID = 1
DELTA_BASE_TOKEN_ID = 32
MAX_ABS_DELTA = 1024
VOCAB_SIZE = 2080
AA_1_TO_3 = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}
AA_ORDER = (
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
)
AA_TO_TOKEN = {aa: AA_BASE_TOKEN_ID + idx for idx, aa in enumerate(AA_ORDER)}

MODEL_CONFIG = Qwen3Config(
    max_seq_len=8192,
    hidden_dim=2048,
    intermediate_dim=8192,
    num_heads=32,
    num_kv_heads=8,
    num_layers=24,
    rope=Llama3RotaryEmbeddingsConfig(),
    use_qk_norm=True,
    attn_backend=AttentionBackend.JAX_FLASH,
)


def delta_to_token(delta: int) -> int:
    if delta == 0:
        raise ValueError("delta=0 is not a contact token")
    ad = abs(delta)
    if ad > MAX_ABS_DELTA:
        raise ValueError(f"abs(delta)={ad} exceeds {MAX_ABS_DELTA}")
    offset = ad - 1 if delta < 0 else MAX_ABS_DELTA + delta - 1
    return DELTA_BASE_TOKEN_ID + offset


def aa_token(ch: str) -> int:
    aa3 = AA_1_TO_3.get(ch.upper())
    if aa3 is None:
        # FoldBench has a few unknown/modified residues represented as X. The
        # training converter would reject these, so skip them at scoring time.
        raise ValueError(f"unknown amino acid {ch!r}")
    return AA_TO_TOKEN[aa3]


def empty_delta_stream(input_seq: str) -> tuple[list[int], list[int]]:
    """Return token ids and positions of each residue AA token."""
    ids: list[int] = []
    aa_positions: list[int] = []
    for ch in input_seq:
        aa_positions.append(len(ids))
        ids.append(aa_token(ch))
        ids.append(STOP_TOKEN_ID)
    return ids, aa_positions


def load_jsonl(path: str) -> list[dict[str, Any]]:
    with fsspec.open(path, "rt") as fh:
        return [json.loads(line) for line in fh]


def load_manifest_sequences(paths: list[str]) -> dict[tuple[str, str], str]:
    import pandas as pd

    seqs: dict[tuple[str, str], str] = {}
    for path in paths:
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            seqs[(str(row["dataset"]), str(row["stem"]))] = str(row["input_seq"])
    return seqs


def save_npz(path: str, score_mean: np.ndarray, score_max: np.ndarray) -> None:
    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        # Keep the historical key as the strict/geomean readout.
        score=score_mean.astype(np.float16),
        score_mean=score_mean.astype(np.float16),
        score_max=score_max.astype(np.float16),
    )
    with fsspec.open(path, "wb") as fh:
        fh.write(buf.getvalue())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gt", default="hf://buckets/open-athena/MarinFold/data/contacts-v1-model-eval-exp89/gt_universe.jsonl")
    parser.add_argument("--manifest", action="append", default=[
        "experiments/exp78_evals_esmfold_contacts/data/eval_manifest_foldbench.csv",
        "experiments/exp78_evals_esmfold_contacts/data/eval_manifest_exp65.csv",
    ])
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=2048)
    args = parser.parse_args()

    records = load_jsonl(args.gt)
    if args.limit is not None:
        records = records[: args.limit]
    seqs = load_manifest_sequences(args.manifest)

    trainer = TrainerConfig(
        tracker=NoopConfig(),
        log_jaxprs=False,
        log_xla_hlo=False,
        mp=jmp.get_policy("p=f32,c=bfloat16"),
        train_batch_size=1,
        per_device_parallelism=1,
        per_device_eval_parallelism=1,
    )
    levanter.trainer.initialize(trainer)
    Pos = Axis("position", args.max_length)
    Batch = Axis("batch", 1)

    with trainer.use_device_mesh():
        key = jax.random.PRNGKey(0)
        Vocab = round_axis_for_partitioning(Axis("vocab", VOCAB_SIZE), trainer.compute_axis_mapping)
        with use_cpu_device():
            model = eqx.filter_eval_shape(MODEL_CONFIG.build, Vocab, key=key)
            checkpoint_path = latest_checkpoint_path(args.checkpoint)
            print(f"[delta-score] loading {checkpoint_path}", flush=True)
            model = load_checkpoint(model, checkpoint_path, subpath="model")
        model = hax.shard(model, trainer.parameter_axis_mapping)
        model = inference_mode(model, True)
        mask = AttentionMask.causal()

        @hax.named_jit(axis_resources=trainer.compute_axis_mapping)
        def forward_logprobs(token_batch):
            logits = model(token_batch, mask, key=key)
            return jax.nn.log_softmax(logits.rearrange((Batch, Pos, Vocab)).array, axis=-1)

        n_ok = 0
        n_skip = 0
        for index, rec in enumerate(records):
            dataset = str(rec["dataset"])
            stem = str(rec["stem"])
            seq = seqs.get((dataset, stem))
            if seq is None:
                print(f"[delta-score] missing sequence for {dataset}/{stem}; skipping", flush=True)
                n_skip += 1
                continue
            try:
                ids, aa_positions = empty_delta_stream(seq)
            except ValueError as exc:
                print(f"[delta-score] {dataset}/{stem}: {exc}; skipping", flush=True)
                n_skip += 1
                continue
            if len(ids) > args.max_length:
                print(f"[delta-score] {dataset}/{stem}: len {len(ids)} > {args.max_length}; skipping", flush=True)
                n_skip += 1
                continue
            padded = np.zeros((1, args.max_length), dtype=np.int32)
            padded[0, : len(ids)] = np.asarray(ids, dtype=np.int32)
            token_batch = hax.named(jnp.asarray(padded), (Batch, Pos))
            lp = np.asarray(forward_logprobs(token_batch)[0], dtype=np.float32)

            L = int(rec["L"])
            directed = np.full((L, L), -1.0e9, dtype=np.float32)
            usable = min(L, len(aa_positions))
            for i in range(usable):
                pos_i = aa_positions[i]
                lo = max(0, i - MAX_ABS_DELTA)
                hi = min(usable, i + MAX_ABS_DELTA + 1)
                for j in range(lo, hi):
                    if i == j:
                        continue
                    directed[i, j] = lp[pos_i, delta_to_token(j - i)]
            score_mean = 0.5 * (directed + directed.T)
            score_max = np.maximum(directed, directed.T)
            save_npz(f"{args.out_dir.rstrip('/')}/{dataset}__{stem}.npz", score_mean, score_max)
            n_ok += 1
            if (index + 1) % 25 == 0:
                print(f"[delta-score] ...{index + 1}/{len(records)} last={dataset}/{stem} L={L}", flush=True)
        print(f"[delta-score] scored={n_ok} skipped={n_skip} out={args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
