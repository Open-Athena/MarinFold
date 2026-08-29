# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Score exp177 checkpoints by contact-suffix position.

This diagnostic reports two binnings for each split/checkpoint/objective:
absolute contact-suffix target index and relative percentile through the contact
suffix. The train split is a deterministic sample with the same number of
proteins as the full validation split.
"""

import argparse
import csv
import dataclasses
import json
import logging
import math
import os
import re
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import equinox as eqx
import fsspec
import haliax as hax
import jax
import jax.numpy as jnp
import jmp
import numpy as np
from haliax import Axis
from huggingface_hub import snapshot_download
from levanter.checkpoint import latest_checkpoint_path, load_checkpoint
from levanter.data.loader import DataLoader
from levanter.data.text.datasets import BlockShuffleConfig, DatasetComponent, LmDataConfig
from levanter.layers.attention_mask import AttentionMask
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.models.lm_model import LmExample, LmHeadModel, split_activations
from levanter.models.qwen import Qwen3Config
from levanter.tracker import NoopConfig
from levanter.trainer import TrainerConfig, initialize as initialize_trainer
from levanter.utils.jax_utils import use_cpu_device
from levanter.utils.mesh import MeshConfig
from levanter.utils.tree_utils import inference_mode

from marinfold.document_structures.contacts_v1.vocab import BEGIN_STRUCTURE, CONTACT, DOC_TYPE, END

LOGGER = logging.getLogger(__name__)

SEQ_LEN = 8192
VOCAB_SIZE = 2845
CONTACTS_TOKENIZER_REPO = "timodonnell/contacts-v1-tokenizer"
CONTACTS_TOKENIZER_REVISION = "5d68a24a899f"
CONTACTS_TOKENIZER = f"{CONTACTS_TOKENIZER_REPO}@{CONTACTS_TOKENIZER_REVISION}"
TOKENIZER_ALLOW_PATTERNS = (
    "tokenizer*",
    "chat_template*",
    "special_tokens*",
    "added_tokens*",
    "vocab*",
    "merges*",
    "spiece*",
    "*.tiktoken",
)
CW_TRAIN_CACHE = "s3://marin-us-east-02a/MarinFold/exp108_qwen_3b_contacts_v1/tokenized/contacts-v1"
CW_VAL_CACHE = "s3://marin-us-east-02a/MarinFold/exp108_qwen_3b_contacts_v1/tokenized/contacts-v1-val"

MODEL_CONFIG = Qwen3Config(
    max_seq_len=SEQ_LEN,
    hidden_dim=2048,
    intermediate_dim=8192,
    num_heads=32,
    num_kv_heads=8,
    num_layers=24,
    rope=Llama3RotaryEmbeddingsConfig(),
)


@dataclass(frozen=True)
class CheckpointSpec:
    name: str
    objective: str
    path: str
    vocab_size: int


@dataclass
class DocMeta:
    row: int
    prediction_start: int
    contact_count: int
    first_ids: np.ndarray
    second_ids: np.ndarray
    second_neighbor_ids: np.ndarray
    second_neighbor_counts: np.ndarray
    second_neighbor_count: np.ndarray


@dataclass
class BinAccumulators:
    abs_sum: np.ndarray
    abs_count: np.ndarray
    pct_sum: np.ndarray
    pct_count: np.ndarray

    @classmethod
    def create(cls, max_abs_bins: int, pct_bins: int) -> "BinAccumulators":
        return cls(
            abs_sum=np.zeros(max_abs_bins, dtype=np.float64),
            abs_count=np.zeros(max_abs_bins, dtype=np.int64),
            pct_sum=np.zeros(pct_bins, dtype=np.float64),
            pct_count=np.zeros(pct_bins, dtype=np.int64),
        )

    def add(self, suffix_indices: np.ndarray, suffix_lengths: np.ndarray, losses: np.ndarray) -> None:
        valid = np.isfinite(losses) & (suffix_indices >= 0) & (suffix_lengths > 0)
        if not np.any(valid):
            return
        idx = suffix_indices[valid].astype(np.int64)
        val = losses[valid].astype(np.float64)
        ok = idx < self.abs_sum.shape[0]
        idx = idx[ok]
        val = val[ok]
        np.add.at(self.abs_sum, idx, val)
        np.add.at(self.abs_count, idx, 1)

        denom = np.maximum(suffix_lengths[valid][ok] - 1, 1)
        pct = np.floor(idx * self.pct_sum.shape[0] / (denom + 1)).astype(np.int64)
        pct = np.clip(pct, 0, self.pct_sum.shape[0] - 1)
        np.add.at(self.pct_sum, pct, val)
        np.add.at(self.pct_count, pct, 1)


def _parse_checkpoint_specs(raw: str) -> list[CheckpointSpec]:
    specs = []
    for item in json.loads(raw):
        specs.append(
            CheckpointSpec(
                name=item["name"],
                objective=item["objective"],
                path=item["path"].rstrip("/"),
                vocab_size=int(item.get("vocab_size", VOCAB_SIZE)),
            )
        )
    return specs


def _latest_if_root(path: str) -> str:
    if re.search(r"/step-\d+$", path):
        return path
    return latest_checkpoint_path(path)


def _component(cache_dir: str, *, flat_cache: bool = False) -> DatasetComponent:
    return DatasetComponent(cache_dir=cache_dir.rstrip("/"), pack=True, flat_cache=flat_cache)


def _data_config(tokenizer_path: str, split: str) -> LmDataConfig:
    if split == "train":
        components = {"train": _component(os.environ.get("EXP177_DIAG_TRAIN_CACHE", CW_TRAIN_CACHE), flat_cache=False)}
        train_weights = {"train": 1.0}
        shuffle: bool | BlockShuffleConfig = BlockShuffleConfig(io_block_size=256, window_blocks=512, perm_type="feistel")
    elif split == "val":
        components = {"val": _component(os.environ.get("EXP177_DIAG_VAL_CACHE", CW_VAL_CACHE))}
        train_weights = {"val": 0.0}
        shuffle = False
    else:
        raise ValueError(f"Unknown split {split!r}")
    return LmDataConfig(
        components=components,
        train_weights=train_weights,
        tokenizer=tokenizer_path,
        cache_dir=None,
        auto_build_caches=False,
        shuffle=shuffle,
        mixture_block_size=1,
        block_cross_document_attention=True,
    )


def _trainer_config(eval_batch_size: int) -> TrainerConfig:
    return TrainerConfig(
        id="exp177-position-loss-diagnostic",
        tracker=NoopConfig(),
        train_batch_size=eval_batch_size,
        per_device_parallelism=1,
        per_device_eval_parallelism=1,
        allow_nondivisible_batch_size=True,
        require_accelerator=True,
        log_jaxprs=False,
        log_xla_hlo=False,
        mesh=MeshConfig(
            axes={"data": -1, "model": 1, "replica": 1},
            dcn_axes={"replica_dcn": 1},
            compute_mapping={"batch": "data"},
            param_mapping={},
        ),
        mp=jmp.get_policy("p=f32,c=bf16,o=bf16"),
    )


def _iter_split(tokenizer_path: str, split: str, Pos: Axis, key: jax.Array):
    config = _data_config(tokenizer_path, split)
    if split == "train":
        return config.train_sets(Pos, key=key)["train"]
    validation = dict((tags[0], ds) for ds, tags in config.tagged_eval_sets(Pos))
    return validation["val"]


def _extract_doc_metas(tokens: np.ndarray, max_contacts: int, max_degree: int) -> list[DocMeta]:
    doc_type = int(DOC_TYPE)
    begin_structure = int(BEGIN_STRUCTURE)
    contact_id = int(CONTACT)
    end_id = int(END)
    metas: list[DocMeta] = []
    batch, seqlen = tokens.shape
    for row in range(batch):
        starts = np.flatnonzero(tokens[row] == doc_type)
        for start in starts:
            ends = np.flatnonzero((np.arange(seqlen) > start) & (tokens[row] == end_id))
            if ends.size == 0:
                continue
            end_pos = int(ends[0])
            begin_positions = np.flatnonzero((np.arange(seqlen) >= start) & (np.arange(seqlen) < end_pos) & (tokens[row] == begin_structure))
            if begin_positions.size == 0:
                continue
            prediction_start = int(begin_positions[-1])
            suffix_len = end_pos - prediction_start
            if suffix_len < 1 or (suffix_len - 1) % 3 != 0:
                continue
            contact_count = (suffix_len - 1) // 3
            if contact_count <= 0 or contact_count > max_contacts:
                continue
            contact_positions = prediction_start + 1 + 3 * np.arange(contact_count)
            if not np.all(tokens[row, contact_positions] == contact_id):
                continue
            first_ids = tokens[row, contact_positions + 1].astype(np.int32)
            second_ids = tokens[row, contact_positions + 2].astype(np.int32)

            neighbor_ids = np.zeros((max_contacts, max_degree), dtype=np.int32)
            neighbor_counts = np.zeros((max_contacts, max_degree), dtype=np.float32)
            neighbor_count = np.zeros(max_contacts, dtype=np.float32)
            for c in range(contact_count):
                actual_first = int(first_ids[c])
                counts: dict[int, int] = {}
                for j in range(c, contact_count):
                    if int(first_ids[j]) == actual_first:
                        counts[int(second_ids[j])] = counts.get(int(second_ids[j]), 0) + 1
                    if int(second_ids[j]) == actual_first:
                        counts[int(first_ids[j])] = counts.get(int(first_ids[j]), 0) + 1
                ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:max_degree]
                for k, (tok, cnt) in enumerate(ordered):
                    neighbor_ids[c, k] = tok
                    neighbor_counts[c, k] = cnt
                    neighbor_count[c] += cnt
            padded_first = np.zeros(max_contacts, dtype=np.int32)
            padded_second = np.zeros(max_contacts, dtype=np.int32)
            padded_first[:contact_count] = first_ids
            padded_second[:contact_count] = second_ids
            metas.append(DocMeta(row, prediction_start, contact_count, padded_first, padded_second, neighbor_ids, neighbor_counts, neighbor_count))
    return metas


def _pad_doc_metas(metas: list[DocMeta], max_contacts: int, max_degree: int) -> dict[str, np.ndarray]:
    n = max(len(metas), 1)
    return {
        "rows": np.asarray([m.row for m in metas] + [0] * (n - len(metas)), dtype=np.int32),
        "prediction_start": np.asarray([m.prediction_start for m in metas] + [0] * (n - len(metas)), dtype=np.int32),
        "contact_count": np.asarray([m.contact_count for m in metas] + [0] * (n - len(metas)), dtype=np.int32),
        "first_ids": np.stack([m.first_ids for m in metas] + [np.zeros(max_contacts, dtype=np.int32)] * (n - len(metas))),
        "second_ids": np.stack([m.second_ids for m in metas] + [np.zeros(max_contacts, dtype=np.int32)] * (n - len(metas))),
        "second_neighbor_ids": np.stack([m.second_neighbor_ids for m in metas] + [np.zeros((max_contacts, max_degree), dtype=np.int32)] * (n - len(metas))),
        "second_neighbor_counts": np.stack([m.second_neighbor_counts for m in metas] + [np.zeros((max_contacts, max_degree), dtype=np.float32)] * (n - len(metas))),
        "second_neighbor_count": np.stack([m.second_neighbor_count for m in metas] + [np.zeros(max_contacts, dtype=np.float32)] * (n - len(metas))),
    }


def _ensure_lm_example(batch, Batch: Axis, Pos: Axis) -> LmExample:
    tokens = batch.tokens
    loss_weight = batch.loss_weight
    if hasattr(tokens, "array"):
        return batch
    named_tokens = hax.named(jnp.asarray(tokens), (Batch, Pos))
    named_loss_weight = hax.named(jnp.asarray(loss_weight), (Batch, Pos))
    mask = batch.attn_mask
    segment_ids = getattr(mask, "segment_ids", None)
    if segment_ids is not None:
        q_seg, kv_seg = segment_ids
        if not hasattr(q_seg, "array"):
            q_seg = hax.named(jnp.asarray(q_seg), (Batch, Pos))
        if not hasattr(kv_seg, "array"):
            kv_seg = hax.named(jnp.asarray(kv_seg), (Batch, Pos))
        mask = AttentionMask.causal(segment_ids=(q_seg, kv_seg))
    return LmExample(tokens=named_tokens, loss_weight=named_loss_weight, attn_mask=mask)


def _tokens_array(batch: LmExample) -> np.ndarray:
    tokens = batch.tokens
    return np.asarray(tokens.array if hasattr(tokens, "array") else tokens)


def _make_ce_fn(axis_mapping):
    @hax.named_jit(axis_resources=axis_mapping)
    def ce_per_position(model: LmHeadModel, batch: LmExample):
        model = inference_mode(model, True)
        return model.compute_next_token_loss(batch, reduction=None, reduction_axis=()).array
    return ce_per_position


def _make_soft_fn(axis_mapping, max_contacts: int):
    contact_axis = jnp.arange(max_contacts, dtype=jnp.int32)

    @hax.named_jit(axis_resources=axis_mapping)
    def soft_per_doc(model: LmHeadModel, batch: LmExample, meta: dict[str, jax.Array]):
        model = inference_mode(model, True)
        activations, aux = split_activations(model.activations(batch.tokens, key=None, attn_mask=batch.attn_mask))
        del aux
        Pos = batch.tokens.axes[-1]
        lm_head = model.get_lm_head()
        target_y = hax.roll(batch.tokens, -1, Pos)
        hard_ce = model.compute_next_token_loss(batch, reduction=None, reduction_axis=())
        lm_head_vocab = lm_head.axes[0]
        target_rows = lm_head.take(lm_head_vocab, target_y)
        z_target = hax.dot(activations, target_rows, axis=model.Embed)
        log_z = (hard_ce + z_target).rearrange((..., Pos)).array
        act = activations.rearrange((..., Pos, model.Embed)).array
        head = lm_head.rearrange((lm_head_vocab, model.Embed)).array

        rows = meta["rows"]
        prediction_start = meta["prediction_start"]
        contact_count = meta["contact_count"]
        first_ids = meta["first_ids"]
        second_ids = meta["second_ids"]
        second_neighbor_ids = meta["second_neighbor_ids"]
        second_neighbor_counts = meta["second_neighbor_counts"]
        second_neighbor_count = meta["second_neighbor_count"]

        valid_contacts = contact_axis[None, :] < contact_count[:, None]
        contact_positions = jnp.clip(prediction_start[:, None] + 1 + 3 * contact_axis[None, :], 0, act.shape[1] - 1)
        first_positions = jnp.clip(contact_positions + 1, 0, act.shape[1] - 1)
        contact_predict_positions = jnp.clip(
            jnp.where(contact_axis[None, :] == 0, prediction_start[:, None], contact_positions - 1), 0, act.shape[1] - 1
        )
        row_idx = rows[:, None]

        contact_act = act[row_idx, contact_predict_positions]
        contact_logits = jnp.sum(contact_act * head[jnp.asarray(int(CONTACT), dtype=jnp.int32)], axis=-1)
        contact_loss = log_z[row_idx, contact_predict_positions] - contact_logits

        endpoint_rows = head[first_ids] + head[second_ids]
        endpoint_rows = jnp.where(valid_contacts[:, :, None], endpoint_rows, 0.0)
        remaining_endpoint_rows = jnp.flip(jnp.cumsum(jnp.flip(endpoint_rows, axis=1), axis=1), axis=1)
        first_act = act[row_idx, contact_positions]
        first_expected = jnp.sum(first_act * remaining_endpoint_rows, axis=-1) / jnp.maximum(2 * (contact_count[:, None] - contact_axis[None, :]), 1)
        first_loss = log_z[row_idx, contact_positions] - first_expected

        second_act = act[row_idx, first_positions]
        neighbor_logits = jnp.sum(second_act[:, :, None, :] * head[second_neighbor_ids], axis=-1)
        second_expected = jnp.sum(second_neighbor_counts * neighbor_logits, axis=-1) / jnp.maximum(second_neighbor_count, 1.0)
        second_loss = log_z[row_idx, first_positions] - second_expected

        end_positions = jnp.clip(prediction_start + 3 * contact_count, 0, act.shape[1] - 1)
        end_act = act[rows, end_positions]
        end_logits = jnp.sum(end_act * head[jnp.asarray(int(END), dtype=jnp.int32)], axis=-1)
        end_loss = log_z[rows, end_positions] - end_logits
        return contact_loss, first_loss, second_loss, end_loss

    return soft_per_doc


def _load_model(checkpoint: CheckpointSpec, trainer: TrainerConfig):
    key = jax.random.PRNGKey(0)
    Vocab = Axis("vocab", checkpoint.vocab_size)
    with use_cpu_device():
        model = eqx.filter_eval_shape(MODEL_CONFIG.build, Vocab, key=key)
        resolved = _latest_if_root(checkpoint.path)
        LOGGER.info("Loading checkpoint %s with vocab_size=%d", resolved, checkpoint.vocab_size)
        model = load_checkpoint(model, resolved, subpath="model")
    return hax.shard_with_axis_mapping(model, trainer.parameter_axis_mapping)


def _add_ce_losses(acc: BinAccumulators, metas: list[DocMeta], ce: np.ndarray) -> None:
    suffix_indices = []
    suffix_lengths = []
    losses = []
    for meta in metas:
        suffix_len = 3 * meta.contact_count + 1
        positions = meta.prediction_start + np.arange(suffix_len)
        suffix_indices.append(np.arange(suffix_len, dtype=np.int64))
        suffix_lengths.append(np.full(suffix_len, suffix_len, dtype=np.int64))
        losses.append(ce[meta.row, positions])
    if losses:
        acc.add(np.concatenate(suffix_indices), np.concatenate(suffix_lengths), np.concatenate(losses))


def _add_soft_losses(acc: BinAccumulators, metas: list[DocMeta], losses_tuple: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> None:
    contact_loss, first_loss, second_loss, end_loss = losses_tuple
    suffix_indices = []
    suffix_lengths = []
    losses = []
    for i, meta in enumerate(metas):
        c = meta.contact_count
        suffix_len = 3 * c + 1
        idx = np.empty(suffix_len, dtype=np.int64)
        val = np.empty(suffix_len, dtype=np.float32)
        if c:
            idx[: 3 * c : 3] = 3 * np.arange(c)
            idx[1 : 3 * c : 3] = 3 * np.arange(c) + 1
            idx[2 : 3 * c : 3] = 3 * np.arange(c) + 2
            val[: 3 * c : 3] = contact_loss[i, :c]
            val[1 : 3 * c : 3] = first_loss[i, :c]
            val[2 : 3 * c : 3] = second_loss[i, :c]
        idx[-1] = 3 * c
        val[-1] = end_loss[i]
        suffix_indices.append(idx)
        suffix_lengths.append(np.full(suffix_len, suffix_len, dtype=np.int64))
        losses.append(val)
    if losses:
        acc.add(np.concatenate(suffix_indices), np.concatenate(suffix_lengths), np.concatenate(losses))


def _write_outputs(output_prefix: str, checkpoint: CheckpointSpec, split: str, acc: BinAccumulators, docs_seen: int) -> None:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{checkpoint.name}_{split}_{checkpoint.objective}")
    abs_uri = f"{output_prefix.rstrip('/')}/{safe}.absolute.csv"
    pct_uri = f"{output_prefix.rstrip('/')}/{safe}.percentile.csv"
    for uri, rows in [
        (abs_uri, ((i, acc.abs_sum[i], acc.abs_count[i]) for i in range(acc.abs_sum.shape[0]) if acc.abs_count[i])),
        (pct_uri, ((i, acc.pct_sum[i], acc.pct_count[i]) for i in range(acc.pct_sum.shape[0]) if acc.pct_count[i])),
    ]:
        with fsspec.open(uri, "wt") as f:
            writer = csv.writer(f)
            writer.writerow(["checkpoint", "objective", "split", "bin_kind", "bin", "loss_sum", "token_count", "loss_mean", "docs_seen"])
            kind = "absolute" if uri == abs_uri else "percentile"
            for b, s, n in rows:
                writer.writerow([checkpoint.name, checkpoint.objective, split, kind, b, f"{s:.9g}", int(n), f"{s / n:.9g}", docs_seen])
    LOGGER.info("Wrote %s and %s", abs_uri, pct_uri)


def run(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    tokenizer_path = snapshot_download(
        repo_id=CONTACTS_TOKENIZER_REPO,
        revision=CONTACTS_TOKENIZER_REVISION,
        allow_patterns=list(TOKENIZER_ALLOW_PATTERNS),
    )
    trainer = _trainer_config(args.eval_batch_size)
    initialize_trainer(dataclasses.replace(trainer, jax_compilation_cache_dir=args.jax_cache_dir))
    Pos = MODEL_CONFIG.max_Pos.resize(SEQ_LEN)
    specs = _parse_checkpoint_specs(args.checkpoints_json)

    with trainer.use_device_mesh():
        ce_fn = _make_ce_fn(trainer.compute_axis_mapping)
        soft_fn = _make_soft_fn(trainer.compute_axis_mapping, args.max_contacts)
        for spec in specs:
            model = _load_model(spec, trainer)
            for split in ("val", "train"):
                target_docs = args.val_docs if split == "val" else args.train_docs
                ds = _iter_split(tokenizer_path, split, Pos, jax.random.PRNGKey(args.train_seed))
                loader = DataLoader(
                    ds,
                    batch_size=args.eval_batch_size,
                    mesh=trainer.device_mesh,
                    axis_resources=trainer.compute_axis_mapping,
                    batch_axis_name=trainer.batch_axis_name,
                    allow_nondivisible_batch_size=True,
                    max_buffered_batches=4,
                )
                acc = BinAccumulators.create(args.max_abs_bins, args.percentile_bins)
                docs_seen = 0
                started = time.time()
                for batch_idx, batch in enumerate(loader):
                    batch = _ensure_lm_example(batch, trainer.EvalBatch, Pos)
                    tokens = _tokens_array(batch)
                    metas = _extract_doc_metas(tokens, args.max_contacts, args.max_degree)
                    if not metas:
                        continue
                    if docs_seen + len(metas) > target_docs:
                        metas = metas[: target_docs - docs_seen]
                    if spec.objective == "ce":
                        ce = np.asarray(ce_fn(model, batch))
                        _add_ce_losses(acc, metas, ce)
                    elif spec.objective == "soft":
                        meta = jax.tree.map(jnp.asarray, _pad_doc_metas(metas, args.max_contacts, args.max_degree))
                        losses_tuple = tuple(np.asarray(x) for x in soft_fn(model, batch, meta))
                        _add_soft_losses(acc, metas, losses_tuple)
                    else:
                        raise ValueError(f"Unknown objective {spec.objective!r}")
                    docs_seen += len(metas)
                    if batch_idx % args.log_every == 0:
                        LOGGER.info("%s %s: docs=%d/%d batches=%d elapsed=%.1fs", spec.name, split, docs_seen, target_docs, batch_idx, time.time() - started)
                    if docs_seen >= target_docs:
                        break
                _write_outputs(args.output_prefix, spec, split, acc, docs_seen)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints-json", default=os.environ.get("EXP177_DIAG_CHECKPOINTS_JSON", "[]"))
    parser.add_argument("--output-prefix", default=os.environ.get("EXP177_DIAG_OUTPUT_PREFIX", "s3://marin-us-east-02a/protein-structure/MarinFold/exp177_soft_target_loss_h2h/position_loss_diagnostic/2026.08.28"))
    parser.add_argument("--val-docs", type=int, default=int(os.environ.get("EXP177_DIAG_VAL_DOCS", "41954")))
    parser.add_argument("--train-docs", type=int, default=int(os.environ.get("EXP177_DIAG_TRAIN_DOCS", "41954")))
    parser.add_argument("--train-seed", type=int, default=int(os.environ.get("EXP177_DIAG_TRAIN_SEED", "177")))
    parser.add_argument("--eval-batch-size", type=int, default=int(os.environ.get("EXP177_DIAG_EVAL_BATCH_SIZE", "8")))
    parser.add_argument("--max-contacts", type=int, default=int(os.environ.get("EXP177_MAX_SPARSE_CONTACTS", "2048")))
    parser.add_argument("--max-degree", type=int, default=int(os.environ.get("EXP177_MAX_SPARSE_DEGREE", "32")))
    parser.add_argument("--max-abs-bins", type=int, default=int(os.environ.get("EXP177_DIAG_MAX_ABS_BINS", "8192")))
    parser.add_argument("--percentile-bins", type=int, default=int(os.environ.get("EXP177_DIAG_PERCENTILE_BINS", "100")))
    parser.add_argument("--log-every", type=int, default=int(os.environ.get("EXP177_DIAG_LOG_EVERY", "10")))
    parser.add_argument("--jax-cache-dir", default=os.environ.get("JAX_COMPILATION_CACHE_DIR", "/tmp/jax-compilation-cache"))
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
