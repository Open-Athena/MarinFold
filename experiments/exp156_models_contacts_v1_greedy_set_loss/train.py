# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Build or launch the exp156 contacts-v1 greedy set-loss experiment.

This follows the exp147 custom-experiment shape: the experiment owns the launch
script and the nonstandard training entrypoint instead of routing through
Levanter's stock ``train_lm.main`` loss function.

The experiment can run either stock next-token CE or the contacts-v1 greedy
latent-set objective. The greedy arm uses a custom Levanter ``Trainer`` loop so
it can replace ``model.compute_next_token_loss`` with contacts-v1 target parsing
and greedy pair matching while still using standard checkpointing, evaluation
hooks, W&B tracking, and optional GPU telemetry.
"""

import argparse
import dataclasses
import os
from collections.abc import Callable
from datetime import timedelta

import jmp
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from fray.types import ResourceConfig
import haliax as hax
from haliax import Axis
from haliax.partitioning import ResourceAxis, named_jit, round_axis_for_partitioning
from huggingface_hub import snapshot_download
from iris.client.client import get_iris_ctx
from levanter.adaptor import NoAdaptorConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.compat.hf_checkpoints import HFCompatConfig
from levanter.data.text.datasets import DatasetComponent, LmDataConfig, UrlDatasetSourceConfig
from levanter.data.text.formats import TextLmDatasetFormat
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.main.train_lm import TrainLmConfig
from levanter.models.qwen import Qwen3Config
from levanter.optim.config import AdamConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import Trainer, TrainerConfig
from levanter.utils.mesh import MeshConfig
from marin.execution.lazy import ArtifactStep, StepContext, lower
from marin.execution.remote import remote
from marin.execution.step_runner import StepRunner
from marin.experiment.data import tokenized
from marin.experiment.namespacing import user_namespaced_name
from marin.processing.tokenize.tokenize import TokenizedCache
from marin.training.training import LevanterCheckpoint, TrainLmOnPodConfig, run_levanter_train_lm
from marinfold.document_structures.contacts_v1.greedy_set_loss import (
    greedy_matched_contact_block_loss,
    parse_contact_block_targets,
)

DEFAULT_COREWEAVE_BASE_PREFIX = "s3://marin-us-east-02a/marin/protein-structure/MarinFold"
DEFAULT_COREWEAVE_CONTACTS_V1_PREFIX = "s3://marin-us-east-02a/MarinFold/data/document_structures/contacts_v1"
DEFAULT_GCS_BASE_PREFIX = "gs://marin-us-east5/protein-structure/MarinFold"


def _default_base_prefix() -> str:
    if "EXP156_BASE_PREFIX" in os.environ:
        return os.environ["EXP156_BASE_PREFIX"].rstrip("/")
    marin_prefix = os.environ.get("MARIN_PREFIX")
    if marin_prefix is not None and marin_prefix.startswith("s3://"):
        return f"{marin_prefix.rstrip('/')}/protein-structure/MarinFold"
    if os.environ.get("EXP156_ACCELERATOR", "gpu") == "gpu":
        return DEFAULT_COREWEAVE_BASE_PREFIX
    return DEFAULT_GCS_BASE_PREFIX


ROOT = _default_base_prefix()
MARIN_PREFIX = f"{ROOT}/exp156_contacts_v1_greedy_set_loss"
os.environ["MARIN_PREFIX"] = MARIN_PREFIX

# CoreWeave jobs should use S3, not GCS. The contacts-v1 mirror currently lives
# under the top-level MarinFold/data prefix, while exp156 outputs default under
# s3://.../marin/protein-structure/MarinFold.
if os.environ.get("EXP156_ACCELERATOR", "gpu") == "gpu":
    default_train_glob = f"{DEFAULT_COREWEAVE_CONTACTS_V1_PREFIX}/train/*.parquet"
    default_val_glob = f"{DEFAULT_COREWEAVE_CONTACTS_V1_PREFIX}/val/*.parquet"
else:
    default_train_glob = f"{ROOT}/exp53_contacts_v1_5x/documents/train/*.parquet"
    default_val_glob = f"{ROOT}/exp53_contacts_v1_5x/documents/val/*.parquet"
CONTACTS_V1_TRAIN_GLOB = os.environ.get("EXP156_TRAIN_GLOB", default_train_glob)
CONTACTS_V1_VAL_GLOB = os.environ.get("EXP156_VAL_GLOB", default_val_glob)
TOKENIZED_TRAIN_CACHE = os.environ.get("EXP156_TOKENIZED_TRAIN_CACHE")
TOKENIZED_VAL_CACHE = os.environ.get("EXP156_TOKENIZED_VAL_CACHE")

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
ARTIFACT_VERSION = os.environ.get("EXP156_VERSION", "exp156-dev")
DATA_VERSION = os.environ.get("EXP156_DATA_VERSION", "2026.07.24")

MODEL_CONFIG = Qwen3Config(
    max_seq_len=8192,
    hidden_dim=2048,
    intermediate_dim=8192,
    num_heads=32,
    num_kv_heads=8,
    num_layers=24,
    rope=Llama3RotaryEmbeddingsConfig(),
)

EXP156_ACCELERATOR = os.environ.get("EXP156_ACCELERATOR", "gpu")
TPU_TYPE = os.environ.get("EXP156_TPU", "v6e-8")
TPU_ZONE = os.environ.get("EXP156_ZONE", "us-east5-b")
GPU_TYPE = os.environ.get("EXP156_GPU", "H100")
GPU_COUNT = int(os.environ.get("EXP156_GPU_COUNT", "1"))

if EXP156_ACCELERATOR == "gpu":
    RESOURCES = ResourceConfig.with_gpu(
        GPU_TYPE,
        count=GPU_COUNT,
        cpu=float(os.environ.get("EXP156_CPU", "16")),
        ram=os.environ.get("EXP156_RAM", "128g"),
        disk=os.environ.get("EXP156_DISK", "200g"),
    )
elif EXP156_ACCELERATOR == "tpu":
    RESOURCES = ResourceConfig.with_tpu(
        TPU_TYPE,
        slice_count=1,
        cpu=32,
        ram="128g",
        disk="50g",
        zone=TPU_ZONE,
    )
else:
    raise ValueError(f"unsupported EXP156_ACCELERATOR={EXP156_ACCELERATOR!r}")

# Start from the current best contacts-v1 HF export by default. On CoreWeave we
# stage this from the open-athena HF bucket to the local worker cache, then pass
# the local directory to Levanter's HF initializer.
INITIALIZE_FROM_HF = os.environ.get("EXP156_INITIALIZE_FROM_HF", "contacts-v1-exp120-1.5B")
LOSS_KIND = os.environ.get("EXP156_LOSS_KIND", "greedy-set")

TAGS = (
    "protein",
    "contacts-v1",
    "greedy-set-loss",
    "latent-order",
    "qwen3",
    "exp156",
    "prototype",
)
TOKEN_AXES = (
    ResourceAxis.REPLICA_DCN,
    ResourceAxis.REPLICA,
    ResourceAxis.DATA,
)


def _tokenized_step(*, split: str, paths: list[str]) -> ArtifactStep[TokenizedCache]:
    return tokenized(
        name=f"tokenized/contacts-v1-{split}",
        paths=paths,
        tokenizer=CONTACTS_TOKENIZER_REPO,
        version=DATA_VERSION,
        validation=split != "train",
        text_key="document",
        resources=ResourceConfig.with_cpu(
            cpu=4,
            ram="16g",
            disk="10g",
            zone=TPU_ZONE if EXP156_ACCELERATOR == "tpu" else None,
        ),
    )


def _mirrored_cache_component(cache_root: str) -> DatasetComponent:
    """Build a Levanter component for a legacy mirrored cache without an artifact record.

    The exp67 cache's ``.artifact.json`` is the legacy JSON literal ``null``, so
    ``TokenizedCache.raw_load`` cannot adopt it. Its Levanter cache layout is
    nevertheless complete and validated; construct the same component that a
    typed cache would expose, with contacts-v1's document-text format.
    """
    source = UrlDatasetSourceConfig(
        tags=[],
        train_urls=[],
        validation_urls=[],
        cache_dir=cache_root,
        format=TextLmDatasetFormat(text_key="document"),
    )
    return DatasetComponent(source=source, cache_dir=source.cache_dir, format=source.format, tags=source.tags)


def _component(cache: TokenizedCache) -> DatasetComponent:
    # Greedy parsing expects one contacts-v1 document per row. Keep the baseline
    # arm unpacked too unless explicitly overridden, so the 1000-step comparison
    # changes the loss rather than the packing semantics.
    pack = os.environ.get("EXP156_PACK", "0") == "1"
    return dataclasses.replace(cache.as_component(), pack=pack)


def _with_local_tokenizer_and_hf_init(
    pod_config: TrainLmOnPodConfig,
    *,
    tokenizer_path: str,
    hf_init_path: str | None,
) -> TrainLmOnPodConfig:
    train_config = dataclasses.replace(
        pod_config.train_config,
        data=dataclasses.replace(
            pod_config.train_config.data,
            tokenizer=tokenizer_path,
        ),
    )
    if hf_init_path is not None:
        train_config = dataclasses.replace(
            train_config,
            initialize_from_hf=hf_init_path,
            pad_tokenizer_to_match_model=True,
        )
    return dataclasses.replace(pod_config, train_config=train_config)


def _token_id(tokenizer, token: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None or int(token_id) < 0:
        raise ValueError(f"token {token!r} is absent from tokenizer")
    return int(token_id)


def _position_token_ids(tokenizer, *, count: int = 2000) -> np.ndarray:
    return np.asarray([_token_id(tokenizer, f"<p{index}>") for index in range(count)], dtype=np.int64)


def contacts_v1_greedy_jax_loss(
    log_probs: jnp.ndarray,
    token_ids: jnp.ndarray,
    *,
    begin_statements_token_id: int,
    contact_token_id: int,
    end_token_id: int,
    max_contact_slots: int,
) -> jnp.ndarray:
    """Differentiable hard-assignment contacts-v1 loss for one token row.

    The contact-slot assignment is discrete (`argmax`) and therefore treated as
    stop-gradient. The returned CE gathers from JAX `log_probs`, so gradients
    flow to the selected token logits.
    """
    seq_len = token_ids.shape[0]
    positions = jnp.arange(seq_len)
    begin_position = jnp.argmax(token_ids == begin_statements_token_id)
    after_begin = positions > begin_position
    end_candidates = jnp.where(after_begin & (token_ids == end_token_id), positions, seq_len - 1)
    end_position = jnp.min(end_candidates)

    prefix_positions = positions < begin_position
    prefix_targets = jnp.roll(token_ids, -1)
    prefix_loss = -jnp.sum(jnp.where(prefix_positions, log_probs[positions, prefix_targets], 0.0))
    prefix_count = jnp.sum(prefix_positions.astype(jnp.float32))

    contact_mask = (token_ids == contact_token_id) & (positions > begin_position) & (positions < end_position)
    slot_positions = jnp.where(contact_mask, size=max_contact_slots, fill_value=0)[0]
    n_slots_total = jnp.sum(contact_mask.astype(jnp.int32))
    n_slots = jnp.minimum(n_slots_total, max_contact_slots)
    slot_index = jnp.arange(max_contact_slots)
    valid_slot = slot_index < n_slots

    left_tokens = token_ids[jnp.minimum(slot_positions + 1, seq_len - 1)]
    right_tokens = token_ids[jnp.minimum(slot_positions + 2, seq_len - 1)]
    pair_left = jnp.minimum(left_tokens, right_tokens)
    pair_right = jnp.maximum(left_tokens, right_tokens)

    marker_scores = log_probs[jnp.maximum(slot_positions - 1, 0), contact_token_id]
    left_pred_positions = slot_positions
    right_pred_positions = jnp.minimum(slot_positions + 1, seq_len - 1)
    forward = log_probs[left_pred_positions[:, None], pair_left[None, :]] + log_probs[
        right_pred_positions[:, None], pair_right[None, :]
    ]
    reverse = log_probs[left_pred_positions[:, None], pair_right[None, :]] + log_probs[
        right_pred_positions[:, None], pair_left[None, :]
    ]
    pair_scores = marker_scores[:, None] + jnp.maximum(forward, reverse)

    def _match_one(carry, slot_i):
        remaining, loss = carry
        candidate_mask = remaining & valid_slot
        masked_scores = jnp.where(candidate_mask, pair_scores[slot_i], -jnp.inf)
        pair_i = jnp.argmax(masked_scores)
        best_score = pair_scores[slot_i, pair_i]
        use_slot = valid_slot[slot_i]
        remaining = remaining.at[pair_i].set(jnp.where(use_slot, False, remaining[pair_i]))
        loss = loss - jnp.where(use_slot, best_score, 0.0)
        return (remaining, loss), None

    initial_remaining = valid_slot
    (_, pair_loss), _ = jax.lax.scan(
        _match_one,
        (initial_remaining, jnp.asarray(0.0, dtype=log_probs.dtype)),
        jnp.arange(max_contact_slots),
    )
    end_loss = -log_probs[jnp.maximum(end_position - 1, 0), end_token_id]
    denom = jnp.maximum(prefix_count + 3.0 * n_slots.astype(jnp.float32) + 1.0, 1.0)
    return (prefix_loss + pair_loss + end_loss) / denom


def contacts_v1_greedy_batch_loss(
    model,
    example,
    tokenizer,
    *,
    key=None,
    max_contact_slots: int,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    logits = model(example.tokens, example.attn_mask, key=key)
    log_probs = jax.nn.log_softmax(logits.array, axis=-1)
    tokens = example.tokens.array
    begin_id = _token_id(tokenizer, "<begin_statements>")
    contact_id = _token_id(tokenizer, "<contact>")
    end_id = _token_id(tokenizer, "<end>")
    losses = jax.vmap(
        lambda row_log_probs, row_tokens: contacts_v1_greedy_jax_loss(
            row_log_probs,
            row_tokens,
            begin_statements_token_id=begin_id,
            contact_token_id=contact_id,
            end_token_id=end_id,
            max_contact_slots=max_contact_slots,
        )
    )(log_probs, tokens)
    loss = jnp.mean(losses)
    return loss, {"greedy_set_loss": loss}


def contacts_v1_greedy_single_pass_smoke_loss(model, example, tokenizer, *, key=None) -> jnp.ndarray:
    """Run one model forward and compute a hard-matched greedy-set loss.

    Host NumPy is used only to choose the greedy pair assignment. The returned
    loss gathers from the original JAX log-probabilities, so gradients flow to
    the selected logits. The assignment itself is hard / stop-gradient.
    """
    logits = model(example.tokens, example.attn_mask, key=key)
    log_probs = jax.nn.log_softmax(logits.array, axis=-1)
    log_probs_np = np.asarray(jax.device_get(log_probs))
    tokens_np = np.asarray(jax.device_get(example.tokens.array))

    begin_id = _token_id(tokenizer, "<begin_statements>")
    contact_id = _token_id(tokenizer, "<contact>")
    end_id = _token_id(tokenizer, "<end>")
    position_ids = _position_token_ids(tokenizer)

    row_losses: list[jnp.ndarray] = []
    for row in range(tokens_np.shape[0]):
        targets = parse_contact_block_targets(
            tokens_np[row],
            begin_statements_token_id=begin_id,
            contact_token_id=contact_id,
            end_token_id=end_id,
            position_token_ids=position_ids,
        )
        matched = greedy_matched_contact_block_loss(
            log_probs_np[row],
            tokens_np[row],
            begin_statements_token_id=begin_id,
            contact_token_id=contact_id,
            end_token_id=end_id,
            position_token_ids=position_ids,
        )

        prefix_positions = jnp.arange(targets.begin_position)
        prefix_targets = jnp.asarray(tokens_np[row, 1 : targets.begin_position + 1])
        row_loss = -jnp.sum(log_probs[row, prefix_positions, prefix_targets])
        for choice in matched.choices:
            marker_pos, left_pos, right_pos = targets.slot_positions[choice.slot_index]
            left_token, right_token = choice.oriented_tokens
            row_loss = row_loss - log_probs[row, int(marker_pos) - 1, contact_id]
            row_loss = row_loss - log_probs[row, int(left_pos) - 1, left_token]
            row_loss = row_loss - log_probs[row, int(right_pos) - 1, right_token]
        row_loss = row_loss - log_probs[row, targets.end_position - 1, end_id]
        row_losses.append(row_loss)
    return jnp.mean(jnp.stack(row_losses))


def run_contacts_v1_greedy_train_lm(config: TrainLmOnPodConfig) -> None:
    """Run Levanter training with the contacts-v1 greedy set loss.

    This is the custom entrypoint that will replace Levanter's stock
    ``train_lm.main`` call. The stock path hardcodes::

        model.compute_next_token_loss(example)

    Here we need to construct ``Trainer(..., contacts_v1_greedy_loss)`` instead.
    The remaining work is to decide the batch object consumed by that loss:

    1. a tokenized ``LmExample`` plus parser state reconstructed from token ids,
       or
    2. a contacts-v1 document batch carrying explicit sequence prefix and contact
       pairs, similar in spirit to PR #144's custom document batch.
    """
    if os.environ.get("EXP156_SINGLE_PASS_SMOKE") == "1":
        _run_single_pass_smoke(config)
        return
    _run_greedy_train(config)


def _run_greedy_train(config: TrainLmOnPodConfig) -> None:
    """Run Levanter Trainer with the contacts-v1 greedy set loss."""
    import equinox as eqx
    import levanter
    import levanter.callbacks as callbacks
    import levanter.trainer
    from levanter.trainer import Trainer
    from marin.training.training import _prepare_training_run

    config, train_config, env = _prepare_training_run(config)
    for key, value in env.items():
        os.environ.setdefault(key, value)

    tokenizer = train_config.data.the_tokenizer
    levanter.trainer.initialize(train_config)
    optimizer = train_config.optimizer.build(train_config.trainer.num_train_steps)
    max_contact_slots = int(os.environ.get("EXP156_MAX_CONTACT_SLOTS", "1024"))

    def _loss(model, example, *, key=None):
        return contacts_v1_greedy_batch_loss(
            model,
            example,
            tokenizer,
            key=key,
            max_contact_slots=max_contact_slots,
        )

    with Trainer(train_config.trainer, optimizer, _loss) as trainer:
        seed = train_config.trainer.seed
        data_key, model_key, training_key = jrandom.split(jrandom.PRNGKey(seed), 3)
        if train_config.data_seed is not None:
            data_key = jrandom.PRNGKey(train_config.data_seed)

        train_length = train_config.train_seq_len or train_config.model.max_seq_len
        Pos = train_config.model.max_Pos.resize(train_length)
        vocab_size = len(tokenizer)
        Vocab = round_axis_for_partitioning(Axis("vocab", vocab_size), trainer.parameter_axis_mapping)
        train_dataset = train_config.data.train_set(Pos, train_config.trainer.batch_schedule, key=data_key)
        validation_sets = train_config.data.validation_sets(Pos)

        model = train_config.model.build(Vocab, key=model_key)
        if train_config.initialize_from_hf:
            if not isinstance(train_config.model, HFCompatConfig):
                raise TypeError("initialize_from_hf requires an HF-compatible model config")
            converter = train_config.model.hf_checkpoint_converter().replaced(
                reference_checkpoint=str(train_config.initialize_from_hf),
                tokenizer=tokenizer,
            )
            if train_config.pad_tokenizer_to_match_model:
                converter = converter.with_tokenizer_padded_to_match_model()
            model = converter.load_pretrained(
                train_config.model.model_type,
                config=train_config.model,
                axis_mapping=trainer.parameter_axis_mapping,
                dtype=trainer.mp.compute_dtype,
            )
            model = named_jit(trainer.mp.cast_to_param, trainer.parameter_axis_mapping)(model)

        state = trainer.initial_state(training_key, model=model)

        @eqx.filter_jit
        def _greedy_eval_loss(eval_model, example):
            eval_model = trainer.mp.cast_to_compute(eval_model)
            loss, _metrics = _loss(eval_model, example, key=None)
            return loss, {}

        @eqx.filter_jit
        def _next_token_eval_loss(eval_model, example):
            eval_model = trainer.mp.cast_to_compute(eval_model)
            loss = eval_model.compute_next_token_loss(example, key=None)
            return loss, {}

        if not validation_sets:
            print("[exp156] no validation datasets provided for greedy-set run")
        for name, dataset in validation_sets.items():
            eval_name = name or "validation"
            greedy_loader = trainer.data_loader(dataset, trainer.EvalBatch)
            trainer.add_hook(
                callbacks.compute_validation_loss(
                    _greedy_eval_loss,
                    greedy_loader,
                    max_batches=train_config.trainer.max_eval_batches,
                    name=f"{eval_name}/greedy_set",
                ),
                every=train_config.trainer.steps_per_eval,
            )
            if os.environ.get("EXP156_ENABLE_GREEDY_NEXT_TOKEN_EVAL", "1") == "1":
                next_token_loader = trainer.data_loader(dataset, trainer.EvalBatch)
                trainer.add_hook(
                    callbacks.compute_validation_loss(
                        _next_token_eval_loss,
                        next_token_loader,
                        max_batches=train_config.trainer.max_eval_batches,
                        name=f"{eval_name}/next_token",
                    ),
                    every=train_config.trainer.steps_per_eval,
                )
            else:
                print(f"[exp156] skipping next-token validation hook for {eval_name}")

        train_loader = trainer.data_loader(train_dataset).iter_from_step(state.step)
        trainer.train(state, train_loader)
    trainer.tracker.finish()


def _run_single_pass_smoke(config: TrainLmOnPodConfig) -> None:
    """Build one batch/model and evaluate the smoke-only greedy loss once."""
    import jax.random as jrandom
    import levanter.trainer
    from haliax import Axis
    from haliax.partitioning import round_axis_for_partitioning
    from levanter.trainer import Trainer
    from marin.training.training import _prepare_training_run

    config, train_config, env = _prepare_training_run(config)
    for key, value in env.items():
        os.environ.setdefault(key, value)

    tokenizer = train_config.data.the_tokenizer
    levanter.trainer.initialize(train_config)
    optimizer = train_config.optimizer.build(train_config.trainer.num_train_steps)

    def _zero_loss(_model, _example, *, key=None):
        return jnp.asarray(0.0, dtype=jnp.float32)

    with Trainer(train_config.trainer, optimizer, _zero_loss) as trainer:
        seed = train_config.trainer.seed
        data_key, model_key, training_key = jrandom.split(jrandom.PRNGKey(seed), 3)
        train_length = train_config.train_seq_len or train_config.model.max_seq_len
        Pos = train_config.model.max_Pos.resize(train_length)
        vocab_size = len(tokenizer)
        Vocab = round_axis_for_partitioning(Axis("vocab", vocab_size), trainer.parameter_axis_mapping)
        train_dataset = train_config.data.train_set(Pos, train_config.trainer.batch_schedule, key=data_key)
        state = trainer.initial_state(training_key, model_init=lambda: train_config.model.build(Vocab, key=model_key))
        loader = trainer.data_loader(train_dataset).iter_from_step(0)
        batch = next(iter(loader))
        example = batch[0] if isinstance(batch, tuple) else batch
        loss = contacts_v1_greedy_single_pass_smoke_loss(state.model, example, tokenizer)
        print(f"[exp156] single-pass greedy-set smoke loss = {float(loss):.6f}")


def _gpu_telemetry_enabled() -> bool:
    return EXP156_ACCELERATOR == "gpu" and os.environ.get("EXP156_ENABLE_GPU_TELEMETRY", "1") != "0"


def _run_with_gpu_telemetry(pod_config: TrainLmOnPodConfig, train_fn: Callable[[TrainLmOnPodConfig], None]) -> None:
    if not _gpu_telemetry_enabled():
        train_fn(pod_config)
        return

    try:
        from marin.monitoring.gpu_telemetry import NvidiaSmiTelemetryConfig, nvidia_smi_telemetry
    except ModuleNotFoundError:
        from experiments.exp156_models_contacts_v1_greedy_set_loss.exp156_gpu_telemetry import (
            NvidiaSmiTelemetryConfig,
            nvidia_smi_telemetry,
        )

    output_uri = f"{pod_config.output_path.rstrip('/')}/telemetry/gpu"
    telemetry_config = NvidiaSmiTelemetryConfig(
        output_uri=output_uri,
        interval=float(os.environ.get("EXP156_GPU_TELEMETRY_INTERVAL", "5")),
        records_per_chunk=int(os.environ.get("EXP156_GPU_TELEMETRY_RECORDS_PER_CHUNK", "120")),
        max_queue_items=int(os.environ.get("EXP156_GPU_TELEMETRY_MAX_QUEUE_ITEMS", "10000")),
        log_every=int(os.environ.get("EXP156_GPU_TELEMETRY_LOG_EVERY", "120")),
        stop_timeout=float(os.environ.get("EXP156_GPU_TELEMETRY_STOP_TIMEOUT", "30")),
    )
    print(f"[exp156] writing GPU telemetry to {output_uri}")
    with nvidia_smi_telemetry(telemetry_config):
        train_fn(pod_config)


def _run_with_pinned_tokenizer(pod_config: TrainLmOnPodConfig) -> None:
    """Stage tokenizer and optional HF warm-start before custom training."""
    from marinfold.registry import resolve_model

    tokenizer_path = snapshot_download(
        repo_id=CONTACTS_TOKENIZER_REPO,
        revision=CONTACTS_TOKENIZER_REVISION,
        allow_patterns=list(TOKENIZER_ALLOW_PATTERNS),
    )
    hf_init = pod_config.train_config.initialize_from_hf
    hf_init_path = str(resolve_model(hf_init)) if isinstance(hf_init, str) and hf_init else None
    run_contacts_v1_greedy_train_lm(
        _with_local_tokenizer_and_hf_init(
            pod_config,
            tokenizer_path=tokenizer_path,
            hf_init_path=hf_init_path,
        )
    )


def _run_stock_next_token_train(pod_config: TrainLmOnPodConfig) -> None:
    """Run the baseline stock Levanter CE loss with the same staged assets."""
    from marinfold.registry import resolve_model

    tokenizer_path = snapshot_download(
        repo_id=CONTACTS_TOKENIZER_REPO,
        revision=CONTACTS_TOKENIZER_REVISION,
        allow_patterns=list(TOKENIZER_ALLOW_PATTERNS),
    )
    hf_init = pod_config.train_config.initialize_from_hf
    hf_init_path = str(resolve_model(hf_init)) if isinstance(hf_init, str) and hf_init else None
    run_levanter_train_lm(
        _with_local_tokenizer_and_hf_init(
            pod_config,
            tokenizer_path=tokenizer_path,
            hf_init_path=hf_init_path,
        )
    )


def _run_stock_next_token_train_with_telemetry(pod_config: TrainLmOnPodConfig) -> None:
    _run_with_gpu_telemetry(pod_config, _run_stock_next_token_train)


def _run_greedy_set_train_with_telemetry(pod_config: TrainLmOnPodConfig) -> None:
    _run_with_gpu_telemetry(pod_config, _run_with_pinned_tokenizer)


def _train_job(pod_config: TrainLmOnPodConfig) -> None:
    if LOSS_KIND == "next-token":
        remote(_run_stock_next_token_train_with_telemetry, resources=pod_config.resources)(pod_config)
        return
    if LOSS_KIND == "greedy-set":
        remote(_run_greedy_set_train_with_telemetry, resources=pod_config.resources)(pod_config)
        return
    raise ValueError(f"unsupported EXP156_LOSS_KIND={LOSS_KIND!r}")


def _identity_config(
    ctx: StepContext,
    train: ArtifactStep[TokenizedCache] | None,
    validation: ArtifactStep[TokenizedCache] | None,
    *,
    name: str,
    steps: int,
    steps_per_eval: int,
    train_batch_size: int,
    per_device_parallelism: int,
    max_eval_batches: int | None,
) -> dict[str, object]:
    """Return stable experiment decisions used for artifact fingerprinting."""
    learning_rate = float(os.environ.get("EXP156_LEARNING_RATE", "3.1623e-3"))
    return {
        "name": name,
        "model": MODEL_CONFIG,
        "initialize_from_hf": INITIALIZE_FROM_HF,
        "optimizer": AdamConfig(
            learning_rate=learning_rate,
            weight_decay=0.2,
            beta1=0.9,
            beta2=0.95,
            warmup=0.1,
            lr_schedule="cosine",
            min_lr_ratio=0.1,
        ),
        "loss": {
            "kind": LOSS_KIND,
            "implementation": "stock Levanter CE" if LOSS_KIND == "next-token" else "custom Trainer greedy set loss",
        },
        "data": {
            "train": TOKENIZED_TRAIN_CACHE or ctx.artifact_path(train),
            "validation": TOKENIZED_VAL_CACHE or ctx.artifact_path(validation),
            "tokenizer": CONTACTS_TOKENIZER,
            "format": "contacts-v1 serialized parquet documents",
            "shuffle": True,
            "mixture_block_size": 1,
            "block_cross_document_attention": True,
        },
        "gpu_telemetry": {
            "enabled": EXP156_ACCELERATOR == "gpu" and os.environ.get("EXP156_ENABLE_GPU_TELEMETRY", "1") != "0",
            "interval": float(os.environ.get("EXP156_GPU_TELEMETRY_INTERVAL", "5")),
            "records_per_chunk": int(os.environ.get("EXP156_GPU_TELEMETRY_RECORDS_PER_CHUNK", "120")),
            "max_queue_items": int(os.environ.get("EXP156_GPU_TELEMETRY_MAX_QUEUE_ITEMS", "10000")),
            "log_every": int(os.environ.get("EXP156_GPU_TELEMETRY_LOG_EVERY", "120")),
        },
        "trainer": {
            "train_batch_size": train_batch_size,
            "per_device_parallelism": per_device_parallelism,
            "num_train_steps": steps,
            "steps_per_eval": steps_per_eval,
            "max_eval_batches": max_eval_batches,
            "precision": "p=f32,c=bfloat16",
            "mesh": {"replica": 1, "data": -1, "model": 1},
            "resources": {
                "accelerator": EXP156_ACCELERATOR,
                "gpu_type": GPU_TYPE if EXP156_ACCELERATOR == "gpu" else None,
                "gpu_count": GPU_COUNT if EXP156_ACCELERATOR == "gpu" else None,
                "tpu_type": TPU_TYPE if EXP156_ACCELERATOR == "tpu" else None,
                "tpu_zone": TPU_ZONE if EXP156_ACCELERATOR == "tpu" else None,
            },
        },
        "wandb": {
            "entity": "open-athena",
            "project": "MarinFold",
            "group": "exp156-contacts-v1-greedy-set-loss",
            "name": name,
            "tags": TAGS,
        },
        "hf_save_steps": steps,
        "data_seed": 0,
    }


def build_step() -> ArtifactStep[LevanterCheckpoint]:
    """Build the lazy training artifact without submitting it."""
    if (TOKENIZED_TRAIN_CACHE is None) != (TOKENIZED_VAL_CACHE is None):
        raise ValueError(
            "EXP156_TOKENIZED_TRAIN_CACHE and EXP156_TOKENIZED_VAL_CACHE must be set together"
        )
    train = None if TOKENIZED_TRAIN_CACHE else _tokenized_step(split="train", paths=[CONTACTS_V1_TRAIN_GLOB])
    validation = None if TOKENIZED_VAL_CACHE else _tokenized_step(split="val", paths=[CONTACTS_V1_VAL_GLOB])
    steps = int(os.environ.get("EXP156_STEPS", "10"))
    steps_per_eval = int(os.environ.get("EXP156_STEPS_PER_EVAL", "10"))
    train_batch_size = int(os.environ.get("EXP156_TRAIN_BATCH_SIZE", "16"))
    per_device_parallelism = int(os.environ.get("EXP156_PER_DEVICE_PARALLELISM", "1"))
    max_eval_batches_env = os.environ.get("EXP156_MAX_EVAL_BATCHES", "1")
    max_eval_batches = (
        int(max_eval_batches_env) if max_eval_batches_env is not None else None
    )
    default_name = f"exp156-contacts-v1-{LOSS_KIND}-from-exp120-{steps}s-bs{train_batch_size}"
    name = os.environ.get("EXP156_NAME", default_name)

    def build_config(ctx: StepContext) -> TrainLmOnPodConfig | dict[str, object]:
        identity = _identity_config(
            ctx,
            train,
            validation,
            name=name,
            steps=steps,
            steps_per_eval=steps_per_eval,
            train_batch_size=train_batch_size,
            per_device_parallelism=per_device_parallelism,
            max_eval_batches=max_eval_batches,
        )
        if ctx.is_fingerprint:
            return identity

        train_key = "tokenized/contacts-v1-train"
        val_key = "tokenized/contacts-v1-val"
        data = LmDataConfig(
            components={
                train_key: _mirrored_cache_component(TOKENIZED_TRAIN_CACHE)
                if TOKENIZED_TRAIN_CACHE
                else _component(ctx.resolved(train)),
                val_key: _mirrored_cache_component(TOKENIZED_VAL_CACHE)
                if TOKENIZED_VAL_CACHE
                else _component(ctx.resolved(validation)),
            },
            train_weights={train_key: 1.0, val_key: 0.0},
            tokenizer=CONTACTS_TOKENIZER,
            cache_dir=None,
            auto_build_caches=False,
            shuffle=True,
            mixture_block_size=1,
            block_cross_document_attention=True,
        )
        trainer = TrainerConfig(
            id=name,
            tracker=WandbConfig(
                entity="open-athena",
                project="MarinFold",
                name=name,
                tags=list(TAGS),
                group="exp156-contacts-v1-greedy-set-loss",
                replicate_path=ctx.output_path,
            ),
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            train_batch_size=train_batch_size,
            per_device_parallelism=per_device_parallelism,
            num_train_steps=steps,
            steps_per_eval=steps_per_eval,
            checkpointer=CheckpointerConfig(
                save_interval=timedelta(minutes=10),
                keep=[{"every": steps}],
            ),
            mesh=MeshConfig(
                axes={"replica": 1, "data": -1, "model": 1},
                compute_mapping={
                    "token": TOKEN_AXES,
                    "token_repeat": TOKEN_AXES,
                },
            ),
            per_device_eval_parallelism=-1,
            max_eval_batches=max_eval_batches,
            allow_nondivisible_batch_size=True,
        )
        train_config = TrainLmConfig(
            data=data,
            trainer=trainer,
            model=MODEL_CONFIG,
            optimizer=identity["optimizer"],
            z_loss_weight=0.0,
            train_seq_len=8192,
            hf_save_steps=steps,
            data_seed=0,
            adapter=NoAdaptorConfig(),
            initialize_from_hf=INITIALIZE_FROM_HF,
            pad_tokenizer_to_match_model=True,
        )
        env_vars = {"WANDB_ENTITY": "open-athena", "WANDB_PROJECT": "MarinFold"}
        for env_name in (
            "WANDB_API_KEY",
            "WANDB_MODE",
            "EXP156_ENABLE_GPU_TELEMETRY",
            "EXP156_GPU_TELEMETRY_INTERVAL",
            "EXP156_GPU_TELEMETRY_RECORDS_PER_CHUNK",
            "EXP156_GPU_TELEMETRY_MAX_QUEUE_ITEMS",
            "EXP156_GPU_TELEMETRY_LOG_EVERY",
            "EXP156_GPU_TELEMETRY_STOP_TIMEOUT",
        ):
            env_value = os.environ.get(env_name)
            if env_value:
                env_vars[env_name] = env_value
        return TrainLmOnPodConfig(
            train_config=train_config,
            resources=ctx.runtime_arg("train_resources"),
            output_path=ctx.output_path,
            env_vars=env_vars,
            auto_build_caches=False,
        )

    return ArtifactStep(
        name=user_namespaced_name(f"checkpoints/{name}", ARTIFACT_VERSION),
        version=ARTIFACT_VERSION,
        artifact_type=LevanterCheckpoint,
        run=_train_job,
        build_config=build_config,
        deps=tuple(step for step in (train, validation) if step is not None),
        runtime_args={"train_resources": RESOURCES},
    )


def build_steps() -> list[ArtifactStep[LevanterCheckpoint]]:
    """Build the single-step launch list used by tests and the CLI."""
    return [build_step()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="store_true",
        help="Submit the lowered graph. Without this flag, only print the plan.",
    )
    args = parser.parse_args(argv)
    lowered = lower(build_step())
    if not args.run:
        print(lowered)
        return 0
    if get_iris_ctx() is None:
        parser.error(
            "--run must execute inside an Iris coordinator job; launch this "
            "experiment as an Iris job first"
        )
    StepRunner().run([lowered])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
