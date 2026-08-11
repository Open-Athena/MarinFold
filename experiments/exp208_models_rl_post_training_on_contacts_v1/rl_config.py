# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the exp208 ``RLJobConfig`` — issue #208.

exp200's assembler, reduced to the base task and pointed at exp199. What changed,
and why each change is load-bearing:

**One lesson, not two.** exp200 pinned a 50:50 plain/multi mix against marin's
adaptive curriculum with ``minimum_sample_probability=1.0``. With a single lesson
the weight is 1 regardless, so that whole mechanism is gone.

**The document term is selectable.** ``doc_term`` picks between #208's
consensus-marginal, exp200's own-F1, and none. That is the experiment's axis, so
it is a first-class config knob rather than a code edit per arm.

**Checkpoints are on a STEP interval.** exp200 checkpointed on a 20-minute timer,
and when two of its three arms stalled, their rolling checkpoints lagged training
by ~30 steps and left nothing clean to evaluate. ``keep=[{"every": N}]`` writes
permanent step-indexed checkpoints; ``save_interval`` still provides the rolling
temporary one for preemption recovery.

Three things carried over verbatim because they were expensive to learn:

**``canonical_model_name`` must contain "qwen" AND be a registered key.**
``vLLMInferenceContext.__init__`` substring-matches it to build a renderer this
path never uses, while ``reload_model`` looks it up as an EXACT key in
``MODEL_MAPPINGS`` on the weight-transfer path. A descriptive invented name
satisfies the first and fails the second — and only during weight transfer, which
pure-generation gates never reach. All three registered Qwen3 entries resolve to
the identical mapping, so borrowing the 1.7B key for this 1.5B model is exact.

**``top_k`` stays ``None``.** ``DecodingConfig`` rejects a non-positive ``top_k``,
so the "disabled" sentinel of -1 cannot be expressed here; the environment
translates ``None`` to vLLM's -1. Do not "fix" this by setting a large finite
top_k — #142 traced under-generation to exactly that, and #82 found sharpening
past T=1.0/p=0.95 collapses the very consensus this experiment optimizes.

**The engine needs an HF repo id, not a ``gs://`` path.** levanter's
``load_tokenizer`` accepts a local directory, a ``mirror://`` ref or a Hub repo
id, and raises ``HFValidationError`` on a URL — deep inside a rollout worker,
after the gang has scheduled. vLLM streaming weights from GCS is what makes this
easy to walk into: the weights path is fine and only the tokenizer path is not.
"""

import datetime
import json
import logging
import os

import fsspec
import jmp
from levanter.checkpoint import CheckpointerConfig
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.models.qwen import Qwen3Config
from levanter.optim.config import AdamConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import MeshConfig, TrainerConfig
from marin.rl.curriculum import CurriculumConfig, LessonConfig
from marin.rl.decoding import DecodingConfig, SamplingParams
from marin.rl.environments.base import EnvConfig
from marin.rl.environments.inference_ctx import (
    VLLMEngineConfig,
    VLLMFallbackSamplingConfig,
    vLLMInferenceContextConfig,
)
from marin.rl.job_config import RLJobConfig, RunConfig, TrainParams
from marin.rl.kl_regularization import KLConfig, KLMode
from marin.rl.replay_buffer import ReplayBufferConfig
from marin.rl.rollout_storage import RolloutStorageConfig, StorageType
from marin.rl.rollout_worker import RolloutTrackerConfig
from marin.rl.weight_transfer import WeightTransferConfig, WeightTransferMode

import contact_rewards as cr
from contacts_env import PLAIN_TOKEN_SLACK, PLAIN_TOKENS_PER_RESIDUE
from dense_loss import ContactsDenseLoss

logger = logging.getLogger(__name__)

# Verified against exp199's published config.json on 2026-08-10: hidden 2048,
# intermediate 8192, 32 heads, 8 kv heads, 24 layers, max_position 8192, llama3
# rope at theta 500000. Identical to the architecture exp200 warm-started from,
# so this block carries over unchanged.
MODEL_CONFIG = Qwen3Config(
    max_seq_len=8192,
    hidden_dim=2048,
    intermediate_dim=8192,
    num_heads=32,
    num_kv_heads=8,
    num_layers=24,
    rope=Llama3RotaryEmbeddingsConfig(),
)
SEQ_LEN = 8192
CANONICAL_MODEL_NAME = "Qwen/Qwen3-1.7B"

WANDB_PROJECT = "MarinFold"
WANDB_ENTITY = "open-athena"
WANDB_GROUP = "exp208-rl-dense-contacts"

DEFAULT_MAX_PROTEIN_LEN = 512
# p_bar's starting value. #208 Phase 0 measured single-rollout per-contact
# precision for THIS model at **0.482** over 10,000 plain rollouts on the eval
# set. exp200's 0.23 came from exp163 arm F in multi-draft mode and is stale by
# a factor of two here: starting p_bar far below the truth makes every correct
# contact look like a large win and every error nearly free, which biases the
# first steps toward over-emission — the opposite of the collapse the design
# guards against, but a bias either way. The environment EMA-tracks it from
# there, so this only shapes the first few steps.
#
# 0.45 rather than 0.482: the training pool is AFDB round-0 with pyconfind
# labels while 0.482 was measured on the PDB-derived eval set, and nothing has
# measured the training distribution directly.
INITIAL_PRECISION = 0.45


def plain_output_tokens(max_protein_len: int) -> int:
    """exp98's single-section response budget."""
    return PLAIN_TOKENS_PER_RESIDUE * max_protein_len + PLAIN_TOKEN_SLACK


def preflight_checkpoint(checkpoint: str) -> int:
    """Validate an HF export and return its vocab size.

    ``rope_theta``: levanter writes the Llama3 rope under ``rope_parameters`` and
    leaves top-level ``rope_theta`` null, and any reader older than transformers 5
    then silently falls back to default rope — a 50x wrong base frequency whose
    error grows with sequence distance. That invalidated a full round of exp163
    evals before anyone noticed, because nothing crashes.

    ``vocab_size``: it must match the checkpoint's embedding matrix, not the
    tokenizer's nominal size, and marin loads HF weights with
    ``resize_vocab_to_match_tokenizer=False``.
    """
    if "://" in checkpoint or os.path.isdir(checkpoint):
        url = f"{checkpoint.rstrip('/')}/config.json"
        with fsspec.open(url, "r") as fh:
            cfg = json.load(fh)
    else:
        from huggingface_hub import hf_hub_download
        url = f"{checkpoint}/config.json"
        with open(hf_hub_download(checkpoint, "config.json")) as fh:
            cfg = json.load(fh)

    if cfg.get("rope_theta") is None:
        raise ValueError(
            f"{url} has no top-level rope_theta (levanter writes it under 'rope_parameters'). "
            "Repair with scripts/repair_checkpoint_config.py before training — readers older "
            "than transformers 5 fall back to default rope SILENTLY, which is how exp163 lost "
            "a round of evals."
        )
    vocab_size = cfg.get("vocab_size")
    if not vocab_size:
        raise ValueError(f"{url} has no vocab_size")
    logger.info("[exp208] checkpoint preflight OK: vocab_size=%d rope_theta=%s",
                vocab_size, cfg["rope_theta"])
    return int(vocab_size)


def check_engine_model_path(checkpoint: str) -> None:
    """Refuse a checkpoint path the rollout worker's tokenizer loader cannot read."""
    if os.path.isdir(checkpoint) or checkpoint.startswith("mirror://"):
        return
    if "://" in checkpoint:
        raise ValueError(
            f"{checkpoint!r} cannot be used as VLLMEngineConfig.model_name: levanter's "
            "load_tokenizer accepts a local directory, a mirror:// ref, or an HF Hub repo "
            "id, and raises HFValidationError on a URL. Publish the export to an HF repo "
            "and pass the repo id."
        )


def check_region_locality(regions: tuple[str, ...], **paths: str) -> None:
    """Refuse a config whose data lives in a different region from the compute.

    Not a soft warning: ``rigging.filesystem.cross_region.TransferBudgetExceeded``
    killed a three-arm sweep an hour in, after the trainers had started and the
    rollout workers had written thousands of rollouts. The trap is that the read
    volume looks trivial — prompts are ~30 KB per protein — while the rollout
    spill and checkpoints on the same prefix are not. Reasoning about only the
    reads is how this was justified the first time.
    """
    allowed = set(regions)
    for name, url in paths.items():
        if not url or not url.startswith("gs://"):
            continue
        bucket = url[len("gs://"):].split("/", 1)[0]
        if not bucket.startswith("marin-"):
            continue
        region = bucket[len("marin-"):]
        if region not in allowed:
            raise ValueError(
                f"{name} is in {region} but the workers run in {sorted(allowed)}: {url}\n"
                "Co-locate the data with the compute (AGENTS.md), or marin aborts the job "
                "with TransferBudgetExceeded once rollout spill and checkpoints start flowing."
            )


def build_curriculum(
    *,
    targets_path: str,
    prompts_path: str,
    n_prompts: int,
    n_generations: int,
    doc_term: str,
    max_protein_len: int = DEFAULT_MAX_PROTEIN_LEN,
    err_decay: float = 0.5,
    temperature: float = 1.0,
    top_p: float = 0.95,
    eval_frequency: int = 50,
    limit: int | None = None,
    trace_path: str | None = None,
) -> CurriculumConfig:
    """One lesson: the base ``<contacts-v1>`` task."""
    lesson = LessonConfig(
        lesson_id="contacts_plain",
        env_config=EnvConfig(
            env_class="contacts_env.ContactsV1RLEnv",
            env_args={
                "targets_path": targets_path,
                "prompts_path": prompts_path,
                "doc_term": doc_term,
                "err_decay": err_decay,
                "initial_precision": INITIAL_PRECISION,
                "max_protein_len": max_protein_len,
                "max_model_len": SEQ_LEN,
                "limit": limit,
                "trace_path": trace_path,
            },
        ),
        sampling_params=SamplingParams(
            n_prompts=n_prompts,
            n_generations_per_prompt=n_generations,
            train_decoding=DecodingConfig(
                temperature=temperature,
                top_p=top_p,
                top_k=None,          # None, not -1: see the module docstring.
                max_output_tokens=plain_output_tokens(max_protein_len),
                stop_token_ids=[cr.END_ID],
                seed=None,           # TPU vLLM rejects per-request seeds.
            ),
            # Reuse the sampling config for eval: a greedy default would collapse
            # the resampled spread the consensus metric is entirely made of.
            eval_decoding=None,
        ),
    )
    return CurriculumConfig(
        lessons={"contacts_plain": lesson},
        max_seq_len=SEQ_LEN,
        eval_frequency=eval_frequency,
        eval_n_examples=32,
        micro_eval_frequency=None,
    )


def build_rl_job_config(
    *,
    run_name: str,
    checkpoint: str,
    tokenizer: str,
    targets_path: str,
    prompts_path: str,
    output_prefix: str,
    learning_rate: float,
    num_train_steps: int,
    doc_term: str = "consensus",
    train_batch_size: int = 64,
    n_prompts: int = 8,
    n_generations: int = 16,
    lam_step: float = 1.0,
    lam_doc: float = 1.0,
    err_decay: float = 0.5,
    kl_beta: float = 0.01,
    weight_decay: float = 0.0,
    max_grad_norm: float = 1.0,
    train_tpu_type: str = "v5p-8",
    inference_tpu_type: str = "v5p-8",
    num_rollout_workers: int = 4,
    inference_tensor_parallel_size: int = 4,
    gpu_memory_utilization: float = 0.90,
    regions: tuple[str, ...] = ("us-central1",),
    steps_per_eval: int = 50,
    checkpoint_every_steps: int = 25,
    sync_interval_steps: int = 8,
    limit: int | None = None,
    seed: int = 0,
) -> RLJobConfig:
    """Assemble the full online RL job for one arm.

    Args:
        learning_rate: marin's DAPO normalisation divides by the batch token
            count and then again by batch size, so learning rates transfer
            neither from other codebases NOR across a change in
            ``train_batch_size`` — and exp208 changes it (32 -> 64) to keep four
            groups per batch at ``n_generations`` 16. Pick it from the KL
            trajectory, not from exp200's sweep.
        n_generations: The group size, which is also the sample size of the
            consensus estimator. exp200 used 8; #208 uses 16 because the
            deployed metric is a consensus over 100 and a rollout's influence on
            a group of 8 is a much cruder proxy for its influence on 100.
        kl_beta: KL anchor to the warm start. Non-zero by default: exp163's
            v1/v2 refiners lost 41-44% of base-task R-precision to a single
            unanchored full fine-tune.
        checkpoint_every_steps: Permanent step-indexed checkpoints. See the
            module docstring for why this is not a timer.
    """
    vocab_size = preflight_checkpoint(checkpoint)
    check_engine_model_path(checkpoint)
    check_region_locality(regions, targets=targets_path, prompts=prompts_path,
                          output_prefix=output_prefix)
    prefix = output_prefix.rstrip("/")

    curriculum = build_curriculum(
        targets_path=targets_path,
        prompts_path=prompts_path,
        n_prompts=n_prompts,
        n_generations=n_generations,
        doc_term=doc_term,
        err_decay=err_decay,
        eval_frequency=steps_per_eval,
        limit=limit,
        # `iris job logs` is empty for a RUNNING child, so the environment
        # reports on itself to object storage instead.
        trace_path=f"{prefix}/trace/{run_name}",
    )

    trainer = TrainerConfig(
        tracker=WandbConfig(
            project=WANDB_PROJECT, entity=WANDB_ENTITY, name=run_name, group=WANDB_GROUP,
            tags=["protein", "contacts-v1", "qwen3", "1_5b", "exp208", "rl", doc_term],
        ),
        log_xla_hlo=False,
        log_jaxprs=False,
        mp=jmp.get_policy("p=f32,c=bfloat16"),
        train_batch_size=train_batch_size,
        num_train_steps=num_train_steps,
        steps_per_eval=steps_per_eval,
        checkpointer=CheckpointerConfig(
            base_path=f"{prefix}/checkpoints/{run_name}",
            # The rolling temporary checkpoint, for preemption recovery...
            save_interval=datetime.timedelta(minutes=20),
            # ...and the permanent step-indexed ones, which are what actually gets
            # evaluated. exp200 had only the timer and lost both stalled arms to it.
            keep=[{"every": checkpoint_every_steps}],
        ),
        mesh=MeshConfig(
            axes={"context": 1, "model": 1},
            shared_mapping={"mlp": "model", "heads": "model", "position": "context"},
        ),
    )

    train_params = TrainParams(
        optimizer=AdamConfig(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup=0.0,
            lr_schedule="constant",
            max_grad_norm=max_grad_norm,
        ),
        rl_loss=ContactsDenseLoss(
            kl=(KLConfig(mode=KLMode.K3_LOSS, beta=kl_beta) if kl_beta > 0
                else KLConfig(mode=KLMode.NONE, beta=0.0)),
            clip_epsilon_low=0.2,
            clip_epsilon_high=0.2,
            lam_step=lam_step,
            lam_doc=lam_doc,
            # Zeroing a whole truncated sequence would discard many fully scored
            # contacts.
            do_overlong_filtering=False,
        ),
        replay_buffer=ReplayBufferConfig(
            capacity=n_prompts * n_generations * 4,
            alpha=3.0,
            # Each rollout trains once: it was drawn from a policy this run is
            # actively moving away from.
            max_samples=1,
            # Must admit rollouts generated since the last weight transfer, or the
            # freshness filter drops everything produced between syncs and the
            # trainer starves.
            max_rollout_step_delay=sync_interval_steps,
            max_rollout_timestamp_delay=3600.0,
            # Keys off episode_reward only. In #208 that field is a consensus
            # marginal which is legitimately zero for a rollout that does not move
            # the vote, and identically zero in the step-only arm — so this filter
            # would drop entire groups carrying perfectly good per-contact signal.
            filter_out_groups_with_no_variance=False,
        ),
    )

    engine = VLLMEngineConfig(
        model_name=checkpoint,
        canonical_model_name=CANONICAL_MODEL_NAME,
        max_model_len=SEQ_LEN,
        tensor_parallel_size=inference_tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        seed=seed,          # engine-level; per-request seeds are rejected on TPU
        load_format="runai_streamer" if checkpoint.startswith("gs://") else "auto",
    )

    config = RLJobConfig(
        model=MODEL_CONFIG,
        trainer=trainer,
        train_params=train_params,
        curriculum=curriculum,
        tokenizer=tokenizer,
        inference_type="vllm",
        seed=seed,
        vocab_size=vocab_size,
        initial_checkpoint=checkpoint,
        rollout_storage=RolloutStorageConfig(
            storage_type=StorageType.FILE,
            path=f"{prefix}/rollouts/{run_name}",
            # The writer reaps its own oldest files past this bound; the default
            # of 32 is far below one step's output at this fan-out.
            max_rollout_files=1024,
        ),
        weight_transfer=WeightTransferConfig(
            mode=WeightTransferMode.ARROW_FLIGHT,
            sync_interval_steps=sync_interval_steps,
            convert_to_bfloat16=True,
        ),
        run_config=RunConfig(
            train_tpu_type=train_tpu_type,
            inference_tpu_type=inference_tpu_type,
            num_rollout_workers=num_rollout_workers,
            regions=list(regions),
        ),
        inference_config=vLLMInferenceContextConfig(
            engine=engine,
            # No stop STRINGS: the environment passes stop_token_ids, and
            # get_stop_tokens only knows llama/qwen chat markers anyway.
            fallback_sampling=VLLMFallbackSamplingConfig(top_k=None, stop_strings=None),
        ),
        run_id=run_name,
        rollout_tracker=RolloutTrackerConfig(
            project=WANDB_PROJECT, name=f"{run_name}-rollout", tags=["exp208", "rollout"],
        ),
        pip_dependency_groups=[],
    )
    # Deliberately NOT config.with_on_policy_training(): that forces
    # sync_interval_steps=1, which exp200 measured at 372 s/step with generation
    # accounting for 0.4% of it.
    return config


__all__ = [
    "CANONICAL_MODEL_NAME",
    "MODEL_CONFIG",
    "SEQ_LEN",
    "build_curriculum",
    "build_rl_job_config",
    "check_engine_model_path",
    "check_region_locality",
    "plain_output_tokens",
    "preflight_checkpoint",
]
