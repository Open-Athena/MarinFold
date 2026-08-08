# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the exp200 ``RLJobConfig`` — issue #200.

Assembles marin.rl's online RL job around exp163's multi-draft contacts-v1
checkpoint: two equally-weighted curriculum lessons (the base ``<contacts-v1>``
task and the ``<contacts-v1.multi>`` task), :class:`ContactsDenseLoss` for the
dense per-contact advantage, and vLLM rollout workers on marin iris v5p.

Three things here are load-bearing and non-obvious.

**The 50:50 mix is enforced by ``minimum_sample_probability=1.0``.** marin's
curriculum is *adaptive*: it reweights lessons by a quadratic that peaks at 50%
success, so left alone it would silently drift away from the mix #200 asks for.
``compute_sampling_weights`` clamps every weight up to the minimum and then
renormalises, and since the pre-clamp weights are always below 1, a minimum of
1.0 makes both lessons exactly 0.5 regardless of reward. This is not
belt-and-braces: success is binarized on ``reward > 0``, so a lesson whose F1 is
sometimes zero (the realistic case for the base task on hard proteins) scores
near the peak of the quadratic while a lesson that always scores something
positive scores at its floor. Measured on this config, the stock minimum of 0.1
sends the mix to roughly 84:16. Note the curriculum
picks ONE lesson per rollout step, so the mix holds in expectation across steps
rather than within a batch — which is what we want anyway, since mixing a ~800
token plain rollout with a ~4,200 token multi rollout in one batch pads the
short one out to the long one's length.

**``canonical_model_name`` must contain "qwen".** ``vLLMInferenceContext.__init__``
builds a renderer chosen by substring match on that name and raises for anything
else. The renderer is then never used — see ``contacts_env`` for why — but it
still has to construct.

**``top_k`` stays ``None``.** ``DecodingConfig`` rejects a non-positive ``top_k``,
so exp163's "disabled" sentinel of -1 cannot be expressed here; the environment
translates ``None`` to vLLM's -1. Do not "fix" this by setting a large finite
top_k: #142 traced under-generation to exactly that.
"""

import datetime
import json
import logging

import fsspec
import jmp
from levanter.checkpoint import CheckpointerConfig
from levanter.models.qwen import Qwen3Config
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.optim.config import AdamConfig
from levanter.trainer import MeshConfig, TrainerConfig
from levanter.tracker.wandb import WandbConfig
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
from contacts_env import (
    MULTI_SECTION_SLACK,
    MULTI_TOKENS_PER_CONTACT,
    PLAIN_TOKEN_SLACK,
    PLAIN_TOKENS_PER_RESIDUE,
)
from dense_loss import ContactsDenseLoss

logger = logging.getLogger(__name__)

# Must match the warm-start checkpoint exactly (exp163 refine_ft_common.MODEL_CONFIG).
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

# See the module docstring: substring-matched to pick a renderer we never use.
CANONICAL_MODEL_NAME = "qwen3-1_5b-contacts-v1-multi"

WANDB_PROJECT = "MarinFold"
WANDB_ENTITY = "open-athena"
WANDB_GROUP = "exp200-rl-best-of-n"

DEFAULT_MAX_SECTIONS = 8
DEFAULT_SECTION_CONTACTS = 220
DEFAULT_MAX_PROTEIN_LEN = 512

# exp163 arm F's measured per-contact precision on held-out proteins. Only the
# starting point for p_bar; the environment tracks it from there.
INITIAL_PRECISION = 0.30


def multi_output_tokens(max_sections: int, section_contacts: int) -> int:
    """exp163's multi-draft response budget."""
    return (MULTI_TOKENS_PER_CONTACT * section_contacts + MULTI_SECTION_SLACK) * max_sections


def plain_output_tokens(max_protein_len: int) -> int:
    """exp98's single-section response budget."""
    return PLAIN_TOKENS_PER_RESIDUE * max_protein_len + PLAIN_TOKEN_SLACK


def preflight_checkpoint(checkpoint: str) -> int:
    """Validate an HF export and return its vocab size.

    Two checks, both from expensive #163 mistakes.

    ``rope_theta``: levanter writes the Llama3 rope under ``rope_parameters`` and
    leaves top-level ``rope_theta`` null, and any reader older than transformers 5
    then silently falls back to default rope — a 50x wrong base frequency whose
    error grows with sequence distance. That invalidated a full round of exp163
    evals before anyone noticed, because nothing crashes.

    ``vocab_size``: it must match the checkpoint's embedding matrix, not the
    tokenizer's nominal size (Eric's contacts-v1 checkpoints pad 2846 -> 2848 for
    TPU efficiency), and marin loads HF weights with
    ``resize_vocab_to_match_tokenizer=False``.

    Args:
        checkpoint: Directory URL of an HF export containing ``config.json``.

    Returns:
        The checkpoint's ``vocab_size``.
    """
    url = f"{checkpoint.rstrip('/')}/config.json"
    with fsspec.open(url, "r") as fh:
        cfg = json.load(fh)

    if cfg.get("rope_theta") is None:
        raise ValueError(
            f"{url} has no top-level rope_theta (levanter wrote it under 'rope_parameters'). "
            "Repair with exp163's stage_v3_to_gcs.py or scripts/repair_checkpoint_config.py "
            "before training — readers older than transformers 5 fall back to default rope "
            "SILENTLY, which is how exp163 lost a round of evals."
        )
    vocab_size = cfg.get("vocab_size")
    if not vocab_size:
        raise ValueError(f"{url} has no vocab_size")
    logger.info("[exp200] checkpoint preflight OK: vocab_size=%d rope_theta=%s", vocab_size, cfg["rope_theta"])
    return int(vocab_size)


def build_curriculum(
    *,
    targets_path: str,
    prompts_path: str,
    n_prompts: int,
    n_generations: int,
    max_sections: int = DEFAULT_MAX_SECTIONS,
    section_contacts: int = DEFAULT_SECTION_CONTACTS,
    max_protein_len: int = DEFAULT_MAX_PROTEIN_LEN,
    err_decay: float = 0.5,
    temperature: float = 1.0,
    top_p: float = 0.95,
    eval_frequency: int = 50,
    limit: int | None = None,
) -> CurriculumConfig:
    """Two lessons at a pinned 50:50 mix: the base task and the multi-draft task."""
    common = {
        "targets_path": targets_path,
        "prompts_path": prompts_path,
        "section_contacts": section_contacts,
        "err_decay": err_decay,
        "initial_precision": INITIAL_PRECISION,
        "max_model_len": SEQ_LEN,
        "limit": limit,
    }

    def lesson(mode: str, max_output_tokens: int) -> LessonConfig:
        return LessonConfig(
            lesson_id=f"contacts_{mode}",
            env_config=EnvConfig(
                env_class="contacts_env.ContactsV1RLEnv",
                env_args={**common, "mode": mode, "max_sections": max_sections},
            ),
            sampling_params=SamplingParams(
                n_prompts=n_prompts,
                n_generations_per_prompt=n_generations,
                train_decoding=DecodingConfig(
                    temperature=temperature,
                    top_p=top_p,
                    # None, not -1: see the module docstring.
                    top_k=None,
                    max_output_tokens=max_output_tokens,
                    stop_token_ids=[cr.END_ID],
                    # TPU vLLM rejects per-request seeds.
                    seed=None,
                ),
                # Reuse the sampling config for eval: a greedy default would collapse
                # the candidate spread that best-of-N is entirely made of.
                eval_decoding=None,
            ),
        )

    lessons = {
        "contacts_plain": lesson("plain", plain_output_tokens(max_protein_len)),
        "contacts_multi": lesson("multi", multi_output_tokens(max_sections, section_contacts)),
    }
    return CurriculumConfig(
        lessons=lessons,
        max_seq_len=SEQ_LEN,
        eval_frequency=eval_frequency,
        eval_n_examples=32,
        micro_eval_frequency=None,
        # Pins the mix at 50:50 against the adaptive reweighting — see the module docstring.
        minimum_sample_probability=1.0,
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
    train_batch_size: int = 128,
    n_prompts: int = 32,
    n_generations: int = 8,
    max_sections: int = DEFAULT_MAX_SECTIONS,
    lam_step: float = 1.0,
    lam_doc: float = 1.0,
    err_decay: float = 0.5,
    kl_beta: float = 0.01,
    weight_decay: float = 0.0,
    max_grad_norm: float = 1.0,
    train_tpu_type: str = "v5p-16",
    inference_tpu_type: str = "v5p-8",
    num_rollout_workers: int = 2,
    inference_tensor_parallel_size: int = 4,
    gpu_memory_utilization: float = 0.90,
    regions: tuple[str, ...] = ("us-east5", "us-central1"),
    steps_per_eval: int = 50,
    limit: int | None = None,
    seed: int = 0,
) -> RLJobConfig:
    """Assemble the full online RL job.

    Args:
        run_name: W&B run name; also the iris job-name stem.
        checkpoint: bf16 HF export to warm start from (exp163 arm F by default).
        tokenizer: contacts-v1 tokenizer with id 7 renamed to ``<contacts-v1.multi>``.
        output_prefix: GCS prefix for checkpoints and rollout spill. Must be in the
            same region the workers run in.
        learning_rate: RL needs far less than exp163's 1e-4 fine-tune LR. Note
            marin's DAPO normalisation divides by the batch token count and then
            again by batch size, so learning rates do not transfer from other
            codebases — sweep it.
        kl_beta: KL anchor to the warm-start checkpoint. Non-zero by default:
            exp163's v1/v2 refiners lost 41-44% of base-task R-precision to a
            single unanchored full fine-tune.
        regions: Regions, NOT zones. exp163 zone-pinning starved three jobs, and
            ``with_tpu`` leaves regions unset so the scheduler may pick one with
            no v5p at all.
    """
    vocab_size = preflight_checkpoint(checkpoint)
    prefix = output_prefix.rstrip("/")

    curriculum = build_curriculum(
        targets_path=targets_path,
        prompts_path=prompts_path,
        n_prompts=n_prompts,
        n_generations=n_generations,
        max_sections=max_sections,
        err_decay=err_decay,
        eval_frequency=steps_per_eval,
        limit=limit,
    )

    trainer = TrainerConfig(
        tracker=WandbConfig(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name=run_name,
            group=WANDB_GROUP,
            tags=["protein", "contacts-v1", "qwen3", "1_5b", "exp200", "rl"],
        ),
        log_xla_hlo=False,
        log_jaxprs=False,
        mp=jmp.get_policy("p=f32,c=bfloat16"),
        train_batch_size=train_batch_size,
        num_train_steps=num_train_steps,
        steps_per_eval=steps_per_eval,
        checkpointer=CheckpointerConfig(
            # Per-run, matching the repo's `checkpoints/<wandb-run-name>/` layout.
            # train_worker also namespaces by `{run_id}-train`, but a sweep writing
            # under one shared prefix is not something to leave to a detail of
            # marin's internals.
            base_path=f"{prefix}/checkpoints/{run_name}",
            save_interval=datetime.timedelta(minutes=20),
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
            kl=KLConfig(mode=KLMode.K3_LOSS, beta=kl_beta) if kl_beta > 0 else KLConfig(mode=KLMode.NONE, beta=0.0),
            clip_epsilon_low=0.2,
            clip_epsilon_high=0.2,
            lam_step=lam_step,
            lam_doc=lam_doc,
            # A truncated multi-draft rollout still contains many fully scored
            # contacts, and ~44% of generations hit the length cap, so dropping
            # whole truncated sequences would discard most of the signal.
            do_overlong_filtering=False,
            # Keys off episode_reward only, which would drop groups whose
            # document returns happen to tie despite informative per-contact
            # advantages.
        ),
        replay_buffer=ReplayBufferConfig(
            capacity=n_prompts * n_generations * 4,
            alpha=3.0,
            max_samples=1,
            max_rollout_step_delay=0,
            max_rollout_timestamp_delay=3600.0,
            filter_out_groups_with_no_variance=False,
        ),
    )

    engine = VLLMEngineConfig(
        model_name=checkpoint,
        # Substring-matched to pick a renderer this path never reaches.
        canonical_model_name=CANONICAL_MODEL_NAME,
        max_model_len=SEQ_LEN,
        tensor_parallel_size=inference_tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        # Engine-level seeding; per-request seeds are rejected on TPU.
        seed=seed,
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
            # The writer reaps its own oldest files past this bound; the default of
            # 32 is far below one step's output at this fan-out.
            max_rollout_files=1024,
        ),
        weight_transfer=WeightTransferConfig(
            mode=WeightTransferMode.ARROW_FLIGHT,
            sync_interval_steps=1,
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
            project=WANDB_PROJECT,
            name=f"{run_name}-rollout",
            tags=["exp200", "rollout"],
        ),
        pip_dependency_groups=[],
    )
    # Lockstep rollout/train: freshest possible on-policy data, which matters here
    # because the dense reward is centred on the policy's OWN recent precision.
    return config.with_on_policy_training()


__all__ = [
    "CANONICAL_MODEL_NAME",
    "MODEL_CONFIG",
    "SEQ_LEN",
    "build_curriculum",
    "build_rl_job_config",
    "multi_output_tokens",
    "plain_output_tokens",
    "preflight_checkpoint",
]
