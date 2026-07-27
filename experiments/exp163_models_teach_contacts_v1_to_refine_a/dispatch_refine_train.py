# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Direct batch-priority Fray dispatch of the exp163 refiner fine-tune (issue #163).

Warm-start Eric's tuned contacts-v1 1.5B (E8, ``prot-exp75-...-bc3084`` step-35679,
eval loss 2.7566) and continue-train it on the rollout-**refinement** corpus with
the answer-span loss mask, on the CoreWeave **rno-2a** H100 cluster (``cw-rno2a``)
at iris **batch** priority. This is the GPU/CoreWeave twin of ``train_exp163.py``
(which runs the SAME recipe on TPU via the marin executor): given a protein's
sequence + K noisy candidate rollouts, emit the TRUE contact set.

Modelled EXACTLY on exp108's ``dispatch_train.py`` (+ its
``train_qwen_3b_contacts_v1_sweep.py`` entry, merged here so this file is the
``python -m`` entry point) and on the sibling ``dispatch_rollouts.py`` (the
exp163-local freshiris dispatch idioms). Like exp108 we submit each LR as its OWN
``fray.types.JobRequest(priority=3)`` gang that WE control -- the marin executor
submits child jobs with NO priority band (-> interactive), and #163/#108 require
the batch band for all CoreWeave work.

Recipe (from ``train_exp163.py``): a dedicated **1-epoch cosine** continue-train
(peak LR decays to the ``min_lr_ratio=0.1`` floor over exactly one epoch) at the
low LRs {1e-4, 3e-4}, ``weight_decay=0.2``, ``warmup=0.1``, global batch **128**
(== Eric's #75 training batch -- no batch-scaling confound), seq **8192**. The
train corpus is answer-span **masked** (loss only on ``<begin_statements> ... <end>``
via :func:`refine_ft_common.answer_span_loss_weight`); the val corpus (original
contacts-v1 val) is monitored **unmasked** every eval -- the step-0 value should
read ~ 2.7566, confirming the E8 warm-start loaded.

Storage -- everything on CoreWeave S3 (``s3://marin-us-east-02a/MarinFold/exp163``):
the pods carry ONE S3 endpoint/credential set (injected by iris) and have NO GCS,
so BOTH the warm-start checkpoint AND the corpus must be staged to S3 first (see
the PREREQUISITES block below). Token caches are built tokenize-on-the-fly on the
training workers (``auto_build_caches=True``), exactly like exp108 -- so NO separate
executor tokenize step runs; the corpus arg points at the RAW refinement parquet.

--------------------------------------------------------------------------------
PREREQUISITES on S3 (must exist BEFORE this can run) -- see also the report:
  1. E8 **Levanter-native** training-state checkpoint dir, staged GCS->S3:
       s3://marin-us-east-02a/MarinFold/exp163/model_levanter/step-35679
     (this is the LEVANTER checkpoint for warm-start -- NOT the HF export at
      .../model/step-35679 that dispatch_rollouts.py feeds to vLLM.)
  2. The refinement corpus (raw parquet, one ``document`` text column):
       s3://marin-us-east-02a/MarinFold/exp163/val10k/refinement_corpus/*.parquet
  3. The unmasked val split (original contacts-v1 val, base-task retention monitor),
     staged to S3 (exp108 already stages this tree):
       s3://marin-us-east-02a/MarinFold/data/document_structures/contacts_v1/val/*.parquet
  (default_tokenize does NOT run as a sub-step: auto_build_caches=True has the
   training workers tokenize the raw parquet into the S3 cache on first read.)

ENV COMPATIBILITY CAVEAT (read the report): this launcher follows the task's
instruction to reuse ``marinfold_models.default_train`` / ``SimpleTrainConfig`` and
``refine_ft_common`` (MODEL_CONFIG + answer_span_loss_weight). On THIS branch those
modules are the executor-era vendor (``from fray import ResourceConfig`` +
``marin.execution.this_output_path``), which do NOT import in the marin-freshiris
env that provides ``fray.types.JobRequest.priority`` (the env this batch dispatch
requires). The dispatch scaffolding here (fray.types / marin.training /
create_environment) is freshiris-correct, but the config-build imports need
``marinfold_models`` + ``refine_ft_common`` ported to freshiris (as origin/main did
with ``build_train_lm_on_pod_config``) -- OR the driver run in an executor-marin env
whose fray carries ``JobRequest.priority``. py_compile passes regardless.
--------------------------------------------------------------------------------

Env knobs (so a smoke run needs no code edit):
  EXP163_LRS              comma list of peak LRs (default "1e-4,3e-4"); set ONE for a smoke
  EXP163_EPOCHS           default 1
  EXP163_STEPS_PER_EPOCH  REQUIRED unless EXP163_TRAIN_TOKENS is set --
                          ceil(train_tokens / (128 * 8192)); the refinement corpus
                          token count (print it during the HF->S3 mirror) is not known here
  EXP163_TRAIN_TOKENS     alternative to STEPS_PER_EPOCH; steps/epoch is derived
  EXP163_MAX_STEPS        cap steps for a smoke run (default: full 1-epoch count)
  EXP163_REPLICAS         number of 8xH100 nodes per run (default 1 = 8 GPUs; <=4)
  EXP163_CORPUS           train corpus glob   (S3 default above)
  EXP163_VAL              unmasked val glob   (S3 default above)
  EXP163_INIT_CKPT        E8 Levanter ckpt dir (S3 default above)
  EXP163_S3_PREFIX        output/cache prefix (default s3://marin-us-east-02a/MarinFold/exp163)
  WANDB_API_KEY           forwarded into the pod (does NOT inherit the launch shell)

Launch (batch priority -- required by #163):

    set -a; source ~/.config/marin/cw-rno2a.env; set +a
    WK=$(python -c "import netrc; print(netrc.netrc().authenticators('api.wandb.ai')[2])")
    uv run iris --cluster=cw-rno2a job run --no-wait --priority batch \\
        --enable-extra-resources --cpu=2 --memory=6GB --disk=16GB --extra gpu \\
        -e WANDB_API_KEY "$WK" -e EXP163_STEPS_PER_EPOCH <N> \\
        -- python -m dispatch_refine_train

Smoke first (ONE LR, ~50 steps, single 8xH100 node) -- confirm the batch fits, the
job reports the batch band, the E8 warm-start loads (step-0 val ~ 2.7566), and the
S3 tokenize-on-the-fly cache builds:

    uv run iris --cluster=cw-rno2a job run --no-wait --priority batch \\
        --enable-extra-resources --cpu=2 --memory=6GB --disk=16GB --extra gpu \\
        -e WANDB_API_KEY "$WK" -e EXP163_LRS 1e-4 -e EXP163_STEPS_PER_EPOCH 50 \\
        -e EXP163_MAX_STEPS 50 -- python -m dispatch_refine_train

Dry-run locally (build + print the JobRequests, no submit):

    EXP163_DRY_RUN=1 EXP163_STEPS_PER_EPOCH=100 python -m dispatch_refine_train
"""

from __future__ import annotations

import dataclasses
import logging
import math
import os

# fray symbols live in ``fray.types`` in the marin-freshiris build -- a plain
# ``from fray import ResourceConfig`` FAILS here (unlike exp108's pinned line).
# Mirrors the sibling dispatch_rollouts.py.
from fray.current_client import current_client
from fray.types import (
    Entrypoint,
    JobRequest,
    JobStatus,
    ResourceConfig,
    create_environment,
)
from marin.training.run_environment import extras_for_resources
from marin.training.training import (
    TrainLmOnPodConfig,
    resolve_training_env,
    run_levanter_train_lm,
)

# tokenize-on-the-fly LM data config (exp108 build_data_config pattern).
from levanter.data.text import (
    DatasetComponent,
    LmDataConfig,
    TextLmDatasetFormat,
    UrlDatasetSourceConfig,
)

# Reuse refine_ft_common's builders (issue #163): the Qwen3 MODEL_CONFIG that MUST
# match the E8 warm-start, the packing-aware answer-span loss mask, the tokenizer,
# and the Feistel data seed. See the ENV COMPATIBILITY CAVEAT above -- these are the
# executor-era imports that must be freshiris-importable on BOTH this driver and the
# GPU worker (answer_span_loss_weight cloudpickles across the fray boundary by
# module reference).
from refine_ft_common import (
    CONTACTS_V1_DATA_SEED,
    CONTACTS_V1_TOKENIZER,
    MODEL_CONFIG,
    answer_span_loss_weight,
)
from marinfold_models import SimpleTrainConfig, default_train

logger = logging.getLogger(__name__)

# iris PriorityBand enum value (iris/rpc/job.proto: PRIORITY_BAND_BATCH = 3). fray
# maps JobRequest.priority (int) straight to the iris band (iris_backend.submit():
# priority_band=request.priority).
IRIS_PRIORITY_BAND_BATCH = 3

# Fail loudly on the frozen 0.99.dev fray, whose JobRequest has no `priority` field,
# so `priority=3` would be silently dropped -> interactive band (which would disrupt
# the very interactive users batch priority protects). Same guard as exp108/exp112/
# dispatch_rollouts.
assert "priority" in {f.name for f in dataclasses.fields(JobRequest)}, (
    "This fray build lacks JobRequest.priority; batch-band dispatch requires the "
    "0.2.x.dev fray line (exp108/exp112 pins), not the frozen 0.99.dev build."
)

# Runtime-tuning env vars forwarded from the driver to the training gang. Iris tasks
# don't inherit the submitter's shell, and the gang runs in a SEPARATE pod from this
# driver, so anything passed to the driver via `iris job run -e XLA_FLAGS ...` (or
# NCCL_*/JAX_*) must be re-exported explicitly onto the gang (verbatim from exp108).
# JAX_PLATFORMS is excluded so the CPU driver's value can't leak onto the GPU gang.
_FORWARD_ENV_PREFIXES = ("XLA_FLAGS", "NCCL_", "JAX_", "LIBTPU_INIT_ARGS")
_FORWARD_ENV_EXCLUDE = ("JAX_PLATFORMS",)


def _forwarded_perf_env() -> dict[str, str]:
    return {
        k: v
        for k, v in os.environ.items()
        if k.startswith(_FORWARD_ENV_PREFIXES) and k not in _FORWARD_ENV_EXCLUDE
    }


# ---------------------------------------------------------------------------
# Storage -- everything under one exp163 S3 prefix (pods have no GCS; #163).
# ---------------------------------------------------------------------------
EXP163_S3_PREFIX = os.environ.get(
    "EXP163_S3_PREFIX", "s3://marin-us-east-02a/MarinFold/exp163"
)
# Train corpus: RAW refinement parquet (one `document` column). tokenize-on-the-fly.
REFINEMENT_CORPUS_GLOB = os.environ.get(
    "EXP163_CORPUS", f"{EXP163_S3_PREFIX}/val10k/refinement_corpus/*.parquet"
)
# Unmasked val = original contacts-v1 val (base-task retention + step-0 E8 anchor).
# exp108 already stages the contacts_v1 tree to this S3 path.
VAL_GLOB = os.environ.get(
    "EXP163_VAL",
    "s3://marin-us-east-02a/MarinFold/data/document_structures/contacts_v1/val/*.parquet",
)
# E8 warm-start -- LEVANTER-native training-state checkpoint dir (NOT the HF export).
# A continue-train (fresh step-0/optimizer/LR schedule/data loader), not a resume.
INIT_CHECKPOINT_S3 = os.environ.get(
    "EXP163_INIT_CKPT", f"{EXP163_S3_PREFIX}/model_levanter/step-35679"
)
# Token caches land here (workers build them on first read; auto_build_caches=True).
CACHE_BASE = f"{EXP163_S3_PREFIX}/tokenized"

# ---------------------------------------------------------------------------
# Device -- a single 8xH100 node on cw-rno2a, FSDP over the 8 GPUs (exp108's
# PROTEIN_RESOURCES_H100, verbatim). The 1.5B refiner shards comfortably on one
# node; scale out via EXP163_REPLICAS after a single-node smoke run is green.
# ---------------------------------------------------------------------------
PROTEIN_RESOURCES_H100 = ResourceConfig.with_gpu(
    "H100",
    count=8,          # GPUs per node (gd-8xh100ib-i128 = 8xH100)
    cpu=32,
    ram="256g",
    disk="256g",
    replicas=1,       # nodes; 1 node = 8 H100. Override via EXP163_REPLICAS.
)

# ---------------------------------------------------------------------------
# Recipe (from train_exp163.py): 1-epoch cosine, low LRs, batch 128, seq 8192.
# ---------------------------------------------------------------------------
SEQ_LEN = 8192
TRAIN_BATCH = int(os.environ.get("EXP163_TRAIN_BATCH", "128"))  # == Eric's #75 batch
WEIGHT_DECAY = 0.2
WARMUP = 0.1
MIN_LR_RATIO = 0.1
LR_SCHEDULE = "cosine"


def _steps_per_epoch_from_tokens(
    train_tokens: int, *, batch: int = TRAIN_BATCH, seq_len: int = SEQ_LEN
) -> int:
    """One full pass over the corpus: ceil(train_tokens / (batch * seq_len))."""
    return math.ceil(train_tokens / (batch * seq_len))


def _steps_per_epoch() -> int:
    """steps/epoch from EXP163_STEPS_PER_EPOCH, else derived from EXP163_TRAIN_TOKENS.

    Mirrors train_exp163.py: the mirrored refinement-corpus token count is not known
    at code time (tokenize-on-the-fly), so one of the two must be provided.
    """
    steps = os.environ.get("EXP163_STEPS_PER_EPOCH")
    if steps:
        return int(steps)
    tokens = os.environ.get("EXP163_TRAIN_TOKENS")
    if tokens:
        return _steps_per_epoch_from_tokens(int(tokens))
    raise SystemExit(
        "EXP163_STEPS_PER_EPOCH is required (or set EXP163_TRAIN_TOKENS and it is "
        "derived). steps/epoch = ceil(train_tokens / (128 * 8192)); print the "
        "refinement-corpus token count during the HF-bucket -> S3 mirror step."
    )


def _lr_tag(lr: float) -> str:
    """`1e-4`-style tag for run names (mirrors train_exp163.py / #75 naming)."""
    return f"{lr:.0e}".replace("e-0", "e-").replace("e+0", "e")


def _lrs() -> list[float]:
    return [float(x) for x in os.environ.get("EXP163_LRS", "1e-4,3e-4").split(",") if x.strip()]


def _epochs() -> int:
    return int(os.environ.get("EXP163_EPOCHS", "1"))


def run_name_for(lr: float, epochs: int) -> str:
    # W&B-safe (alnum + hyphens, < 64 chars).
    return f"plm-exp163-refine-cv1-1_5b-lr{_lr_tag(lr)}-e{epochs}-cos"


# ---------------------------------------------------------------------------
# Data -- tokenize-on-the-fly LM config; train answer-span MASKED, val UNMASKED.
# ---------------------------------------------------------------------------
def build_data_config(*, corpus_glob: str, val_glob: str) -> LmDataConfig:
    """Concrete-path, tokenize-on-the-fly LM data config for the refinement run.

    Mirrors exp108's ``build_data_config`` (raw-parquet ``UrlDatasetSourceConfig`` +
    concrete S3 ``cache_dir`` + ``auto_build_caches``), with the exp163 additions:
      * the TRAIN component carries ``loss_weight_fn=answer_span_loss_weight`` -- LM
        loss armed ONLY on each document's ``<begin_statements> ... <end>`` answer span
        (zeroed on the sequence header AND every ``<CAND>`` candidate block);
      * the VAL component is UNMASKED (every token contributes) -- the original
        contacts-v1 val, a base-task-retention monitor + step-0 E8 warm-start anchor.
    Caches land at ``<cache_dir>/train`` and ``<cache_dir>/validation``.
    """
    train_source = UrlDatasetSourceConfig(
        train_urls=[corpus_glob],
        validation_urls=[],
        cache_dir=f"{CACHE_BASE}/refinement-train",
        format=TextLmDatasetFormat(text_key="document"),
    )
    val_source = UrlDatasetSourceConfig(
        train_urls=[],
        validation_urls=[val_glob],
        cache_dir=f"{CACHE_BASE}/contacts-v1-val",
        format=TextLmDatasetFormat(text_key="document"),
    )
    # pack=True: never concat-and-split (partial protein docs are nonsensical). The
    # answer-span mask is packing-safe (two running cumsums; see refine_ft_common).
    train_component = DatasetComponent(
        source=train_source,
        cache_dir=train_source.cache_dir,
        format=train_source.format,
        pack=True,
        split="train",
        loss_weight_fn=answer_span_loss_weight,  # THE exp163 masked-loss addition
    )
    val_component = DatasetComponent(
        source=val_source,
        cache_dir=val_source.cache_dir,
        format=val_source.format,
        pack=True,
        split="validation",
        # no loss_weight_fn -> unmasked
    )
    return LmDataConfig(
        tokenizer=CONTACTS_V1_TOKENIZER,  # bare id: timodonnell/contacts-v1-tokenizer
        cache_dir=None,                   # each component sets its own concrete cache_dir
        auto_build_caches=True,           # build caches on the training workers
        shuffle=True,                     # full Feistel permutation (data_seed on the config)
        block_cross_document_attention=True,
        components={"refinement-train": train_component, "contacts-v1-val": val_component},
        train_weights={"refinement-train": 1.0, "contacts-v1-val": 0.0},  # val weight 0
    )


# ---------------------------------------------------------------------------
# On-pod training config -- SimpleTrainConfig + default_train, patched to a
# CONCRETE S3 output_path for direct dispatch (the executor is bypassed).
# ---------------------------------------------------------------------------
def build_on_pod_config(
    *,
    run_name: str,
    learning_rate: float,
    num_train_steps: int,
    output_path: str,
    corpus_glob: str,
    val_glob: str,
    init_checkpoint: str,
    resources: ResourceConfig,
    env_vars: dict[str, str] | None,
    steps_per_eval: int,
    steps_per_export: int,
    tags: tuple[str, ...],
) -> TrainLmOnPodConfig:
    """Build the ``TrainLmOnPodConfig`` for one LR via ``default_train`` + a
    ``SimpleTrainConfig`` (warm-start + masked loss), then inject a CONCRETE S3
    ``output_path`` so nothing routes through the executor.

    ``SimpleTrainConfig.initialize_from_checkpoint_path`` warm-starts model weights
    from the E8 Levanter checkpoint with a fresh step-0 / optimizer / LR schedule /
    data loader (``reset_data_loader_on_init`` default True -> a continue-train, not a
    resume), so ``learning_rate`` + ``WARMUP`` + ``LR_SCHEDULE`` define this run's
    schedule. ``pad_tokenizer_to_match_model=True`` pads the 2846-token tokenizer up
    to E8's TPU-padded 2848 vocab so the warm-started shapes match (verify at launch
    via the step-0 sanity eval ~ 2.7566).
    """
    train_config = SimpleTrainConfig(
        resources=resources,
        train_batch_size=TRAIN_BATCH,
        num_train_steps=num_train_steps,
        learning_rate=learning_rate,
        lr_schedule=LR_SCHEDULE,
        min_lr_ratio=MIN_LR_RATIO,
        weight_decay=WEIGHT_DECAY,
        warmup=WARMUP,
        train_seq_len=SEQ_LEN,
        steps_per_eval=steps_per_eval,
        steps_per_export=steps_per_export,
        max_eval_batches=None,  # eval the FULL val split each time (issue #163)
        data_seed=CONTACTS_V1_DATA_SEED,
        env_vars=env_vars,
        initialize_from_checkpoint_path=init_checkpoint,  # E8 Levanter ckpt (S3)
        pad_tokenizer_to_match_model=True,
    )

    # default_train returns an ExecutorStep whose `.config` is a TrainLmOnPodConfig
    # (see marinfold_models.defaults.default_train). We pass override_output_path so
    # the step's own path is concrete too, then patch the pod config below.
    step = default_train(
        name=run_name,
        tokenized=build_data_config(corpus_glob=corpus_glob, val_glob=val_glob),
        model_config=MODEL_CONFIG,
        train_config=train_config,
        tags=list(tags),
        eval_harness_tasks=[],
        use_default_validation=False,
        wandb_group="exp163-rollout-refinement",
        wandb_name=run_name,
        override_output_path=output_path,
    )
    pod_config: TrainLmOnPodConfig = step.config

    # Direct dispatch bypasses the executor's path resolution, so replace the lazy
    # `this_output_path()` placeholders with the concrete S3 output_path:
    #   * TrainLmOnPodConfig.output_path -- run_levanter_train_lm.apply_output_path
    #     derives the checkpointer base_path (+ hf_save_path) from it at runtime;
    #   * WandbConfig.replicate_path -- NOT touched by apply_output_path, so set it
    #     here (default_train wired it to this_output_path()).
    # Also force auto_build_caches so the workers build the tokenize-on-the-fly cache.
    inner = pod_config.train_config
    trainer = inner.trainer
    tracker = dataclasses.replace(trainer.tracker, replicate_path=output_path)
    inner = dataclasses.replace(inner, trainer=dataclasses.replace(trainer, tracker=tracker))
    pod_config = dataclasses.replace(
        pod_config, train_config=inner, output_path=output_path, auto_build_caches=True
    )
    return pod_config


# ---------------------------------------------------------------------------
# One batch-band JobRequest (built separately so DRY_RUN can print it un-submitted).
# ---------------------------------------------------------------------------
def build_request(
    *,
    run_name: str,
    learning_rate: float,
    num_train_steps: int,
    output_path: str,
    corpus_glob: str,
    val_glob: str,
    init_checkpoint: str,
    resources: ResourceConfig,
    replicas: int,
    env_vars: dict[str, str],
    steps_per_eval: int,
    steps_per_export: int,
    tags: tuple[str, ...],
    max_retries_failure: int = 3,
) -> JobRequest:
    """Assemble one batch-priority Fray ``JobRequest`` for a single LR gang."""
    # Merge driver-forwarded perf env (XLA_FLAGS/NCCL_/JAX_) under the explicit
    # env_vars (WANDB_*), which win on conflict (exp108).
    env_vars = {**_forwarded_perf_env(), **env_vars}

    on_pod_config = build_on_pod_config(
        run_name=run_name,
        learning_rate=learning_rate,
        num_train_steps=num_train_steps,
        output_path=output_path,
        corpus_glob=corpus_glob,
        val_glob=val_glob,
        init_checkpoint=init_checkpoint,
        resources=resources,
        env_vars=env_vars,
        steps_per_eval=steps_per_eval,
        steps_per_export=steps_per_export,
        tags=tags,
    )

    environment = create_environment(
        # resolve_training_env: hardware defaults + GIT_COMMIT + JAX compile cache.
        env_vars=resolve_training_env(base_env=dict(env_vars), resources=resources),
        extras=extras_for_resources(resources),  # GpuConfig -> ["gpu"] (marin-core[gpu])
    )

    return JobRequest(
        name=run_name,  # fray/iris-safe (alnum + hyphens)
        entrypoint=Entrypoint.from_callable(run_levanter_train_lm, args=[on_pod_config]),
        resources=resources,                 # with_gpu("H100", count=8); replicas set below too
        environment=environment,
        replicas=replicas,                   # nodes in the gang (freshiris JobRequest carries it)
        priority=IRIS_PRIORITY_BAND_BATCH,   # -> iris BATCH band (the whole point)
        processes_per_task=1,                # one JAX process driving all 8 local GPUs
        max_retries_failure=max_retries_failure,
    )


def main() -> None:
    lrs = _lrs()
    epochs = _epochs()
    spe = _steps_per_epoch()
    num_train_steps = spe * epochs
    _max_steps = os.environ.get("EXP163_MAX_STEPS")
    if _max_steps:
        num_train_steps = min(num_train_steps, int(_max_steps))

    replicas = int(os.environ.get("EXP163_REPLICAS", "1"))
    assert 1 <= replicas <= 4, f"EXP163_REPLICAS must be in [1, 4], got {replicas}"
    resources = PROTEIN_RESOURCES_H100
    if replicas != 1:
        resources = dataclasses.replace(PROTEIN_RESOURCES_H100, replicas=replicas)

    # W&B routing -- the pod does NOT inherit the launcher's shell, so forward the key
    # from the driver env (set at launch with `-e WANDB_API_KEY <key>`). Never hard-coded.
    env_vars: dict[str, str] = {"WANDB_ENTITY": "open-athena"}
    if os.environ.get("WANDB_API_KEY"):
        env_vars["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]

    # steps_per_eval / steps_per_export track train_exp163.py (quarter / half epoch).
    steps_per_eval = max(1, spe // 4)
    steps_per_export = max(1, spe // 2)

    print(
        f"[exp163] refiner fine-tune (direct batch dispatch): "
        f"LRs={[_lr_tag(l) for l in lrs]} wd={WEIGHT_DECAY} warmup={WARMUP} "
        f"sched={LR_SCHEDULE} batch={TRAIN_BATCH} seq={SEQ_LEN} "
        f"replicas={replicas} ({replicas * 8} H100/run) | "
        f"{spe} steps/epoch x {epochs} epoch(s) = {spe * epochs} steps"
        + (f" (capped to {num_train_steps} for smoke)" if _max_steps else "")
        + f" | {len(lrs)} job(s)\n"
        f"         warm-start (E8 Levanter): {INIT_CHECKPOINT_S3}\n"
        f"         corpus (masked train):    {REFINEMENT_CORPUS_GLOB}\n"
        f"         val (unmasked):           {VAL_GLOB}"
    )

    requests: list[JobRequest] = []
    for lr in lrs:
        name = run_name_for(lr, epochs)
        output_path = f"{EXP163_S3_PREFIX}/checkpoints/{name}"
        tags = (
            "protein", "contacts-v1", "qwen3", "1_5b", "answer-masked",
            "exp163", "refinement", "coreweave", f"e{epochs}", f"lr{_lr_tag(lr)}",
        )
        requests.append(
            build_request(
                run_name=name,
                learning_rate=lr,
                num_train_steps=num_train_steps,
                output_path=output_path,
                corpus_glob=REFINEMENT_CORPUS_GLOB,
                val_glob=VAL_GLOB,
                init_checkpoint=INIT_CHECKPOINT_S3,
                resources=resources,
                replicas=replicas,
                env_vars=env_vars,
                steps_per_eval=steps_per_eval,
                steps_per_export=steps_per_export,
                tags=tags,
            )
        )

    if os.environ.get("EXP163_DRY_RUN"):
        print("[exp163] DRY RUN -- JobRequests built, not submitting.")
        for req in requests:
            dev = req.resources.device
            print(
                f"  {req.name}: priority={req.priority} replicas={req.replicas} "
                f"gpu={getattr(dev, 'variant', '?')}x{getattr(dev, 'count', '?')} "
                f"cpu={req.resources.cpu} ram={req.resources.ram} disk={req.resources.disk} "
                f"steps={num_train_steps} extras={extras_for_resources(req.resources)} "
                f"entrypoint={type(req.entrypoint).__name__}"
            )
        return

    # Runs as an in-cluster CPU driver job -> current_client() is the controller.
    # CRITICAL: the training gangs are CHILDREN of this driver job. If the driver
    # exits first, iris finalizes (kills) them. So submit ALL first (they run
    # concurrently), then block on every one -- the driver must outlive the gangs.
    client = current_client()
    handles = []
    for req in requests:
        job = client.submit(req)
        handles.append((req.name, job))
        print(f"[exp163] submitted {req.name} (job_id={job.job_id})", flush=True)
    print(f"[exp163] submitted {len(handles)} gang(s) at iris batch priority; awaiting completion.")

    # Wait for every gang, reporting per-run status WITHOUT aborting siblings on one
    # failure. Fail the driver at the end if any run did not succeed (visible in iris).
    results: dict[str, JobStatus] = {}
    for name, job in handles:
        status = job.wait(raise_on_failure=False)
        results[name] = status
        print(f"[exp163] {name}: {status}", flush=True)

    failed = [name for name, st in results.items() if st != JobStatus.SUCCEEDED]
    if failed:
        raise SystemExit(f"[exp163] {len(failed)}/{len(handles)} run(s) did not succeed: {failed}")
    print(f"[exp163] all {len(handles)} run(s) SUCCEEDED -> checkpoints under {EXP163_S3_PREFIX}/checkpoints/")


if __name__ == "__main__":
    main()
