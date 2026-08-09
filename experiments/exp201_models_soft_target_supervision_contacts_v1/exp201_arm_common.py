# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""exp201 Phase 1b: shared training recipe for the masked arm and its control.

Two arms that differ in **exactly one thing** — whether the sequence-statement
head slots contribute to the training loss:

* ``control`` — a plain ``qwen3``; #117's recipe, unchanged.
* ``masked``  — ``Qwen3StatementHeadMaskedConfig``, which drops 23.7 % of
  supervised slots (1.17 nats/token, 43 % of the training loss) that are a
  uniform draw over the not-yet-emitted sequence statements. See #201.

Everything else is byte-identical between them: same model shape, same token
cache, same shuffle, same ``data_seed``, same optimizer. The control is re-run
here rather than read off #150's curve so the training *harness* is not a
confound — #150 drove marin's older ``ExecutorStep`` / ``default_train`` path,
this drives ``build_train_lm_on_pod_config`` with direct Fray dispatch.

**Recipe provenance.** Every constant below is #117's, via
[exp150](../exp150_models_reproduce_eric_contacts_v1/contacts_v1_repro_common.py),
which reproduced his 2.7112. Values #117 leaves to a library default are stated
explicitly at that default, so a future library change cannot silently move the
recipe.

**The token cache is #150's, reused.** Both arms read
``exp150_reproduce_eric_contacts_v1/tokenized/...`` — verified complete
(``is_finished: true``; 4,129,682 train rows, 41,954 val rows). Reusing it means
both arms see exactly the data #117/#150 trained on, costs no re-tokenization,
and removes any chance of the two arms disagreeing on their corpus.
``auto_build_caches`` is therefore **off**: if a cache is ever missing we want a
loud failure, not a silent rebuild into another experiment's prefix.
"""

import os

from fray.types import ResourceConfig
from levanter.data.text.datasets import (
    BlockShuffleConfig,
    DatasetComponent,
    LmDataConfig,
    UrlDatasetSourceConfig,
)
from levanter.data.text.formats import TextLmDatasetFormat
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.models.lm_model import LmConfig
from levanter.models.qwen import Qwen3Config
from levanter.optim.config import AdamConfig

from marin.training.training import run_levanter_train_lm

from marinfold_models import (
    Qwen3StatementHeadMaskedConfig,
    build_train_lm_on_pod_config,
)

ARMS = ("control", "masked")

# --- Storage (us-east5: where exp53's corpus and #150's caches live) ---------
_BUCKET = os.environ.get("EXP201_BUCKET", "gs://marin-us-east5").rstrip("/")
_ROOT = f"{_BUCKET}/protein-structure/MarinFold"
OUTPUT_PREFIX = f"{_ROOT}/exp201_soft_target_supervision"

# #150's completed token caches. The trailing hash is part of the path marin's
# executor minted; it is load-bearing, not decoration.
_EXP150 = f"{_ROOT}/exp150_reproduce_eric_contacts_v1/tokenized"
TRAIN_CACHE_DIR = f"{_EXP150}/contacts-v1-train-dfe81b"
VAL_CACHE_DIR = f"{_EXP150}/contacts-v1-val-92827b"
TRAIN_CACHE_ROWS = 4_129_682
VAL_CACHE_ROWS = 41_954

# Raw sources the caches were built from. Recorded for provenance; with
# auto_build_caches off nothing reads them at train time.
_DATA_PREFIX = f"{_ROOT}/exp53_contacts_v1_5x/documents"
TRAIN_GLOB = f"{_DATA_PREFIX}/train/*.parquet"
VAL_GLOB = f"{_DATA_PREFIX}/val/*.parquet"

# Bare repo id: MarinFold's tokenizer-load path rejects `repo@rev`
# (huggingface_hub's validate_repo_id -- the recurring exp85/exp120 gotcha).
CONTACTS_V1_TOKENIZER = "timodonnell/contacts-v1-tokenizer"

TRAIN_COMPONENT_KEY = "contacts-v1-train"
VAL_COMPONENT_KEY = "contacts-v1-val"

# --- #117 recipe constants ---------------------------------------------------
SEQ_LEN = 8192
TRAIN_BATCH = 128
DATA_SEED = 0
WARMUP = 0.1
LR_SCHEDULE = "cosine"
WEIGHT_DECAY = 0.2
MIN_LR_RATIO = 0.1
BASE_LR = 3.1623e-3          # #117's tuned peak LR; the sweep is multiples of this
NUM_EVALS_PER_EPOCH = 2
TRAIN_TOKENS = 4_676_753_425  # #117 TRAIN_TOKENS; steps/epoch derive from it

# #117's hierarchical Feistel BLOCK shuffle -- NOT `shuffle=True`, which levanter
# routes to a FULL permutation. A real divergence on this pLDDT-ordered corpus,
# and the discrepancy exp137 found and exp150 pinned down.
SHUFFLE = BlockShuffleConfig(io_block_size=256, window_blocks=512, perm_type="feistel")

_MODEL_KWARGS = dict(
    max_seq_len=SEQ_LEN,
    hidden_dim=2048,
    intermediate_dim=8192,
    num_heads=32,
    num_kv_heads=8,
    num_layers=24,
    rope=Llama3RotaryEmbeddingsConfig(),
)

# contacts-v1 token ids the mask keys off. Defaults in
# ``marinfold_models.loss_masks``; passed explicitly so the value that trained a
# checkpoint is recorded in its config, and verified against the real tokenizer
# by ``verify_mask.py``.
BEGIN_SEQUENCE_ID = 8
BEGIN_STRUCTURE_ID = 9
END_ID = 10

# --- TPU slice ---------------------------------------------------------------
# Must be co-located with us-east5 (marin blocks cross-region training). v5p
# gangs only register autoscaler demand under the REGION form -- a zone= request
# sits in coscheduling with Demand=0 forever (exp150).
TPU_TYPE = os.environ.get("EXP201_TPU", "v5p-8")
TPU_SLICES = int(os.environ.get("EXP201_SLICES", "1"))
TPU_REGION = os.environ.get("EXP201_REGION", "us-east5").lower()


def build_resources(
    tpu_type: str = TPU_TYPE,
    slices: int = TPU_SLICES,
    region: str = TPU_REGION,
) -> ResourceConfig:
    """The TPU slice for one arm.

    Built by a function rather than held as a module constant because this module
    is imported on BOTH sides -- the launcher (to size the ``JobRequest``) and the
    training pod (to assemble its own config). Each side constructs it with its
    own fray.

    #163 measured that the 1.5B at batch 128 x seq 8192 fits a **v5p-8**, which is
    also what places: a v5p-128 request never registered autoscaler demand and the
    scheduler spent an hour trying to squeeze the 16-host gang onto an idle v5p-8.
    """
    return ResourceConfig.with_tpu(
        tpu_type,
        slice_count=slices,
        cpu=32,
        ram="128g",
        disk="50g",
        regions=[region],
    )


PROTEIN_RESOURCES = build_resources()


def steps_per_epoch(batch_size: int = TRAIN_BATCH, seq_len: int = SEQ_LEN) -> int:
    """One pass over the train corpus -- #117's ``round(TRAIN_TOKENS / (bs * seq))``.

    At bs128 / seq8192 this is **4,460**. Cross-check: #75's 8-epoch run ended at
    ``step-35679``, i.e. 35,680 = 8 x 4,460.
    """
    return round(TRAIN_TOKENS / (batch_size * seq_len))


def steps_for_epochs(epochs: int, batch_size: int = TRAIN_BATCH) -> int:
    """``epochs * steps_per_epoch`` -- #117's ``Point.num_train_steps``."""
    return epochs * steps_per_epoch(batch_size)


def evals_per_epoch_steps(batch_size: int = TRAIN_BATCH) -> int:
    """#117's ``round(steps_per_epoch / 2)`` -> 2,230."""
    return max(1, round(steps_per_epoch(batch_size) / NUM_EVALS_PER_EPOCH))


def model_config(arm: str) -> LmConfig:
    """The only thing that differs between the two arms.

    Args:
        arm: ``"control"`` or ``"masked"``.

    Returns:
        A Qwen3 1.47B config; the masked arm's subclass overrides only
        ``compute_next_token_loss``, so the two are architecturally identical and
        their checkpoints are interchangeable.
    """
    if arm == "control":
        return Qwen3Config(**_MODEL_KWARGS)
    if arm == "masked":
        return Qwen3StatementHeadMaskedConfig(
            **_MODEL_KWARGS,
            begin_sequence_id=BEGIN_SEQUENCE_ID,
            begin_structure_id=BEGIN_STRUCTURE_ID,
            end_id=END_ID,
        )
    raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")


def build_data_config() -> LmDataConfig:
    """#150's completed caches, read-only, packed prefix-only.

    ``pack=True`` on every component is #117's ``_apply_recipe_overrides``:
    documents are never concat-and-split, so a training window never contains a
    partial protein.
    """
    fmt = TextLmDatasetFormat(text_key="document")

    def component(cache_dir: str, urls: list[str], split: str) -> DatasetComponent:
        source = UrlDatasetSourceConfig(
            train_urls=urls if split == "train" else [],
            validation_urls=urls if split == "validation" else [],
            cache_dir=cache_dir,
            format=fmt,
        )
        return DatasetComponent(
            source=source, cache_dir=cache_dir, format=fmt, pack=True, split=split
        )

    return LmDataConfig(
        tokenizer=CONTACTS_V1_TOKENIZER,
        cache_dir=None,  # each component names its own concrete cache
        auto_build_caches=False,  # caches exist; fail loudly rather than rebuild
        shuffle=SHUFFLE,
        block_cross_document_attention=True,
        components={
            TRAIN_COMPONENT_KEY: component(TRAIN_CACHE_DIR, [TRAIN_GLOB], "train"),
            VAL_COMPONENT_KEY: component(VAL_CACHE_DIR, [VAL_GLOB], "validation"),
        },
        train_weights={TRAIN_COMPONENT_KEY: 1.0, VAL_COMPONENT_KEY: 0.0},
    )


def build_on_pod_config(
    *,
    arm: str,
    run_name: str,
    learning_rate: float,
    num_train_steps: int,
    output_path: str,
    resources=PROTEIN_RESOURCES,
    env_vars: dict[str, str] | None = None,
    steps_per_eval: int | None = None,
    steps_per_checkpoint: int | None = None,
    tags: tuple[str, ...] = (),
):
    """Assemble the ``TrainLmOnPodConfig`` for one arm at one learning rate.

    Optimizer values #117 leaves to levanter's ``AdamConfig`` defaults -- betas
    0.9/0.95, epsilon 1e-8, max_grad_norm 1.0 -- are levanter defaults and are
    not restated here; ``min_lr_ratio``, ``warmup`` and the schedule are set
    because #117 sets them.
    """
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")
    if steps_per_eval is None:
        steps_per_eval = evals_per_epoch_steps()
    if steps_per_checkpoint is None:
        steps_per_checkpoint = steps_per_epoch()

    optimizer = AdamConfig(
        learning_rate=learning_rate,
        weight_decay=WEIGHT_DECAY,
        warmup=WARMUP,
        min_lr_ratio=MIN_LR_RATIO,
        lr_schedule=LR_SCHEDULE,
    )

    return build_train_lm_on_pod_config(
        run_name=run_name,
        model=model_config(arm),
        optimizer=optimizer,
        data=build_data_config(),
        resources=resources,
        output_path=output_path,
        num_train_steps=num_train_steps,
        train_batch_size=TRAIN_BATCH,
        seq_len=SEQ_LEN,
        auto_build_caches=False,  # OVERRIDES LmDataConfig.auto_build_caches
        steps_per_eval=steps_per_eval,
        max_eval_batches=None,  # #117 pins the FULL val split
        steps_per_checkpoint=steps_per_checkpoint,
        z_loss_weight=None,  # #117 train_lm(z_loss_weight=None)
        data_seed=DATA_SEED,
        wandb_project="MarinFold",
        wandb_group="exp201-statement-head-mask",
        wandb_name=run_name,
        tags=("protein", "contacts-v1", "qwen3", "exp201", f"arm-{arm}", *tags),
        env_vars=env_vars,
    )


def train_arm_on_pod(
    *,
    arm: str,
    run_name: str,
    learning_rate: float,
    num_train_steps: int,
    output_path: str,
    tpu_type: str,
    tpu_slices: int,
    tpu_region: str,
    steps_per_eval: int,
    steps_per_checkpoint: int,
    env_vars: dict[str, str],
    tags: tuple[str, ...],
) -> None:
    """Assemble the training config **on the pod**, then run it.

    This is the job entrypoint, and every argument is a primitive on purpose.
    iris cloudpickles ``(fn, args, kwargs)``; passing an assembled
    ``TrainLmOnPodConfig`` instead would mean the levanter that *builds* it and
    the levanter that *loads* it have to agree, and they cannot: iris rejects a
    ``marin-iris`` client older than 14 days, which forces the launcher onto
    marin ``origin/main``, while the pod installs ``marin-levanter`` from PyPI,
    whose newest release under the ``<0.3`` pin is seven weeks older. That skew
    failed with ``AttributeError: Can't get attribute 'XprofUploadConfig'``
    before the trainer ever started.

    Because this function lives in a module (not ``__main__``) that iris bundles
    into the workspace, cloudpickle stores it **by reference** — so the pod
    imports this file fresh and builds the config with its own libraries. Only
    strings and numbers cross the wire.
    """
    config = build_on_pod_config(
        arm=arm,
        run_name=run_name,
        learning_rate=learning_rate,
        num_train_steps=num_train_steps,
        output_path=output_path,
        resources=build_resources(tpu_type, tpu_slices, tpu_region),
        env_vars=dict(env_vars),
        steps_per_eval=steps_per_eval,
        steps_per_checkpoint=steps_per_checkpoint,
        tags=tuple(tags),
    )
    run_levanter_train_lm(config)


__all__ = [
    "ARMS",
    "BASE_LR",
    "OUTPUT_PREFIX",
    "PROTEIN_RESOURCES",
    "SEQ_LEN",
    "TRAIN_BATCH",
    "TRAIN_CACHE_DIR",
    "VAL_CACHE_DIR",
    "build_data_config",
    "build_resources",
    "build_on_pod_config",
    "evals_per_epoch_steps",
    "model_config",
    "steps_for_epochs",
    "steps_per_epoch",
    "train_arm_on_pod",
]
