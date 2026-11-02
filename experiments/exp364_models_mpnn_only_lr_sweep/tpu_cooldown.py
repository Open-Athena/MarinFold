"""Independent v5p replica of the existing one-epoch cooldown fork.

The scientific configuration is assembled by cooldown_train.py. Only storage,
run identity, hardware microbatching, and the TPU attention kernel change. The
regional mirrors must be verified before submission; this entry point never
reads training data or initialization state from CoreWeave.
"""

import hashlib
import json
import os
import sys
import uuid
from dataclasses import replace
from functools import partial

import click
import fsspec
from fray.current_client import current_client
from fray.types import Entrypoint, JobRequest, ResourceConfig, create_environment
from iris.cluster.setup_scripts import default_setup_script
from levanter.layers.attention import AttentionBackend
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.cli import build_options
from marin.processing.tokenize.cache_stats import read_tokenized_cache_stats
from marin.training.training import (
    LevanterCheckpoint,
    TrainLmOnPodConfig,
    resolve_training_env,
    run_levanter_train_lm,
)

from experiments.exp277_models_single_mpnn_pilot.prepare import verify_cache
from experiments.exp364_models_mpnn_only_lr_sweep.config import CORPORA
from experiments.exp364_models_mpnn_only_lr_sweep.cooldown_train import (
    RUN_ID as GPU_RUN_ID,
)
from experiments.exp364_models_mpnn_only_lr_sweep.cooldown_train import (
    SOURCE_RUN_ID,
    SOURCE_STEP,
    build_cooldown_run,
)
from experiments.exp364_models_mpnn_only_lr_sweep.stage_tpu_inputs import PLAN_PATH
from experiments.exp364_models_mpnn_only_lr_sweep.verify_final import verify_export

REGION = "us-east5"
ZONE = "us-east5-a"
PREFIX = (
    "gs://marin-us-east5/protein-structure/MarinFold/exp364_models_mpnn_only_lr_sweep"
)
VALIDATION_CACHE = (
    "gs://marin-us-east5/protein-structure/MarinFold/exp232_train_trc/"
    "tokenized/contacts-v1-val/2026.07.25"
)
REGIONAL_CORPORA = tuple(
    replace(
        corpus,
        source="",
        cache=f"{PREFIX}/input-mirrors/exp277/{corpus.name}/2026.09.09.1",
    )
    for corpus in CORPORA
)
SOURCE_ROOT = f"{PREFIX}/fork-sources/{SOURCE_RUN_ID}"
RUN_ID = f"{GPU_RUN_ID}-tpu-{REGION}"
# v5p names count cores: v5p-64 is 32 chips on eight four-chip VMs.
# Microbatch one also fits the previously validated 5.86B v5p-64 model
# (exp363); global batch 128 is preserved by gradient accumulation.
DEFAULT_TPU = "v5p-64"
SUPPORTED_TPUS = ("v5p-8", "v5p-64")


def validate_inputs() -> None:
    """Refuse incomplete regional caches or an unverified checkpoint mirror."""
    for corpus in REGIONAL_CORPORA:
        if not verify_cache(corpus):
            raise ValueError(f"Missing regional cache: {corpus.cache}")
    stats = read_tokenized_cache_stats(VALIDATION_CACHE, "validation")
    if (stats.total_elements, stats.total_tokens) != (41_954, 47_821_958):
        raise ValueError(f"Regional validation cache changed: {stats}")
    source = f"{SOURCE_ROOT}/checkpoints/step-{SOURCE_STEP}"
    with fsspec.open(f"{source}/metadata.json", "rt") as handle:
        metadata = json.load(handle)
    if metadata["step"] != SOURCE_STEP:
        raise ValueError(f"Unexpected initialization step: {metadata}")
    # Written only after checking every copied object's bytes against the
    # pinned transfer inventory. Counts alone cannot prove a complete mirror.
    with fsspec.open(f"{PREFIX}/input-mirrors/verified.json", "rt") as handle:
        verification = json.load(handle)
    if (
        verification["source_step"] != SOURCE_STEP
        or not verification["all_verified"]
        or verification["plan_sha256"]
        != hashlib.sha256(PLAN_PATH.read_bytes()).hexdigest()
    ):
        raise ValueError(f"Regional mirror verification failed: {verification}")


def run_tpu_job(config: TrainLmOnPodConfig) -> None:
    """Submit a batch-priority TPU gang and wait for its final checkpoint."""
    env = resolve_training_env(config.env_vars, config.resources)
    env["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]
    smoke = "smoke" in config.train_config.trainer.tracker.tags
    handle = current_client().submit(
        JobRequest(
            name=f"exp364-tpu-cooldown-{uuid.uuid4().hex[:8]}",
            entrypoint=Entrypoint.from_callable(partial(run_levanter_train_lm, config)),
            resources=config.resources,
            environment=create_environment(
                extras=["tpu"],
                env_vars=env,
                setup_scripts=[
                    default_setup_script(
                        extras=["tpu"],
                        python_version=f"{sys.version_info.major}.{sys.version_info.minor}",
                    )
                ],
            ),
            priority=3,
            max_task_failures=0 if smoke else 20,
            max_retries_failure=0 if smoke else 5,
        )
    )
    handle.wait(raise_on_failure=True)
    final_step = config.train_config.trainer.num_train_steps - 1
    with fsspec.open(
        f"{config.output_path}/checkpoints/step-{final_step}/metadata.json", "rt"
    ) as handle:
        metadata = json.load(handle)
    if metadata["step"] != final_step:
        raise ValueError(f"TPU job did not finish at expected step {final_step}")
    verification = verify_export(f"{config.output_path}/hf/step-{final_step}")
    with fsspec.open(f"{config.output_path}/final_verification.json", "wt") as stream:
        json.dump(verification, stream, indent=2)


def build_run(
    *,
    smoke: bool,
    attempt: int = 1,
    tpu: str = DEFAULT_TPU,
    resume_step: int | None = None,
) -> ArtifactStep[LevanterCheckpoint]:
    """Keep the GPU fork's absolute step, data order, optimizer, and cooldown."""
    if tpu not in SUPPORTED_TPUS:
        raise ValueError(f"Unsupported v5p shape: {tpu}")
    if resume_step is not None and (smoke or resume_step != 186880):
        raise ValueError("Final-batch recovery must restore verified step 186880")
    parent = build_cooldown_run(
        smoke=smoke,
        nodes=4,
        attempt=attempt,
        corpora=REGIONAL_CORPORA,
        validation_cache=VALIDATION_CACHE,
        prefix=PREFIX,
        source_root=SOURCE_ROOT,
    )
    run_id = f"{RUN_ID}-smoke-a{attempt:02d}" if smoke else RUN_ID

    def build_config(ctx: StepContext) -> TrainLmOnPodConfig:
        if not ctx.is_fingerprint and ctx.prefix.rstrip("/") != PREFIX:
            raise ValueError(f"TPU execution requires regional prefix {PREFIX}")
        pod = parent.build_config(ctx)
        config = pod.train_config
        trainer = config.trainer
        tags = [
            tag
            for tag in trainer.tracker.tags
            if not tag.startswith(("cluster=", "gpu=", "nodes=", "trial="))
        ]
        trainer = replace(
            trainer,
            id=run_id,
            per_device_parallelism=1,
            per_device_eval_parallelism=1,
            tracker=replace(
                trainer.tracker,
                name=run_id,
                tags=[
                    *tags,
                    "trial=lr5e-4-1ep-cooldown-tpu",
                    "tpu-replica",
                    f"region={REGION}",
                    *([] if ctx.is_fingerprint else [f"tpu={tpu}"]),
                ],
            ),
        )
        if resume_step is not None:
            trainer = replace(
                trainer,
                load_checkpoint=True,
                load_checkpoint_path=(
                    f"{PREFIX}/runs/{RUN_ID}/checkpoints/step-{resume_step}"
                ),
            )
        return replace(
            pod,
            env_vars={
                "MARIN_PREFIX": PREFIX,
                "WANDB_ENTITY": "open-athena",
                "WANDB_PROJECT": "MarinFold",
            },
            train_config=replace(
                config,
                trainer=trainer,
                model=replace(config.model, attn_backend=AttentionBackend.SPLASH),
            ),
        )

    return replace(
        parent,
        name=f"runs/{run_id}",
        build_config=build_config,
        run=run_tpu_job,
        runtime_args={
            **parent.runtime_args,
            "train_resources": ResourceConfig.with_tpu(
                tpu,
                zone=ZONE,
                regions=[REGION],
                preemptible=True,
            ),
        },
        override_path=f"{PREFIX}/runs/{run_id}",
    )


@click.command()
@build_options
def main() -> ArtifactStep[LevanterCheckpoint]:
    if os.environ.get("MARIN_PREFIX") != PREFIX:
        raise ValueError(f"MARIN_PREFIX must be {PREFIX}")
    validate_inputs()
    return build_run(
        smoke=os.environ.get("SMOKE") == "1",
        attempt=int(os.environ["ATTEMPT"]),
        tpu=os.environ.get("TPU", DEFAULT_TPU),
        resume_step=int(os.environ["RESUME_STEP"])
        if "RESUME_STEP" in os.environ
        else None,
    )


if __name__ == "__main__":
    main()
