"""Fork the 5e-4 run's full state and finish exactly its first data epoch."""

import os
from dataclasses import replace

import click
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.cli import build_options
from marin.training.training import LevanterCheckpoint, TrainLmOnPodConfig

from experiments.exp277_models_single_mpnn_pilot.epoch_data import one_epoch_data
from experiments.exp277_models_single_mpnn_pilot.prepare import verify_cache
from experiments.exp364_models_mpnn_only_lr_sweep.config import (
    CORPORA,
    EPOCH_PACKED_EXAMPLES,
    PREFIX,
    TRAIN_STEPS,
    VERSION,
    run_id_for,
)
from experiments.exp364_models_mpnn_only_lr_sweep.train import build_run

SOURCE_STEP = 139_053
SOURCE_RUN_ID = run_id_for("lr5e-4")
SOURCE_ROOT = f"{PREFIX}/fork-sources/{SOURCE_RUN_ID}"
RUN_ID = "contacts-v1-exp364-mpnn-only-1ep-lr5e-4-cooldown-1.5B"
EPOCH_STEPS = (EPOCH_PACKED_EXAMPLES + 127) // 128


def build_cooldown_run(
    *, smoke: bool, nodes: int, attempt: int = 1
) -> ArtifactStep[LevanterCheckpoint]:
    """Keep the parent's recipe and data prefix, changing only its future LR/end.

    The source is an immutable, same-region copy of a rolling checkpoint. Full
    trainer restoration preserves the optimizer, RNG, and absolute data offset.
    The original augmentation ramp and warmup duration remain unchanged. A smoke
    uses ten additional updates with an accelerated ten-update cooldown.
    """
    parent = build_run(trial="lr5e-4", smoke=False, nodes=nodes, attempt=attempt)
    run_id = f"{RUN_ID}-smoke-a{attempt:02d}" if smoke else RUN_ID
    end_step = SOURCE_STEP + 1 + 10 if smoke else EPOCH_STEPS
    source = ArtifactStep[LevanterCheckpoint].adopt(
        "input/one-epoch-cooldown/source-from139053",
        VERSION,
        source=SOURCE_ROOT,
        kind=LevanterCheckpoint,
        config={"source_run": SOURCE_RUN_ID, "checkpoint_step": SOURCE_STEP},
    )

    def build_config(ctx: StepContext) -> TrainLmOnPodConfig:
        pod = parent.build_config(ctx)
        config = pod.train_config
        trainer = config.trainer
        # Converting the parent's fraction to an absolute duration preserves
        # its historical schedule when shortening the total training horizon.
        optimizer = replace(
            config.optimizer,
            warmup=int(config.optimizer.warmup * TRAIN_STEPS),
            decay=10 if smoke else int(config.optimizer.decay * EPOCH_STEPS),
        )
        source_dir = LevanterCheckpoint(path=ctx.artifact_path(source)).checkpoint_dir
        tags = [
            tag
            for tag in trainer.tracker.tags
            if tag
            not in {"scratch", "mpnn-only-two-epochs", "trial=lr5e-4", "production"}
        ]
        trainer = replace(
            trainer,
            id=run_id,
            tracker=replace(
                trainer.tracker,
                name=run_id,
                tags=[
                    *tags,
                    "mpnn-only-one-epoch-cooldown",
                    "trial=lr5e-4-1ep-cooldown",
                    f"source-step={SOURCE_STEP}",
                    "smoke" if smoke else "production",
                ],
            ),
            num_train_steps=end_step,
            initialize_from=f"{source_dir}/step-{SOURCE_STEP}",
            allow_partial_checkpoint=False,
            max_eval_batches=2 if smoke else trainer.max_eval_batches,
            steps_per_eval=10 if smoke else trainer.steps_per_eval,
            checkpointer=replace(
                trainer.checkpointer,
                keep=[] if smoke else [{"every": EPOCH_STEPS // 10}],
            ),
        )
        return replace(
            pod,
            train_config=replace(
                config,
                trainer=trainer,
                optimizer=optimizer,
                data=one_epoch_data(
                    config.data,
                    num_train_steps=TRAIN_STEPS,
                    expected_packed_examples=EPOCH_PACKED_EXAMPLES,
                ),
                hf_save_steps=end_step + 1,
            ),
        )

    return replace(
        parent,
        name=f"runs/{run_id}",
        deps=(*parent.deps, source),
        build_config=build_config,
        override_path=f"{PREFIX}/runs/{run_id}",
    )


@click.command()
@build_options
def main() -> ArtifactStep[LevanterCheckpoint]:
    for corpus in CORPORA:
        if not verify_cache(corpus):
            raise ValueError(f"Incomplete cache: {corpus.cache}")
    return build_cooldown_run(
        smoke=os.environ.get("SMOKE") == "1",
        nodes=int(os.environ["NODES"]),
        attempt=int(os.environ["ATTEMPT"]),
    )


if __name__ == "__main__":
    main()
