"""Fork full optimizer/data state and reduce LR without moving the final cooldown."""

import os
from dataclasses import replace

import click
from marin.execution.lazy import ArtifactStep
from marin.experiment.cli import build_options
from marin.training.training import LevanterCheckpoint

from experiments.exp277_models_single_mpnn_pilot.config import CORPORA, EPOCH_TRAIN_STEPS
from experiments.exp277_models_single_mpnn_pilot.prepare import verify_cache
from experiments.exp363_models_double_channel_width.config import PREFIX, RUN_ID
from experiments.exp363_models_double_channel_width.fork_optimizer import (
    RESUME_STEP,
    SOURCE_CHECKPOINT_STEP,
    TARGET_LR,
    ForkAdamConfig,
)
from experiments.exp363_models_double_channel_width.train import build_run


SOURCE_CHECKPOINT = f"{PREFIX}/fork_sources/{RUN_ID}/step-{SOURCE_CHECKPOINT_STEP}"
FORK_RUN_ID = f"{RUN_ID}-fork{SOURCE_CHECKPOINT_STEP}-lr7p07e-4"


def build_fork_run(
    *, nodes: int, smoke: bool = False, attempt: int = 1,
) -> ArtifactStep[LevanterCheckpoint]:
    """Restore full state with the same finite corpus, seeds and data offset."""
    run_id = f"{FORK_RUN_ID}-smoke-a{attempt:02d}" if smoke else FORK_RUN_ID
    step = build_run(smoke=False, nodes=nodes, run_id=run_id)
    base_build_config = step.build_config

    def build_config(ctx):
        pod = base_build_config(ctx)
        config = pod.train_config
        optimizer = ForkAdamConfig(**vars(config.optimizer))
        end_step = RESUME_STEP + 20 if smoke else EPOCH_TRAIN_STEPS
        tracker = replace(
            config.trainer.tracker,
            tags=[
                "exp363", "double-channel-width", "lr-fork", "full-state-resume",
                f"source-step={SOURCE_CHECKPOINT_STEP}", f"target-lr={TARGET_LR}",
                "smoke" if smoke else "production", f"nodes={nodes}",
            ],
        )
        trainer = replace(
            config.trainer,
            initialize_from=SOURCE_CHECKPOINT,
            allow_partial_checkpoint=False,
            num_train_steps=end_step,
            steps_per_eval=end_step if smoke else config.trainer.steps_per_eval,
            max_eval_batches=2 if smoke else None,
            tracker=tracker,
        )
        # TrainerConfig.initialize_from retains the absolute step/data cursor.
        # initialize_from_checkpoint_path in train_lm would reset that cursor.
        config = replace(config, optimizer=optimizer, trainer=trainer, hf_save_steps=end_step + 1)
        return replace(pod, train_config=config)

    return replace(step, build_config=build_config)


@click.command()
@build_options
def main() -> ArtifactStep[LevanterCheckpoint]:
    for corpus in CORPORA:
        if not verify_cache(corpus):
            raise ValueError(f"Incomplete cache: {corpus.cache}")
    return build_fork_run(
        nodes=int(os.environ["NODES"]),
        smoke=os.environ.get("SMOKE") == "1",
        attempt=int(os.environ.get("ATTEMPT", "1")),
    )


if __name__ == "__main__":
    main()
