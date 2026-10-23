"""Exercise full-state restoration and the fork's absolute LR/data timeline."""

import json
import runpy
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import cloudpickle
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from levanter.checkpoint import load_checkpoint, save_checkpoint
from marin.execution.build_context import BuildContext, VersionCodex, build_context
from marin.execution.lazy import StepContext
from marin.training.training import TrainLmOnPodConfig, apply_output_path

from experiments.exp277_models_single_mpnn_pilot.config import EPOCH_TRAIN_STEPS
from experiments.exp363_models_double_channel_width.config import PREFIX, VERSION
from experiments.exp363_models_double_channel_width.fork_optimizer import (
    FINAL_COOLDOWN_START,
    RESUME_STEP,
    SOURCE_CHECKPOINT_STEP,
    TARGET_LR,
    TRANSITION_STEPS,
    ForkAdamConfig,
)
from experiments.exp363_models_double_channel_width.fork_train import (
    FORK_RUN_ID,
    SOURCE_CHECKPOINT,
    build_fork_run,
)
from experiments.exp363_models_double_channel_width.test_train import resolved


def resolved_fork(*, smoke: bool = False) -> TrainLmOnPodConfig:
    with build_context(BuildContext(VersionCodex(VERSION))):
        step = build_fork_run(nodes=4, smoke=smoke)
        step.fingerprint()
        ctx = StepContext.for_run(
            output_path=step.path(PREFIX), prefix=PREFIX,
            runtime_args=step.runtime_args, deps=step.deps,
        )
        pod = step.build_config(ctx)
        return replace(pod, train_config=apply_output_path(pod.train_config, pod.output_path))


def test_fork_preserves_data_contract_and_isolates_output() -> None:
    source = resolved(smoke=False, nodes=4)
    fork = resolved_fork()
    c = fork.train_config
    assert asdict(c.data) == asdict(source.train_config.data)
    assert c.model == source.train_config.model
    assert c.data_seed == source.train_config.data_seed
    assert c.trainer.num_train_steps == EPOCH_TRAIN_STEPS
    assert c.trainer.train_batch_size == source.train_config.trainer.train_batch_size
    assert c.trainer.initialize_from == SOURCE_CHECKPOINT
    assert c.trainer.load_checkpoint is None
    assert not c.trainer.allow_partial_checkpoint
    assert c.initialize_from_checkpoint_path is c.initialize_model_from_checkpoint_path is None
    assert c.trainer.id == FORK_RUN_ID
    assert fork.output_path != source.output_path
    assert c.trainer.checkpointer.base_path == f"{fork.output_path}/checkpoints"
    assert c.trainer.checkpointer.base_path not in SOURCE_CHECKPOINT
    assert c.trainer.tracker.project == "MarinFold"


def test_schedule_reaches_target_then_keeps_original_final_cooldown() -> None:
    config = resolved_fork().train_config.optimizer
    schedule = config.lr_scheduler(EPOCH_TRAIN_STEPS)
    assert float(schedule(RESUME_STEP)) == pytest.approx(0.001)
    assert float(schedule(RESUME_STEP + TRANSITION_STEPS // 2)) == pytest.approx(0.0008535)
    assert float(schedule(RESUME_STEP + TRANSITION_STEPS)) == pytest.approx(TARGET_LR)
    assert float(schedule(FINAL_COOLDOWN_START - 1)) == pytest.approx(TARGET_LR)
    assert float(schedule(FINAL_COOLDOWN_START)) == pytest.approx(TARGET_LR)
    assert float(schedule(FINAL_COOLDOWN_START + 1000)) < TARGET_LR
    assert float(schedule(EPOCH_TRAIN_STEPS)) == pytest.approx(TARGET_LR * 0.1)
    smoke = resolved_fork(smoke=True).train_config
    assert smoke.trainer.num_train_steps == RESUME_STEP + 20
    assert smoke.data.augmentation_num_train_steps == EPOCH_TRAIN_STEPS
    assert float(smoke.optimizer.lr_scheduler(smoke.trainer.num_train_steps)(RESUME_STEP)) == pytest.approx(0.001)


def test_checkpoint_restores_adam_moments_and_uses_new_schedule(tmp_path: Path) -> None:
    source_config = resolved(smoke=False, nodes=4).train_config.optimizer
    fork_config = ForkAdamConfig(**vars(source_config))
    source = source_config.build(EPOCH_TRAIN_STEPS)
    fork = fork_config.build(EPOCH_TRAIN_STEPS)
    params = {"weight": jnp.ones((2, 2))}
    grads = {"weight": jnp.full((2, 2), 0.25)}

    @jax.jit
    def advance(optimizer_state):
        def update(_, state):
            return source.update(grads, state, params)[1]
        return jax.lax.fori_loop(0, RESUME_STEP, update, optimizer_state)

    old_state = advance(source.init(params))
    tree = {
        "model": params, "opt_state": old_state, "step": jnp.array(RESUME_STEP),
        "training_key": jax.random.PRNGKey(19),
    }
    path = str(tmp_path / "source")
    save_checkpoint(tree, step=SOURCE_CHECKPOINT_STEP, checkpoint_path=path, is_temporary=False)
    exemplar = {**tree, "opt_state": fork.init(params), "step": jnp.array(0)}
    restored = load_checkpoint(exemplar, path)
    for expected, actual in zip(jax.tree.leaves(tree), jax.tree.leaves(restored), strict=True):
        np.testing.assert_array_equal(expected, actual)
    assert int(restored["step"]) == RESUME_STEP

    @jax.jit
    def run_fork(state):
        def update(carry, _):
            updates, new_state = fork.update(grads, carry, params)
            return new_state, (new_state.hyperparams["learning_rate"], updates["weight"])
        return jax.lax.scan(update, state, None, length=TRANSITION_STEPS + 1)

    final, (rates, updates) = run_fork(restored["opt_state"])
    assert float(rates[0]) == pytest.approx(0.001)
    assert float(rates[-1]) == pytest.approx(TARGET_LR)
    assert np.all(np.diff(rates) <= 0)
    assert int(final.count) == RESUME_STEP + TRANSITION_STEPS + 1
    np.testing.assert_allclose(updates[-1] / updates[0], TARGET_LR / 0.001, rtol=1e-5)
    # A later restart uses the fork's saved schedule counter, not its source offset.
    restart_path = str(tmp_path / "fork")
    save_checkpoint({"opt_state": final}, step=int(final.count) - 1, checkpoint_path=restart_path)
    resumed = load_checkpoint({"opt_state": fork.init(params)}, restart_path)["opt_state"]
    _, next_state = fork.update(grads, resumed, params)
    assert float(next_state.hyperparams["learning_rate"]) == pytest.approx(TARGET_LR)
    assert int(next_state.count) == int(final.count) + 1


def test_entrypoint_optimizer_serializes_in_remote_process() -> None:
    driver = runpy.run_path(
        str(Path(__file__).with_name("fork_train.py")),
        run_name="remote_driver_entrypoint",
    )
    source = resolved(smoke=False, nodes=4).train_config.optimizer
    optimizer = driver["ForkAdamConfig"](**vars(source))
    result = subprocess.run(
        [
            sys.executable, "-c",
            "import cloudpickle, draccus, json, sys; "
            "from levanter.optim.config import OptimizerConfig; "
            "config = cloudpickle.loads(sys.stdin.buffer.read()); "
            "encoded = draccus.encode(config, OptimizerConfig); "
            "assert draccus.decode(OptimizerConfig, encoded) == config; "
            "print(json.dumps(encoded))",
        ],
        input=cloudpickle.dumps(optimizer), capture_output=True, check=True,
    )
    encoded = json.loads(result.stdout)
    assert encoded["type"] == "exp363-fork-adam"
    assert encoded["target_learning_rate"] == TARGET_LR
    assert encoded["source_resume_step"] == RESUME_STEP
