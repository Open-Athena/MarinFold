"""Full-state LR forks, matched data, and independently checked diagnostics."""

import json
import subprocess
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import haliax as hax
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import pytest
from conftest import make_document
from levanter.checkpoint import CheckpointerConfig, save_checkpoint
from levanter.data.dataset import ListAsyncDataset
from levanter.data.text.datasets import DatasetComponent
from levanter.data.text.formats import TextLmDatasetFormat
from levanter.main.train_lm import TrainLmConfig
from levanter.main.train_lm import main as train_main
from levanter.tracker.tracker import NoopConfig, NoopTracker, TrackerConfig
from levanter.trainer import Trainer
from test_data import make_cache
from test_loss import example_and_oracle
from test_trainer import assert_state_equal, model_loss, tiny_config, trainer_config

from experiments.exp279_models_exact_soft_contact_targets.checkpoints import (
    validate_restore,
)
from experiments.exp279_models_exact_soft_contact_targets.launch_lr import trial_request
from experiments.exp279_models_exact_soft_contact_targets.data import ContactDataConfig
from experiments.exp279_models_exact_soft_contact_targets.inputs import (
    resolve_tokenizer,
)
from experiments.exp279_models_exact_soft_contact_targets.lr_metrics import (
    LrWatchConfig,
    endpoint_example,
    gradient_metrics,
)
from experiments.exp279_models_exact_soft_contact_targets.lr_trial import (
    ContinuationAdamConfig,
    RATES,
    START,
    STOP,
    build_trial_config,
    select_resume,
)
from experiments.exp279_models_exact_soft_contact_targets.recipe import (
    PHASES,
    optimizer_for_phase,
)


def test_full_state_forks_preserve_moments_rng_data_and_apply_lr_immediately(
    tmp_path, vocab
):
    examples = [
        example_and_oracle(
            vocab, edges=((143, 144),) if i % 2 else ((144, 145), (143, 145))
        )[0]
        for i in range(24)
    ]
    dataset = ListAsyncDataset(examples)
    model = tiny_config().build(
        hax.Axis("vocab", vocab.size), key=jax.random.PRNGKey(4)
    )
    base = optimizer_for_phase(PHASES["base"])
    checkpoint = str(tmp_path / "step-1")
    with Trainer(
        trainer_config(tmp_path),
        base.build(217801),
        model_loss,
        add_default_hooks=False,
    ) as trainer:
        state = trainer.initial_state(jax.random.PRNGKey(5), model=model)
        iterator = trainer.data_loader(dataset).iter_from_step(0)
        for _ in range(2):
            state = trainer.train_step(state, next(iterator)).state
        save_checkpoint(state.saveable_state, 1, checkpoint, is_temporary=False)
        snapshot = jax.tree.map(lambda x: np.array(x), state)
        expected_batch = next(iterator)
    updated = []
    for index, rate in enumerate(RATES.values()):
        optimizer = ContinuationAdamConfig(
            **{**asdict(base), "learning_rate": rate}
        ).build(STOP)
        config = trainer_config(tmp_path / str(index), resume=checkpoint)
        with Trainer(config, optimizer, model_loss, add_default_hooks=False) as trainer:
            restored = trainer.initial_state(jax.random.PRNGKey(999), model=model)
            validate_restore(restored.saveable_state, checkpoint)
            assert_state_equal(restored, snapshot)
            batch = next(
                trainer.data_loader(dataset).iter_from_step(int(restored.step))
            )
            assert_state_equal(batch, expected_batch)
            restored_count = int(restored.opt_state.count)
            result = trainer.train_step(restored, batch).state
            np.testing.assert_allclose(
                result.opt_state.hyperparams["learning_rate"], rate, rtol=1e-7
            )
            assert int(result.opt_state.count) == restored_count + 1
            updated.append(result)
    for index, factor in ((1, 1.5), (2, 2.0)):
        assert_state_equal(
            updated[0].opt_state.inner_state, updated[index].opt_state.inner_state
        )
        assert_state_equal(updated[0].training_key, updated[index].training_key)
        for before, a, b in zip(
            jax.tree.leaves(snapshot.model),
            jax.tree.leaves(updated[0].model),
            jax.tree.leaves(updated[index].model),
            strict=True,
        ):
            np.testing.assert_allclose(
                np.asarray(b) - before,
                factor * (np.asarray(a) - before),
                atol=1.3e-7,
                rtol=2e-5,
            )


def test_trial_configs_match_except_rate_identity_and_diagnostics(tmp_path):
    checkpoint = tmp_path / "step-14520"
    checkpoint.mkdir()
    (checkpoint / "metadata.json").write_text(json.dumps({"step": 14520}))
    manifest = {
        "inputs": {
            name: {
                "cache_dir": f"/frozen/{name}/"
                + ("validation" if name == "val" else "train")
            }
            for name in ("afdb", "esm", "val")
        }
    }
    configs = [
        build_trial_config(
            manifest,
            trial=trial,
            run_name=f"exp279-soft-{trial}-test",
            output=str(tmp_path),
            resume=str(checkpoint),
        )
        for trial in RATES
    ]
    for config, rate in zip(configs, RATES.values(), strict=True):
        assert config.model == configs[0].model and config.data == configs[0].data
        assert config.trainer.num_train_steps == STOP and STOP - START == 5000
        assert config.trainer.train_batch_size == 128
        assert config.trainer.per_device_parallelism == 1
        assert config.trainer.allow_partial_checkpoint is False
        assert config.trainer.watch.start == START and config.trainer.watch.stop == STOP
        actual = config.optimizer.lr_scheduler(STOP)(jnp.asarray([START, STOP - 1]))
        np.testing.assert_allclose(actual, rate, rtol=1e-7)
        a, b = asdict(config.optimizer), asdict(configs[0].optimizer)
        a.pop("learning_rate")
        b.pop("learning_rate")
        assert a == b
    pilot = build_trial_config(
        manifest,
        trial="lr100",
        run_name="exp279-soft-lr100-test",
        output=str(tmp_path),
        resume=str(checkpoint),
        stop_after=START + 2,
    )
    assert pilot.optimizer == configs[0].optimizer
    assert pilot.trainer.watch.stop == STOP
    assert pilot.trainer.watch.final_update == START + 2


def test_endpoint_mask_matches_serialized_contact_markers(vocab):
    original, _ = example_and_oracle(vocab, edges=((143, 144), (143, 145), (144, 145)))
    endpoint = endpoint_example(original)
    expected = np.zeros(original.tokens.size, dtype=np.float32)
    markers = np.flatnonzero(np.asarray(original.tokens.array) == vocab.contact)
    expected[markers] = 1
    expected[markers + 1] = 1
    expected *= np.asarray(original.loss_weight.array)
    np.testing.assert_array_equal(endpoint.loss_weight.array, expected)
    assert (
        endpoint.tokens is original.tokens and endpoint.attn_mask is original.attn_mask
    )
    assert type(endpoint).__name__ == "LmExample"
    assert expected.sum() == 6
    model = tiny_config().build(
        hax.Axis("vocab", vocab.size), key=jax.random.PRNGKey(7)
    )
    logits = model(original.tokens, original.attn_mask).array
    hard = np.roll(np.asarray(original.tokens.array), -1)
    nll = -np.asarray(jax.nn.log_softmax(logits))[np.arange(len(hard)), hard]
    actual = float(model.compute_next_token_loss(endpoint).array)
    np.testing.assert_allclose(
        actual, (nll * expected).sum() / expected.sum(), rtol=2e-6
    )


def test_clipping_metrics_measure_pre_clip_norm_and_actual_progress():
    metrics = gradient_metrics(
        jnp.asarray([3.0, 4.0]),
        jnp.asarray([0.3, 0.4]),
        completed=START + 2500,
        start=START,
        stop=STOP,
    )
    assert float(metrics["grad/norm/total"]) == 5
    assert float(metrics["grad/clipped"]) == 1
    assert float(metrics["updates/norm/total"]) == 0.5
    assert float(metrics["lr_trial/progress"]) == 0.5
    np.testing.assert_allclose(metrics["grad/clip_scale"], 0.2)


def test_resume_rejects_changed_identity_and_selects_own_checkpoint(tmp_path):
    root = str(tmp_path / "trial")
    identity = {"trial": "lr100", "parent": "frozen"}
    assert select_resume(root, identity, "/parent/step-14520") == (
        "/parent/step-14520",
        START,
    )
    (tmp_path / "trial").mkdir()
    record = tmp_path / "trial/experiment.json"
    record.write_text(json.dumps(identity))
    checkpoint = tmp_path / "trial/step-15000"
    checkpoint.mkdir()
    (checkpoint / "metadata.json").write_text(
        json.dumps(
            {"step": 15000, "is_temporary": False, "timestamp": "2026-09-10T14:00:00"}
        )
    )
    assert select_resume(root, identity, "/parent/step-14520") == (
        str(checkpoint),
        15001,
    )
    with pytest.raises(ValueError, match="identity differs"):
        select_resume(root, {**identity, "trial": "lr200"}, "/parent/step-14520")


@pytest.mark.parametrize("cluster", ["cw-us-east-02a", "cw-rno2a"])
def test_lr_request_uses_existing_gpu_setup_and_fixed_target(cluster):
    request = trial_request(
        trial="lr150",
        run_name="exp279-soft-lr150-test",
        job_name="test",
        env={},
        cluster=cluster,
    )
    assert request.replicas == 4 and request.resources.device.count == 8
    assert request.resources.target_cluster == cluster
    assert request.priority == 3
    script = request.entrypoint.binary_entrypoint.args[-1]
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)
    assert (
        ".lr_trial" in script
        and "--trial lr150" in script
        and f"--stop-after {STOP}" in script
    )
    assert "--extra gpu" in script and "--locked" in script


class RecordingTracker(NoopTracker):
    """Capture every callback log without merging away duplicate-step records."""

    def __init__(self, path: str):
        self.path = Path(path)

    def log(self, metrics, *, step, commit=None):
        with self.path.open("a") as stream:
            stream.write(
                json.dumps({"step": step, "metrics": metrics}, default=float) + "\n"
            )


@TrackerConfig.register_subclass("exp279_test_recording")
@dataclass
class RecordingConfig(NoopConfig):
    path: str = ""

    def init(self, run_id):
        return RecordingTracker(self.path)


@pytest.mark.accelerator
def test_stock_gpu_loop_logs_ordinary_endpoint_ce_and_completed_updates(tmp_path):
    assert jax.default_backend() != "cpu"
    tokenizer = resolve_tokenizer()
    cache = tmp_path / "cache" / "validation"
    make_cache(cache, [make_document([(143, 144), (143, 145)])] * 128)
    data = ContactDataConfig(
        tokenizer=tokenizer,
        shuffle=False,
        auto_build_caches=False,
        edge_capacity=8,
        components={
            name: DatasetComponent(
                cache_dir=str(cache if split == "train" else cache.parent),
                flat_cache=split == "train",
                split=split,
                pack=True,
                format=TextLmDatasetFormat(text_key="document"),
            )
            for name, split in (("synthetic", "train"), ("validation", "validation"))
        },
        train_weights={"synthetic": 1.0},
        required_validation_names=("validation",),
    )
    records = tmp_path / "metrics.jsonl"
    config = TrainLmConfig(
        data=data,
        model=replace(tiny_config(), tokenizer=tokenizer),
        optimizer=ContinuationAdamConfig(learning_rate=0.0015),
        trainer=replace(
            trainer_config(tmp_path, microbatch=2),
            tracker=RecordingConfig(path=str(records)),
            require_accelerator=True,
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            num_train_steps=3,
            steps_per_eval=2,
            max_eval_batches=1,
            checkpointer=CheckpointerConfig(
                base_path=str(tmp_path / "checkpoints"),
                append_run_id_to_base_path=False,
                save_interval=None,
                keep=[{"every": 1}],
            ),
            watch=LrWatchConfig(
                start=0,
                stop=3,
                final_update=3,
                sequence_length=64,
                endpoint_examples=4,
                endpoint_interval=1,
                eval_per_device=1,
                validation_cache=str(cache),
                tokenizer=tokenizer,
                checkpoint_root=str(tmp_path / "contact_diagnostics"),
            ),
        ),
    )
    train_main(config)
    rows = [json.loads(line) for line in records.read_text().splitlines()]
    for key in ("eval/loss", "contact_eval/loss", "grad/norm/total", "grad/clipped"):
        values = [row["metrics"][key] for row in rows if key in row["metrics"]]
        assert values, f"The real training loop never logged {key}"
        assert all(np.isfinite(value) for value in values)
    progress = [row for row in rows if "lr_trial/progress" in row["metrics"]]
    assert progress[-1]["step"] == 2
    assert progress[-1]["metrics"]["lr_trial/progress"] == 1
    assert progress[-1]["metrics"]["lr_trial/completed_updates"] == 3
    assert (tmp_path / "checkpoints/step-2/manifest.json").is_file()
