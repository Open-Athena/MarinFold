# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

import math

from levanter.optim.config import LrScheduleContext

from experiments.exp232_sweep_cv1_decontam.exp232_sweep import (
    NUM_TRAIN_STEPS,
    augmentation_probability,
)
from experiments.exp232_sweep_cv1_decontam.tpu.exp232_cw_to_trc import (
    ARTIFACTS,
    REGION_BUCKETS,
    SUPPORTED_REGIONS,
    destination,
)
from experiments.exp232_sweep_cv1_decontam.tpu.exp232_train_trc import (
    END_STEP,
    FINAL_CHECKPOINT_STEP,
    FINAL_COOLDOWN_START,
    INITIAL_LR_HOLD_START,
    INITIAL_LR_TRANSITION_STEPS,
    PEAK_LEARNING_RATE,
    RESUME_STEP,
    VARIANTS,
    Exp232RecoveryLrSchedule,
    _training_env,
    batch_fit,
)


def _schedule(ratio: float):
    return Exp232RecoveryLrSchedule(ratio).build(
        LrScheduleContext(
            warmup_steps=0,
            decay_steps=END_STEP,
            learning_rate=PEAK_LEARNING_RATE,
            min_lr_ratio=0.0,
            min_lr=0.0,
        )
    )


def test_production_boundaries() -> None:
    assert RESUME_STEP == 333_961
    assert END_STEP == 551_761
    assert FINAL_CHECKPOINT_STEP == 551_760
    assert INITIAL_LR_TRANSITION_STEPS == 10_890
    assert INITIAL_LR_HOLD_START == 344_851
    assert FINAL_COOLDOWN_START == 464_641


def test_lr_variants_hit_every_inclusive_endpoint() -> None:
    for variant in VARIANTS.values():
        target = variant.target_learning_rate
        schedule = _schedule(variant.target_ratio)
        observed = {
            "before_resume": float(schedule(RESUME_STEP - 1)),
            "resume": float(schedule(RESUME_STEP)),
            "target_last_transition": float(schedule(INITIAL_LR_HOLD_START - 1)),
            "target_first_hold": float(schedule(INITIAL_LR_HOLD_START)),
            "target_before_final": float(schedule(FINAL_COOLDOWN_START - 1)),
            "target_final_start": float(schedule(FINAL_COOLDOWN_START)),
            "zero_final_checkpoint": float(schedule(FINAL_CHECKPOINT_STEP)),
            "zero_after_end": float(schedule(END_STEP)),
        }
        assert math.isclose(
            observed["before_resume"], PEAK_LEARNING_RATE, abs_tol=1e-10
        )
        assert math.isclose(observed["resume"], PEAK_LEARNING_RATE, abs_tol=1e-10)
        for key in (
            "target_last_transition",
            "target_first_hold",
            "target_before_final",
            "target_final_start",
        ):
            assert math.isclose(observed[key], target, abs_tol=1e-10), (key, observed)
        assert math.isclose(observed["zero_final_checkpoint"], 0.0, abs_tol=1e-10)
        assert math.isclose(observed["zero_after_end"], 0.0, abs_tol=1e-10)


def test_augmentation_is_continuously_full_rate() -> None:
    assert augmentation_probability(RESUME_STEP - 1, NUM_TRAIN_STEPS) == 1.0
    assert augmentation_probability(RESUME_STEP, NUM_TRAIN_STEPS) == 1.0
    assert augmentation_probability(FINAL_CHECKPOINT_STEP, NUM_TRAIN_STEPS) == 1.0


def test_ingress_is_direct_and_region_local() -> None:
    destinations: set[str] = set()
    for region in SUPPORTED_REGIONS:
        for artifact in ARTIFACTS:
            assert artifact.source.startswith("s3://marin-us-east-02a/")
            target = destination(region, artifact)
            assert target.startswith(f"gs://{REGION_BUCKETS[region]}/")
            assert target not in destinations
            destinations.add(target)


def test_wandb_routing_isolated_by_run_mode(monkeypatch) -> None:
    monkeypatch.setenv("WANDB_ENTITY", "eric-czech")
    monkeypatch.setenv("WANDB_PROJECT", "marin")
    smoke = _training_env("us-east1", smoke=True)
    assert (smoke["WANDB_ENTITY"], smoke["WANDB_PROJECT"]) == (
        "eric-czech",
        "marin",
    )

    monkeypatch.setenv("WANDB_ENTITY", "open-athena")
    monkeypatch.setenv("WANDB_PROJECT", "MarinFold")
    production = _training_env("us-east1", smoke=False)
    assert (production["WANDB_ENTITY"], production["WANDB_PROJECT"]) == (
        "open-athena",
        "MarinFold",
    )


def test_v6e_batch_fit_reflects_measured_small_slice_memory() -> None:
    small = batch_fit("v6e-4")
    assert small.per_device_parallelism == 8
    assert small.gradient_accumulation == 4

    production = batch_fit("v6e-32")
    assert production.per_device_parallelism == 4
    assert production.gradient_accumulation == 1
