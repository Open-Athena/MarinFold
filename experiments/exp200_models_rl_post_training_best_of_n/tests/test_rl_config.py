# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the exp200 RL job config.

These pin the settings that are silent when wrong: the 50:50 lesson mix (marin's
curriculum is adaptive and would drift), the response budgets, the vocab size,
and the checkpoint preflight that exp163 learned to want the expensive way.
"""

import json

import pytest
from marin.rl.kl_regularization import KLMode

import contact_rewards as cr
import rl_config


@pytest.fixture
def checkpoint(tmp_path):
    path = tmp_path / "ckpt"
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps({"model_type": "qwen3", "vocab_size": 2848, "rope_theta": 1_000_000.0})
    )
    return str(path)


def build(checkpoint, **kwargs):
    defaults = dict(
        run_name="plm-exp200-test",
        checkpoint=checkpoint,
        tokenizer=checkpoint,
        targets_path="gs://bucket/targets.parquet",
        prompts_path="gs://bucket/prompts",
        output_prefix="gs://bucket/exp200",
        learning_rate=3e-6,
        num_train_steps=150,
    )
    return rl_config.build_rl_job_config(**{**defaults, **kwargs})


def test_preflight_returns_the_checkpoint_vocab_size(checkpoint):
    assert rl_config.preflight_checkpoint(checkpoint) == 2848


def test_preflight_rejects_the_levanter_rope_export_bug(tmp_path):
    """levanter writes rope under `rope_parameters` and leaves rope_theta null.

    Older readers then silently use default rope — a 50x wrong base frequency that
    invalidated a full round of exp163 evals without anything crashing.
    """
    path = tmp_path / "bad"
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps({"vocab_size": 2848, "rope_theta": None, "rope_parameters": {"rope_theta": 1e6}})
    )
    with pytest.raises(ValueError, match="no top-level rope_theta"):
        rl_config.preflight_checkpoint(str(path))


def test_preflight_rejects_a_missing_vocab_size(tmp_path):
    path = tmp_path / "novocab"
    path.mkdir()
    (path / "config.json").write_text(json.dumps({"rope_theta": 1e6}))
    with pytest.raises(ValueError, match="no vocab_size"):
        rl_config.preflight_checkpoint(str(path))


def test_vocab_size_comes_from_the_checkpoint_not_the_tokenizer(checkpoint):
    # Eric's contacts-v1 checkpoints pad the embedding for TPU efficiency, and
    # marin loads HF weights with resize_vocab_to_match_tokenizer=False.
    assert build(checkpoint).vocab_size == 2848


def test_exactly_two_lessons_pinned_to_a_fifty_fifty_mix(checkpoint):
    curriculum = build(checkpoint).curriculum
    assert set(curriculum.lessons) == {"contacts_plain", "contacts_multi"}
    # marin's curriculum reweights lessons by a quadratic peaking at 50% success;
    # a minimum of 1.0 clamps both weights equal before renormalisation.
    assert curriculum.minimum_sample_probability == 1.0


def test_lessons_point_at_this_experiments_env_with_the_right_mode(checkpoint):
    lessons = build(checkpoint).curriculum.lessons
    for lesson_id, lesson in lessons.items():
        assert lesson.env_config.env_class == "contacts_env.ContactsV1RLEnv"
        assert lesson.env_config.env_args["mode"] == lesson_id.removeprefix("contacts_")
    assert lessons["contacts_multi"].env_config.env_args["max_sections"] == 8


def test_response_budgets_match_the_published_formulas(checkpoint):
    lessons = build(checkpoint, max_sections=8).curriculum.lessons
    # exp163: (3 * 220 + 8) * 8
    assert lessons["contacts_multi"].sampling_params.max_output_tokens == 5344
    # exp98: 4 * 512 + 64
    assert lessons["contacts_plain"].sampling_params.max_output_tokens == 2112


def test_decoding_disables_top_k_and_stops_on_end(checkpoint):
    for lesson in build(checkpoint).curriculum.lessons.values():
        decoding = lesson.sampling_params.train_decoding
        # DecodingConfig rejects a non-positive top_k, so exp163's -1 sentinel
        # cannot be expressed here; the env translates None to vLLM's -1. #142
        # traced under-generation to a finite top_k, so this must not become 4096.
        assert decoding.top_k is None
        assert decoding.stop_token_ids == [cr.END_ID]
        # TPU vLLM rejects per-request seeds.
        assert decoding.seed is None
        assert decoding.temperature == 1.0


def test_on_policy_training_is_applied(checkpoint):
    config = build(checkpoint)
    assert config.weight_transfer.sync_interval_steps == 1
    assert config.train_params.replay_buffer.max_rollout_step_delay == 0
    assert config.train_params.replay_buffer.max_samples == 1


def test_loss_is_the_dense_module_with_a_kl_anchor(checkpoint):
    loss = build(checkpoint, kl_beta=0.01).train_params.rl_loss
    assert type(loss).__name__ == "ContactsDenseLoss"
    assert loss.kl.mode == KLMode.K3_LOSS and loss.kl.beta == 0.01
    # A truncated multi-draft rollout still holds many fully scored contacts, and
    # ~44% of generations hit the length cap.
    assert loss.do_overlong_filtering is False
    assert loss.lam_step == 1.0 and loss.lam_doc == 1.0


def test_kl_can_be_disabled(checkpoint):
    assert build(checkpoint, kl_beta=0.0).train_params.rl_loss.kl.mode == KLMode.NONE


def test_renderer_workaround_is_wired(checkpoint):
    """vLLMInferenceContext picks a renderer by substring on canonical_model_name
    and raises for anything that is not qwen/llama. It is never used."""
    engine = build(checkpoint).inference_config.engine
    assert "qwen" in engine.canonical_model_name.lower()
    assert engine.max_model_len == rl_config.SEQ_LEN


def test_sweep_arms_do_not_share_a_checkpoint_path(checkpoint):
    a = build(checkpoint, run_name="arm-a")
    b = build(checkpoint, run_name="arm-b")
    assert str(a.trainer.checkpointer.base_path) != str(b.trainer.checkpointer.base_path)
    assert a.rollout_storage.path != b.rollout_storage.path


def test_rollout_file_cap_is_above_one_steps_output(checkpoint):
    """FileRolloutWriter reaps its own oldest files past max_rollout_files; the
    default of 32 would silently discard most of a step at this fan-out."""
    config = build(checkpoint, n_prompts=32, n_generations=8)
    assert config.rollout_storage.max_rollout_files > 32


def test_regions_are_pinned_and_no_zone_is_set(checkpoint):
    """exp163: zone-pinning starved three jobs, and with_tpu leaves regions unset
    so the scheduler may pick a region with no v5p at all."""
    run_config = build(checkpoint).run_config
    assert run_config.regions == ["us-east5", "us-central1"]
    assert run_config.zone is None


def test_the_fifty_fifty_pin_actually_resists_adaptive_drift(checkpoint):
    """Behavioural check, not a config read.

    marin's curriculum reweights lessons by a quadratic peaking at 50% binarized
    success, so realistic lopsided performance DOES move the mix: with the stock
    minimum of 0.1 the arms below land at roughly 84:16. The pin has to hold the
    mix #200 asked for regardless of which lesson is doing better.
    """
    import dataclasses

    import numpy as np
    from marin.rl.curriculum import Curriculum

    curriculum = build(checkpoint).curriculum
    # Success is binarized on reward > 0: alternating gives ~0.5 (peak weight),
    # all-positive gives 1.0 (zero weight).
    history = {
        "contacts_plain": np.tile([0.0, 0.4], 100),
        "contacts_multi": np.full(200, 0.4),
    }

    def weights(minimum_sample_probability):
        c = Curriculum(
            config=dataclasses.replace(
                curriculum, minimum_sample_probability=minimum_sample_probability
            )
        )
        for name, rewards in history.items():
            stats = c.stats[name].training_stats
            stats.reward_history, stats.total_samples = rewards, 200
        return c.compute_sampling_weights()

    drifted = weights(0.1)
    assert abs(drifted["contacts_plain"] - drifted["contacts_multi"]) > 0.5, (
        "the control no longer drifts, so this test proves nothing — pick a case that does"
    )

    pinned = weights(curriculum.minimum_sample_probability)
    assert pinned["contacts_plain"] == pytest.approx(0.5)
    assert pinned["contacts_multi"] == pytest.approx(0.5)


def test_rejects_a_gcs_checkpoint_for_the_inference_engine(checkpoint, tmp_path, monkeypatch):
    """levanter's load_tokenizer cannot read gs://, and vLLM's weight loader can.

    That asymmetry makes this easy to walk into: the weights path works, and only
    the tokenizer path fails — inside a rollout worker, after the gang scheduled.
    """
    with pytest.raises(ValueError, match="load_tokenizer accepts a local directory"):
        rl_config.check_engine_model_path("gs://bucket/exp163/tpuF-bf16/step-404")


def test_accepts_local_dirs_mirrors_and_repo_ids(checkpoint):
    rl_config.check_engine_model_path(checkpoint)                       # local dir
    rl_config.check_engine_model_path("mirror://tokenizers/x/y")        # mirror ref
    rl_config.check_engine_model_path("timodonnell/contacts-v1-multi")  # HF repo id
