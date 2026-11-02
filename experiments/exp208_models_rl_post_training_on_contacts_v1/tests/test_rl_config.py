# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the exp208 RL job config.

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
        run_name="plm-exp208-test",
        checkpoint=checkpoint,
        tokenizer=checkpoint,
        targets_path="gs://bucket/targets.parquet",
        prompts_path="gs://bucket/prompts",
        output_prefix="gs://bucket/exp208",
        learning_rate=3e-6,
        num_train_steps=400,
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


def test_exactly_one_lesson(checkpoint):
    """exp200 trained a pinned 50:50 plain/multi mix; #208 drops multi entirely.

    With a single lesson the sampling weight is 1 regardless, so exp200's
    `minimum_sample_probability=1.0` pin against marin's adaptive reweighting is
    not merely unnecessary here — there is nothing left for it to hold.
    """
    curriculum = build(checkpoint).curriculum
    assert set(curriculum.lessons) == {"contacts_plain"}


def test_lesson_points_at_this_experiments_env_in_plain_mode(checkpoint):
    lesson = build(checkpoint).curriculum.lessons["contacts_plain"]
    assert lesson.env_config.env_class == "contacts_env.ContactsV1RLEnv"
    args = lesson.env_config.env_args
    assert args["doc_term"] == "consensus"
    # No `mode` / `max_sections` any more: multi-draft is gone, and a leftover
    # key here would silently be ignored by the env constructor.
    assert "mode" not in args and "max_sections" not in args


def test_doc_term_is_configurable_because_it_is_the_experiments_axis(checkpoint):
    for doc_term in ("consensus", "own_f1", "none"):
        lesson = build(checkpoint, doc_term=doc_term).curriculum.lessons["contacts_plain"]
        assert lesson.env_config.env_args["doc_term"] == doc_term


def test_response_budget_matches_the_published_formula(checkpoint):
    lesson = build(checkpoint).curriculum.lessons["contacts_plain"]
    # exp98: 4 * 512 + 64
    assert lesson.sampling_params.max_output_tokens == 2112


def test_checkpoints_are_kept_on_a_step_interval_not_only_a_timer(checkpoint):
    """exp200 checkpointed on a 20-minute timer only.

    When two of its three arms stalled, their rolling checkpoints lagged training
    by ~30 steps and left nothing clean to evaluate. The timer stays for
    preemption recovery; `keep` is what produces evaluable step-indexed artifacts.
    """
    checkpointer = build(checkpoint).trainer.checkpointer
    assert checkpointer.keep and checkpointer.keep[0]["every"] == 25
    assert checkpointer.save_interval is not None


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


def test_weight_sync_is_amortized_and_the_buffer_admits_that_staleness(checkpoint):
    """sync_interval_steps=1 measured at 6.2 min/step with generation at 0.4% of it.

    The freshness window must match the sync interval: max_rollout_step_delay=0
    (what with_on_policy_training forces) drops everything the rollout worker
    produces between syncs, and the trainer starves.
    """
    config = build(checkpoint, sync_interval_steps=8)
    assert config.weight_transfer.sync_interval_steps == 8
    assert config.train_params.replay_buffer.max_rollout_step_delay == 8
    # Each rollout still trains exactly once.
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
    """exp163: `with_tpu` leaves regions unset and the scheduler may pick one with
    no v5p at all, while zone-pinning starved three jobs. Pin the region, and pin
    it to where capacity was actually measured — us-central1 held 103 ready v5p-8
    on 2026-08-09 against 3 in us-east5."""
    run_config = build(checkpoint).run_config
    assert run_config.regions == ["us-central1"]
    assert run_config.zone is None


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


def test_canonical_model_name_satisfies_both_of_its_consumers():
    """It feeds a substring match AND an exact-key lookup, which pull apart.

    `_get_renderer` only needs "qwen" or "llama" in the string, so a descriptive
    invented name passes it — and then `reload_model` raises
    `KeyError: No MODEL_MAPPING registered` when it looks the same string up as an
    exact key. That failure appears only on the weight-transfer path, so a
    generation-only run (the Phase 1 gate) cannot catch it.
    """
    from marin.rl.environments.inference_ctx.vllm_utils import (
        MODEL_MAPPINGS,
        MODEL_TRANSPOSE_KEYS,
    )

    name = rl_config.CANONICAL_MODEL_NAME
    assert "qwen" in name.lower() or "llama" in name.lower(), "renderer selection would raise"
    assert MODEL_MAPPINGS[name], "weight transfer would raise KeyError"
    assert MODEL_TRANSPOSE_KEYS[name], "weight transfer would raise KeyError"


def test_borrowed_qwen3_key_is_exact_not_approximate(checkpoint):
    """Borrowing the 1.7B key for a 1.5B model is only safe because the mapping is
    per-architecture. If upstream ever makes it size-dependent, this fails."""
    from marin.rl.environments.inference_ctx.vllm_utils import (
        _MODEL_MAPPINGS,
        _MODEL_TRANSPOSE_KEYS,
    )

    qwen3 = [k for k in _MODEL_MAPPINGS if "Qwen3" in k]
    assert len(qwen3) > 1
    assert all(_MODEL_MAPPINGS[k] == _MODEL_MAPPINGS[qwen3[0]] for k in qwen3)
    assert all(_MODEL_TRANSPOSE_KEYS[k] == _MODEL_TRANSPOSE_KEYS[qwen3[0]] for k in qwen3)
    assert rl_config.CANONICAL_MODEL_NAME in qwen3


def test_rejects_data_in_a_different_region_from_the_compute(checkpoint):
    """marin aborts with TransferBudgetExceeded, an hour into a run, after the
    rollout workers have already written thousands of rollouts."""
    with pytest.raises(ValueError, match="but the workers run in"):
        build(
            checkpoint,
            targets_path="gs://marin-us-east5/protein-structure/MarinFold/exp208/train/targets.parquet",
            regions=("us-central1",),
        )


def test_accepts_colocated_data(checkpoint):
    build(
        checkpoint,
        targets_path="gs://marin-us-central1/x/targets.parquet",
        prompts_path="gs://marin-us-central1/x/prompts",
        output_prefix="gs://marin-us-central1/x",
        regions=("us-central1",),
    )


def test_region_check_ignores_non_marin_buckets(checkpoint):
    # HF repo ids and third-party buckets carry no region signal.
    rl_config.check_region_locality(("us-central1",), a="gs://some-other-bucket/x", b="hf://repo/x")


def test_rejects_a_step_count_the_weight_transfer_hook_cannot_align_with(checkpoint):
    """exp208's exp163 control trained 10 clean steps and then died at step 9 with
    "weight transfer hook ran at step 9, which is not aligned with
    sync_interval_steps=8".

    The final transfer fires at num_train_steps - 1, so any step count that is not
    a multiple of the sync interval throws that away 25 minutes into a gang. Fail
    at config time instead.
    """
    with pytest.raises(ValueError, match="not a multiple of"):
        build(checkpoint, num_train_steps=10, sync_interval_steps=8)
    build(checkpoint, num_train_steps=16, sync_interval_steps=8)   # aligned: fine
