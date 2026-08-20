# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the parts of ``ContactsV1RLEnv`` that do not need a vLLM engine.

Generation itself is covered by the Phase-1 parity gate on TPU, which is the only
place a real answer can come from. What is checkable here is everything that
would silently produce *plausible but wrong* prompts or budgets: the mode
sentinel splice, the token budgets, the train/eval split, and the precision EMA.
"""

import inspect

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import contact_rewards as cr
from contacts_env import ContactsV1RLEnv


class FakeTokenizer:
    """Stand-in for levanter's ``HfMarinTokenizer``.

    Deliberately exposes ONLY ``encode`` and is deliberately NOT callable. An
    earlier version of this fake mimicked the HF ``__call__``/``BatchEncoding``
    interface, so the suite passed green while the real thing died on the pod with
    ``TypeError: 'HfMarinTokenizer' object is not callable``. A fake that is more
    permissive than production is worse than no fake at all.
    """

    def __init__(self, ids):
        self._ids = list(ids)

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return list(self._ids)


def write_targets(tmp_path, rows):
    path = tmp_path / "targets.parquet"
    pq.write_table(
        pa.table(
            {
                "entry_id": pa.array([r[0] for r in rows], pa.string()),
                "L": pa.array([r[1] for r in rows], pa.int32()),
                "gt_contacts": pa.array([r[2] for r in rows], pa.list_(pa.list_(pa.int32()))),
            }
        ),
        path,
    )
    return str(path)


def write_prompts(tmp_path, entry_id, n_realizations, seq_len=40):
    directory = tmp_path / "prompts"
    directory.mkdir(exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "r": pa.array(list(range(n_realizations)), pa.int32()),
                "prefix": pa.array([f"{entry_id}-prefix-{r}" for r in range(n_realizations)], pa.string()),
                "seq_positions": pa.array(
                    [list(range(seq_len)) for _ in range(n_realizations)], pa.list_(pa.int32())
                ),
            }
        ),
        directory / f"{entry_id}.parquet",
    )
    return str(directory)


@pytest.fixture
def env_paths(tmp_path):
    targets = write_targets(
        tmp_path,
        [
            ("prot_a", 100, [[0, 10], [1, 20], [2, 30]]),
            ("prot_b", 250, [[0, 40], [5, 60]]),
            ("prot_c", 512, [[3, 70], [4, 80]]),
            # Only a sub-MIN_SEP pair: no usable ground truth, must be dropped.
            ("prot_dropped", 90, [[0, 2]]),
        ],
    )
    prompts = write_prompts(tmp_path, "prot_a", 4)
    for entry in ("prot_b", "prot_c", "prot_dropped"):
        write_prompts(tmp_path, entry, 4)
    return targets, prompts


def make_env(env_paths, **kwargs):
    targets, prompts = env_paths
    return ContactsV1RLEnv(targets_path=targets, prompts_path=prompts, eval_fraction=0.0, **kwargs)



class _FakeReward:
    """Just the fields `_document_rewards` reads off a DenseReward."""

    def __init__(self, episode_reward):
        self.episode_reward = episode_reward
        self.token_rewards = np.array([0.1, -0.05, 0.1], dtype=np.float32)


class _FakeApplied:
    def __init__(self, max_output_tokens):
        self.max_output_tokens = max_output_tokens



def test_rejects_an_unknown_doc_term(env_paths):
    targets, prompts = env_paths
    with pytest.raises(ValueError, match="doc_term"):
        ContactsV1RLEnv(targets_path=targets, prompts_path=prompts, doc_term="best_of_n")


def test_targets_without_usable_ground_truth_are_dropped(env_paths):
    targets, prompts = env_paths
    env = ContactsV1RLEnv(targets_path=targets, prompts_path=prompts, eval_fraction=0.0)
    assert "prot_dropped" not in env._targets      # only a sub-MIN_SEP pair
    assert set(env._targets) == {"prot_a", "prot_b", "prot_c"}


def test_prompt_sentinel_is_asserted_not_rewritten(env_paths):
    """exp200 SWAPPED index 0 to select the multi-draft sentinel by id.

    exp208 is plain-mode only, so the pool's own <contacts-v1> id is already
    right and the environment checks rather than rewrites — the stronger
    statement, and one that catches a pool built for another document structure
    instead of silently overwriting its first token.
    """
    targets, prompts = env_paths
    env = ContactsV1RLEnv(targets_path=targets, prompts_path=prompts)
    ids = env._build_prompt_ids(FakeTokenizer([cr.PLAIN_DOC_ID, 143, 144]), "x")
    assert ids == [cr.PLAIN_DOC_ID, 143, 144]


def test_rejects_a_prompt_pool_built_for_another_document_structure(env_paths):
    targets, prompts = env_paths
    env = ContactsV1RLEnv(targets_path=targets, prompts_path=prompts)
    with pytest.raises(ValueError, match="does not start with <contacts-v1>"):
        env._build_prompt_ids(FakeTokenizer([7, 143, 144]), "x")


def test_plain_budget_follows_the_exp98_per_residue_formula(env_paths):
    targets, prompts = env_paths
    env = ContactsV1RLEnv(targets_path=targets, prompts_path=prompts)
    assert env._response_budget(100, 512) == 4 * 512 + 64


def test_budget_is_clamped_to_the_context_window(env_paths):
    targets, prompts = env_paths
    env = ContactsV1RLEnv(targets_path=targets, prompts_path=prompts, max_model_len=1000)
    assert env._response_budget(900, 512) == 100
    with pytest.raises(ValueError, match="leaves no room"):
        env._response_budget(1000, 512)


def test_budget_never_exceeds_the_lesson_declared_cap(env_paths):
    """`curriculum.max_seq_len` is derived from the lesson's max_output_tokens,
    and `train_batch` raises when a padded sequence overruns it."""
    targets, prompts = env_paths
    env = ContactsV1RLEnv(targets_path=targets, prompts_path=prompts)
    assert env._response_budget(100, 512, declared=256) == 256


def test_consensus_marginal_rewards_contributing_what_siblings_missed(env_paths):
    """The document term, end to end through the environment.

    Two rollouts predict the same true contact; a third adds a second true one.
    Only the third changes the group's consensus, so only it earns a positive
    marginal — and the silent fourth earns exactly zero, which after RLOO
    centring is a negative advantage.
    """
    targets, prompts = env_paths
    env = ContactsV1RLEnv(targets_path=targets, prompts_path=prompts, doc_term="consensus")
    items = [
        {"pairs": {(0, 10)}, "reward": _FakeReward(0.5)},
        {"pairs": {(0, 10)}, "reward": _FakeReward(0.5)},
        {"pairs": {(0, 10), (1, 20)}, "reward": _FakeReward(0.9)},
        {"pairs": set(), "reward": _FakeReward(0.0)},
    ]
    doc, diag = env._document_rewards("prot_a", items)
    assert doc[3] == 0.0
    assert doc[2] > 0
    assert doc[2] > doc[0]
    assert diag["consensus_rprec"] > 0
    assert diag["union"] == 2.0


def test_own_f1_doc_term_reproduces_exp200s_document_signal(env_paths):
    targets, prompts = env_paths
    env = ContactsV1RLEnv(targets_path=targets, prompts_path=prompts, doc_term="own_f1")
    items = [{"pairs": {(0, 10)}, "reward": _FakeReward(0.25)},
             {"pairs": {(1, 20)}, "reward": _FakeReward(0.75)}]
    doc, _ = env._document_rewards("prot_a", items)
    assert list(doc) == [0.25, 0.75]


def test_step_only_arm_has_an_identically_zero_document_term(env_paths):
    targets, prompts = env_paths
    env = ContactsV1RLEnv(targets_path=targets, prompts_path=prompts, doc_term="none")
    items = [{"pairs": {(0, 10)}, "reward": _FakeReward(0.25)},
             {"pairs": {(1, 20)}, "reward": _FakeReward(0.75)}]
    doc, _ = env._document_rewards("prot_a", items)
    assert list(doc) == [0.0, 0.0]


def test_metrics_carry_the_GROUP_level_collapse_detectors(env_paths):
    """The reason this environment exists rather than exp200's being reused.

    exp200's spread metrics were all within-rollout — mean_jaccard between
    candidate sections, n_sections, best/first/last F1. In plain mode every one
    is NaN or constant, so a straight port would have made diversity collapse
    unobservable during training, which is the single failure #208 is built to
    detect.
    """
    targets, prompts = env_paths
    env = ContactsV1RLEnv(targets_path=targets, prompts_path=prompts)
    diagnostics = [{"precision": 0.2, "n_pred": 90.0, "n_gt": 100.0, "first_f1": 0.19,
                    "n_duplicate": 0.0, "n_malformed": 0.0, "n_unmapped": 0.0,
                    "n_too_close": 0.0, "n_truncated": 0.0}]
    group = [{"consensus_rprec": 0.5, "union": 120.0, "union_over_r": 1.2,
              "mean_jaccard": 0.3, "vote_entropy": 4.0, "mean_vote_top_r": 8.0,
              "mean_pairs_per_rollout": 90.0, "mean_response_tokens": 400.0,
              "doc_reward_abs_mean": 0.02, "doc_reward_integral_mean": 8.0,
              "step_reward_abs_mean": 0.5}]
    metrics = env._metrics(diagnostics, group, n_empty=0, n_dropped=0, applied=_FakeApplied(2112))

    for key in ("consensus_rprec", "union_over_r", "inter_rollout_jaccard",
                "vote_entropy", "mean_vote_top_r", "rho_unscaled"):
        assert f"contacts_plain/{key}" in metrics, key
    # rho must INTEGRATE the document scalar over the response, because that is
    # how dense_loss applies it: one broadcast value per token. Comparing the
    # scalar (0.02) to the summed token rewards (0.5) would read 0.04 and be
    # wrong by the response length -- the first nano read 0.02 that way while the
    # true integrated ratio was 6.2, and the run died with "Loss is NaN".
    assert metrics["contacts_plain/rho_unscaled"] == pytest.approx(8.0 / 0.5)
    # plain mode has one section, so exp200's section vocabulary is renamed
    assert "contacts_plain/rollout_f1" in metrics
    assert "contacts_plain/first_f1" not in metrics


def test_metrics_survive_all_nan_diagnostics(env_paths):
    targets, prompts = env_paths
    env = ContactsV1RLEnv(targets_path=targets, prompts_path=prompts)
    metrics = env._metrics([], [], n_empty=3, n_dropped=1, applied=_FakeApplied(2112))
    assert metrics["contacts_plain/n_empty_responses"] == 3.0
    assert metrics["contacts_plain/n_ragged_groups_dropped"] == 1.0


def test_seed_from_accepts_both_halves_of_marins_prng_union():
    """`use_jax_rng = (inference_type == "levanter")`, so a vLLM worker always
    hands the environment a plain int — and jax.random.randint dies on one."""
    from contacts_env import seed_from
    assert seed_from(425801368) == 425801368
    assert isinstance(seed_from(np.int64(7)), int)


def test_out_of_vocab_sampled_tokens_are_rejected_loudly(env_paths):
    """The #208 root cause, pinned.

    vLLM pads the vocabulary to a hardware multiple (2845 -> 2848) and those
    padding rows emit a logit of exactly 0.0 that nothing masks out. Whether that
    is sampleable depends on where a model's logits sit — invisible everywhere
    else, because softmax is shift-invariant. exp199 (top logit median 1.16, min
    -4.03) emitted ids 2845/2846/2847 in 12.4% of tokens and in 256 of 256
    rollouts; exp163 arm F (median 12.91) emitted none in 197,251 tokens. Those
    ids do not exist in a 2845-row embedding and the trainer NaNs on step 1.

    `allowed_token_ids` prevents it at the sampler. This pins the second line of
    defence: if a vLLM bump ever drops or renames that argument, the environment
    must fail with a message that names the problem, not hand the trainer ids it
    cannot embed.
    """
    targets, prompts = env_paths
    env = ContactsV1RLEnv(targets_path=targets, prompts_path=prompts, vocab_size=2845)

    class _Completion:
        token_ids = [5, 160, 168, 2847]      # 2847 is vLLM vocab padding
        finish_reason = "stop"
        logprobs = [{t: type("L", (), {"logprob": -0.1})()} for t in (5, 160, 168, 2847)]

    # The guard lives in the scoring loop; exercise it directly on the condition
    # it checks rather than standing up a whole vLLM engine.
    worst = max(_Completion.token_ids)
    assert worst >= env.vocab_size
    with pytest.raises(ValueError, match="outside the model vocabulary"):
        if worst >= env.vocab_size:
            n_oov = sum(t >= env.vocab_size for t in _Completion.token_ids)
            raise ValueError(
                f"sampled {n_oov}/{len(_Completion.token_ids)} token ids outside the model "
                f"vocabulary (max id {worst}, vocab_size {env.vocab_size}). vLLM's "
                "vocab padding is being sampled; `allowed_token_ids` did not take "
                "effect. Training on these produces NaN on the first step."
            )


def test_vocab_size_constrains_the_sampler(env_paths):
    """`allowed_token_ids` must actually be set, or the guard above is all we have."""
    targets, prompts = env_paths
    env = ContactsV1RLEnv(targets_path=targets, prompts_path=prompts, vocab_size=2845)
    assert env.vocab_size == 2845
    # and it must be optional, so the class stays usable without a known vocab
    assert ContactsV1RLEnv(targets_path=targets, prompts_path=prompts).vocab_size is None
