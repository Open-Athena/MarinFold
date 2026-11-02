# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Pin the per-token advantage path through marin.rl.

exp208's whole reward design rests on one property of marin's ingestion code:
``train_batch.py`` fills a response row with ``np.full(len(response_tokens), advantage)``,
and ``np.full`` broadcasts when ``advantage`` is an ARRAY rather than a scalar. That
is not something marin documents or promises — ``RolloutWithAdvantage.advantage`` is
even annotated ``float``.

If it regresses, nothing crashes. The run trains, W&B logs, and every rollout just
silently gets a constant advantage — i.e. plain RLOO wearing a dense-reward costume,
which would read as "the dense reward didn't help" rather than as a bug. Hence this
test, which asserts the dense signal survives all the way into ``TrainingBatch``.
"""

import numpy as np
import pytest
from marin.rl.decoding import RolloutDecodingTrace
from marin.rl.kl_regularization import KLConfig, KLMode
from marin.rl.train_batch import create_training_batch_from_rollouts
from marin.rl.types import Rollout, RolloutWithAdvantage

from dense_loss import ContactsDenseLoss

TRACE = RolloutDecodingTrace(
    strategy="sample",
    temperature=1.0,
    top_k=-1,
    top_p=0.95,
    min_p=None,
    repetition_penalty=None,
    presence_penalty=None,
    frequency_penalty=None,
    max_output_tokens=64,
    min_output_tokens=None,
    stop_strings=None,
    stop_token_ids=(10,),
    ignore_eos=False,
    seed=None,
)


def make_rollout(example_id: str, prompt_len: int, token_rewards, episode_reward: float) -> Rollout:
    token_rewards = np.asarray(token_rewards, dtype=np.float32)
    n = len(token_rewards)
    return Rollout(
        env_name="contacts-v1-multi",
        env_example_id=example_id,
        prompt_tokens=np.arange(prompt_len, dtype=np.int32),
        response_tokens=np.arange(100, 100 + n, dtype=np.int32),
        response_logprobs=np.full(n, -0.5, dtype=np.float32),
        token_rewards=token_rewards,
        episode_reward=episode_reward,
        decoding=TRACE,
        is_truncated=False,
    )


def loss_module(**kwargs) -> ContactsDenseLoss:
    return ContactsDenseLoss(kl=KLConfig(mode=KLMode.NONE, beta=0.0), **kwargs)


def test_compute_advantages_returns_one_array_per_rollout():
    group = [
        make_rollout("a", 4, [1.0, -1.0, 0.5, 0.0], 0.6),
        make_rollout("b", 4, [0.0, 0.25, -0.75, 2.0], 0.2),
    ]
    advantages = loss_module().compute_advantages(group)

    assert len(advantages) == 2
    for adv, rollout in zip(advantages, group):
        assert adv.shape == rollout.response_tokens.shape
        assert adv.dtype == np.float32


def test_document_term_is_rloo_baselined_across_the_group():
    # Two rollouts, rewards 0.6 and 0.2 -> leave-one-out advantages +0.4 / -0.4.
    group = [
        make_rollout("a", 2, [0.0, 0.0, 0.0], 0.6),
        make_rollout("b", 2, [0.0, 0.0, 0.0], 0.2),
    ]
    advantages = loss_module(lam_step=0.0, lam_doc=1.0).compute_advantages(group)
    assert advantages[0] == pytest.approx([0.4] * 3)
    assert advantages[1] == pytest.approx([-0.4] * 3)


def test_lambdas_scale_the_two_terms_independently():
    group = [
        make_rollout("a", 2, [1.0, 0.0, -1.0], 0.6),
        make_rollout("b", 2, [0.0, 0.0, 0.0], 0.2),
    ]
    advantages = loss_module(lam_step=2.0, lam_doc=0.5).compute_advantages(group)
    # step * 2 + doc(0.4) * 0.5
    assert advantages[0] == pytest.approx([2.2, 0.2, -1.8])


def test_dense_advantage_survives_into_the_training_batch():
    """The load-bearing one: loss_weights must VARY within a response row."""
    step_a = [1.0, -1.0, 0.5, 0.0, -0.25]
    group = [
        make_rollout("a", 3, step_a, 0.6),
        make_rollout("b", 3, [0.0, 0.1, 0.2, 0.3, 0.4], 0.2),
    ]
    advantages = loss_module(lam_step=1.0, lam_doc=0.0).compute_advantages(group)

    batch = create_training_batch_from_rollouts(
        [RolloutWithAdvantage(rollout=r, advantage=a) for r, a in zip(group, advantages)],
        max_tokens=64,
        pad_token_id=0,
    )

    weights = np.asarray(batch.loss_weights.array)
    masks = np.asarray(batch.loss_masks.array)

    row = weights[0]
    assert len(np.unique(row[masks[0] > 0])) > 1, "advantage collapsed to a constant per row"
    # Prompt is 3 tokens, so the response occupies columns 3..7 in emission order.
    assert row[3:8] == pytest.approx(step_a)
    assert np.all(row[:3] == 0.0), "prompt tokens must carry no advantage"
    assert np.all(masks[0][:3] == 0.0)


def test_rejects_a_reward_vector_that_does_not_match_the_response():
    group = [make_rollout("a", 2, [1.0, 2.0], 0.5)]
    bad = Rollout(
        env_name=group[0].env_name,
        env_example_id="short",
        prompt_tokens=group[0].prompt_tokens,
        response_tokens=np.arange(5, dtype=np.int32),
        response_logprobs=np.zeros(5, dtype=np.float32),
        token_rewards=np.zeros(3, dtype=np.float32),
        episode_reward=0.5,
        decoding=TRACE,
        is_truncated=False,
    )
    with pytest.raises(ValueError, match="token_rewards has shape"):
        loss_module().compute_advantages([bad])


def test_rejects_marins_default_constant_token_rewards():
    """`create_rollout_from_choice` fills token_rewards with the episode reward.

    An environment that forgets to replace it would otherwise train fine and
    quietly be a constant-advantage run.
    """
    group = [
        make_rollout("a", 2, [0.6, 0.6, 0.6], 0.6),
        make_rollout("b", 2, [0.2, 0.2, 0.2], 0.2),
    ]
    with pytest.raises(ValueError, match="constant and equal to its episode_reward"):
        loss_module().compute_advantages(group)


def test_a_single_flat_rollout_does_not_trip_the_guard():
    """The exp208 correction to exp200's per-rollout version of this check.

    A rollout that emits no scoreable contact legitimately has all-zero
    token_rewards, and in #208 `episode_reward` is a consensus marginal that is
    exactly zero whenever dropping that rollout does not move the vote — always,
    in the step-only arm. Per rollout that is indistinguishable from marin's
    constant default, so exp200's check would have crashed a healthy run. Across
    a group it IS distinguishable, because marin's default flattens every member.
    """
    loss = ContactsDenseLoss(kl=KLConfig(mode=KLMode.NONE, beta=0.0))
    group = [
        make_rollout("silent", 2, [0.0, 0.0, 0.0], 0.0),      # emitted nothing
        make_rollout("dense", 2, [0.26, -0.07, 0.26], 0.0),   # real per-contact signal
    ]
    advantages = loss.compute_advantages(group)
    assert len(advantages) == 2
    assert np.allclose(advantages[0], 0.0)
    assert not np.allclose(advantages[1], advantages[1][0])
