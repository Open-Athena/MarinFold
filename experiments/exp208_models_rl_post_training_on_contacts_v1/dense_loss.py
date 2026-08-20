# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""exp208's RL loss: dense per-token advantages over marin's RLOO objective.

marin.rl's objective is ALREADY per-token — ``rl_losses.compute_ppo_loss_objective``
multiplies ``loss_weights`` (shape ``[batch, position]``) elementwise, with no
broadcast from a per-sequence scalar. The only thing that flattens the signal is
the ingestion path, which fills each response row with one constant
(``train_batch.py``: ``np.full(len(response_tokens), advantage)``).

``np.full`` broadcasts an array ``fill_value``, and ``replay_buffer``'s
``zip(group.rollouts, advantages, strict=True)`` only checks the OUTER length. So
returning one array per rollout from :meth:`compute_advantages` carries a distinct
advantage for every response token straight through to the loss, with no changes
to marin. That behaviour is load-bearing and not something marin promises, which
is why ``tests/test_dense_advantage_broadcast.py`` pins it: if it ever regresses,
the run still trains and still logs — it just silently degrades to a constant
advantage, which looks like "RL didn't help" rather than like a bug.

    A_t = lam_step * token_rewards[t]  +  lam_doc * (R_doc - RLOO baseline)

``token_rewards`` is the dense per-contact term from :mod:`contact_rewards`.
``R_doc`` is what the environment put in ``episode_reward``, which for #208 is
the rollout's **leave-one-out marginal contribution to the group's consensus**
rather than its own F1 — see :mod:`contacts_env`. RLOO's centring is applied on
top of that, which is legitimate: a marginal is still a per-rollout scalar, and
subtracting the group mean is variance reduction, not a change of objective.
"""

from dataclasses import dataclass

import numpy as np
from marin.rl.rl_losses import RLOOLoss, compute_rloo_advantages
from marin.rl.types import Rollout


@dataclass
class ContactsDenseLoss(RLOOLoss):
    """RLOO with a dense per-contact term added to the per-sequence advantage.

    Inherits ``build`` / ``needs_reference_model`` / ``create_loss_fn`` from
    :class:`~marin.rl.rl_losses.RLOOLoss` unchanged — only advantage computation
    differs. All of RLOO's knobs (``kl``, PPO clipping, ``synchronous``,
    ``vocab_tile_size``, ``do_overlong_filtering``) still apply.

    ``do_overlong_filtering`` should stay False: it zeroes an entire truncated
    sequence, but a truncated contacts-v1 rollout still contains many fully
    scored contacts.

    Attributes:
        lam_step: Scale on the dense per-contact term.
        lam_doc: Scale on the document-level term. #208's primary axis is the
            RATIO of the two, and because they differ by roughly an order of
            magnitude in natural scale, that ratio is measured (``rho`` in the
            environment's metrics) rather than read off these numbers.
    """

    lam_step: float = 1.0
    lam_doc: float = 1.0

    def compute_advantages(self, rollout_group: list[Rollout]) -> list[np.ndarray]:
        """Per-token advantages, one float32 array per rollout in the group.

        Args:
            rollout_group: Generations for a single prompt. RLOO's leave-one-out
                baseline is taken across exactly this group.

        Returns:
            A list of ``(response_length,)`` arrays. Lengths must match
            ``rollout.response_tokens`` exactly — ``np.full`` raises otherwise,
            which is the failure mode we want (loud) rather than a silent reshape.
        """
        doc_advantages = compute_rloo_advantages(rollout_group)
        steps, out = [], []
        for rollout, doc in zip(rollout_group, doc_advantages, strict=True):
            step = np.asarray(rollout.token_rewards, dtype=np.float32)
            n_response = len(rollout.response_tokens)
            if step.shape != (n_response,):
                raise ValueError(
                    f"token_rewards has shape {step.shape} but the response is "
                    f"{n_response} tokens ({rollout.env_example_id}). The environment must "
                    "write one dense reward per response token."
                )
            steps.append((rollout, step))
            out.append((self.lam_step * step + self.lam_doc * np.float32(doc)).astype(np.float32))

        self._assert_dense(steps)
        return out

    @staticmethod
    def _assert_dense(steps: list[tuple[Rollout, np.ndarray]]) -> None:
        """Refuse a group whose token rewards are marin's constant default.

        The signature of ``create_rollout_from_choice`` never having been replaced
        is a response row filled with one constant equal to ``episode_reward``.
        Training on that is a constant-advantage run wearing a dense-reward
        costume, so it has to fail loudly.

        CHECKED PER GROUP, NOT PER ROLLOUT — this is a correction to exp200,
        where the same check was per rollout and would have crashed a legitimate
        run. A rollout that emits no scoreable contact has all-zero
        ``token_rewards``, and in #208 ``episode_reward`` is a consensus marginal
        that is *exactly* zero whenever dropping that rollout does not move the
        vote — which Phase 0 measured at a large fraction of rollouts, and which
        is always the case in the step-only arm where the document term is
        identically 0. Per rollout, "all zeros and episode_reward is 0" is
        indistinguishable from the bug; across a whole group it is not, because
        marin's default would flatten every member.
        """
        if not steps:
            return
        for rollout, step in steps:
            if len(step) and (np.ptp(step) > 0 or step[0] != np.float32(rollout.episode_reward)):
                return                     # at least one genuinely dense row
        ids = ", ".join(r.env_example_id for r, _ in steps[:3])
        raise ValueError(
            "every rollout in this group has token_rewards constant and equal to its "
            f"episode_reward ({ids}) — the signature of marin's default "
            "`create_rollout_from_choice`, i.e. the environment never wrote the dense "
            "per-contact reward. Training on this would be a constant-advantage run."
        )


__all__ = ["ContactsDenseLoss"]
