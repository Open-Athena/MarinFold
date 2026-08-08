# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""exp200's RL loss: dense per-token advantages over marin's RLOO objective.

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

The advantage combines the two terms #200 calls for::

    A_t = lam_step * token_rewards[t]  +  lam_doc * (R_doc - RLOO baseline)

``token_rewards`` is the dense per-contact term from :mod:`contact_rewards`;
``R_doc`` is the document-level return (best-of-N section F1 for multi-draft
rollouts), baselined leave-one-out across the group of generations for the same
protein. The document term is what rewards *spread* — it is the only part of the
signal that pays for a candidate being different from its siblings.
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

    Note ``do_overlong_filtering`` should stay False here: it zeroes an entire
    truncated sequence, but a truncated multi-draft rollout still contains many
    fully-scored contacts, and ~44% of generations hit the length cap.

    Attributes:
        lam_step: Scale on the dense per-contact term.
        lam_doc: Scale on the document-level best-of-N term.
    """

    lam_step: float = 1.0
    lam_doc: float = 1.0

    def compute_advantages(self, rollout_group: list[Rollout]) -> list[np.ndarray]:
        """Per-token advantages, one float32 array per rollout in the group.

        Args:
            rollout_group: Generations for a single prompt. RLOO's leave-one-out
                baseline is taken across exactly this group, so groups must hold
                more than one rollout for the document term to carry any signal.

        Returns:
            A list of ``(response_length,)`` arrays. Lengths must match
            ``rollout.response_tokens`` exactly — ``np.full`` raises otherwise,
            which is the failure mode we want (loud) rather than a silent
            reshape.
        """
        doc_advantages = compute_rloo_advantages(rollout_group)
        out: list[np.ndarray] = []
        for rollout, doc in zip(rollout_group, doc_advantages, strict=True):
            step = np.asarray(rollout.token_rewards, dtype=np.float32)
            n_response = len(rollout.response_tokens)
            if step.shape != (n_response,):
                raise ValueError(
                    f"token_rewards has shape {step.shape} but the response is "
                    f"{n_response} tokens ({rollout.env_example_id}). The environment must "
                    "write one dense reward per response token."
                )
            if n_response and np.all(step == np.float32(rollout.episode_reward)):
                raise ValueError(
                    f"token_rewards for {rollout.env_example_id} is uniformly equal to "
                    "episode_reward — this is the signature of marin's default "
                    "`create_rollout_from_choice`, i.e. the environment never replaced it "
                    "with the dense per-contact reward. Training on this would be a "
                    "constant-advantage run wearing a dense-reward costume."
                )
            out.append((self.lam_step * step + self.lam_doc * np.float32(doc)).astype(np.float32))
        return out


__all__ = ["ContactsDenseLoss"]
