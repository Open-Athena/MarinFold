# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""exp208's advantage estimator for SkyRL — issue #208, SkyRL port.

DELIBERATELY A PASS-THROUGH, AND THAT IS THE WHOLE DESIGN DECISION.

SkyRL hands an estimator `(token_level_rewards, response_mask, index)` and nothing
else — no response token ids, no per-protein ground truth, no rollout pair sets.
exp208's two terms need different things:

* the **stepwise** term needs response token ids, to land a reward on each
  `<contact> <pI> <pJ>` triple;
* the **document** term needs the whole rollout group's pair sets, to compute
  `C(all) - C(all \\ {i})`.

Neither is reachable from this signature. Both ARE reachable in the generator:
`response_ids` are right there, and `trajectory_id.instance_id` /
`repetition_id` identify protein and rollout-within-protein. So
`DenseContactsGenerator` computes the finished per-token advantage — both terms,
already combined and already group-centred — and this estimator passes it through
unchanged against the response mask.

The alternative, splitting the terms across generator and estimator, would mean
encoding the document scalar into a reserved token position and recovering it
here. That is exactly the kind of implicit contract that made marin.rl's dense
path fragile, and it buys nothing: the generator has strictly more information
than the estimator does.

Contrast the built-in GRPO estimator, which does `token_level_rewards.sum(dim=-1)`
and broadcasts one scalar back across the response. That is right for a trajectory
reward and wrong here — collapsing would discard the per-contact signal that is
the entire point of #208.

Note SkyRL applies loss reduction to advantages before the policy loss sees them,
so a custom loss should perform a masked SUM rather than an average.
"""

import numpy as np
import torch


def compute_contacts_dense_advantage(
    token_level_rewards: torch.Tensor,      # [batch, response_len]
    response_mask: torch.Tensor,            # [batch, response_len]
    index: np.ndarray,                      # [batch] — group id per rollout
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the generator's per-token advantage, masked.

    Args:
        token_level_rewards: The finished advantage from
            :class:`DenseContactsGenerator` — ``lam_step * r_step(t)`` plus the
            group-centred ``lam_doc * A_consensus`` already folded in.
        response_mask: Zeroes padding so ragged batches do not contribute.
        index: Unused here; the grouping it encodes was already applied in the
            generator, where the pair sets live. Accepted so the signature
            matches SkyRL's registry contract.

    Returns:
        ``(advantages, returns)``, both ``[batch, response_len]``. SkyRL's
        built-ins return the same tensor twice for non-critic estimators.
    """
    if token_level_rewards.shape != response_mask.shape:
        raise ValueError(
            f"token_level_rewards {tuple(token_level_rewards.shape)} does not match "
            f"response_mask {tuple(response_mask.shape)}"
        )
    with torch.no_grad():
        advantages = token_level_rewards * response_mask
        # A constant advantage across every response token is the signature of a
        # generator that never wrote the dense reward -- the SkyRL analogue of
        # marin.rl's `np.full` failure mode, which reads as "RL didn't help"
        # rather than as a bug. Refuse it.
        live = response_mask.sum(dim=-1) > 1
        if live.any():
            varying = (advantages * response_mask).std(dim=-1)[live]
            if torch.all(varying == 0):
                raise ValueError(
                    "every rollout's advantage is constant across its response tokens; "
                    "the dense per-contact reward was not written. Check that "
                    "DenseContactsGenerator is the configured generator."
                )
    return advantages, advantages


def register(name: str = "contacts_dense") -> str:
    """Register with SkyRL's advantage registry. Import-time side effects are
    avoided so this module stays importable in tests without SkyRL installed."""
    from skyrl.backends.skyrl_train.utils.ppo_utils import AdvantageEstimatorRegistry

    AdvantageEstimatorRegistry.register(name, compute_contacts_dense_advantage)
    return name


__all__ = ["compute_contacts_dense_advantage", "register"]
