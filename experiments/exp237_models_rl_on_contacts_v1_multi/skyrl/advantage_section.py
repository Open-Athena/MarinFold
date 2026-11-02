# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""exp237's advantage estimator for SkyRL — issue #237.

A pass-through, and that is the design decision rather than a shortcut. SkyRL
hands an estimator ``(token_level_rewards, response_mask, index)`` and nothing
else — no response token ids, no ground truth, no section boundaries. All three
are needed to know which tokens belong to which candidate contact set, and all
three exist in the generator, so `MultiSectionGenerator` produces the finished
per-token advantage (already centred and scaled against its prompt group) and
this passes it through against the response mask.

Contrast SkyRL's built-in GRPO, which does ``token_level_rewards.sum(dim=-1)``
and broadcasts one scalar back across the response. That is right for arms M-F
and M-B, whose reward genuinely is one number per rollout, and wrong for M-C,
where collapsing would discard the per-section credit assignment that is the
entire point of #237.

The guard below is inherited from #208 with its fix intact: the standard
deviation must be taken over the **response tokens only**. Taking ``.std()``
across the full padded row silently defeats it — padding contributes zeros, so a
genuinely constant-per-token advantage still reads as varying. #208's arm C
trained 125 steps past the one assertion written to catch it for exactly that
reason.
"""

import numpy as np
import torch


def compute_section_advantage(
    token_level_rewards: torch.Tensor,      # [batch, response_len]
    response_mask: torch.Tensor,            # [batch, response_len]
    index: np.ndarray,                      # [batch] — group id per rollout
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the generator's per-token section advantage, masked.

    Args:
        token_level_rewards: The finished advantage from
            :class:`multi_generator.MultiSectionGenerator` — one value per
            response token, constant within a section, centred and scaled across
            the prompt group.
        response_mask: Zeroes padding so ragged batches do not contribute.
        index: Unused. The grouping it encodes was already applied in the
            generator, where the section boundaries live. Accepted so the
            signature matches SkyRL's registry contract.

    Returns:
        ``(advantages, returns)``, both ``[batch, response_len]``. SkyRL's
        built-ins return the same tensor twice for non-critic estimators.
    """
    if token_level_rewards.shape != response_mask.shape:
        raise ValueError(
            f"token_level_rewards {tuple(token_level_rewards.shape)} does not match "
            f"response_mask {tuple(response_mask.shape)}")
    with torch.no_grad():
        advantages = token_level_rewards * response_mask
        live = response_mask.sum(dim=-1) > 1
        if live.any():
            n = response_mask.sum(dim=-1, keepdim=True).clamp(min=1)
            mean = (advantages * response_mask).sum(dim=-1, keepdim=True) / n
            var = (((advantages - mean) ** 2) * response_mask).sum(dim=-1, keepdim=True) / n
            varying = var.sqrt().squeeze(-1)[live]
            if torch.all(varying == 0):
                raise ValueError(
                    "every rollout's advantage is constant across its response tokens; the "
                    "per-section reward was not written. Either MultiSectionGenerator is not "
                    "the configured generator, or every rollout emitted a single section — "
                    "which is the multi format collapsing, not a numerical problem.")
    return advantages, advantages


def compute_rollout_advantage(
    token_level_rewards: torch.Tensor,      # [batch, response_len]
    response_mask: torch.Tensor,            # [batch, response_len]
    index: np.ndarray,                      # [batch] — group id per rollout
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pass-through for arm M-BC, whose advantage is constant within a rollout.

    Separate from :func:`compute_section_advantage` **because its guard is the
    opposite one**. A per-section reward that comes out constant across a
    rollout's response means the dense signal was never written, and that must
    raise. A rollout-level blend is constant across the response *by
    construction* — it is one number per rollout — so the same guard would fire
    on every correct run.

    What can still go silently wrong here is the group pass not running at all,
    which leaves every advantage at zero. That is what this checks.
    """
    if token_level_rewards.shape != response_mask.shape:
        raise ValueError(
            f"token_level_rewards {tuple(token_level_rewards.shape)} does not match "
            f"response_mask {tuple(response_mask.shape)}")
    with torch.no_grad():
        advantages = token_level_rewards * response_mask
        live = response_mask.sum(dim=-1) > 0
        if live.any() and torch.all(advantages[live].abs().sum(dim=-1) == 0):
            raise ValueError(
                "every rollout's advantage is exactly zero; the group blend never ran. "
                "Check that MultiSectionGenerator is the configured generator and that "
                "reward_mode is a rollout-level mode.")
    return advantages, advantages


def register(name: str = "contacts_section") -> str:
    """Register with SkyRL's advantage registry.

    Import-time side effects are avoided so this module stays importable in tests
    without SkyRL installed.
    """
    from skyrl.backends.skyrl_train.utils.ppo_utils import AdvantageEstimatorRegistry

    AdvantageEstimatorRegistry.register(name, compute_section_advantage)
    return name


def register_rollout(name: str = "contacts_rollout") -> str:
    """Register the rollout-level pass-through (arm M-BC)."""
    from skyrl.backends.skyrl_train.utils.ppo_utils import AdvantageEstimatorRegistry

    AdvantageEstimatorRegistry.register(name, compute_rollout_advantage)
    return name


__all__ = ["compute_rollout_advantage", "compute_section_advantage", "register",
           "register_rollout"]
