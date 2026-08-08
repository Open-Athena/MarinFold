# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Qwen3 trained against contacts-v1's exact conditional next-token targets.

The Phase 2 arm of `#201 <https://github.com/Open-Athena/MarinFold/issues/201>`_.
Where the Phase 1b arm (``masked_loss_model``) *drops* the sequence-statement
slots, this one keeps them and supervises them with the distribution the
generator actually drew from — and does the same for contact first endpoints.
Second endpoints stay one-hot; see ``soft_targets`` for why that is 94 % of the
benefit for a twentieth of the work.

Both losses have the same optimum and the same expected value; the soft one is
the lower-variance estimator of it. It is **not** a smaller number, and the floor
of both is the target entropy ``H(q)`` — measured at 2.09 nats/token over the
exp53 validation split.

No custom kernel, no materialized logits
----------------------------------------

Soft cross-entropy is ``logsumexp(z) - <q, z>``, and levanter's fused kernel
already returns the hard cross-entropy ``logsumexp(z) - z[y]``. So::

    z_y       = dot(h, W[y])            # one gather + one dot, per position
    logsumexp = hard_ce + z_y           # recovered, never materialised as logits
    soft_ce   = logsumexp - dot(h, u) / norm

where ``(u, norm)`` come from :func:`marinfold_models.soft_targets
.soft_target_directions`. The extra cost is two per-position dot products against
a single embedding row each — negligible beside the transformer, and the
``[position, vocab]`` logits are never built.

Loss reporting
--------------

``train/loss`` is the soft loss (what is optimized). ``train/loss_hard`` is the
ordinary one-hot loss from the same forward pass — the series comparable with
every historical contacts-v1 run — and ``train/target_entropy`` is their
difference in expectation, i.e. the nuisance the soft target absorbs.

Evaluation (``reduction=None``) returns the **hard** per-position loss, so
``eval/.../loss`` stays directly comparable with #117/#150. Unlike the masked
arm this is not merely a reporting convention: the soft arm fits every slot the
control fits, so its val loss should be directly competitive rather than
expected-worse.
"""

from dataclasses import dataclass
from typing import Optional, Type, cast

import jax
import jax.numpy as jnp

import haliax as hax
from haliax import Axis, NamedArray
from jaxtyping import PRNGKeyArray
from levanter.models.lm_model import LmConfig, LmExample
from levanter.models.qwen import Qwen3Config, Qwen3LMHeadModel
from levanter.tracker import jit_log

from marinfold_models.soft_targets import (
    BEGIN_SEQUENCE_ID,
    BEGIN_STRUCTURE_ID,
    CONTACT_ID,
    END_ID,
    soft_target_directions,
)


@LmConfig.register_subclass("qwen3_contacts_v1_soft_targets")
@dataclass(frozen=True)
class Qwen3SoftTargetConfig(Qwen3Config):
    """``Qwen3Config`` trained against contacts-v1's order-marginalized targets.

    Token ids default to the published contacts-v1 tokenizer and are config
    fields rather than imports, so ``models/`` keeps no dependency on the
    ``marinfold`` inference package inside the training environment. Resolve them
    from the tokenizer you actually train with and assert the values.
    """

    contact_id: int = CONTACT_ID
    begin_sequence_id: int = BEGIN_SEQUENCE_ID
    begin_structure_id: int = BEGIN_STRUCTURE_ID
    end_id: int = END_ID

    @property  # type: ignore[override]
    def model_type(self) -> Type["Qwen3SoftTargetLMHeadModel"]:
        return Qwen3SoftTargetLMHeadModel


class Qwen3SoftTargetLMHeadModel(Qwen3LMHeadModel):
    """``Qwen3LMHeadModel`` whose training loss uses the exact conditional targets."""

    @classmethod
    def init(  # type: ignore[override]
        cls, Vocab: Axis, config: Qwen3SoftTargetConfig, *, key: PRNGKeyArray
    ) -> "Qwen3SoftTargetLMHeadModel":
        # Delegate the architecture, then rewrap: Qwen3LMHeadModel.init hard-codes
        # its own class rather than using cls, and the module tree is identical.
        base = Qwen3LMHeadModel.init(Vocab, config, key=key)
        return cls(base.transformer, base.embeddings, base.lm_head)

    def compute_next_token_loss(
        self,
        example: LmExample,
        *,
        key=None,
        reduction: Optional[hax.ReductionFunction] = cast(
            Optional[hax.ReductionFunction], hax.mean
        ),
        reduction_axis: Optional[hax.AxisSelection] = None,
        **kwargs,
    ) -> jnp.ndarray | NamedArray:
        """Order-marginalized next-token loss.

        ``reduction=None`` is the evaluation path and returns the ordinary hard
        per-position loss, keeping ``eval/.../loss`` comparable with #117/#150.
        A non-mean reduction is rejected rather than silently mixing the two
        losses' denominators.
        """
        if reduction is None:
            return super().compute_next_token_loss(
                example, key=key, reduction=None, reduction_axis=reduction_axis, **kwargs
            )
        if reduction is not hax.mean:
            raise ValueError(
                "Qwen3SoftTargetLMHeadModel supports reduction=hax.mean or "
                f"reduction=None; got {reduction!r}."
            )

        config = cast(Qwen3SoftTargetConfig, self.config)
        Pos = example.tokens.axes[-1]

        activations = self.activations(example.tokens, example.attn_mask, key=key)
        aux_loss = 0
        if isinstance(activations, tuple):
            activations, aux_loss = activations
        lm_head = self.get_lm_head()

        # loss_i * w_i at every position: levanter's fused hard cross-entropy.
        weighted_hard = super().compute_next_token_loss(
            example, key=key, reduction=None, reduction_axis=None, **kwargs
        )
        weight = _next_token_weight(Pos, example.loss_weight)
        hard = _unweight(weighted_hard, weight)

        # Recover logsumexp(z) without ever materialising the logits: the fused
        # kernel returned logsumexp - z[y], and z[y] is one gather plus one dot.
        target_ids = hax.roll(example.tokens, -1, Pos)
        z_target = hax.dot(
            activations, lm_head.take(self.Vocab, target_ids), axis=self.Embed
        )
        log_normalizer = hard + z_target

        direction, normalizer, is_soft = soft_target_directions(
            example.tokens,
            lm_head,
            Vocab=self.Vocab,
            Embed=self.Embed,
            contact_id=config.contact_id,
            begin_sequence_id=config.begin_sequence_id,
            begin_structure_id=config.begin_structure_id,
            end_id=config.end_id,
        )
        safe_normalizer = hax.where(normalizer > 0, normalizer, 1.0)
        z_soft = hax.dot(activations, direction, axis=self.Embed) / safe_normalizer
        soft = hax.where(is_soft & (normalizer > 0), log_normalizer - z_soft, hard)

        soft_loss = _weighted_mean(soft * weight, weight, reduction_axis)
        if reduction_axis is None:
            hard_loss = _weighted_mean(hard * weight, weight, None)
            jit_log(
                {
                    "train/loss_hard": hard_loss,
                    "train/target_entropy": hard_loss - soft_loss,
                    "train/soft_slot_fraction": (
                        hax.sum(weight * is_soft.astype(weight.dtype)) / hax.sum(weight)
                    ),
                }
            )
        return soft_loss + aux_loss


def _next_token_weight(Pos: Axis, loss_weight: NamedArray) -> NamedArray:
    """The caller's weight with the final position masked off.

    The last position has no next token to predict. Mirrors
    ``levanter.models.loss.next_token_loss_weight``, reimplemented here rather
    than imported because that helper is recent and this module has to keep
    working across the levanter versions the experiment venvs pin.
    """
    not_last = hax.logical_not(hax.nn.one_hot(-1, Pos, dtype=jnp.bool_))
    return loss_weight * not_last.astype(loss_weight.dtype)


def _unweight(weighted: NamedArray, weight: NamedArray) -> NamedArray:
    """Recover the per-position loss from levanter's weight-multiplied output.

    Zero-weight positions are dropped by every downstream reduction, so their
    value only has to be finite.
    """
    safe = hax.where(weight != 0, weight, 1.0)
    return hax.where(weight != 0, weighted / safe, 0.0)


def _weighted_mean(
    weighted_values: NamedArray,
    weight: NamedArray,
    axis: Optional[hax.AxisSelection],
) -> jnp.ndarray | NamedArray:
    """``sum(values * w) / sum(w)``, matching haliax's weighted-mean semantics."""
    numerator = hax.sum(weighted_values, axis=axis)
    denominator = hax.sum(weight, axis=axis)
    return hax.where(denominator != 0, numerator / denominator, hax.zeros_like(numerator))


__all__ = [
    "Qwen3SoftTargetConfig",
    "Qwen3SoftTargetLMHeadModel",
]
