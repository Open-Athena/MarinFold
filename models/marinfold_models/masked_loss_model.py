# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Qwen3 with the contacts-v1 sequence-statement-head slots dropped from the loss.

The mask-only arm of `#201 <https://github.com/Open-Athena/MarinFold/issues/201>`_.
Phase 0 of that experiment measured the contacts-v1 loss budget over the exp53
validation split and found the largest single component is not the contact list
at all: the **sequence-statement shuffle is 1.13 nats/token, 42 % of the whole
training loss**, spent predicting which residue statement the generator happened
to emit next. Those slots are prompt, not prediction.

Dropping them needs no soft-target machinery — just a modified
``LmExample.loss_weight``, after which levanter's ordinary fused cross-entropy
runs unchanged. This module is the whole intervention.

**The architecture is untouched.** Only ``compute_next_token_loss`` differs, so
checkpoints, the HF export path and the eval harness are interchangeable with a
plain ``qwen3`` run — in particular a checkpoint trained here can be evaluated
under the unmasked config to get a loss directly comparable to #117/#150.

Two things about the loss numbers, both easy to misread:

**Train loss is not comparable to the control.** levanter's weighted mean
divides by the *sum of weights*, so ``train/loss`` here is the mean over the
slots that survive the mask — a different denominator and a different slot mix
than every historical contacts-v1 run. The unmasked mean is logged alongside as
``train/loss_unmasked`` (and the surviving fraction as
``train/kept_slot_fraction``); that series is the comparable one, and both come
from a single forward pass.

**Evaluation is deliberately NOT masked.** ``levanter.eval`` pairs the
per-position loss it requests (``reduction=None``) with ``example.loss_weight``
— the *unmasked* weight — so returning a masked numerator there would give a
meaningless hybrid of the two denominators. The eval path therefore returns the
standard loss, keeping ``eval/.../loss`` directly comparable with #117/#150.
Expect the masked arm to score **worse** on it: it deliberately stopped fitting
~23 % of the slots, and those slots are pure permutation noise. That is the
intervention working, not failing — which is why R-precision on the #82/#89 eval
set, not val loss, is this experiment's primary endpoint (#169).
"""

from dataclasses import dataclass
from typing import Optional, Type, cast

import jax.numpy as jnp

import haliax as hax
from haliax import Axis, NamedArray
from jaxtyping import PRNGKeyArray
from levanter.models.lm_model import LmConfig, LmExample
from levanter.models.qwen import Qwen3Config, Qwen3LMHeadModel

from marinfold_models.loss_masks import (
    BEGIN_SEQUENCE_ID,
    BEGIN_STRUCTURE_ID,
    END_ID,
    contacts_v1_statement_head_mask,
)


@LmConfig.register_subclass("qwen3_contacts_v1_statement_head_masked")
@dataclass(frozen=True)
class Qwen3StatementHeadMaskedConfig(Qwen3Config):
    """``Qwen3Config`` whose loss ignores contacts-v1 statement-head slots.

    The token ids default to the published contacts-v1 tokenizer
    (``timodonnell/contacts-v1-tokenizer``). They are config fields rather than
    imports so ``models/`` keeps no dependency on the ``marinfold`` inference
    package (and its gemmi / transformers / pyarrow stack) inside the TPU
    training environment. Callers should resolve them from the tokenizer they
    actually train with and assert the values — see the experiment's
    ``verify_config.py``.
    """

    begin_sequence_id: int = BEGIN_SEQUENCE_ID
    begin_structure_id: int = BEGIN_STRUCTURE_ID
    end_id: int = END_ID

    @property  # type: ignore[override]
    def model_type(self) -> Type["Qwen3StatementHeadMaskedLMHeadModel"]:
        return Qwen3StatementHeadMaskedLMHeadModel

    @property
    def section_closer_ids(self) -> tuple[int, ...]:
        """Ids that close a sequence section: full document, then sequence-only."""
        return (self.begin_structure_id, self.end_id)


class Qwen3StatementHeadMaskedLMHeadModel(Qwen3LMHeadModel):
    """``Qwen3LMHeadModel`` with the statement-head slots masked out of the loss."""

    @classmethod
    def init(  # type: ignore[override]
        cls, Vocab: Axis, config: Qwen3StatementHeadMaskedConfig, *, key: PRNGKeyArray
    ) -> "Qwen3StatementHeadMaskedLMHeadModel":
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
        """Next-token loss with the sequence-statement-head slots zeroed.

        Computes the per-position loss once and reduces it twice: with the mask
        (the value trained on) and without it (logged for comparability with
        every historical contacts-v1 run). The second reduction is free — it
        reuses the same forward pass.

        ``reduction=None`` is the evaluation path and returns the **unmasked**
        loss; see the module docstring. A non-mean reduction is rejected: the
        mask changes the denominator, so "sum of a masked loss" has no
        comparable meaning.
        """
        if reduction is None:
            return super().compute_next_token_loss(
                example, key=key, reduction=None, reduction_axis=reduction_axis, **kwargs
            )

        if reduction is not hax.mean:
            raise ValueError(
                "Qwen3StatementHeadMaskedLMHeadModel supports reduction=hax.mean or "
                f"reduction=None; got {reduction!r}. A masked loss has no meaningful "
                "sum reduction (the mask changes the denominator, not the scale)."
            )

        config = cast(Qwen3StatementHeadMaskedConfig, self.config)
        Pos = example.tokens.axes[-1]
        keep = contacts_v1_statement_head_mask(
            example.tokens,
            begin_sequence_id=config.begin_sequence_id,
            section_closer_ids=config.section_closer_ids,
            dtype=example.loss_weight.dtype,
        )

        # loss_i * w_i at every position, where w is levanter's causal/packing weight.
        weighted = super().compute_next_token_loss(
            example, key=key, reduction=None, reduction_axis=None, **kwargs
        )
        weight = _next_token_weight(Pos, example.loss_weight)

        return _weighted_mean(weighted * keep, weight * keep, reduction_axis)


def _next_token_weight(Pos: Axis, loss_weight: NamedArray) -> NamedArray:
    """The caller's weight with the final position masked off.

    The last position has no next token to predict. Mirrors
    ``levanter.models.loss.next_token_loss_weight``, reimplemented here rather
    than imported because that helper is recent and this module has to keep
    working across the levanter versions the experiment venvs pin.
    """
    not_last = hax.logical_not(hax.nn.one_hot(-1, Pos, dtype=jnp.bool_))
    return loss_weight * not_last.astype(loss_weight.dtype)


def _weighted_mean(
    weighted_values: NamedArray,
    weight: NamedArray,
    axis: Optional[hax.AxisSelection],
) -> jnp.ndarray | NamedArray:
    """``sum(values * w) / sum(w)``, matching haliax's weighted-mean semantics.

    ``weighted_values`` is already multiplied by ``weight``. Zero-weight
    reductions yield 0 rather than NaN, as ``haliax.nn.loss.maybe_reduce_loss``
    does.
    """
    numerator = hax.sum(weighted_values, axis=axis)
    denominator = hax.sum(weight, axis=axis)
    return hax.where(denominator != 0, numerator / denominator, hax.zeros_like(numerator))


__all__ = [
    "Qwen3StatementHeadMaskedConfig",
    "Qwen3StatementHeadMaskedLMHeadModel",
]
