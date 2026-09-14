# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Stock Qwen3 forward/weights with an exact, chunked categorical training loss."""

from dataclasses import dataclass, fields
from typing import cast

import haliax as hax
import jax
import jax.numpy as jnp
from levanter.models.lm_model import LmConfig, split_activations
from levanter.models.loss import DEFAULT_REDUCTION, next_token_loss_weight
from levanter.models.qwen import Qwen3Config, Qwen3LMHeadModel

from experiments.exp232_sweep_cv1_decontam.training_contract import MODEL_CONFIG

from .recipe import TOKENIZER
from .targets import ContactExample, distributions


def reference_model_config(*, soft_targets: bool) -> "ContactQwen3Config":
    """Copy the reference architecture, changing only training targets."""
    values = {
        field.name: getattr(MODEL_CONFIG, field.name) for field in fields(MODEL_CONFIG)
    }
    values["tokenizer"] = TOKENIZER
    return ContactQwen3Config(**values, soft_targets=soft_targets)


@LmConfig.register_subclass("exp279_qwen3")
@dataclass(frozen=True)
class ContactQwen3Config(Qwen3Config):
    """Loss-only variant. HF exports are ordinary Qwen3 checkpoints."""

    soft_targets: bool = True
    loss_block_size: int = 128

    @property
    def model_type(self):
        return ContactQwen3Model


class ContactQwen3Model(Qwen3LMHeadModel):
    """Same parameter tree and activations as stock Qwen3."""

    @property
    def config(self) -> ContactQwen3Config:
        return cast(ContactQwen3Config, self.transformer.config)

    @classmethod
    def init(
        cls, Vocab: hax.Axis, config: Qwen3Config, *, key: jax.Array
    ) -> "ContactQwen3Model":
        base = Qwen3LMHeadModel.init(Vocab, config, key=key)
        return cls(base.transformer, base.embeddings, base.lm_head)

    def compute_next_token_loss(
        self,
        example,
        *,
        key=None,
        reduction: hax.ReductionFunction | None = DEFAULT_REDUCTION,
        reduction_axis=None,
        logsumexp_weight=None,
        loss_dtype=jnp.float32,
        logit_soft_cap=None,
    ):
        # Stock validation examples deliberately retain ordinary held-out CE.
        if not self.config.soft_targets or not isinstance(example, ContactExample):
            return super().compute_next_token_loss(
                example,
                key=key,
                reduction=reduction,
                reduction_axis=reduction_axis,
                logsumexp_weight=logsumexp_weight,
                loss_dtype=loss_dtype,
                logit_soft_cap=logit_soft_cap,
            )
        if loss_dtype != jnp.float32 or logit_soft_cap is not None:
            raise ValueError("exp279 uses FP32 loss without logit soft capping")
        activations, auxiliary = split_activations(
            self.activations(example.tokens, example.attn_mask, key=key)
        )
        losses, _ = contact_loss_terms(
            activations,
            self.get_lm_head(),
            example,
            block_size=self.config.loss_block_size,
            logsumexp_weight=logsumexp_weight,
        )
        weight = next_token_loss_weight(self.Pos, example.loss_weight)
        return (
            hax.nn.loss.reduce_loss(losses, reduction, reduction_axis, weight=weight)
            + auxiliary
        )


def contact_loss_terms(
    activations: hax.NamedArray,
    head: hax.NamedArray,
    example: ContactExample,
    *,
    block_size: int = 128,
    logsumexp_weight: float | None = None,
) -> tuple[hax.NamedArray, hax.NamedArray]:
    """Return per-token CE and target entropy with bounded intermediate memory.

    Dot products and log-softmax run in FP32, including when the transformer
    uses BF16. Rematerializing each vocabulary block during backprop bounds
    temporary logits/q storage instead of retaining [batch,8192,2845] arrays.
    Batch axes are vmapped here before the common weighted reduction, so the
    stock trainer and evaluator can pass a whole batch (including microbatches).
    """
    Pos = example.tokens.resolve_axis("position")
    batch_axes = tuple(axis for axis in example.tokens.axes if axis != Pos)
    if batch_axes:
        head = head.astype(jnp.float32)

        def per_example(x, ex):
            return contact_loss_terms(
                x, head, ex, block_size=block_size, logsumexp_weight=logsumexp_weight
            )

        mapped = per_example
        for axis in reversed(batch_axes):
            mapped = hax.vmap(mapped, axis)
        return mapped(activations, example)
    if block_size <= 0:
        raise ValueError("Expected a positive loss block size")
    Embed = head.resolve_axis("embed")
    Vocab = head.resolve_axis("vocab")
    hidden = activations.rearrange((Pos, Embed)).array
    weights = head.rearrange((Embed, Vocab)).array.astype(jnp.float32)
    targets = example.targets
    padding = (-Pos.size) % block_size

    def blocks(x):
        x = jnp.pad(x, ((0, padding),) + ((0, 0),) * (x.ndim - 1))
        return x.reshape((-1, block_size) + x.shape[1:])

    tokens = example.tokens.array
    inputs = tuple(
        blocks(x)
        for x in (
            hidden,
            tokens,
            jnp.roll(tokens, -1),
            targets.start.array,
            targets.stop.array,
            targets.kind.array,
        )
    )

    @jax.checkpoint
    def block_loss(xs):
        x, observed, hard, start, stop, kind = xs
        logits = jnp.matmul(
            x.astype(jnp.float32), weights, precision=jax.lax.Precision.HIGHEST
        )
        q = distributions(
            targets.first.array,
            targets.second.array,
            start,
            stop,
            kind,
            observed,
            hard,
            Vocab.size,
        )
        log_p = jax.nn.log_softmax(logits, axis=-1)
        ce = -jnp.sum(q * log_p, axis=-1)
        if logsumexp_weight:
            ce += logsumexp_weight * jax.nn.logsumexp(logits, axis=-1) ** 2
        entropy = -jnp.sum(jax.scipy.special.xlogy(q, q), axis=-1)
        return ce, entropy

    ce, entropy = jax.lax.map(block_loss, inputs)
    return (
        hax.named(ce.reshape(-1)[: Pos.size], Pos),
        hax.named(entropy.reshape(-1)[: Pos.size], Pos),
    )
