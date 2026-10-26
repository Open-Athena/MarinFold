"""Keep the Qwen3 loss finite for the exact-coverage loader's padding tail.

The pinned Haliax weighted mean selects zero for an empty denominator, but its
inactive division still produces NaN reverse-mode gradients. Levanter can apply
the loss to entirely padded examples/microbatches in the final finite batch.
Give only those empty losses a harmless denominator, then select constant zero.
Nonempty examples retain their original weights, reduction and model behavior.
"""

from dataclasses import dataclass, fields, replace
from typing import Any

import haliax as hax
from jaxtyping import PRNGKeyArray
from levanter.models.lm_model import LmConfig, LmExample
from levanter.models.qwen import Qwen3Config, Qwen3LMHeadModel


class PaddingSafeQwen3LMHeadModel(Qwen3LMHeadModel):
    """Qwen3 with zero loss and zero gradient for completely masked examples."""

    @classmethod
    def init(
        cls, Vocab: hax.Axis, config: Qwen3Config, *, key: PRNGKeyArray
    ) -> "PaddingSafeQwen3LMHeadModel":
        model = Qwen3LMHeadModel.init(Vocab, config, key=key)
        return cls(model.transformer, model.embeddings, model.lm_head)

    def compute_next_token_loss(
        self, example: LmExample, **kwargs: Any
    ) -> hax.NamedArray:
        has_weight = hax.any(example.loss_weight != 0)
        safe_example = replace(
            example,
            loss_weight=hax.where(
                has_weight, example.loss_weight, hax.ones_like(example.loss_weight)
            ),
        )
        loss = super().compute_next_token_loss(safe_example, **kwargs)
        return hax.where(has_weight, loss, hax.zeros_like(loss))


@LmConfig.register_subclass("exp364_padding_safe_qwen3")
@dataclass(frozen=True)
class PaddingSafeQwen3Config(Qwen3Config):
    """Preserve every architecture field and checkpoint leaf of Qwen3."""

    @property
    def model_type(self) -> type[PaddingSafeQwen3LMHeadModel]:
        return PaddingSafeQwen3LMHeadModel


def padding_safe_model(config: Qwen3Config) -> PaddingSafeQwen3Config:
    """Copy the architecture unchanged into the padding-safe loss implementation."""
    return PaddingSafeQwen3Config(
        **{field.name: getattr(config, field.name) for field in fields(Qwen3Config)}
    )
