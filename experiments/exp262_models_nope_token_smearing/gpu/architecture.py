# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Levanter architecture variants for issue #262: token smearing and NoPE.

Two independent changes to the exp232 Qwen3, each expressible on its own so the
ablation can separate them:

**Token smear.** A learned causal depthwise mix at the embedding, adapted from
the nanogpt speedrun's smear module. Every token gets a gated, per-channel share
of the previous ``smear_width`` tokens' embeddings folded into its own::

    x_t = e_t + sum_k sigmoid(gate_k(e_t)) * (w_k * e_{t-k})    for k = 1..width

Two design points that are not negotiable, both of which the tests enforce:

* **Separate per-channel weights per offset.** ``w_1`` and ``w_2`` are distinct
  ``Embed``-shaped vectors. A single shared scalar would make offsets 1 and 2
  indistinguishable and destroy the arg1/arg2 distinction inside a
  ``<contact> <pX> <pY>`` statement, which is the entire point of the smear for
  this document format.
* **Strictly causal.** The shift is masked at the start of the sequence rather
  than wrapped. ``hax.roll`` alone would wrap the *last* tokens of a sequence
  into the *first* positions, which leaks the future into the past and yields a
  beautiful loss curve attached to a worthless model.

``w_k`` is initialised to zero, so a freshly initialised smear model is exactly
the baseline model and the arms start from the same point. The gradient with
respect to ``w_k`` is still non-zero, so it trains away from zero immediately.

**NoPE.** ``NoRotaryEmbeddingsConfig`` is a rotary choice that returns queries
and keys untouched. It refuses to produce an HF rope config: a NoPE model is not
a Qwen3, and silently emitting a theta would produce an export that loads as a
different model (exactly the failure mode of the transformers-5 ``rope_parameters``
bug in ``marinfold.inference._config``). Runs using these configs must set
``hf_save_steps=None``; the smear weights have no HF Qwen3 home either.
"""

import dataclasses
from dataclasses import dataclass
from typing import Dict, Optional, Type

import equinox as eqx
import haliax as hax
import haliax.nn as hnn
import jax.numpy as jnp
import jax.random as jrandom
from haliax import Axis, NamedArray
from haliax.jax_utils import named_call
from haliax.state_dict import ModuleWithStateDictSerialization
from levanter.layers.rotary import RotaryEmbeddings, RotaryEmbeddingsConfig
from levanter.models.llama import LlamaConfig, LlamaLMHeadModel, LlamaTransformer
from levanter.models.lm_model import LmConfig
from levanter.models.qwen import Qwen3Config

# The gate reads a slice of the token embedding, as in the speedrun's smear
# module (there, the first 12 dimensions). A slice rather than the whole vector
# keeps the gate cheap and keeps it a function of token identity alone.
GATE_DIMS = 16


class NoRotaryEmbeddings(RotaryEmbeddings):
    """The identity. No positional information enters attention."""

    def __call__(self, q: NamedArray, position_ids: NamedArray) -> NamedArray:
        return q


@dataclass(frozen=True)
class NoRotaryEmbeddingsConfig(RotaryEmbeddingsConfig):
    """Rotary choice that applies no rotation at all (NoPE)."""

    def build(self, HeadSize: Axis) -> RotaryEmbeddings:
        return NoRotaryEmbeddings()

    @classmethod
    def make_from_hf_config(cls, rope_theta: float, config: dict) -> "RotaryEmbeddingsConfig":
        raise NotImplementedError(
            "NoPE has no HF rope config to be built from; a checkpoint that needs this "
            "was not trained by exp262"
        )

    def to_hf_config(self) -> tuple[float, dict | None]:
        raise NotImplementedError(
            "a NoPE model cannot be written as an HF Qwen3 config — every rope_theta "
            "would load as a *different* model. Set hf_save_steps=None on runs that use "
            "this config, and write a real exporter before promoting one."
        )


RotaryEmbeddingsConfig.register_subclass("nope", NoRotaryEmbeddingsConfig)


class SmearEmbedding(ModuleWithStateDictSerialization, eqx.Module):
    """Token embedding followed by a gated, per-channel causal smear.

    Mirrors ``levanter.models.llama.LlamaEmbedding`` (same ``token_embeddings``
    field and state-dict key, so a baseline checkpoint's embedding table still
    loads) and adds the smear on top.
    """

    token_embeddings: hnn.Embedding
    norm: Optional[hnn.RmsNorm]
    gate: Optional[hnn.Linear]
    weights: Optional[NamedArray]  # {SmearOffset, Embed}
    SmearOffset: Axis = eqx.field(static=True)
    GateSlice: Axis = eqx.field(static=True)

    @staticmethod
    def init(Vocab: Axis, config: "SmearQwen3Config", *, key) -> "SmearEmbedding":
        k_emb, k_gate = jrandom.split(key, 2)
        token_embeddings = hnn.Embedding.init(Vocab, config.Embed, key=k_emb)
        norm = config.mk_LayerNorm(config.Embed) if config.input_embedding_norm else None

        SmearOffset = Axis("smear_offset", config.smear_width)
        GateSlice = Axis("smear_gate_in", min(GATE_DIMS, config.Embed.size))
        if config.smear_width == 0:
            return SmearEmbedding(token_embeddings, norm, None, None, SmearOffset, GateSlice)

        gate = hnn.Linear.init(In=GateSlice, Out=SmearOffset, key=k_gate, use_bias=True, out_first=True)
        # Zero init: the smear contributes nothing at step 0, so a smear arm and
        # its control start from an identical function.
        weights = hax.zeros((SmearOffset, config.Embed))
        return SmearEmbedding(token_embeddings, norm, gate, weights, SmearOffset, GateSlice)

    @property
    def Vocab(self) -> Axis:
        return self.token_embeddings.Vocab

    @property
    def Embed(self) -> Axis:
        return self.token_embeddings.Embed

    @named_call
    def embed(self, input_ids, *args):
        embeddings = self.token_embeddings(input_ids)
        if self.gate is not None:
            embeddings = self._smear(embeddings)
        if self.norm is not None:
            embeddings = self.norm(embeddings)
        return embeddings

    def _smear(self, embeddings: NamedArray) -> NamedArray:
        """Fold the previous ``smear_width`` tokens into each position."""
        Pos = embeddings.resolve_axis("position")
        gate_input = hax.slice(embeddings, self.Embed, self.GateSlice, start=0)
        gates = hnn.sigmoid(self.gate(gate_input))
        positions = hax.arange(Pos)
        total = embeddings
        for offset in range(1, self.SmearOffset.size + 1):
            shifted = hax.roll(embeddings, offset, Pos)
            # Mask the wrapped-around head of the sequence: without this the
            # last tokens leak into the first positions and the model is no
            # longer causal.
            shifted = hax.where(positions >= offset, shifted, 0.0)
            weight = self.weights[{self.SmearOffset.name: offset - 1}]  # {Embed}
            gate = gates[{self.SmearOffset.name: offset - 1}]  # {Pos}
            # Multiply in this order: each step broadcasts a strict subset of
            # ``shifted``'s axes. Folding weight and gate together first would
            # try to combine {Embed} with {Pos}, which haliax refuses.
            total = total + gate * (weight * shifted)
        return total

    def unembed(self, x: NamedArray):
        return self.token_embeddings.unembed(x)

    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        return {"token_embeddings": "model.embed_tokens", "gate": "model.smear_gate"}

    def resize_embeddings(self, new_size: int, key=None):
        new_weights = self.token_embeddings.resize_embeddings(new_size, key=key)
        return dataclasses.replace(self, token_embeddings=new_weights)


@LmConfig.register_subclass("smear_qwen3")
@dataclass(frozen=True)
class SmearQwen3Config(Qwen3Config):
    """exp232's Qwen3 plus an optional causal token smear at the embedding.

    ``smear_width=0`` with the exp232 rope reproduces the control arm exactly.
    Pair ``smear_width=2`` with ``rope=NoRotaryEmbeddingsConfig()`` for the NoPE
    arm the issue proposes.
    """

    smear_width: int = 0

    def __post_init__(self):
        if self.smear_width < 0:
            raise ValueError(f"smear_width must be non-negative, got {self.smear_width}")

    @property  # type: ignore[override]
    def model_type(self) -> Type["SmearQwen3LMHeadModel"]:
        return SmearQwen3LMHeadModel

    @property
    def uses_rope(self) -> bool:
        return not isinstance(self.rope, NoRotaryEmbeddingsConfig)

    def total_trainable_params(self, vocab_size: int) -> int:
        """Base parameters plus the smear's gate and per-channel weights."""
        base = super().total_trainable_params(vocab_size)
        if self.smear_width == 0:
            return base
        gate_in = min(GATE_DIMS, self.Embed.size)
        return base + self.smear_width * (self.Embed.size + gate_in + 1)


class SmearQwen3LMHeadModel(LlamaLMHeadModel):
    """Qwen3 with :class:`SmearEmbedding` in place of ``LlamaEmbedding``."""

    @classmethod
    def init(cls, Vocab: Axis, config: SmearQwen3Config, *, key):  # type: ignore[override]
        k_t, k_emb = jrandom.split(key, 2)
        transformer = LlamaTransformer.init(config, key=k_t)
        embeddings = SmearEmbedding.init(Vocab, config, key=k_emb)
        if config.tie_word_embeddings:
            lm_head = None
        else:
            lm_head = hnn.Linear.init(In=config.Embed, Out=Vocab, key=k_emb, use_bias=False, out_first=True)
        return SmearQwen3LMHeadModel(transformer, embeddings, lm_head)


__all__ = [
    "GATE_DIMS",
    "NoRotaryEmbeddings",
    "NoRotaryEmbeddingsConfig",
    "SmearEmbedding",
    "SmearQwen3Config",
    "SmearQwen3LMHeadModel",
]
