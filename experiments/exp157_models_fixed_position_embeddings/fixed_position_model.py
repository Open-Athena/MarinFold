"""Llama variant with fixed residue-position input embeddings.

The contacts-v1 tokenizer represents residue locations as ordinary vocabulary
items (``<p0>`` ... ``<p1999>``). A stock LM therefore learns a free vector for
every residue-location token. This experiment removes those rows from the
trainable input embedding table and synthesizes their input vectors from the
residue index instead.
"""

import dataclasses
from dataclasses import dataclass
from typing import Dict, Optional, Type, Union, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import PRNGKeyArray

import haliax as hax
import haliax.nn as hnn
from haliax import Axis, AxisSpec, NamedArray
from haliax.jax_utils import maybe_rng_split, named_call
from haliax.nn.normalization import LayerNormBase
from haliax.state_dict import ModuleWithStateDictSerialization

from levanter.layers.attention import AttentionMask
from levanter.models.lm_model import LmConfig, LmHeadModel, resize_embeddings_and_lm_head
from levanter.models.llama import LlamaConfig, LlamaTransformer
from levanter.models.qwen import Qwen3Config


@dataclass(frozen=True)
class ResiduePositionEmbeddingSpec:
    """Vocabulary span whose input embeddings are deterministic residue vectors.

    Attributes:
        start_token_id: Token id for ``<p0>``.
        num_tokens: Number of residue-position tokens in the contiguous span.
        base: Frequency base used for the sinusoidal/RoPE-style features.
    """

    start_token_id: int
    num_tokens: int
    base: float = 10_000.0

    def validate(self, vocab_size: int) -> None:
        """Raise ``ValueError`` if the configured span is not inside the vocab."""
        if self.start_token_id < 0:
            raise ValueError("start_token_id must be non-negative")
        if self.num_tokens <= 0:
            raise ValueError("num_tokens must be positive")
        end_token_id = self.start_token_id + self.num_tokens
        if end_token_id > vocab_size:
            raise ValueError(
                f"fixed position span [{self.start_token_id}, {end_token_id}) exceeds "
                f"vocab size {vocab_size}"
            )


def fixed_rope_position_vectors(
    position_ids: NamedArray,
    Embed: Axis,
    *,
    base: float = 10_000.0,
) -> NamedArray:
    """Return deterministic RoPE-style vectors for residue position numbers.

    The returned vector is the usual sinusoidal feature map used by rotary and
    absolute sinusoidal position encodings: even channels are sine terms and odd
    channels are cosine terms at geometrically spaced frequencies. It is a pure
    function of the residue index, so it has no optimizer state and cannot drift
    during training.
    """
    if Embed.size % 2 != 0:
        raise ValueError(f"fixed position embedding requires an even Embed size, got {Embed.size}")

    half_dim = Embed.size // 2
    raw_positions = position_ids.array.astype(jnp.float32)[..., None]
    channel = jnp.arange(half_dim, dtype=jnp.float32)
    inv_freq = base ** (-2.0 * channel / Embed.size)
    angles = raw_positions * inv_freq
    vectors = jnp.stack((jnp.sin(angles), jnp.cos(angles)), axis=-1).reshape(
        *position_ids.array.shape,
        Embed.size,
    )
    return hax.NamedArray(vectors, (*position_ids.axes, Embed))


def residue_position_spec_from_tokenizer(
    tokenizer,
    *,
    num_tokens: int = 2000,
    token_template: str = "<p{}>",
    base: float = 10_000.0,
) -> ResiduePositionEmbeddingSpec:
    """Build a fixed-position spec from a tokenizer with contiguous ``<pN>`` ids."""
    ids = [int(tokenizer.convert_tokens_to_ids(token_template.format(i))) for i in range(num_tokens)]
    expected = list(range(ids[0], ids[0] + num_tokens))
    if ids != expected:
        raise ValueError(
            f"position token ids are not contiguous for {token_template!r}: "
            f"first ids={ids[:8]}, last ids={ids[-8:]}"
        )
    return ResiduePositionEmbeddingSpec(start_token_id=ids[0], num_tokens=num_tokens, base=base)


class FixedResiduePositionEmbedding(eqx.Module):
    """Input embedding table with deterministic vectors for residue locations.

    Non-position tokens are stored in a compact learned table with the fixed span
    removed. Position-token inputs are ignored by that table and replaced with a
    deterministic sinusoidal vector derived from ``token_id - start_token_id``.
    """

    token_embeddings: hnn.Embedding

    Vocab: Axis = eqx.field(static=True)
    Embed: AxisSpec = eqx.field(static=True)
    position_spec: ResiduePositionEmbeddingSpec = eqx.field(static=True)
    norm: Optional[LayerNormBase] = None

    @staticmethod
    def init(
        Vocab: Axis,
        Embed: Axis,
        position_spec: ResiduePositionEmbeddingSpec,
        *,
        key: PRNGKeyArray,
        norm: Optional[LayerNormBase] = None,
    ) -> "FixedResiduePositionEmbedding":
        """Initialize trainable non-position embeddings and static position span."""
        position_spec.validate(Vocab.size)
        NonPositionVocab = Axis("non_position_vocab", Vocab.size - position_spec.num_tokens)
        token_embeddings = hnn.Embedding.init(NonPositionVocab, Embed, key=key)
        return FixedResiduePositionEmbedding(token_embeddings, Vocab, Embed, position_spec, norm)

    def _compact_ids(self, input_ids: NamedArray) -> NamedArray:
        """Map full-vocab token ids to the compact non-position embedding table."""
        start = self.position_spec.start_token_id
        stop = start + self.position_spec.num_tokens
        ids_after_fixed_span = input_ids >= stop
        compact = hax.where(ids_after_fixed_span, input_ids - self.position_spec.num_tokens, input_ids)
        return hax.where(self.is_position_token(input_ids), 0, compact).astype(jnp.int32)

    def is_position_token(self, input_ids: NamedArray) -> NamedArray:
        """Return a boolean mask for ids in the fixed residue-position span."""
        start = self.position_spec.start_token_id
        stop = start + self.position_spec.num_tokens
        return (input_ids >= start) & (input_ids < stop)

    @named_call
    def embed(self, input_ids: NamedArray) -> NamedArray:
        """Embed full-vocabulary token ids, synthesizing fixed position vectors."""
        learned = self.token_embeddings(self._compact_ids(input_ids))
        residue_positions = (input_ids - self.position_spec.start_token_id).astype(jnp.float32)
        fixed = fixed_rope_position_vectors(
            residue_positions,
            cast(Axis, self.Embed),
            base=self.position_spec.base,
        ).astype(learned.dtype)
        embedded = hax.where(self.is_position_token(input_ids), fixed, learned)
        if self.norm is not None:
            embedded = self.norm(embedded)
        return embedded

    def __call__(self, input_ids: NamedArray, *, key: PRNGKeyArray | None = None) -> NamedArray:
        """Alias for ``embed``; ``key`` is ignored for hnn.Embedding compatibility."""
        return self.embed(input_ids)

    def unembed(self, input_embeds: NamedArray) -> NamedArray:
        """Tied output embeddings are intentionally unsupported for this experiment."""
        raise NotImplementedError(
            "FixedResiduePositionEmbedding removes position rows from the input table; "
            "use an untied lm_head for output logits."
        )

    def resize_embeddings(self, new_size: int, key: Optional[PRNGKeyArray] = None):
        """Resize the full vocabulary while preserving the fixed position span."""
        self.position_spec.validate(new_size)
        new_non_position_size = new_size - self.position_spec.num_tokens
        new_token_embeddings = self.token_embeddings.resize_embeddings(new_non_position_size, key=key)
        return dataclasses.replace(
            self,
            Vocab=self.Vocab.resize(new_size),
            token_embeddings=new_token_embeddings,
        )


@LmConfig.register_subclass("fixed_residue_position_llama")
@dataclass(frozen=True)
class FixedResiduePositionLlamaConfig(LlamaConfig):
    """Llama config whose residue-position input vectors are fixed features."""

    position_embedding: ResiduePositionEmbeddingSpec = dataclasses.field(
        default_factory=lambda: ResiduePositionEmbeddingSpec(start_token_id=0, num_tokens=2000)
    )

    @property
    def model_type(self) -> Type["FixedResiduePositionLlamaLMHeadModel"]:  # pyrefly: ignore[bad-override]
        return FixedResiduePositionLlamaLMHeadModel

    def total_trainable_params(self, vocab_size):
        """Return the stock count minus removed trainable input-position rows."""
        stock = super().total_trainable_params(vocab_size)
        removed = self.position_embedding.num_tokens * self.hidden_dim
        return stock - removed


@LmConfig.register_subclass("fixed_residue_position_qwen3")
@dataclass(frozen=True)
class FixedResiduePositionQwen3Config(Qwen3Config):
    """Qwen3 config whose residue-position input vectors are fixed features."""

    position_embedding: ResiduePositionEmbeddingSpec = dataclasses.field(
        default_factory=lambda: ResiduePositionEmbeddingSpec(start_token_id=0, num_tokens=2000)
    )

    @property
    def model_type(self) -> Type["FixedResiduePositionLlamaLMHeadModel"]:  # pyrefly: ignore[bad-override]
        return FixedResiduePositionLlamaLMHeadModel

    def total_trainable_params(self, vocab_size):
        """Return the stock count minus removed trainable input-position rows."""
        stock = super().total_trainable_params(vocab_size)
        removed = self.position_embedding.num_tokens * self.hidden_dim
        return stock - removed


class FixedResiduePositionLlamaLMHeadModel(ModuleWithStateDictSerialization, LmHeadModel[FixedResiduePositionLlamaConfig]):
    """Llama/Qwen3 LM with deterministic input embeddings for residue-position tokens."""

    transformer: LlamaTransformer
    embeddings: FixedResiduePositionEmbedding
    lm_head: Optional[hnn.Linear]

    @property
    def config(self):
        return self.transformer.config

    @property
    def vocab_size(self) -> int:
        return self.Vocab.size

    @property
    def Vocab(self) -> Axis:
        return self.embeddings.Vocab

    @classmethod
    def init(
        cls,
        Vocab: Axis,
        config: FixedResiduePositionLlamaConfig | FixedResiduePositionQwen3Config,
        *,
        key: PRNGKeyArray,
    ) -> "FixedResiduePositionLlamaLMHeadModel":
        """Initialize the transformer, compact input embeddings, and LM head."""
        if config.tie_word_embeddings:
            raise ValueError("fixed residue-position embeddings require tie_word_embeddings=False")
        k_t, k_emb, k_head = jrandom.split(key, 3)
        transformer = LlamaTransformer.init(config, key=k_t)
        norm = None
        if config.input_embedding_norm:
            norm = config.mk_LayerNorm(config.Embed)
        embeddings = FixedResiduePositionEmbedding.init(
            Vocab,
            config.Embed,
            config.position_embedding,
            key=k_emb,
            norm=norm,
        )
        lm_head = hnn.Linear.init(
            In=config.Embed,
            Out=Vocab,
            key=k_head,
            use_bias=False,
            out_first=True,
        )
        return FixedResiduePositionLlamaLMHeadModel(transformer, embeddings, lm_head)

    def __call__(
        self,
        input_ids: NamedArray,
        attn_mask: Optional[Union[NamedArray, AttentionMask]] = None,
        pos_ids: NamedArray | None = None,
        *,
        key=None,
    ) -> NamedArray:
        """Return next-token logits for ``input_ids``."""
        k_t, k_head = maybe_rng_split(key, 2)
        x = self.embeddings.embed(input_ids)
        x = self.transformer(x, attn_mask=attn_mask, key=k_t, pos_ids=pos_ids)
        if self.lm_head is None:
            raise RuntimeError("fixed residue-position model always uses an untied lm_head")
        return self.lm_head(x, key=k_head)

    def activations(
        self,
        input_ids: NamedArray,
        attn_mask: Optional[AttentionMask | NamedArray] = None,
        *,
        key=None,
        pos_ids: NamedArray | None = None,
    ) -> NamedArray:
        """Return hidden states before the LM head."""
        x = self.embeddings.embed(input_ids)
        return self.transformer(x, attn_mask=attn_mask, key=key, pos_ids=pos_ids)

    def get_lm_head(self) -> hax.NamedArray:
        """Return the untied output projection matrix."""
        if self.lm_head is None:
            raise RuntimeError("fixed residue-position model always uses an untied lm_head")
        return self.lm_head.weight

    def resize_vocab(self, new_size: int, key=None) -> "FixedResiduePositionLlamaLMHeadModel":
        """Resize the compact input table and full output LM head."""
        new_embeddings, new_lm_head = resize_embeddings_and_lm_head(
            self.Vocab,
            self.embeddings,
            self.lm_head,
            new_size,
            key,
        )
        return dataclasses.replace(self, embeddings=new_embeddings, lm_head=new_lm_head)

    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        return {"transformer": "model", "embeddings": None}
