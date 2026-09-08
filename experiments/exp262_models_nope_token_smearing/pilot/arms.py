# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""The issue #262 architecture variants, in PyTorch, for the local pilot.

A second implementation of what ``gpu/architecture.py`` does in levanter. It
exists because the pilot runs on one local GPU with the HF stack rather than on
a pod with levanter, and it is deliberately a thin wrapper around a stock
``Qwen3ForCausalLM`` so the pilot measures the *architecture change* and not a
reimplementation of Qwen3.

The two knobs are the 2x2 of the ablation:

* ``smear_width`` — 0, or 2 for the width-3 causal smear.
* ``use_rope`` — True for the production Llama3 rope, False for NoPE.

``tests/test_arms.py`` holds the same causality and independence tests as the
levanter side, plus a direct numerical cross-check between the two smears.
"""

from dataclasses import dataclass

from smear import GATE_DIMS, NoRotaryEmbedding, SmearEmbedding
from transformers import Qwen3Config, Qwen3ForCausalLM


@dataclass(frozen=True)
class Arm:
    """One cell of the ablation."""

    key: str
    use_rope: bool
    smear_width: int
    label: str


ARMS = (
    Arm("a-rope", use_rope=True, smear_width=0, label="RoPE, no smear (control)"),
    Arm("b-rope-smear", use_rope=True, smear_width=2, label="RoPE + smear(2)"),
    Arm("c-nope-smear", use_rope=False, smear_width=2, label="NoPE + smear(2)"),
    Arm("d-nope", use_rope=False, smear_width=0, label="NoPE, no smear"),
)
ARMS_BY_KEY = {arm.key: arm for arm in ARMS}


def build_config(
    *,
    vocab_size: int,
    hidden: int,
    layers: int,
    heads: int,
    kv_heads: int,
    intermediate: int,
    max_seq_len: int,
) -> Qwen3Config:
    """A scaled-down twin of the exp232 production config.

    Same family (RMSNorm, SwiGLU, GQA, QK-norm, untied embeddings) and the same
    rope base and llama3 scaling, so the only difference from production is
    size — and, in the NoPE arms, the thing under test.
    """
    return Qwen3Config(
        vocab_size=vocab_size,
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        head_dim=hidden // heads,
        max_position_embeddings=max_seq_len,
        rope_theta=500_000,
        rope_scaling={
            "rope_type": "llama3",
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": max_seq_len,
        },
        tie_word_embeddings=False,
        attention_bias=False,
        attention_dropout=0.0,
    )


def build_model(config: Qwen3Config, arm: Arm) -> Qwen3ForCausalLM:
    """Instantiate a Qwen3 and apply the arm's two modifications."""
    model = Qwen3ForCausalLM(config)
    if arm.smear_width:
        model.model.embed_tokens = SmearEmbedding(model.model.embed_tokens, arm.smear_width)
    if not arm.use_rope:
        model.model.rotary_emb = NoRotaryEmbedding(model.model.rotary_emb)
    return model


__all__ = ["ARMS", "ARMS_BY_KEY", "Arm", "GATE_DIMS", "NoRotaryEmbedding", "SmearEmbedding", "build_config", "build_model"]
