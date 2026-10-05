"""An inference export must reload with the original Qwen weight sharing."""

from pathlib import Path

import pytest
import torch
from train import ContactLM, inference_state_dict
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig


@pytest.mark.parametrize("tied", [True, False])
def test_qwen_export_retains_weights_and_embedding_sharing(
    tmp_path: Path, tied: bool
) -> None:
    config = Qwen3_5TextConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        layer_types=["full_attention"],
        tie_word_embeddings=tied,
    )
    model = Qwen3_5ForCausalLM(config)
    original_input = model.get_input_embeddings().weight.detach().clone()
    original_output = model.get_output_embeddings().weight.detach().clone()
    model.save_pretrained(tmp_path, state_dict=inference_state_dict(model))
    restored = Qwen3_5ForCausalLM.from_pretrained(tmp_path, dtype=torch.bfloat16)
    assert (
        restored.get_input_embeddings().weight
        is restored.get_output_embeddings().weight
    ) == tied
    torch.testing.assert_close(
        restored.get_input_embeddings().weight,
        original_input.bfloat16(),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        restored.get_output_embeddings().weight,
        original_output.bfloat16(),
        rtol=0,
        atol=0,
    )
    assert model.get_input_embeddings().weight.dtype == torch.float32


def test_loaded_backbone_enables_training_checkpointing(tmp_path: Path) -> None:
    config = Qwen3_5TextConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        layer_types=["full_attention"],
    )
    Qwen3_5ForCausalLM(config).save_pretrained(tmp_path)
    loaded = Qwen3_5ForCausalLM.from_pretrained(tmp_path)
    loaded.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    assert not loaded.training
    wrapped = ContactLM(loaded)
    assert loaded.is_gradient_checkpointing
    assert all(module.training for module in wrapped.modules())
    wrapped.eval()
    assert not loaded.training
    wrapped.train()
    assert loaded.training
