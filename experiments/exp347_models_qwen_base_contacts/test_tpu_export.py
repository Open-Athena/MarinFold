"""Exercise inverse mappings and a real Transformers reload of the TPU export."""

import json
from pathlib import Path

import fsspec
import numpy as np
import torch
from flax import nnx
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

from tpu_checkpoint import ROOT
from tpu_export import export_checkpoint, hf_parameters
from tpu_model import abstract_model, configuration, convert_parameters
from validate_tpu_math import probe_config


def test_tpu_hf_export_reload(tmp_path: Path) -> None:
    """Recover every FP32 coordinate exactly and serialize BF16 without omissions."""
    torch.set_num_threads(4)
    torch.manual_seed(347)
    source = probe_config()
    model = Qwen3_5ForCausalLM(Qwen3_5TextConfig(**source)).eval()
    original = {
        name: value.detach().numpy() for name, value in model.named_parameters()
    }
    config = configuration(source, length=128, batch=2, dtype="float32")
    _, abstract = abstract_model(config)
    _, template, _ = nnx.split(abstract, nnx.Param, ...)
    params = convert_parameters(original, source, config, template).to_pure_dict()
    restored = dict(hf_parameters(params, template, source, config))
    assert set(restored) == set(original)
    for name in original:
        np.testing.assert_array_equal(original[name], restored[name])
    fs = fsspec.filesystem("memory")
    name = "test-exp347-tpu-export"
    root = f"{ROOT}/checkpoints/{name}"
    model.config.save_pretrained(tmp_path)
    fs.pipe_file(root + "/source/config.json", (tmp_path / "config.json").read_bytes())
    fs.pipe_file(root + "/source/tokenizer.json", b'{"test": true}')
    metadata = {
        "step": 59321,
        "tokens": 9123000000,
        "format": "contacts_v1",
        "target_tokens": 100000000000,
    }
    request = export_checkpoint(
        fs, root, params, template, source, config, metadata, root + "/source", name
    )
    destination = tmp_path / "export"
    destination.mkdir()
    for path in fs.glob(request["checkpoint"] + "/*"):
        fs.get_file(path, str(destination / path.rsplit("/", 1)[-1]))
    reloaded = Qwen3_5ForCausalLM.from_pretrained(destination, local_files_only=True)
    assert set(dict(reloaded.named_parameters())) == set(original)
    for key, value in reloaded.named_parameters():
        torch.testing.assert_close(
            value, torch.tensor(original[key]).to(torch.bfloat16), rtol=0, atol=0
        )
    assert json.loads((destination / "tokenizer.json").read_text()) == {"test": True}
    assert (
        export_checkpoint(
            fs, root, params, template, source, config, metadata, root + "/source", name
        )
        == request
    )
