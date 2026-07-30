# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Make a levanter HF export loadable by the vLLM 0.9.2 eval workers, in bf16.

The `hf/step-*/` exports published for #117 and #146 were written by levanter
under **transformers 5.12.1**, which serialises two things the eval stack (vLLM
0.9.2 / transformers 4.57.x) cannot read:

* ``config.json`` carries the 5.x ``rope_parameters`` block instead of the 4.x
  ``rope_theta`` + ``rope_scaling`` pair, so a 4.x ``Qwen3Config`` silently loses
  the llama3 rope scaling — the model would load and produce garbage.
* ``tokenizer_config.json`` declares ``"tokenizer_class": "TokenizersBackend"``,
  a levanter export class name ``AutoTokenizer`` cannot resolve at all.

Both are repaired here, on disk, *before* upload — deliberately, so the eval
worker (``exp82/score_rollout_worker_cw.py``) stays byte-identical to the one
that produced the published #75 / #117 numbers. The tokenizer repair reuses
``marinfold.inference._tokenizer``, the repo's canonical implementation.

The weights are recast fp32 -> bf16 at the same time. vLLM loads with
``dtype="bfloat16"`` regardless, so this is the exact rounding it would do at
load time — but it halves what has to cross the ~2.5 MB/s workstation uplink.

    uv run python prepare_hf_export.py --src <dir> --dst <dir>
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoConfig

from marinfold.inference._tokenizer import load_tokenizer

# Written by the 5.x exporter and meaningless (or actively misleading) to a 4.x
# config: `dtype: null` overrides the load-time dtype, and the label maps are
# sequence-classification boilerplate the LM head never uses.
DROP_CONFIG_KEYS = ("dtype", "id2label", "label2id", "transformers_version")


def downgrade_config(src: Path, dst: Path) -> dict:
    """Rewrite a transformers-5.x ``config.json`` into 4.x shape.

    ``rope_parameters`` is split back into the ``rope_theta`` scalar and the
    ``rope_scaling`` dict, then the whole thing is round-tripped through the
    installed (4.57.x) ``AutoConfig`` so the output is exactly what that version
    would have written — rather than a hand-edited JSON that merely looks right.
    """
    raw = json.loads((src / "config.json").read_text())
    rope = raw.pop("rope_parameters", None)
    if rope is not None:
        rope = dict(rope)
        raw["rope_theta"] = rope.pop("rope_theta")
        raw["rope_scaling"] = rope
    for key in DROP_CONFIG_KEYS:
        raw.pop(key, None)

    config = AutoConfig.for_model(**raw)
    config.save_pretrained(dst)
    print(f"[prepare] config: rope_theta={config.rope_theta} "
          f"rope_scaling={config.rope_scaling} layers={config.num_hidden_layers} "
          f"hidden={config.hidden_size} vocab={config.vocab_size}")
    return raw


def repair_tokenizer(src: Path, dst: Path) -> None:
    """Write a tokenizer whose ``tokenizer_class`` ``AutoTokenizer`` can resolve."""
    tokenizer = load_tokenizer(src)
    tokenizer.save_pretrained(dst)
    print(f"[prepare] tokenizer: {type(tokenizer).__name__} vocab={tokenizer.vocab_size} "
          f"eos={tokenizer.eos_token!r} pad={tokenizer.pad_token!r}")


def recast_weights(src: Path, dst: Path) -> int:
    """Copy the safetensors shards with every floating tensor cast to bf16."""
    index_path = src / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"no model.safetensors.index.json under {src}")
    index = json.loads(index_path.read_text())

    total = 0
    for shard in sorted(set(index["weight_map"].values())):
        tensors = load_file(src / shard)
        recast = {
            name: (t.to(torch.bfloat16) if t.is_floating_point() else t)
            for name, t in tensors.items()
        }
        save_file(recast, dst / shard, metadata={"format": "pt"})
        size = (dst / shard).stat().st_size
        total += size
        print(f"[prepare]   {shard}: {size / 2**30:.2f} GiB")

    index["metadata"]["total_size"] = total
    (dst / "model.safetensors.index.json").write_text(json.dumps(index))
    return total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="published hf/step-N export")
    ap.add_argument("--dst", type=Path, required=True, help="vLLM-loadable bf16 output dir")
    a = ap.parse_args()

    a.dst.mkdir(parents=True, exist_ok=True)
    downgrade_config(a.src, a.dst)
    repair_tokenizer(a.src, a.dst)
    total = recast_weights(a.src, a.dst)

    # generation_config.json, if present, is what bakes HF `generate` defaults
    # (notably top_k=50) into the checkpoint. The eval workers pass
    # generation_config="vllm" and explicit SamplingParams, so it is inert —
    # but copy it if it exists so the export stays a faithful mirror.
    for extra in ("generation_config.json",):
        if (a.src / extra).exists():
            shutil.copy(a.src / extra, a.dst / extra)

    print(f"[prepare] {total / 2**30:.2f} GiB -> {a.dst}")
    print("[prepare] files: " + ", ".join(sorted(p.name for p in a.dst.iterdir())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
