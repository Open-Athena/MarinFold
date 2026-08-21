# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert a SkyRL FSDP checkpoint to an HF model directory — issue #208.

SkyRL writes `policy/model_world_size_N_rank_R.pt` plus a `policy/huggingface/`
directory holding the config and tokenizer. At `world_size=1` (the only
configuration exp208 may use — sharding diverges from the inference engines, see
README) rank 0 holds the whole model, so this is a load-and-save rather than a
gather.

The tokenizer is copied alongside the weights, not left behind: a contacts-v1
checkpoint without its tokenizer is unusable, and separating them is a mistake
this project has already paid for once.

    python export_skyrl_checkpoint.py --ckpt ~/ckpts_armS_final/global_step_40 \
        --out ~/hf_armS_step40
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="a global_step_N directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    ckpt = Path(args.ckpt).expanduser()
    policy = ckpt / "policy"
    hf_src = policy / "huggingface"

    cfg_json = json.loads((hf_src / "config.json").read_text())
    world = json.loads((policy / "fsdp_config.json").read_text()).get("world_size", 1)
    if world != 1:
        raise SystemExit(
            f"world_size={world}: this checkpoint is sharded across ranks and rank 0 does not "
            "hold the whole model. exp208 only trains unsharded (see README)."
        )

    shard = policy / "model_world_size_1_rank_0.pt"
    print(f"loading {shard} ...")
    state = torch.load(shard, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise SystemExit(f"unexpected checkpoint payload: {type(state)}")
    # SkyRL may nest the tensors under a key; take the mapping of tensors.
    if not any(isinstance(v, torch.Tensor) for v in state.values()):
        for key in ("model", "state_dict", "module"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    state = {k.removeprefix("_orig_mod.").removeprefix("module."): v for k, v in state.items()}

    # FSDP2 stores parameters as DTensor even at world_size=1, and copying a
    # DTensor into a plain module raises "mixed torch.Tensor and DTensor". At one
    # rank the local shard IS the whole tensor, so to_local() is exact and needs
    # no process group; full_tensor() would try to collective-gather and hang.
    def _local(v):
        if hasattr(v, "to_local"):
            placements = getattr(v, "placements", ())
            if any(getattr(p, "is_shard", lambda: False)() for p in placements) and \
                    getattr(getattr(v, "device_mesh", None), "size", lambda: 1)() > 1:
                raise SystemExit("DTensor is genuinely sharded across >1 rank; cannot to_local()")
            return v.to_local()
        return v

    state = {k: _local(v) for k, v in state.items()}

    config = AutoConfig.from_pretrained(hf_src)
    dtype = getattr(torch, args.dtype)
    print(f"materialising {config.model_type} ({cfg_json.get('vocab_size')} vocab) ...")
    model = AutoModelForCausalLM.from_config(config, dtype=dtype)
    missing, unexpected = model.load_state_dict(
        {k: v.to(dtype) for k, v in state.items() if isinstance(v, torch.Tensor)}, strict=False)
    # Loud: a silently partial load is a random-init lm_head, which reads as a
    # trained model right up until the numbers make no sense.
    real_missing = [k for k in missing if "rotary" not in k and "inv_freq" not in k]
    if real_missing:
        raise SystemExit(f"{len(real_missing)} parameters not found in the checkpoint: {real_missing[:6]}")
    if unexpected:
        print(f"  note: {len(unexpected)} unexpected keys ignored, e.g. {unexpected[:3]}")

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                 "special_tokens_map.json"):
        src = hf_src / name
        if src.exists():
            shutil.copy2(src, out / name)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
