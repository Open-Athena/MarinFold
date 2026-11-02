# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Turn a trained levanter checkpoint into something vLLM can evaluate — issue #200.

marin.rl's train path writes only levanter-native checkpoints; there is no HF
export anywhere in it. So each arm needs two steps before it can be scored:

1. **levanter -> HF.** ``levanter.main.export_lm_to_hf`` with ``use_cpu=True``, so
   this runs on a CPU pod rather than occupying a v5p. The tokenizer comes from the
   published repo, which is the one where id 7 spells ``<contacts-v1.multi>``.
2. **f32 -> bf16, and repair the rope.** The trainer's parameters are f32
   (``mp="p=f32,c=bfloat16"``), and handing TPU vLLM f32 weights is a known
   failure rather than a silent cast. levanter also writes the Llama3 rope under
   ``rope_parameters`` leaving top-level ``rope_theta`` null, which makes any
   reader older than transformers 5 fall back to default rope — a 50x wrong base
   frequency that already cost exp163 a round of evals.

The bf16-and-rope logic is exp163's ``stage_v3_to_gcs.stage_model``, vendored
because iris bundles one directory and a sibling-experiment import fails on the pod.

Evaluation itself can read the ``gs://`` export directly: ``phase1_parity.stage()``
copies to local disk first, so the tokenizer loader never sees a URL. Only the RL
rollout worker needed an HF repo id.
"""

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path

import fsspec

# Copied verbatim from exp163: never re-serialize these, or transformers rewrites
# the rope key and emits a tokenizer_class the marin worker cannot import.
VERBATIM = ("config.json", "tokenizer.json", "tokenizer_config.json",
            "special_tokens_map.json", "generation_config.json")


def latest_step(checkpoint_dir: str) -> str:
    """The highest step-N under a levanter checkpoint directory."""
    fs, _ = fsspec.core.url_to_fs(checkpoint_dir)
    steps = []
    for path in fs.ls(checkpoint_dir.rstrip("/"), detail=False):
        name = path.rsplit("/", 1)[-1]
        if name.startswith("step-") and name[5:].isdigit():
            steps.append((int(name[5:]), name))
    if not steps:
        raise SystemExit(f"no step-N checkpoints under {checkpoint_dir}")
    return max(steps)[1]


def cast_bf16_and_fix_rope(src: str, dst: str) -> None:
    """bf16 the tensors, translate the rope config, copy metadata verbatim."""
    import torch
    from safetensors.torch import load_file, save_file

    fs_src, _ = fsspec.core.url_to_fs(src)
    fs_dst, _ = fsspec.core.url_to_fs(dst)
    names = [p.rsplit("/", 1)[-1] for p in fs_src.ls(src.rstrip("/"), detail=False)]

    with tempfile.TemporaryDirectory(prefix="exp200-cast-") as tmp:
        local = Path(tmp)
        for name in names:
            if not name:
                continue
            with fs_src.open(f"{src.rstrip('/')}/{name}", "rb") as fh, open(local / name, "wb") as out:
                shutil.copyfileobj(fh, out, length=32 << 20)

        config = json.loads((local / "config.json").read_text())
        if config.get("rope_theta") is None and config.get("rope_parameters"):
            params = dict(config["rope_parameters"])
            theta = params.pop("rope_theta", None)
            config["rope_theta"] = theta
            config["rope_scaling"] = params or None
            config.pop("rope_parameters", None)
            print(f"[export] repaired rope: rope_theta={theta}", flush=True)
        if config.get("rope_theta") is None:
            raise SystemExit(
                "[export] FATAL: no rope_theta and no rope_parameters to derive it from; "
                "vLLM would silently use default rope."
            )
        (local / "config.json").write_text(json.dumps(config, indent=2))

        total = 0
        for name in names:
            if not name.endswith(".safetensors"):
                continue
            tensors = load_file(str(local / name))
            cast = {
                k: (v.to(torch.bfloat16) if v.dtype in (torch.float32, torch.float64, torch.float16) else v)
                for k, v in tensors.items()
            }
            save_file(cast, str(local / name), metadata={"format": "pt"})
            total += (local / name).stat().st_size
            print(f"[export] cast {name} -> bf16", flush=True)

        index = local / "model.safetensors.index.json"
        if index.exists():
            meta = json.loads(index.read_text())
            meta.setdefault("metadata", {})["total_size"] = total
            index.write_text(json.dumps(meta, indent=2))

        for name in names:
            if not name:
                continue
            with open(local / name, "rb") as fh, fs_dst.open(f"{dst.rstrip('/')}/{name}", "wb") as out:
                shutil.copyfileobj(fh, out, length=32 << 20)
    print(f"[export] wrote {dst}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", required=True,
                    help="levanter checkpoint dir holding step-N subdirectories")
    ap.add_argument("--out", required=True, help="destination for the bf16 HF export")
    ap.add_argument("--step", default=None, help="step-N to export; default is the highest")
    ap.add_argument("--tokenizer", required=True, help="HF repo id of the renamed tokenizer")
    ap.add_argument("--vocab-size", type=int, default=2845)
    a = ap.parse_args()

    step = a.step or latest_step(a.checkpoint_dir)
    src = f"{a.checkpoint_dir.rstrip('/')}/{step}"
    print(f"[export] exporting {src}", flush=True)

    import jmp
    from levanter.main.export_lm_to_hf import ConvertLmConfig
    from levanter.main.export_lm_to_hf import main as export_main
    from levanter.trainer import TrainerConfig

    import rl_config

    t0 = time.time()
    staging = f"{a.out.rstrip('/')}-f32"
    export_main(
        ConvertLmConfig(
            trainer=TrainerConfig(mp=jmp.get_policy("p=f32")),
            checkpoint_path=src,
            output_dir=staging,
            model=rl_config.MODEL_CONFIG,
            tokenizer=a.tokenizer,
            override_vocab_size=a.vocab_size,
            save_tokenizer=True,
            use_cpu=True,
        )
    )
    print(f"[export] HF export done in {time.time() - t0:.0f}s -> {staging}", flush=True)

    cast_bf16_and_fix_rope(staging, a.out)
    resolved = rl_config.preflight_checkpoint(a.out)
    print(f"[export] DONE {a.out} (step={step}, vocab_size={resolved})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
