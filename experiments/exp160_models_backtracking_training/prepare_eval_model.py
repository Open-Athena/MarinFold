# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Make a levanter HF export loadable by the vLLM eval workers, in bf16 (#160).

Same job as exp169's ``prepare_hf_export.py`` — levanter writes its exports
under **transformers 5.12.1**, which serialises two things the vLLM eval stack
(transformers 4.57.x) cannot read:

* ``config.json`` carries the 5.x ``rope_parameters`` block instead of the 4.x
  ``rope_theta`` + ``rope_scaling`` pair, so a 4.x ``Qwen3Config`` silently
  loses the llama3 rope scaling — the model loads and degrades *with sequence
  length*, which is the failure mode most likely to be mistaken for a result;
* ``tokenizer_config.json`` declares ``"tokenizer_class": "TokenizersBackend"``,
  a levanter export class name ``AutoTokenizer`` cannot resolve at all.

The weights are recast fp32 -> bf16 at the same time; TPU parameters are bf16
and vLLM shards the checkpoint as it loads, so handing it fp32 is a hard
failure rather than a silent cast.

**Why this is not exp169's script.** That one needs torch (for the cast) and a
transformers 4.57 runtime (to round-trip the config); this one runs in a marin
checkout's venv, which has neither — it casts through ``jax`` and writes bf16
safetensors through ``safetensors.flax``, and rewrites the config as plain JSON.
That keeps export (``export_trained_to_hf.py``, levanter) and preparation in one
environment, so a trained checkpoint reaches a vLLM-ready directory without a
second venv. ``verify_eval_model.py`` then checks the result under a real
transformers 4.57, which is the assertion that actually matters.

    /home/bizon/git/marin-freshiris/.venv/bin/python prepare_eval_model.py \\
        --src <fp32 export dir> --dst <bf16 dir> [--tokenizer <dir>]
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

# Written by the 5.x exporter and meaningless (or actively misleading) to a 4.x
# config: `dtype: null` overrides the load-time dtype, and the label maps are
# sequence-classification boilerplate the LM head never uses.
DROP_CONFIG_KEYS = ("dtype", "id2label", "label2id", "transformers_version")

TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")


def downgrade_config(src: Path, dst: Path, *, vocab_size: int | None) -> dict:
    """Rewrite a transformers-5.x ``config.json`` into 4.x shape.

    ``rope_parameters`` is split back into the ``rope_theta`` scalar and the
    ``rope_scaling`` dict. Idempotent: a config already in 4.x shape passes
    through untouched.
    """
    raw = json.loads((src / "config.json").read_text())
    rope = raw.pop("rope_parameters", None)
    if rope is not None:
        rope = dict(rope)
        raw["rope_theta"] = rope.pop("rope_theta")
        raw["rope_scaling"] = rope
    for key in DROP_CONFIG_KEYS:
        raw.pop(key, None)
    if "rope_scaling" not in raw or "rope_theta" not in raw:
        raise SystemExit(
            f"{src}/config.json has neither rope_parameters nor rope_theta+rope_scaling; "
            "refusing to write a config whose rope would silently default"
        )
    if vocab_size is not None and raw.get("vocab_size") != vocab_size:
        raise SystemExit(f"config vocab_size={raw.get('vocab_size')} != expected {vocab_size}")
    (dst / "config.json").write_text(json.dumps(raw, indent=2))
    print(f"[prepare] config: rope_theta={raw['rope_theta']} rope_scaling={raw['rope_scaling']} "
          f"layers={raw['num_hidden_layers']} hidden={raw['hidden_size']} "
          f"vocab={raw['vocab_size']}")
    return raw


def copy_tokenizer(src: Path, dst: Path) -> None:
    """Copy the tokenizer files, forcing a ``tokenizer_class`` AutoTokenizer resolves.

    Only ``tokenizer_class`` is rewritten (plus the two levanter-only markers
    that ride with it); the vocabulary in ``tokenizer.json`` is never touched,
    because it is the thing that has to stay byte-identical to what the model
    was trained against.
    """
    if not (src / "tokenizer.json").exists():
        raise SystemExit(f"no tokenizer.json under {src} — the tokenizer travels with the model")
    shutil.copy(src / "tokenizer.json", dst / "tokenizer.json")

    cfg_path = src / "tokenizer_config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    was = cfg.get("tokenizer_class")
    cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
    for key in ("backend", "is_local", "local_files_only"):
        cfg.pop(key, None)
    (dst / "tokenizer_config.json").write_text(json.dumps(cfg, indent=2))

    stm = src / "special_tokens_map.json"
    if stm.exists():
        shutil.copy(stm, dst / "special_tokens_map.json")
    else:
        # Derivable from tokenizer_config, and AutoTokenizer is happier with it
        # present. Only the three contacts-v1 specials exist.
        keep = {k: cfg[k] for k in ("eos_token", "pad_token", "unk_token") if k in cfg}
        (dst / "special_tokens_map.json").write_text(json.dumps(keep, indent=2))
    print(f"[prepare] tokenizer: tokenizer_class {was!r} -> 'PreTrainedTokenizerFast' "
          f"(from {src})")


def recast_weights(src: Path, dst: Path, max_shard_bytes: int | None = None) -> int:
    """Copy the safetensors shards with every floating tensor cast to bf16.

    Tensors are streamed one at a time (``safe_open``) rather than loading a
    whole 5 GiB shard as fp32 and then a bf16 copy beside it.
    """
    import jax.numpy as jnp
    from safetensors import safe_open
    from safetensors.flax import save_file

    index_path = src / "model.safetensors.index.json"
    if not index_path.exists():
        raise SystemExit(f"no model.safetensors.index.json under {src}")
    index = json.loads(index_path.read_text())

    # Load every tensor once, cast, and (optionally) repack into shards below
    # `max_shard_bytes`. Repacking matters for the HF *bucket* uploader, which
    # panics partway through a multi-GB object ("is not fully completed:
    # 2257162240/2261618688 bytes") -- so a 2.3 GiB shard is not publishable
    # even though it is a perfectly good file locally.
    tensors: dict = {}
    for shard in sorted(set(index["weight_map"].values())):
        with safe_open(str(src / shard), framework="np") as fh:
            for name in fh.keys():                                  # noqa: SIM118
                arr = jnp.asarray(fh.get_tensor(name))
                tensors[name] = arr.astype(jnp.bfloat16) if jnp.issubdtype(
                    arr.dtype, jnp.floating) else arr

    def nbytes(a):
        return a.size * a.dtype.itemsize

    limit = max_shard_bytes or float("inf")
    groups, cur, cur_bytes = [], {}, 0
    for name in index["weight_map"]:            # preserve the original order
        a = tensors[name]
        if cur and cur_bytes + nbytes(a) > limit:
            groups.append(cur); cur, cur_bytes = {}, 0
        cur[name] = a
        cur_bytes += nbytes(a)
    if cur:
        groups.append(cur)

    total, weight_map = 0, {}
    n = len(groups)
    for i, group in enumerate(groups, 1):
        shard = f"model-{i:05d}-of-{n:05d}.safetensors"
        # metadata format=pt is what transformers/vLLM expect to see; the file
        # is framework-neutral either way.
        save_file(group, str(dst / shard), metadata={"format": "pt"})
        size = (dst / shard).stat().st_size
        total += size
        for name in group:
            weight_map[name] = shard
        print(f"[prepare]   {shard}: {size / 2**30:.2f} GiB ({len(group)} tensors)")

    (dst / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total}, "weight_map": weight_map}))
    return total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="fp32 levanter HF export dir")
    ap.add_argument("--dst", type=Path, required=True, help="vLLM-loadable bf16 output dir")
    ap.add_argument("--tokenizer", type=Path, default=None,
                    help="take tokenizer files from here instead of --src")
    ap.add_argument("--max-shard-bytes", type=int, default=None,
                    help="repack into shards below this size; needed to publish to "
                         "the HF bucket, whose uploader fails on multi-GB objects")
    ap.add_argument("--expect-vocab", type=int, default=None,
                    help="assert config.json vocab_size (3849 superset / 2845 contacts-v1)")
    a = ap.parse_args()

    a.dst.mkdir(parents=True, exist_ok=True)
    downgrade_config(a.src, a.dst, vocab_size=a.expect_vocab)
    copy_tokenizer(a.tokenizer or a.src, a.dst)
    total = recast_weights(a.src, a.dst, a.max_shard_bytes)

    for extra in ("generation_config.json",):
        if (a.src / extra).exists():
            shutil.copy(a.src / extra, a.dst / extra)

    print(f"[prepare] {total / 2**30:.2f} GiB -> {a.dst}")
    print("[prepare] files: " + ", ".join(sorted(p.name for p in a.dst.iterdir())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
