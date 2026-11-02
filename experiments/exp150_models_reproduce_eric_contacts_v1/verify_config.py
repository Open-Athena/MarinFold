# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Assert the built exp150 train step matches Eric's exp117 point, knob for knob.

The preview in ``train_contacts_v1_repro`` prints what we *intend*; this asserts
what ``default_train`` actually *built*, after every default, override and
``versioned()`` wrapper has been applied. That gap is exactly where a
reproduction silently stops being one — e.g. a levanter default drifting, or
``steps_per_hf_export`` inheriting ``steps_per_export``.

Every expected value below is transcribed from ``marin@origin/eac/plm-exp117:
experiments/protein/exp117_sweep.py`` (``e0f3da1``); the ``source`` column in
each row says where. Run it before launching, and again after any dependency
bump::

    uv run --no-sync python verify_config.py

Exits non-zero on the first mismatch, listing every failure.
"""
from __future__ import annotations

import sys

from levanter.data.text.datasets import BlockShuffleConfig
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig

import contacts_v1_repro_common as common
from train_contacts_v1_repro import build_steps


def _unwrap(value):
    """Unwrap marin's ``versioned()`` wrapper (``VersionedValue``) if present."""
    return getattr(value, "value", value)


def main() -> int:
    step = build_steps()[0]
    pod = step.config
    tc = pod.train_config
    trainer = tc.trainer
    opt = tc.optimizer
    data = tc.data

    # (label, actual, expected, source in exp117_sweep.py)
    checks = [
        # --- the swept point --------------------------------------------------
        ("learning_rate", _unwrap(opt.learning_rate), 3.1623e-3, "point e16-lr3p162e-3-wd0p2-bs128"),
        ("weight_decay", opt.weight_decay, 0.2, "point"),
        ("train_batch_size", trainer.train_batch_size, 128, "point"),
        ("num_train_steps", _unwrap(trainer.num_train_steps), 71_360, "Point.num_train_steps = 16 * 4460"),
        ("train_seq_len", tc.train_seq_len, 8192, "SEQ_LEN"),
        # --- schedule ---------------------------------------------------------
        ("lr_schedule", _unwrap(opt.lr_schedule), "cosine", "LR_SCHEDULE"),
        ("warmup", opt.warmup, 0.1, "WARMUP"),
        # --- AdamConfig values Eric takes as levanter defaults ----------------
        ("beta1", opt.beta1, 0.9, "levanter AdamConfig default"),
        ("beta2", opt.beta2, 0.95, "levanter AdamConfig default"),
        ("epsilon", opt.epsilon, 1e-8, "levanter AdamConfig default"),
        ("max_grad_norm", opt.max_grad_norm, 1.0, "levanter AdamConfig default"),
        ("min_lr_ratio", opt.min_lr_ratio, 0.1, "levanter OptimizerConfig default"),
        ("z_loss_weight", tc.z_loss_weight, None, "train_lm(z_loss_weight=None)"),
        # --- eval + checkpoint cadence ----------------------------------------
        ("steps_per_eval", trainer.steps_per_eval, 2_230, "Point.steps_per_eval = round(4460/2)"),
        ("max_eval_batches", trainer.max_eval_batches, None, "_apply_recipe_overrides (full val)"),
        ("checkpointer.keep", trainer.checkpointer.keep, [{"every": 4_460}],
         "production_shape: 1 permanent ckpt/epoch"),
        # No in-training HF export. We pass steps_per_hf_export=-1, which
        # marinfold's _resolve_hf_export_steps maps to hf_save_steps=None (the
        # disabled state). Left unset it would INHERIT steps_per_export and
        # export every epoch, which Eric's run does not do.
        ("hf_save_steps", tc.hf_save_steps, None, "not in exp117; disabled deliberately"),
        # --- data -------------------------------------------------------------
        ("data_seed", _unwrap(tc.data_seed), 0, "DATA_SEED"),
        ("shuffle", data.shuffle,
         BlockShuffleConfig(io_block_size=256, window_blocks=512, perm_type="feistel"), "SHUFFLE"),
        ("block_cross_document_attention", data.block_cross_document_attention, True,
         "levanter default (exp117 leaves it unset)"),
        ("tokenizer", data.tokenizer, "timodonnell/contacts-v1-tokenizer", "TOKENIZER (bare id; see common)"),
        ("train weight", data.train_weights.get("contacts-v1-train"), 1.0, "datasets={train_cache: 1.0}"),
        ("val weight", data.train_weights.get("contacts-v1-val"), 0.0, "validation=[val_cache]"),
        # --- model ------------------------------------------------------------
        ("model.max_seq_len", tc.model.max_seq_len, 8192, "MODEL_CONFIG"),
        ("model.hidden_dim", tc.model.hidden_dim, 2048, "MODEL_CONFIG"),
        ("model.intermediate_dim", tc.model.intermediate_dim, 8192, "MODEL_CONFIG"),
        ("model.num_heads", tc.model.num_heads, 32, "MODEL_CONFIG"),
        ("model.num_kv_heads", tc.model.num_kv_heads, 8, "MODEL_CONFIG"),
        ("model.num_layers", tc.model.num_layers, 24, "MODEL_CONFIG"),
        ("model.rope", tc.model.rope, Llama3RotaryEmbeddingsConfig(), "MODEL_CONFIG"),
        # --- from scratch -----------------------------------------------------
        ("initialize_from_checkpoint_path", tc.initialize_from_checkpoint_path, None, "from scratch"),
        # levanter types this ``bool | str = False``; False is the "don't" state.
        ("initialize_from_hf", tc.initialize_from_hf, False, "from scratch"),
    ]

    failures = []
    for label, actual, expected, source in checks:
        ok = actual == expected
        print(f"  {'ok ' if ok else 'FAIL'}  {label:32} {actual!r:>52}   [{source}]")
        if not ok:
            failures.append((label, actual, expected))

    # Packing is per-component; assert it on every one (exp117 _apply_recipe_overrides).
    for key, component in data.components.items():
        ok = component.pack is True
        print(f"  {'ok ' if ok else 'FAIL'}  {'pack[' + key + ']':32} {component.pack!r:>52}   [pack-prefix-only]")
        if not ok:
            failures.append((f"pack[{key}]", component.pack, True))

    # Derivation sanity: the formulas, not just their current outputs.
    ok = common.steps_per_epoch() == 4_460 and common.steps_for_epochs() == 71_360
    print(f"  {'ok ' if ok else 'FAIL'}  {'steps formula':32} "
          f"{f'{common.steps_per_epoch()}/epoch, {common.steps_for_epochs()} total':>52}   "
          f"[round(TRAIN_TOKENS / (bs*seq)) * epochs]")
    if not ok:
        failures.append(("steps formula", common.steps_for_epochs(), 71_360))

    print()
    if failures:
        print(f"{len(failures)} MISMATCH(ES) vs Eric's exp117 point:")
        for label, actual, expected in failures:
            print(f"  {label}: got {actual!r}, expected {expected!r}")
        return 1
    print(f"All {len(checks) + len(data.components) + 1} checks pass — "
          f"config matches exp117 e16-lr3p162e-3-wd0p2-bs128.")
    print(f"Target: eval/contacts-v1-val/loss = {common.TARGET_VAL_LOSS} (+-0.01)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
