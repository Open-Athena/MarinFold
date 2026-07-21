# exp137 — train a 1.5B protein LM on contacts-and-crops-v1 (issue #137)

Train a **Qwen3 1.47B from scratch** on the **contacts-and-crops-v1** corpus
(exp130/#130 format, exp132/#132 corpus — 8192-token coordinate documents:
contacts + Pass-1 coarse boxes + a few Pass-2 fine crops), **reproducing Eric's
exp117 best-so-far training config** on **TPU (v5p) on the `marin` iris cluster**.

This is the 8k-crops analog of #121 (which trained the 32k
contacts-and-coordinates-v1 corpus). Because crops is an **8192-token** format —
exp117's *native* sequence length — Eric's recipe transfers 1:1 with **no RoPE
re-tuning**.

## Recipe (exp117 best config, verbatim)

| knob | value | source |
|---|---|---|
| model | Qwen3 1.47B (hidden 2048, inter 8192, 24 layers, 32 heads, 8 kv, Llama3 rope θ=500000, qk-norm) | exp117 `MODEL_CONFIG` |
| vocab | **3848** (crops tokenizer) | — |
| seq len | 8192 | — |
| global batch | 128 | — |
| learning rate | **3.1623e-3** (√10·1e-3) | exp117 leader |
| weight decay | **0.2** | exp117 leader |
| optimizer | AdamW, betas 0.9/0.95, cosine, `min_lr_ratio=0.1`, 10% warmup, no z-loss | exp117 |
| loss | **unmasked** (coordinates are ordinary body tokens) | SPEC |
| packing | `pack=True`, block-cross-document-attention, Feistel shuffle `data_seed=0` | — |
| **num_train_steps** | **71,359** | Eric's 16-epoch leader (see budget) |

**Token budget.** Eric's exp117 leader (contacts-v1-val **2.7112**) ran 16 epochs
of contacts-v1 = 71,359 steps @ 128×8192 = **74.8B tokens**. crops has **33.82B**
train tokens (exp132 `stats.json`), so 71,359 steps matches Eric's *token budget*
(≈ **2.21 crops epochs**), not his epoch count — the right apples-to-apples
reference for a long from-scratch run, and well past Chinchilla for 1.5B.

## Data & tokenizer

| | value |
|---|---|
| Train | `gs://marin-us-east5/protein-structure/MarinFold/exp132_contacts_and_crops_v1/documents/train/*.parquet` (2067 shards, 4,129,682 docs, 33.82B tok) |
| Val (primary) | `.../exp132_contacts_and_crops_v1/documents/val/*.parquet` (22 shards, 41,954 docs) |
| Val (secondary) | `.../exp53_contacts_v1_5x/documents/val/*.parquet` (contacts-v1 val) |
| Tokenizer | `timodonnell/contacts-and-crops-v1-tokenizer` @ `80fe4ee` (3848 vocab) |
| Text column | `document` |

The crops corpus is published to the open-athena HF **bucket** (not
levanter-addressable on the worker), so `mirror_crops_corpus.py` mirrors the
parquet shards byte-for-byte to GCS. The exp137 launch mirrored from exp132's
local generation scratch (`--source local`), whose shards are byte-identical to
the published bucket (verified by shard size).

**Two validation losses are tracked** (`eval/<key>/loss`):
- `contacts-and-crops-v1-val` — **primary** (the corpus we train on).
- `contacts-v1-val` — **secondary**. The crops tokenizer is a strict superset of
  the contacts-v1 tokenizer (ids 2–2845 byte-identical), so contacts-v1 documents
  tokenize identically and this loss is **directly comparable to Eric's exp117
  leader (2.7112)**.

## Files

- `crops_train_common.py` — shared config: Qwen3 model, tokenizer, data globs,
  resources, tokenize steps, `build_train_step` (from scratch, exp117 recipe).
- `train_crops_1_5b.py` — launcher (`executor_main`); env-tunable steps/LR/WD/slice.
- `mirror_crops_corpus.py` — HF-bucket→GCS (or local→GCS) corpus mirror.
- `export_checkpoints.py` — CPU HF export of a `step-N` checkpoint (tokenizer co-located).
- `pyproject.toml` / `uv.lock` — pinned marin (PyPI `0.2.x.dev`, `<0.3`) +
  `marinfold-models` @ main `cc1e478`.

## Launch

```bash
cd experiments/exp137_models_contacts_and_crops_v1_1_5b
uv venv && uv sync --extra tpu

# Mirror the corpus to GCS (once). The launch used the local fast path:
uv run python mirror_crops_corpus.py --source local \
    --local-dir /data/exp132_contacts_and_crops_v1_scratch/documents

# Smoke run (few steps) then the full run. Submit with the FRESH marin
# checkout's iris binary (iris-freshness gate below); it bundles THIS dir's
# old-marin pyproject, and its nested train/tokenize gangs are freshness-exempt.
EXP137_STEPS=20 EXP137_STEPS_PER_EVAL=10 EXP137_STEPS_PER_EXPORT=10 \
EXP137_NAME=exp137-smoke WANDB_API_KEY=<key> \
  /home/bizon/git/marin/.venv/bin/iris --cluster marin job run --no-wait \
    --enable-extra-resources --cpu=1 --memory=16GB --disk=16GB \
    --extra=tpu --zone=us-east5-a \
    -e WANDB_API_KEY <key> -e WANDB_ENTITY open-athena \
    -e EXP137_STEPS 20 -e EXP137_STEPS_PER_EVAL 10 -e EXP137_STEPS_PER_EXPORT 10 \
    -e EXP137_NAME exp137-smoke \
    -- python -m train_crops_1_5b

# Full run (default 71,359 steps). Bump the slice with EXP137_TPU / EXP137_ZONE.
WANDB_API_KEY=<key> \
  /home/bizon/git/marin/.venv/bin/iris --cluster marin job run --no-wait \
    --enable-extra-resources --cpu=1 --memory=16GB --disk=16GB \
    --extra=tpu --zone=us-east5-a \
    -e WANDB_API_KEY <key> -e WANDB_ENTITY open-athena \
    -- python -m train_crops_1_5b
```

**iris-freshness gate.** This dir pins an old marin (the last one carrying
`default_train`/`executor_main`); its pinned iris client is >14 days old and root
job submission is rejected. Only ROOT submissions are gated — submit the driver
with the fresh marin checkout's iris binary (`/home/bizon/git/marin/.venv/bin/iris`)
while cwd bundles this dir's pyproject. The driver pod runs old marin
(`default_train` works); its nested train/tokenize gang submissions are exempt.
Pass all env via `-e` (the driver pod does not inherit the launch shell).

**TPU slice / allocation fallback.** Default `v5p-128` @ `us-east5-a`. If a big
slice won't allocate, downsize via `EXP137_TPU` (`v5p-64` → `v5p-32` → `v6e-8`;
set `EXP137_ZONE=us-east5-b` for v6e). Global batch 128 fits all of these (larger
slices just have smaller per-device batch).

## Success criteria (issue #137)

1. Converged 1.5B crops model, exp117 recipe, checkpoints to GCS, W&B `MarinFold`.
2. Two val losses tracked (crops-val primary, contacts-v1-val secondary vs 2.7112).
3. Smoke run first.
4. Report s/step, tok/s, MFU, slice.
5. HF export co-located with the crops tokenizer; downstream exp89 contact eval.

## Status

See the PR / issue #137 for launch status.
