---
marinfold_experiment:
  issue: 150
  title: "exp: reproduce Eric's best contacts-v1 1.5B run exactly (exp117 e16-lr3p162e-3-wd0p2-bs128) on MarinFold's own training path"
  kind: models
  branch: claude/eric-contacts-v1-reproduction-1c8c53
---

# exp: reproduce Eric's best contacts-v1 1.5B run exactly

**Issue:** [#150](https://github.com/Open-Athena/MarinFold/issues/150) · **Kind:** `models` · **Branch:** `claude/eric-contacts-v1-reproduction-1c8c53`

## Question

Given Eric's best contacts-v1 training config and the **exact same corpus**, does
MarinFold's own training path (`marinfold_models.defaults.default_train`)
reproduce his validation loss — i.e. is the harness a no-op, or does it cost us
something?

Target: **`eval/contacts-v1-val/loss` = 2.7112**, from
`prot-exp117-cv1-s02-1_5b-e16-lr3p162e-3-wd0p2-bs128-us-east5`
(W&B `eric-czech/marin`, tag `exp117`).

## Hypothesis

Both paths configure the same levanter `TrainLmConfig`, so the run should land
**within ±0.01 of 2.7112**. A larger gap means some part of what we've been
calling a *data* effect in #120 / #121 / #137 is actually a *harness* effect —
which would be the most useful possible outcome, since it's currently an
untested assumption underneath every cross-experiment comparison we make
against Eric's numbers.

## Background

We had never run a straight reproduction. Every prior MarinFold training
experiment changed at least one axis, so none is a clean control:

| exp | reused from Eric | changed |
|---|---|---|
| #67 / #85 | contacts-v1 corpus | Llama not Qwen3, untuned LR, ~2.7 ep; predates his sweep |
| #108 / #109 | corpus; framed as a #75 scale-up | 3B not 1.5B, CoreWeave H100, direct dispatch |
| #120 Arm A | his actual checkpoint + his exact documents | **continue-train** from `step-35679` on a 941k round-0 subset |
| #121 | #75 architecture verbatim | ccoord mix, seq 32768, batch 256 |
| #137 | **exp117's winning hyperparameters**, from scratch, native `default_train` | trained on contacts-and-crops-v1 |

#137 is the closest — right recipe, wrong corpus. This experiment is #137 with
the corpus swapped back, which makes it simultaneously a harness-validation
control **and the matched control for #137**, turning that run from an isolated
number into a clean corpus A/B at identical recipe and token budget.

## Approach

Adapted from exp137's `crops_train_common.py` (same from-scratch native path,
same recipe) with the corpus and tokenizer swapped back to contacts-v1. Every
recipe value was read out of
`marin@origin/eac/plm-exp117:experiments/protein/exp117_sweep.py` (`e0f3da1`) —
not inferred from issue prose.

### Recipe — exp117 `e16-lr3p162e-3-wd0p2-bs128`, verbatim

| knob | value | source in `exp117_sweep.py` |
|---|---|---|
| model | Qwen3 1.47B — hidden 2048, inter 8192, 24 layers, 32 heads, 8 kv, Llama3 rope | `MODEL_CONFIG` |
| vocab | 2845 | `VOCAB_SIZE` |
| tokenizer | `timodonnell/contacts-v1-tokenizer@5d68a24a899f` | `TOKENIZER` |
| seq len / global batch | 8192 / 128 | `SEQ_LEN`, point |
| learning rate | **3.1623e-3** | point |
| weight decay | **0.2** | point |
| schedule | cosine, warmup 0.1 | `LR_SCHEDULE`, `WARMUP` |
| optimizer | AdamW — betas 0.9/0.95, eps 1e-8, `max_grad_norm` 1.0, `min_lr_ratio` 0.1 | levanter `AdamConfig` defaults† |
| z-loss | `None` | `train_lm(z_loss_weight=None)` |
| loss | unmasked | contacts-v1 has no `<distance>` statements |
| packing | `pack=True` on every component | `_apply_recipe_overrides` |
| shuffle | `BlockShuffleConfig(io_block_size=256, window_blocks=512, perm_type="feistel")`, `data_seed=0` | `SHUFFLE`, `DATA_SEED` |
| eval | full val, `max_eval_batches=None`, 2/epoch → every **2,230** steps | `NUM_EVALS_PER_EPOCH` |
| permanent ckpt | every **4,460** steps (1/epoch) | `production_shape` |

† Eric passes only lr/wd/warmup/schedule to `AdamConfig` and takes the rest as
defaults. We pass those same values **explicitly** (verified against
`levanter/optim/config.py`) so a future levanter default change can't silently
move the recipe out from under the reproduction.

### Step budget — derived, not estimated

`steps_per_epoch = round(TRAIN_TOKENS / (batch × seq))` with
`TRAIN_TOKENS = 4,676,753,425`:

- `round(4,676,753,425 / 1,048,576)` = **4,460** steps/epoch
- 16 epochs → **`num_train_steps` = 71,360** = 74.86B tokens

Cross-check: #75's 8-epoch run ended at `step-35679`, i.e. 35,680 = 8 × 4,460.
(This is 71,3**60**, not the 71,359 exp137 used — that appears to take the final
step *index* for the count. A 1-step difference, immaterial there, but pointless
to inherit when exactness is the whole point.)

### Data

- Train `gs://marin-us-east5/protein-structure/MarinFold/exp53_contacts_v1_5x/documents/train/*.parquet`
- Val `.../documents/val/*.parquet`
- Text key `document`. These are the same shards Eric's region-relative
  `TRAIN_DOCS` / `VAL_DOCS` resolve to under `marin-us-east5`.
- Region **us-east5**, co-located with the corpus and checkpoint bucket (marin
  hard-blocks cross-region training).

**Cache equality check:** our tokenize step must report exactly
**4,676,753,425** train tokens. If it does, corpus + tokenizer are bit-identical
to his and the only remaining variable is the training harness. Run
`verify_cache.py` after the tokenize step materializes.

### Known divergences (documented, not hidden)

1. **The harness — the thing under test.** Eric drives marin's newer
   `marin.experiment.train.train_lm` + `StepRunner` + `fray`; we drive
   `marinfold_models.defaults.default_train` + `executor_main` on marin `<0.3`.
2. **Tokenizer pin** — MarinFold's tokenizer-load path rejects `repo@rev` (the
   exp85/exp120 gotcha), so we pass the bare repo id. Single-revision repo →
   same bytes; the pin is documentation.
3. **Token cache** — we build our own under our `MARIN_PREFIX` rather than
   reusing `gs://marin-us-east5/tokenized/contacts-v1/2026.07.13.1/`. The token
   count check above is what makes that safe.
4. **Batch placement** — Eric fits per-device batch from his calibration table;
   we take the default fit. Execution-only: same global batch, same math.
5. **Metric name** — his component key is `tokenized/contacts-v1-val` →
   `eval/tokenized/contacts-v1-val/loss`; ours is `eval/contacts-v1-val/loss`.
   Same series, cosmetically renamed (his key carries a storage prefix).

## Success criteria

- **Primary:** final `eval/contacts-v1-val/loss` within **±0.01** of **2.7112**.
  Report the signed delta either way.
- If |Δ| > 0.02, treat it as a finding and bisect (shuffle → cache → optimizer →
  packing) before using the number.
- Tokenize step reports **4,676,753,425** train tokens.
- HF export of the final checkpoint with the tokenizer co-located.
- **Secondary:** exp89 downstream harness (R-precision / AUC on the 554-protein
  curated eval set) vs. Eric's E8 numbers and vs. #137's crops run at matched
  budget.

## Files

- `contacts_v1_repro_common.py` — the recipe. Model, corpus, tokenizer, shuffle,
  optimizer, and the `steps_per_epoch` / `steps_for_epochs` formulas transcribed
  from `exp117_sweep.py` with per-knob provenance.
- `train_contacts_v1_repro.py` — launcher. `EXP150_PREVIEW=yes` prints the fully
  resolved config and submits nothing; `EXP150_SMOKE=yes` runs a tiny
  identity-isolated end-to-end check.
- `verify_cache.py` — reads the materialized token-cache ledgers and asserts the
  train split is exactly `TRAIN_TOKENS`.
- `export_checkpoints.py` — CPU HF export of a `step-N` checkpoint, contacts-v1
  tokenizer co-located.
- `pyproject.toml` / `uv.lock` — marin pinned `<0.3`, `marinfold-models` pinned
  to the `models/` tree this was written against.

## Running it

```bash
cd experiments/exp150_models_reproduce_eric_contacts_v1
uv venv && uv sync --extra tpu

# 1. resolved config, nothing submitted
EXP150_PREVIEW=yes uv run python -m train_contacts_v1_repro

# 2. smoke (tiny, isolated identity)
EXP150_SMOKE=yes EXP150_TPU=v5p-8 ... iris job run ... -- python -m train_contacts_v1_repro

# 3. the real run
... iris job run ... -- python -m train_contacts_v1_repro
```

Full iris invocations are in `train_contacts_v1_repro.py`'s module docstring.
Note the launcher must be submitted with the **fresh marin checkout's** `iris`
(`/home/bizon/git/marin/.venv/bin/iris`) — the cluster rejects clients older
than 14 days, while the job itself runs this dir's pinned marin.

## Results

_Pending._

## Conclusion

_Pending._
