---
marinfold_experiment:
  issue: 199
  title: 'exp: optimize 1.5B contacts-v1 AFDB + ESM models'
  kind: models
  branch: exp/199-optimize-contacts-v1
---

# exp: optimize 1.5B contacts-v1 AFDB + ESM models

**Issue:** [#199](https://github.com/Open-Athena/MarinFold/issues/199) · **Kind:** `models` · **Branch:** `exp/199-optimize-contacts-v1`

## Question

Which optimization settings improve the 1.5B no-crops contacts-v1 model trained on the AFDB and ESM-Atlas corpora?

## Approach

- `exp199_sweep_trc.py` defines 24 trials: six #166 optimization points,
  two AFDB/ESM mixtures, and base versus scheduled amino-acid augmentation.
- Every trial starts at step zero from its corresponding region-local #117
  model checkpoint, with a fresh optimizer, schedule, and data order.
- Training uses only the existing AFDB, ESM-Atlas, and contacts-v1 validation
  token caches. Cache auto-building is disabled; the graph has no tokenization
  or copy path.
- The model and optimizer follow #166. The stable tokenizer is
  `eczech/contacts-v1-tokenizer-5d68a24a899f` with 2,845 vocabulary rows.
- A production trial trains for 72,600 steps at global batch 128 and sequence
  length 8,192 (76,126,617,600 tokens). Full validation runs every 2,230 steps.
- Permanent native checkpoints are retained every four evaluations (8,920
  steps), plus the forced final checkpoint.
- Marin comes only from the packages pinned by this directory's `uv.lock`.

## Runbook

Run from this directory after manually checking the selected region's three
caches and all six seed checkpoints. `TRIAL` is one of `m1-p01-base` through
`m2-p06-aug`. The numeric CalVer suffix determines the sweep subversion (`.1`
becomes `s01`).

Preview one trial:

```bash
set -a
source /home/exedev/marin.env
set +a
export REGION=us-east5
export TPU=v6e-64
export TRIAL=m1-p01-base
export MARIN_PREFIX=gs://marin-us-east5/protein-structure/MarinFold/exp199_optimize_contacts_v1_afdb_esm

uv run --extra tpu --frozen python exp199_sweep_trc.py \
  --version 2026.08.07.1
```

Add `--run` only after reviewing the lowered plan. For a short isolated run,
set `SMOKE=yes` and optionally `SMOKE_STEPS` (default 10); smoke runs still use
the complete validation cache.
