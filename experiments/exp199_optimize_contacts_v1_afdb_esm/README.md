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

## Contact evaluation

The `exp/199-evals` branch keeps a reusable catalog and worker for contact
evaluation of exp199 checkpoints. Exp199 jobs restore Levanter checkpoints
directly from their region-local GCS bucket. The control downloads a pinned HF
export. Each job prepares BF16 weights on ephemeral worker disk and runs the
complete settled 554-protein, 100-rollout recipe.

The catalog now includes every permanent checkpoint from the completed
`prot-exp199-cv1-s01-m1-p03-aug-us-east1` run and the exp117 control used by PR
#190. It also includes the final checkpoint from the completed
`prot-exp199-cv1-s01-m1-p06-aug-us-east1` run. Submit one checkpoint per job
from the isolated evaluation workspace:

```bash
cd evals/contact_prediction
uv run --frozen python submit_contact_eval.py \
  --checkpoint s01-m1-p03-aug-step72599 \
  --run-tag <unique-tag> \
  --cluster marin-dev \
  --user eczech
```

The submitter always passes `--user eczech`, defaults to `marin-dev` and
`v6e-4`, and uses the checkpoint's source region. Results live under
`hf://buckets/open-athena/MarinFold/data/contacts-v1-model-eval-exp199/replicates/<run-tag>/runs/`.
The final p03 checkpoint is step 72,599 in `us-east1`. The control is the pinned
exp117 1.5B, 16-epoch HF export at step 35,679 and runs in `europe-west4`.
The p06-aug final checkpoint is also step 72,599 in `us-east1`. A shared helper
keeps later finished exp199 runs to a one-line catalog entry.

The current scorer exactly reproduces PR #190's archived control votes at
R-all `0.5335961341539802`. Three fresh generations of that checkpoint reached
`0.5347972614575084`, `0.535215598085612`, and `0.5328883690891095`. The four
evaluations span `0.002327228996502506`, and every fresh result passed the
declared 0.006 tolerance. The final p03-aug, p06-aug, p03-base, and CoreWeave
p06-aug checkpoints reached `0.5743326909766765`, `0.5244069975064393`,
`0.5779648259578162`, and `0.587348377794962`, respectively. The combined
[boxplot and loss scatter](plots/final_checkpoint_rprecision.png) shows all
four control evaluations separately. Historical losses are converted to the
current scale with the empirical offset documented in the detailed eval
README. A bounded sigmoid fit across the unique 1.5B checkpoints has an upper
asymptote of `0.595529`, below the Protenix-v2 baseline `0.603158`. The range
metrics, public artifacts, run order, and reusable layout are
in [`evals/contact_prediction/README.md`](evals/contact_prediction/README.md) and
[`evals/contact_prediction/PLAN.md`](evals/contact_prediction/PLAN.md).

The first completed evaluation covers
`prot-exp199-cv1-s01-m1-p06-base` at step 26,760. Its mean all-range
R-precision is 0.461997 across the fixed 554-protein set. The full run record,
range metrics, timings, and public artifact links are in
[`evals/contact_prediction/README.md`](evals/contact_prediction/README.md).
