---
marinfold_experiment:
  issue: 177
  title: 'exp: compare soft-target contact loss against next-token CE on TPUs'
  kind: models
  branch: exp176/soft-target-loss-h2h
---

# exp: compare soft-target contact loss against next-token CE on TPUs

**Issue:** [#177](https://github.com/Open-Athena/MarinFold/issues/177) · **Kind:** `models` · **Branch:** `exp176/soft-target-loss-h2h`

## Question

Does PR #144's contacts-v1 soft-target/document loss improve training over ordinary next-token cross-entropy when both arms train from scratch on TPUs under Eric's best known exp117 recipe?

## Hypothesis

The soft-target objective removes arbitrary contact ordering/orientation from the loss. At matched architecture, token budget, data order, optimizer, and TPU placement, it should reduce validation loss faster than stock CE, or at least preserve downstream contact quality while making the training target less noisy.

## Approach

- Start a new experiment/branch rather than extending abandoned exp156.
- Import the reusable PR #144 document/soft-target machinery into shared code:
  - `marinfold.document_structures.documents`
  - contacts-v1 training-document builders
  - `marinfold_models.document_loss`
  - fixed-quota shard document loading helpers
- Train two from-scratch contacts-v1 Qwen3 1.47B arms on TPUs:
  1. stock next-token CE on canonical serialized contacts-v1 documents;
  2. soft-target contact loss using PR #144's weighted target distributions.
- Use Eric exp117 best-known recipe as the baseline config, without loading exp117 weights:
  - Qwen3 1.47B, seq 8192
  - 16 epochs / 35,680 steps at global batch 256
  - AdamW, LR `3.1623e-3`, WD `0.2`, betas `0.9/0.95`
  - cosine schedule, 10% warmup, `data_seed=0`
- Evaluate both arms against the ordinary tokenized contacts-v1 validation loss. If the soft-target training path cannot use that metric directly, add an auxiliary CE validation hook so the curves are comparable.

## Success criteria

- Both arms launch and train from scratch on TPU with no HF/checkpoint warm start.
- Both arms log matched train loss, validation loss, throughput, and checkpoint/export artifacts.
- The comparison uses the same train corpus, validation corpus, step budget, batch size, optimizer schedule, and TPU shape.
- A short smoke validates the soft-target batch/loss path before the full run.

## Results

_(Fill in after the run completes.)_

## Conclusion

_(Fill in after results are in.)_
