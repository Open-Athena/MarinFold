---
marinfold_experiment:
  issue: 364
  title: 'exp: ProteinMPNN-only training for two epochs with a learning-rate sweep'
  kind: models
  branch: main
---

# exp: ProteinMPNN-only training for two epochs with a learning-rate sweep

**Issue:** [#364](https://github.com/Open-Athena/MarinFold/issues/364) · **Kind:** `models` · **Branch:** `main`

## Question

How does the current default 1.47B contacts-v1 model train when its training data contains only ProteinMPNN redesigns, with no native-sequence training documents?

## Hypothesis

Training exclusively on inverse-folded sequences may retain structural learning while changing generalization to natural versus designed proteins. A three-point learning-rate sweep tests optimization sensitivity without changing the default architecture or other training settings.

## Background

The current default is #277's scratch-trained `contacts-v1-exp277-m2-p06-full-epoch-1.5B` (final checkpoint step 266344), as recorded in LATEST_RESULTS.md. Its #266 redesign subset contains exactly 162,574,724 documents (31,702,680 AFDB plus 130,872,044 ESM-Atlas), approximately the requested 166 million. Existing audited token caches reside on CoreWeave and contain 174,107,898,831 raw tokens. Native documents are excluded from training; the default validation set remains unchanged.

## Approach

Train three models from scratch for two complete epochs of the redesign subset, with peak learning rates 0.0005, 0.001, and 0.002. Preserve exp277's 24-layer Qwen3 (hidden 2048, MLP 8192, 32 attention / 8 KV heads, QK norm and RoPE), tokenizer, sequence length 8192, global batch 128, AdamW weight decay 0.2 and other optimizer parameters, WSD warmup 10% / cooldown 20% / minimum ratio 0.1, model/data/augmentation seeds, document packing and clipping, blocked cross-document attention, validation, and scheduled amino-acid statement-order augmentation. Scale the schedule over the requested two-epoch duration. Traverse the complete concatenated redesign corpus once per epoch with a fresh deterministic shuffle, without equal-source reweighting, oversampling, or native training documents.

Audit the packed-example counts before launch, enforce exact finite two-epoch coverage, and retain stable per-trial W&B/checkpoint identities for recovery. Use existing CoreWeave inputs and available batch-priority GPUs; TRC remains permitted if complete regional data already exists. No bulk cross-region transfer is authorized. Keep W&B routing at open-athena/MarinFold. Launch separate smoke identities before production.

## Success criteria

All three trials complete two full redesign epochs with verified final native checkpoints. Preserve per-trial run history, learning curves, provenance, and final HF exports with co-located tokenizers. Report validation losses at matched exposure and any operational failures. Contact benchmark evaluation is a follow-up rather than a prerequisite for launching these requested runs.

## Results

_(Fill in after the run completes.)_

## Conclusion

_(Fill in after results are in.)_
