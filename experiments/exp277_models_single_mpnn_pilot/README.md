---
marinfold_experiment:
  issue: 277
  title: 'exp: single 1.5B native + MPNN redesign pilot using the exp232 winner'
  kind: models
  branch: exp/277-single-mpnn-pilot
---

# exp: single 1.5B native + MPNN redesign pilot using the exp232 winner

**Issue:** [#277](https://github.com/Open-Athena/MarinFold/issues/277) · **Kind:** `models` · **Branch:** `exp/277-single-mpnn-pilot`

## Question

Does adding ProteinMPNN-redesigned sequences improve contacts-v1 prediction in one straightforward 1.5B training run?

## Hypothesis

Structure-aware sequence redesign may improve generalization at fixed training budget; preserving source weights avoids confusing sequence diversity with repeated-backbone weighting.

## Background

A deliberately smaller companion to #274, whose mixture/hyperparameter sweep belongs to Zack. Use the published decontaminated native and redesigned AFDB/ESM-Atlas corpora from #225 and #266. Do not wait for multimer data (#145).

## Approach

Train one Qwen3 1.5B model from scratch using exp232's winning m2-p06 settings: LR 0.001, weight decay 0.2, sequence length 8192, global batch 128, packed documents with blocked cross-document attention, and scheduled amino-acid order augmentation. Preserve native token-proportional source balance (5.9522% AFDB / 94.0478% ESM), splitting each source 50:50 native:redesigned. This avoids weighting AFDB's eight redesigns as eight independent backbones.

Use exp232's original 145,200-step / 152.253B-token budget for the first answer, with WSD warmup 10%, decay 20%, minimum LR ratio 0.1. This is a single-model pilot, not a full replication of the longer continued-training baseline. Compare first with the matched-budget exp232 m2-p06 sweep checkpoint; report the longer best baseline separately.

Choose placement from live Iris inventory, preferring existing regional data over a bulk cross-region transfer. Reuse native token caches; tokenize redesigned corpora on their existing CoreWeave storage. Validate data and a short isolated training smoke before the production run. Record exact Iris/W&B identities, configuration, and checkpoints in the experiment and run history.

## Success criteria

A healthy launched production run and reproducible checkpoint trail. Subsequently score with the current exp245/exp82 routine eval-val protocol, reporting R-precision and uncertainty against the matched-budget native baseline. No held-out eval-test access for this pilot. A null result is useful; avoid treating a single seed as a definitive answer for #274.

## Results

Tokenizer smoke `/bizon/exp277-prepare-smoke-a04` succeeded on US-EAST-02A: all 160,000 AFDB documents (21,584,525 tokens) and all 39,180 ESM documents (40,616,645 tokens) exactly matched independent fresh tokenization. See `data/tokenization_smoke.csv`. Full preparation is running as `/bizon/exp277-prepare-a01`. No model-quality result is available.

## Conclusion

Pending training and routine eval-val scoring.

## Runtime and placement

The live Iris inventory on 2026-09-09 showed US-EAST-02A with 208 unallocated H100s across 26 empty nodes and zero pending GPU requests. The planned 16-node / 128-H100 batch gang leaves ten nodes free. Exp232 already completed training with this gang size. RNO2A adds remote data access; the pinned CUDA stack does not support the ARM GB200 fleet. The redesigned corpora and native caches are already in the regional `marin-us-east-02a` bucket, so no large data transfer is required.

The controller now rejects exp232's old Iris client. This experiment pins Marin 0.2.86.dev32222909699 (2026-08-19) and its matching Iris/Fray/Levanter packages, with a committed lock. Other resolved training dependencies retain their exp232 versions where compatible. This runtime change is an additional comparison caveat. The cluster's default storage now points to R2; the launcher explicitly supplies the existing CoreWeave credentials and regional LOTA endpoint to all experiment workers.

`launch.py` submits one named phase at a time: `prepare-smoke`, `prepare`, `train-smoke`, or `train`. It builds a minimal workspace containing the experiment and its imported exp232 recipe helpers. `prepare.py` verifies source shard counts and parquet-footer document totals, rejects malformed/OOV tokenization records, and independently audits every smoke-cache row. `train.py` verifies completed caches before constructing the mixture and preserves full production validation. The smoke uses ten optimizer updates and two validation batches.

Checkpoints are written under `s3://marin-us-east-02a/MarinFold/exp277_models_single_mpnn_pilot/runs/<wandb-run-name>/checkpoints/step-<N>/`. Tokenizer files must accompany any exported or published weights.
