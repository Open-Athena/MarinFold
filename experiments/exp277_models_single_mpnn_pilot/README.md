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

Structure-aware sequence redesign may improve generalization when the model trains once on all available native and redesigned documents.

## Background

A deliberately smaller companion to #274, whose mixture/hyperparameter sweep belongs to Zack. Use the published decontaminated native and redesigned AFDB/ESM-Atlas corpora from #225 and #266. Do not wait for multimer data (#145).

## Approach

Train one Qwen3 1.5B model from scratch using exp232's winning m2-p06 settings: LR 0.001, weight decay 0.2, sequence length 8192, global batch 128, packed documents with blocked cross-document attention, and scheduled amino-acid order augmentation. Concatenate the complete native AFDB, native ESM, redesigned AFDB, and redesigned ESM caches, then shuffle the combined packed-example index. Each packed example is visited once; no 50:50 mixture weighting or source cycling is used.

The in-region packing audit counted 232,090,905 documents / 248,583,762,834 raw tokens and 34,092,146 packed examples. One epoch is exactly 266,345 optimizer steps, with 114 real examples and 14 padding examples in the last batch. The existing 8192-token document limit trims one trailing token from each of 862 documents; no document is dropped. Counts are recorded in `data/epoch_corpus_counts.csv`; `audit_epoch.py` reproduces the audit using the trainer's packer. The data configuration checks the packed-example count at startup and exposes a finite dataset to the loader.

Use WSD warmup 10%, decay 20%, minimum LR ratio 0.1 over the full epoch. At the earlier observed throughput, allow about 66–68 hours for this run. Native/redesigned proportions follow corpus size (approximately 30%/70% by raw tokens). Compare to the native-only exp232 winner while reporting the different training exposure explicitly.

Choose placement from live Iris inventory, preferring existing regional data over a bulk cross-region transfer. Reuse native token caches; tokenize redesigned corpora on their existing CoreWeave storage. Validate data and a short isolated training smoke before the production run. Record exact Iris/W&B identities, configuration, and checkpoints in the experiment and run history.

## Success criteria

A healthy launched production run and reproducible checkpoint trail. Subsequently score with the current exp245/exp82 routine eval-val protocol, reporting R-precision and uncertainty against the native baseline, reporting the different token budgets. No held-out eval-test access for this pilot. A null result is useful; avoid treating a single seed as a definitive answer for #274.

## Results

Full data preparation succeeded as `/bizon/exp277-prepare-a03`. The redesigned AFDB cache has 31,702,680 documents / 35,352,543,972 tokens; ESM has 130,872,044 documents / 138,755,354,859 tokens. Native and redesigned cache cardinalities, token totals, locations, and initial-run mixture weights are recorded in `data/tokenized_corpora.csv`. Both isolated tokenizer smoke caches exactly matched independent fresh tokenization for every row (199,180 documents total; `data/tokenization_smoke.csv`).

The 16-node / 128-H100 GPU smoke `/bizon/exp277-train-smoke-a01` succeeded: ten updates, finite final train loss 6.43476, two-batch eval loss 6.31338, and approximately 0.866 seconds per step. Both step-9 native and HF checkpoints are present, including tokenizer files. See [`smoke W&B`](https://wandb.ai/open-athena/MarinFold/runs/contacts-v1-exp277-m2-p06-native-mpnn-1.5B-smoke) and `data/training_smoke.csv`.

Production was submitted on 2026-09-09 at 19:32 UTC as [`/bizon/exp277-train-a01`](https://iris-cw-us-east-02a.oa.dev/#/job/%2Fbizon%2Fexp277-train-a01), using committed source `a8582d2d`. Startup verification passed: W&B reached step 23 with finite train loss 6.70270 and 0.86358 seconds per step (1.214M tokens/s). See [production W&B](https://wandb.ai/open-athena/MarinFold/runs/contacts-v1-exp277-m2-p06-native-mpnn-1.5B) and `data/training_startup.csv`. The GPU child is `/bizon/exp277-train-a01/exp277-train-1fca4ea5`. Training compute is approximately 35 hours at this early rate, plus validation and checkpoint overhead. The first full language-model validation pass completed at step 2,114 with loss 3.76525 (`data/validation_progress.csv`). That weighted-mixture run was intentionally stopped on 2026-09-10 at 13:38 UTC after the user requested one full-corpus epoch; its last observed step was 69,556. Contact-prediction quality has not yet been evaluated.

Preparation required streaming document-only 128-row parquet batches and 32 GB workers for AFDB's long-document tail after an 8 GB memory failure. ESM used 512 workers with 8 GB after measured peak usage of 1.3 GB, reusing completed shards across the resize. No corpus was transferred across regions.

Replacement status (2026-09-10 13:38 UTC): the full-corpus packing audit and four configuration/coverage tests passed. The new run identity is `contacts-v1-exp277-m2-p06-full-epoch-1.5B`; it starts from scratch. GPU startup validation and submission are pending. The earlier run had no production failures; its permanent checkpoints at steps 14,520, 29,040, 43,560, and 58,080 remain available.

## Conclusion

Pending training and routine eval-val scoring.

## Runtime and placement

The live Iris inventory on 2026-09-09 showed US-EAST-02A with 208 unallocated H100s across 26 empty nodes and zero pending GPU requests. Immediately before production launch, the inventory showed 248 free H100s on 31 empty nodes and no pending GPU requests. The 16-node / 128-H100 batch gang leaves 15 nodes free. Exp232 already completed training with this gang size. RNO2A adds remote data access; the pinned CUDA stack does not support the ARM GB200 fleet. The redesigned corpora and native caches are already in the regional `marin-us-east-02a` bucket, so no large data transfer is required.

The controller now rejects exp232's old Iris client. This experiment pins Marin 0.2.86.dev32222909699 (2026-08-19) and its matching Iris/Fray/Levanter packages, with a committed lock. Other resolved training dependencies retain their exp232 versions where compatible. This runtime change is an additional comparison caveat. The cluster's default storage now points to R2; the launcher explicitly supplies the existing CoreWeave credentials and regional LOTA endpoint to all experiment workers.

`launch.py` submits one named phase at a time: `prepare-smoke`, `prepare`, `train-smoke`, or `train`. It builds a minimal workspace containing the experiment and its imported exp232 recipe helpers. `prepare.py` verifies source shard counts and parquet-footer document totals, rejects malformed/OOV tokenization records, and independently audits every smoke-cache row. `train.py` verifies completed caches before constructing the finite concatenation and preserves full production validation. The replacement smoke uses the full training caches, ten optimizer updates, and two validation batches; it has a separate output identity. `epoch_data.py` preserves scheduled augmentation while bypassing the cycling mixture.

Checkpoints are written under `s3://marin-us-east-02a/MarinFold/exp277_models_single_mpnn_pilot/runs/<wandb-run-name>/checkpoints/step-<N>/`. Tokenizer files must accompany any exported or published weights.
