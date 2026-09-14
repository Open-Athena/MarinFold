## One model, one complete corpus epoch
Train contacts-v1 Qwen3 1.5B from scratch on all native and ProteinMPNN-redesigned AFDB and ESM-Atlas documents. Companion to Zack's broader issue #274 sweep; tracked in issue #277.

## Full-corpus coverage
232,090,905 documents and 248.584B raw tokens produce 34,092,146 packed examples. Concatenate the four complete caches, shuffle, and visit each example exactly once. No native/redesigned mixture weighting. One epoch is 266,345 updates at batch 128; the final batch has 114 real and 14 padding examples.

## Optimizer and runtime
Retain exp232 m2-p06: LR 0.001, weight decay 0.2, sequence length 8192, blocked attention, and scheduled amino-acid order augmentation. WSD warmup 10%, decay 20%, minimum LR ratio 0.1 spans the full epoch. The run completed in just under 73 hours on 128 H100s at batch priority in cw-us-east-02a, including validation, recovery, checkpointing, and export.

## Earlier run retained
The original 50:50 weighted-mixture run was stopped intentionally at about step 69,556 on 2026-09-10 at 13:38 UTC when the requested objective changed. Latest validation loss: 3.10302. Permanent checkpoints through step 58,080 remain available. No production failures. Its metrics and history remain a separate record.

## Validation and completed training
All four full caches are verified. The in-region packing audit used the trainer's exact implementation. Four tests passed, including finite coverage across unequal source sizes and a partial shuffle block. The existing document-length policy trims one trailing token in 862 documents; every document remains represented. The full-corpus ten-step GPU smoke succeeded with finite losses and native/HF checkpoints including the tokenizer. Production finished all 266,345 updates on September 13. Across 126 full validations, loss improved from 3.90598 at step 2,114 to a best of 2.98274 at step 264,250; final loss was 2.98441 at step 266,344. The final LR reached the configured 0.1x WSD floor. Permanent native and HF checkpoints at step 266,344 are verified, including tokenizer files. The first job failed near step 73,702 during a distributed checkpoint barrier; attempt a03 restored step 72,744 and completed successfully. Iris also recovered two worker preemptions automatically. Both batch-priority Iris jobs ended successfully with all 16 workers terminal-successful.

## Second epoch continuation
Restore the complete trainer state from permanent checkpoint step 213,072, immediately before cooldown. Add one full 266,345-update epoch over a new data-seed-1 permutation of all 34,092,146 packed examples. The first update is at absolute step 213,073 and the final checkpoint is step 479,417. LR returns to the restored 0.001 peak for the added epoch's first 80%, then linearly cools to 0.0001 over its final 20%. Seven contract tests cover the strict full-state restore, offset finite loader, reshuffle, exact update count, and LR boundaries. The one-node batch-priority restore smoke is submitted as `/bizon/exp277-continue-smoke-a01`; its GPU child is capacity-queued, and production remains gated on its success.

## Interpretation
The first-epoch contact evaluation is complete: 670 units, 67,000 requested rollouts, 66,999 usable, and one explicitly accounted capped rollout. Compared with the native-only decontaminated exp232 winner, exp277 R-precision improves by +0.01503 / +0.02202 on legacy all / long contacts, is tied on eval-val at +0.00204 / +0.00211 under the 0.005 rule, and improves by +0.08599 / +0.10375 on eval-denovo. This gives no material natural eval-val gain but a strong designed-protein signal. The single seed and different training exposure limit causal attribution. A reshuffled second epoch is queued from the last pre-cooldown checkpoint.

## New default and published artifacts
The first-epoch step-266344 checkpoint is the new default, contacts-v1-exp277-m2-p06-full-epoch-1.5B. Its public HF copy includes the original float32 weight bytes, tokenizer, corrected Transformers 4.x metadata, and a publication manifest. The previous exp232 model remains registered. New plots compare identical eval-val and eval-denovo proteins and show paired gains over exp232, the saved Top7 contact map, and training validation loss. Helico structure metrics still belong to exp232.
