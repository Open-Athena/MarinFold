## One model, one complete corpus epoch
Train contacts-v1 Qwen3 1.5B from scratch on all native and ProteinMPNN-redesigned AFDB and ESM-Atlas documents. Companion to Zack's broader issue #274 sweep; tracked in issue #277.

## Full-corpus coverage
232,090,905 documents and 248.584B raw tokens produce 34,092,146 packed examples. Concatenate the four complete caches, shuffle, and visit each example exactly once. No native/redesigned mixture weighting. One epoch is 266,345 updates at batch 128; the final batch has 114 real and 14 padding examples.

## Optimizer and runtime
Retain exp232 m2-p06: LR 0.001, weight decay 0.2, sequence length 8192, blocked attention, and scheduled amino-acid order augmentation. WSD warmup 10%, decay 20%, minimum LR ratio 0.1 spans the full epoch. Estimated duration: 66–68 hours on 128 H100s at batch priority in cw-us-east-02a.

## Earlier run retained
The original 50:50 weighted-mixture run was stopped intentionally at about step 69,556 on 2026-09-10 at 13:38 UTC when the requested objective changed. Latest validation loss: 3.10302. Permanent checkpoints through step 58,080 remain available. No production failures. Its metrics and history remain a separate record.

## Validation and current status
All four full caches are verified. The in-region packing audit used the trainer's exact implementation. Four tests passed, including finite coverage across unequal source sizes and a partial shuffle block. The existing document-length policy trims one trailing token in 862 documents; every document remains represented. The full-corpus ten-step GPU smoke succeeded with finite losses and native/HF checkpoints including the tokenizer. Production reached step 73,702 / 266,345 (27.67%) on September 11. All 34 validation passes improved, reaching 3.08621 at step 71,876. The first failure occurred at 09:10 UTC during a checkpoint save: distributed shutdown timed out after worker 6 began shutdown; initiating cause unconfirmed. Step 72,744 is the last completed recovery checkpoint. Attempt a03 launched at 09:17:36 UTC; all 16 workers are running at batch priority. Checkpoint restore succeeded and training resumed from step 72,745 at 09:27 UTC with finite loss around 2.88. Post-restart checkpoint 73,199 saved successfully. W&B caught up by 09:46 UTC: step 73,958 (27.77%), loss 2.87417, 0.877 seconds/update. Same W&B run: contacts-v1-exp277-m2-p06-full-epoch-1.5B. Training remains in the stable LR phase. Permanent checkpoints 26,634 and 53,268 are preserved. Continuous monitoring is active.

## Interpretation
One seed provides an initial answer to the redesign question. Compare with the native-only winner while reporting the different training exposure. Contact-prediction quality has not yet been evaluated.
