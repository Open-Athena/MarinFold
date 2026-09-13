## One model, one complete corpus epoch
Train contacts-v1 Qwen3 1.5B from scratch on all native and ProteinMPNN-redesigned AFDB and ESM-Atlas documents. Companion to Zack's broader issue #274 sweep; tracked in issue #277.

## Full-corpus coverage
232,090,905 documents and 248.584B raw tokens produce 34,092,146 packed examples. Concatenate the four complete caches, shuffle, and visit each example exactly once. No native/redesigned mixture weighting. One epoch is 266,345 updates at batch 128; the final batch has 114 real and 14 padding examples.

## Optimizer and runtime
Retain exp232 m2-p06: LR 0.001, weight decay 0.2, sequence length 8192, blocked attention, and scheduled amino-acid order augmentation. WSD warmup 10%, decay 20%, minimum LR ratio 0.1 spans the full epoch. Estimated duration: 66–68 hours on 128 H100s at batch priority in cw-us-east-02a.

## Earlier run retained
The original 50:50 weighted-mixture run was stopped intentionally at about step 69,556 on 2026-09-10 at 13:38 UTC when the requested objective changed. Latest validation loss: 3.10302. Permanent checkpoints through step 58,080 remain available. No production failures. Its metrics and history remain a separate record.

## Validation and current status
All four full caches are verified. The in-region packing audit used the trainer's exact implementation. Four tests passed, including finite coverage across unequal source sizes and a partial shuffle block. The existing document-length policy trims one trailing token in 862 documents; every document remains represented. The full-corpus ten-step GPU smoke succeeded with finite losses and native/HF checkpoints including the tokenizer. Production reached step 213,223 / 266,345 (80.06%) at 01:16 UTC on September 13. WSD linear decay is active at learning rate 0.00099752. Across 100 full validations, the best loss is 3.05770 at step 207,172 and the latest is 3.05887 at step 211,400. Permanent checkpoint 213,072 committed successfully. The first job failed at step 73,702 during a distributed checkpoint barrier; attempt a03 restored step 72,744 and resumed at step 72,745. Iris later handled two worker preemptions automatically. All 16 workers are running on 128 H100s at batch priority with no task failures. Same W&B run: contacts-v1-exp277-m2-p06-full-epoch-1.5B. Continuous monitoring is active.

## Interpretation
One seed provides an initial answer to the redesign question. Compare with the native-only winner while reporting the different training exposure. Contact-prediction quality has not yet been evaluated.
