## One model, one complete corpus epoch
Train contacts-v1 Qwen3 1.5B from scratch on all native and ProteinMPNN-redesigned AFDB and ESM-Atlas documents. Companion to Zack's broader issue #274 sweep; tracked in issue #277.

## Full-corpus coverage
232,090,905 documents and 248.584B raw tokens produce 34,092,146 packed examples. Concatenate the four complete caches, shuffle, and visit each example exactly once. No native/redesigned mixture weighting. One epoch is 266,345 updates at batch 128; the final batch has 114 real and 14 padding examples.

## Optimizer and runtime
Retain exp232 m2-p06: LR 0.001, weight decay 0.2, sequence length 8192, blocked attention, and scheduled amino-acid order augmentation. WSD warmup 10%, decay 20%, minimum LR ratio 0.1 spans the full epoch. Estimated duration: 66–68 hours on 128 H100s at batch priority in cw-us-east-02a.

## Earlier run retained
The original 50:50 weighted-mixture run was stopped intentionally at about step 69,556 on 2026-09-10 at 13:38 UTC when the requested objective changed. Latest validation loss: 3.10302. Permanent checkpoints through step 58,080 remain available. No production failures. Its metrics and history remain a separate record.

## Validation and current status
All four full caches are verified. The in-region packing audit used the trainer's exact implementation. Four tests passed, including finite coverage across unequal source sizes and a partial shuffle block. The existing document-length policy trims one trailing token in 862 documents; every document remains represented. The full-corpus ten-step GPU smoke succeeded with finite losses and native/HF checkpoints including the tokenizer. Replacement production job /bizon/exp277-train-a02 was submitted at 13:54:56 UTC on 2026-09-10; all 16 workers are running at batch priority. W&B run: contacts-v1-exp277-m2-p06-full-epoch-1.5B. Startup passed at step 65 / 266,345 with train loss 5.90443 and 0.858 seconds/step. At 21:03 UTC, training reached step 27,003 (10.14%). All twelve full validation passes improved: 3.90598 at step 2,114 to 3.18098 at step 25,368. Warmup is complete; the first permanent checkpoint at step 26,634 saved successfully. No failures or restarts. Continuous monitoring is active.

## Interpretation
One seed provides an initial answer to the redesign question. Compare with the native-only winner while reporting the different training exposure. Contact-prediction quality has not yet been evaluated.
