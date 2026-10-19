# Double the default model channel width

## Question and design
Does model width improve contact accuracy at the same one-epoch exposure?
Baseline: exp277, native plus MPNN corpus, 24 layers.
Hidden width 2048 -> 4096; MLP width 8192 -> 16384.
Keep 32 query / 8 KV heads; head width 64 -> 128.
Keep data, optimizer schedule, seeds and global batch fixed.

## Architecture and smoke verified
5,861,984,256 parameters, versus exp277's 1,471,374,336 (3.984x).
Architecture, tokenizer, data and recovery checks passed.
64-H100 smoke: ten updates, train loss 8.56948, eval loss 8.03575.
2.821 seconds/update; native and HF checkpoints saved with tokenizer.
Compressed smoke schedule was spiky; this is an execution check.

## Current status — October 19
GPU production running: step 48,451, 18.19% of one epoch.
Train loss 2.893; validation loss 3.123, previously 3.465.
Complete rolling checkpoint step 48,452; permanent step 26,634.
32 H100s: 5.61 s/update; about 14.2 compute days remaining.
Queueing, validation, checkpoints and preemptions add wall time.

## TPU trial succeeded
Interactive v5p-64: 32 chips across 8 VMs; all 20 updates completed.
Finished October 16; no worker failures or preemptions.
Native and HF step-19 checkpoints verified; tokenizer included.
Median steps 5–19: 7.40 s/update, about 32% longer than GPU steps.
Short native-only smoke; throughput is preliminary, not cost-normalized.
Production continues on GPUs. No bulk transfer or migration performed.

## Matched validation: no width benefit yet
Same native + MPNN corpus, global batch, validation and optimizer schedule.
Wider model led early; default overtook it between steps 33,824 and 35,938.
Default has lower loss at all six subsequent full evaluations.
At step 46,508: default 3.11447; wider 3.12334.
Gap +0.00887 nats/token, or 0.89% higher perplexity.
Single seed, about 18% of the epoch; baseline LR retained without retuning.
Both losses keep falling. No contact-accuracy evaluation for the wider model yet.

## Other architectures: context, not a common leaderboard
exp108 depth-scaled 2.93B: all three LR trials unstable around 6.8k updates.
exp108 width-scaled 2.78B: all three trials finished; older recipe and validation.
exp262 NoPE + smear: final loss 3.00207 versus its control's 2.97454.
Its early advantage reversed, illustrating why interim rankings can mislead.
exp157 learned RoPE deltas: corrected CE 3.0654 versus vanilla 3.0815.
Only exp277 provides the same native + MPNN recipe as the current wider run.
Raw historical validation losses cannot be pooled across cache/logging paths.
