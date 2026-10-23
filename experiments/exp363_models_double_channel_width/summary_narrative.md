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

## Current status — October 23
Credentials refreshed; fsspec and TensorStore reads verified.
Original a04 submitted at 10:34 EDT; restored checkpoint 75,031 -> step 75,032.
Fork a04 submitted at 10:41 EDT; source checkpoint 74,178 -> step 74,179.
Both use four nodes / 32 H100s, batch, with independent identities and paths.
Both are training: original first resumed update 10:43 EDT; fork 10:51 EDT.
Fork step 74,190: LR 0.00099847547, approximately 5.59 seconds/update.
Fixed credential forwarding and remote optimizer class serialization; 14 tests pass.
Latest pre-restart full validation: original 3.09826 at step 73,990.

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
Default has lower loss at all 18 subsequent full evaluations.
At step 71,876: default 3.08621; wider 3.09976.
Gap +0.01354 nats/token, or 1.36% higher perplexity.
Gap narrowed from its +0.01685 peak at step 61,306.
Single seed, incomplete epoch; baseline LR retained without retuning.
Both losses keep falling. No contact-accuracy evaluation for the wider model yet.

## Other architectures: context, not a common leaderboard
exp108 depth-scaled 2.93B: all three LR trials unstable around 6.8k updates.
exp108 width-scaled 2.78B: all three trials finished; older recipe and validation.
exp262 NoPE + smear: final loss 3.00207 versus its control's 2.97454.
Its early advantage reversed, illustrating why interim rankings can mislead.
exp157 learned RoPE deltas: corrected CE 3.0654 versus vanilla 3.0815.
Only exp277 provides the same native + MPNN recipe as the current wider run.
Raw historical validation losses cannot be pooled across cache/logging paths.

## Parallel LR fork — October 22
Original schedule retains its 0.001 plateau and final cooldown when resumed.
Fork restores checkpoint 74,178, resuming at absolute step 74,179.
Weights, Adam moments/counters, RNG, data position and augmentation are preserved.
Linear reduction to 0.000707 over 2,114 updates; target reached at step 76,293.
Hold until original final cooldown at step 213,076; final LR target 0.0000707.
Same 266,345-update horizon, separate W&B and checkpoint identities.
Source copy and counters verified; 14 local configuration/restore/serialization tests passed.
October 22 a02 failed before training; October 23 credentials refreshed.
Retry a03 exposed CLI-class serialization; moved optimizer into an importable module.
Corrected a04 is training; first nine LRs match the schedule within 1.14e-10.
First loss at 74,179 exactly matches the original; no new full validation yet.
