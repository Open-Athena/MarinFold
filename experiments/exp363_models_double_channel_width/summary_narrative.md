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

## Current status — October 29, 10:51 EDT
The lower-LR fork has also diverged. Original remains intentionally stopped.
Fork still running on four nodes / 32 H100s, batch; all four workers on attempt 10.
Latest fork metrics at 10:50:16 EDT: step 149,414 (56.10%), LR 0.000707.
Train loss 17.63110; 5.59 seconds/update. Final cooldown has not begun.
Validation rose from 3.06352 at 128,954 to 6.80370 at 133,182.
It peaked at 28.84608 and remains 8.69887 at step 147,980.
Recommend stopping the fork. It remains active; no cancellation was requested.
The numerical cause remains undiagnosed; no replacement run was launched.

## TPU trial succeeded
Interactive v5p-64: 32 chips across 8 VMs; all 20 updates completed.
Finished October 16; no worker failures or preemptions.
Native and HF step-19 checkpoints verified; tokenizer included.
Median steps 5–19: 7.40 s/update, about 32% longer than GPU steps.
Short native-only smoke; throughput is preliminary, not cost-normalized.
Production continues on GPUs. No bulk transfer or migration performed.

## Matched validation: early gain did not survive
The fork beat the default at its first 26 evaluations, through step 128,954.
At that step: default 3.069638; fork 3.063521.
All nine later evaluations are worse than the default.
At step 147,980: default 3.066265; fork 8.698871.
First training loss above 4 in the inspected window: step 131,426, loss 6.26793.
First above 10: step 131,912, loss 28.72058; LR stayed at 0.000707.
Neither tested LR sustained the larger model through the full epoch.
Single seed; protein-contact accuracy remains unevaluated.
The plot appendix predates fork divergence: historical October 26 snapshot.

## Other architectures: context, not a common leaderboard
exp108 depth-scaled 2.93B: all three LR trials unstable around 6.8k updates.
exp108 width-scaled 2.78B: all three trials finished; older recipe and validation.
exp262 NoPE + smear: final loss 3.00207 versus its control's 2.97454.
Its early advantage reversed, illustrating why interim rankings can mislead.
exp157 learned RoPE deltas: corrected CE 3.0654 versus vanilla 3.0815.
Only exp277 provides the same native + MPNN recipe as the current wider run.
Raw historical validation losses cannot be pooled across cache/logging paths.

## Parallel LR fork — October 22
Original schedule retains its 0.001 plateau and final cooldown.
Fork restores checkpoint 74,178, resuming at absolute step 74,179.
Weights, Adam moments/counters, RNG, data position and augmentation are preserved.
Linear reduction to 0.000707 over 2,114 updates; target reached at step 76,293.
Hold until original final cooldown at step 213,076; final LR target 0.0000707.
Same 266,345-update horizon, separate W&B and checkpoint identities.
Source copy and counters verified; 14 local configuration/restore/serialization tests passed.
October 22 a02 failed before training; October 23 credentials refreshed.
Retry a03 exposed CLI-class serialization; moved optimizer into an importable module.
Corrected a04 is training; first nine LRs match the schedule within 1.14e-10.
First loss at 74,179 exactly matches the original, confirming the starting comparison.
