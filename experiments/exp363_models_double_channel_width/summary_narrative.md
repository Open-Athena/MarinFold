# Double the default model channel width

## Question and design
Does model width improve contact accuracy at the same one-epoch exposure?
Baseline: exp277, native plus MPNN corpus, 24 layers.
Hidden width 2048 -> 4096; MLP width 8192 -> 16384.
Keep 32 query / 8 KV heads; head width 64 -> 128.
Keep data, optimizer schedule, seeds and global batch fixed.

## Architecture and smoke verified
5,861,984,256 parameters, versus exp277's 1,471,374,336 (3.984x).
12 local configuration/tokenizer/data checks passed.
64-H100 smoke: ten updates, train loss 8.56948, eval loss 8.03575.
2.821 seconds/update; native and HF checkpoints saved with tokenizer.
Compressed smoke schedule was spiky; this is an execution check.

## Current status
Full 266,345-update epoch started on 32 H100s in US-EAST-02A.
W&B initialized at 2026-10-13 15:15:36 UTC; data initialization in progress.
Other production jobs left 56 free GPUs; allocation reduced to fit.
Global batch 128 retained through four-way gradient accumulation.
Production uses the original 10% warmup and 1e-3 peak LR.
Linear compute-only estimate: 17.4 days, pending measured production speed.
Contact-accuracy results are pending.
