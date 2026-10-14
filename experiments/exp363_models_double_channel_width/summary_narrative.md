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
GPU production queued after eleven preemptions; step-3848 checkpoint saved.
Measured speed: 5.6 seconds/update, excluding interruptions.
TPU trial submitted on 2026-10-14: v6e-32, 32 chips, us-east5-b.
Twenty updates on existing regional native caches; same model/batch shape.
TPU Splash attention, fresh initialization, compressed smoke schedule.
Production migration requires 320.4 GB of MPNN caches and optimizer state.
No bulk transfer performed; explicit cross-region transfer approval required.
Contact-accuracy results are pending.
