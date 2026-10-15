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
GPU production queued after 12 preemptions; step-3848 checkpoint saved.
Last logged step 3918 (1.47%); train loss 3.452, validation loss 3.685.
No overnight progress confirmed; measured compute is 5.6 seconds/update.
First TPU attempt never trained: CPU parent preemptions, then timeout.
Replacement v6e-32 trial submitted 2026-10-15 with a stable CPU parent.
Twenty native-only updates; TPU throughput remains unmeasured.
Production migration: 320.4 GB; transfer approval still pending.
Contact-accuracy results are pending.
