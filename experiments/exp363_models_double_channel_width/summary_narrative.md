# Double the default model channel width

## Question and design
Does model width improve contact accuracy at the same one-epoch exposure?
Baseline: exp277, native plus MPNN corpus, 24 layers.
Hidden width 2048 -> 4096; MLP width 8192 -> 16384.
Keep 32 query / 8 KV heads; head width 64 -> 128.
Keep data, optimizer schedule, seeds and global batch fixed.

## Architecture verified
Actual Levanter parameter-tree counts:
Baseline: 1,471,374,336 parameters.
Wider model: 5,861,984,256 parameters (3.984x).
12 configuration, tokenizer and finite-epoch dataset tests passed.
Count reproduction: count_parameters.py via the README command.

## Current status
64-H100 smoke active in CoreWeave US-EAST-02A.
Resized from 128 GPUs after concurrent jobs reduced free capacity.
Global batch remains 128 through gradient accumulation.
Pinned tokenizer fix verified; distributed initialization succeeded.
Checking full-corpus packing, then ten updates and checkpoint output.
Production training and contact results are still pending.
