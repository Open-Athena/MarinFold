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
11 configuration and finite-epoch dataset tests passed.
Count reproduction: count_parameters.py via the README command.

## Current status
A 128-H100 smoke is being prepared in CoreWeave US-EAST-02A.
248 available H100s observed at 2026-10-13 14:43 UTC.
Production follows only after the ten-update smoke succeeds.
No training or contact-accuracy results yet.
