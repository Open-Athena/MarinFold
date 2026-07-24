# Summary slides — contacts-v1 greedy set loss

## What we're doing

Prototype a contacts-v1-specific greedy latent-order loss that treats the contact list as an unordered set during training.

## Why

The current serialized next-token objective penalizes arbitrary contact ordering and orientation choices. A greedy set loss may let the model learn easy contacts first and use those emitted contacts as context for harder ones.

## Results so far

Implementation in progress.
