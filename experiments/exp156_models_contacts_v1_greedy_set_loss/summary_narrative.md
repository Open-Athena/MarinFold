# Summary slides — contacts-v1 greedy set loss

## What we're doing

Prototype a contacts-v1-specific greedy latent-order loss that treats the contact list as an unordered set during training.

## Why

The current serialized next-token objective penalizes arbitrary contact ordering and orientation choices. A greedy set loss may let the model learn easy contacts first and use those emitted contacts as context for harder ones.

## Results so far

The greedy-set objective now trains inside Levanter's jitted `Trainer` loop and logs greedy-set validation loss. On a short 1×H100 rerun, greedy validation reached `5.393` and auxiliary next-token CE validation reached `5.419` at step 74. The earlier 300-step stock next-token H100 run reached CE `4.19192`, so this is not yet a fair same-step win for greedy.

On 4×GB200, high-LR (`3.1623e-3`) runs looked unstable/overfit: next-token finished with CE eval `5.516`, while greedy finished with greedy-set eval `5.674` after disabling the auxiliary next-token CE hook for multi-GPU greedy runs. That hook currently hits a fused-CE axis mismatch on local multi-GPU meshes.

## Next

Run lower-LR (`3e-4`) 8×H100 comparisons and keep multi-GPU greedy validation on the greedy-set metric until next-token CE eval is made non-fused or otherwise mesh-safe.
