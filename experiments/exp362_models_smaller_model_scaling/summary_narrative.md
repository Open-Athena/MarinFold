## Does scale still help at 1.47B?
Train three smaller Qwen3 models on the exact default corpus and full schedule.
Compare 255M, 498M, 985M, and the completed 1.47B control.
Only depth varies: 4, 8, 16, and 24 layers.

## Equal training exposure
232,090,905 documents; 248.584B raw tokens.
34,092,146 packed examples; 266,345 optimizer updates.
Preserve exp277 optimizer, augmentation, shuffle, tokenizer and seeds.

## Current state
Experiment #362 registered. Resolved configuration and finite-data tests pass.
TRC preflight found no complete regional MPNN caches.
Proceeding with existing CoreWeave inputs at batch priority.
Training and contact accuracy results are pending.

## Interpretation
Use final validation loss and matched eval-val / eval-denovo contact accuracy.
A fixed-recipe depth sweep can motivate larger runs, but cannot establish their performance.
One seed per size; no held-out eval-test access.
