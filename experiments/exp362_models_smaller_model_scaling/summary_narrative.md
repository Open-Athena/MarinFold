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
All three smoke gates passed, including native/HF checkpoints and tokenizers.
All three production runs retain the full 266,345-update schedule.
Fresh W&B/Iris observation at 2026-10-19 13:18 UTC:
255M and 498M completed all 266,345 updates; Iris jobs succeeded.
Final validation loss: 255M 3.2483; 498M 3.0834; 1.47B control 2.9844.
Both final native checkpoints and HF exports include the expected tokenizer.
Metadata, tensor headers and object sizes verified; payloads not yet hashed.
985M reached 70.4% and is advancing; latest validation loss 3.0998.
Its final 20% learning-rate decay is still ahead.
64 H100s remain active; preserve the progressing eight-node dispatch.
Scheduled monitoring remains blocked by runtime permissions.
Contact evaluations, public publication and final conclusions remain pending.

## Interpretation
Use final validation loss and matched eval-val / eval-denovo contact accuracy.
A fixed-recipe depth sweep can motivate larger runs, but cannot establish their performance.
One seed per size; no held-out eval-test access.
