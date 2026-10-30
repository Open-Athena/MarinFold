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
Fresh W&B/Iris observation at 2026-10-29 14:50 UTC:
All three models completed all 266,345 updates; Iris jobs succeeded.
Final loss: 255M 3.2483; 498M 3.0834; 985M 3.0182; 1.47B 2.9844.
All final native checkpoints and HF exports passed structural verification.
Each HF export includes the expected contacts-v1 tokenizer.
Metadata, tensor headers and object sizes verified; payloads not yet hashed.
985M final checkpoint was written October 22; completion verified October 29.
No training dispatches remain active; zero GPUs requested by this sweep.
Scheduled monitoring remains blocked by runtime permissions.
Contact evaluations, timings, publication and contact conclusions remain pending.

## Interpretation
Use final validation loss and matched eval-val / eval-denovo contact accuracy.
Final loss improves at every size; 985M to 1.47B improves by 0.0338.
The shared learning rate was not tuned separately for each size.
Separate tuning could change the observed gaps or ranking.
A fixed-recipe depth sweep can motivate larger runs, but cannot establish their performance.
One seed per size; no held-out eval-test access.
