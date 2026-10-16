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
Fresh W&B/Iris observation at 2026-10-16 15:34 UTC:
255M reached 68.6%; 498M reached 67.2% and is advancing.
985M reached 25.4%; waiting for eight nodes after another preemption.
255M resumed training on one node, then was preempted again.
255M / 985M report 13 / 3 preemptions and zero job failures.
104 H100s requested; east02 has no free GPUs for a better placement.
Latest temporary native checkpoints: 182441 (255M), 67180 (985M).
Metadata/manifests reachable; latest checkpoint payloads not load-tested.
Scheduled monitoring remains blocked by runtime permissions.
Full training and contact accuracy results are pending.

## Interpretation
Use final validation loss and matched eval-val / eval-denovo contact accuracy.
A fixed-recipe depth sweep can motivate larger runs, but cannot establish their performance.
One seed per size; no held-out eval-test access.
