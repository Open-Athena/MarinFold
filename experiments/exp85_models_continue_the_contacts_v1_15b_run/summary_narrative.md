# Summary slides — exp: continue the contacts-v1 1.5B run for another epoch (LR re-heat / warm restart)

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide.
     Keep this current as the experiment progresses. -->

## What we're doing

Does continuing the quick #67 contacts-v1 1.5B run (`protein-contacts-1_5b-3.5e-4-contacts-v1-unmasked-3b5cf2`) for **another epoch with a re-heated learning rate** (a cosine warm-restart from the final `step-11999` checkpoint) lower eval loss — and does it improve contact recapitulation on the exp82 benchmark?

## Why

The #67 run was a single un-tuned cosine decay over ~2.7 epochs (final eval/loss **2.980**, bpb 0.4232). Its LR decayed to its floor, so the last steps made little progress. A **warm restart** — reload the weights, re-heat the LR, and run ~1 more epoch of shuffled data — should squeeze out additional loss reduction "for free" (no new data, ~1 epoch of compute) and give a second contacts-v1 checkpoint to compare against both #67 and the #61/#75 tuned sweep. Whether the loss gain translates into better contact prediction is the open question (exp82 showed #67 is near-chance at *de novo* contact prediction).

## Results so far

Recipe + code complete (v5p-32, batch 512, re-heat LR 4.0e-4, ~1125 steps, warm-start
from #67 step-11999). **BLOCKED** before step 0 by a marin/levanter cache-reader bug on
the iris TPU worker: `Sharded cache ledger missing input_ids/0 count` (ledger key is
`input_ids`). Reproduced locally; analysis + fix in `MARIN_CACHE_READER_BUG.md`. The
worker ran the *fixed* marin (`0.2.19.dev`) and still failed, and it doesn't reproduce
locally — so the open question is why the worker derives `input_ids/0`. Handed off for
infra/marin investigation; PR #86 left open.
