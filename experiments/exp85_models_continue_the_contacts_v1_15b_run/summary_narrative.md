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
from #67 step-11999). The 2026-06-17 attempts were blocked before step 0 by a
marin/levanter cache-reader mismatch on the iris TPU worker:
`Sharded cache ledger missing input_ids/0 count` while the ledger key is `input_ids`.
The branch now avoids that path with `ArrayExemplarTextLmDatasetFormat`: it keeps
Levanter's packed text-dataset path but makes the cache-reader exemplar an ndarray
`input_ids` leaf, and disables automatic cache rebuilds. Local direct loads of the
real train and validation GCS caches both derive `input_ids`, pass offset
construction, and return `PackedTokenDataset`. Relaunch
`/tim/iris-run-job-20260618-151053` is now running on v5p-32, resumed W&B run
`protein-contacts-1_5b-contacts-v1-unmasked-reheat-e3-bs512-5fc77c`, reached
train step 9/1125, and logged loss ~2.96 by 2026-06-18 15:18 UTC.
