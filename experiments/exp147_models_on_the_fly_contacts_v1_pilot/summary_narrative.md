# Summary slides — exp: pilot on-the-fly contacts-v1 training from ESM Atlas contacts

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide.
     Keep this current as the experiment progresses. -->

## What we're doing

Can we train the exp117 Qwen3 1.47B recipe directly from the reusable ESMFold2 Atlas pyconfind-contact rows, constructing the canonical `contacts-v1` documents and token sequences at read time rather than materializing a text corpus?

## Why

A shard-local direct Levanter dataset can reconstruct `AnalyzedStructure`, call the existing deterministic `contacts-v1` builder, encode with the pinned contacts tokenizer vocabulary, and greedily pack the same document stream quickly enough to feed a TPU training pilot. The ordinary full `contacts-v1` validation loss should remain directly comparable to exp117 (interesting below 2.8).

## Results so far

A stateful streaming prototype now constructs documents directly from contact
rows and carries partial best-fit bins across loader calls. It owns shard/row
shuffling and partitions shards across JAX processes. Synthetic tests pass; no
real data transfer or training run has started. Exact checkpoint resume remains
the central unresolved consequence because Levanter prefetches ahead of the
optimizer.
