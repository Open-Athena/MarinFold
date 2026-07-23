# Summary slides — exp: pilot on-the-fly contacts-v1 training from ESM Atlas contacts

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide.
     Keep this current as the experiment progresses. -->

## What we're doing

Can we train the exp117 Qwen3 1.47B recipe directly from the reusable ESMFold2 Atlas pyconfind-contact rows, constructing the canonical `contacts-v1` documents and token sequences at read time rather than materializing a text corpus?

## Why

A shard-local direct Levanter dataset can reconstruct `AnalyzedStructure`, call the existing deterministic `contacts-v1` builder, encode with the pinned contacts tokenizer vocabulary, and greedily pack the same document stream quickly enough to feed a TPU training pilot. The ordinary full `contacts-v1` validation loss should remain directly comparable to exp117 (interesting below 2.8).

## Results so far

A stateless prototype now maps every example index to a deterministic
`(epoch, shard, slot)`, constructs documents directly from contact rows, and
best-fit packs an entire shard into exactly 2,650 output slots. Overfull shards
uniformly sample packed bins; underfull shards use zero-loss padding. Synthetic
tests pass.

The first real GCS shard smoke converted 20,000 documents into 2,608 packs in
55.88 seconds (47.42 examples/s), with no drops or truncations and 98.2469%
real-token utilization. The fixed quota added 42 zero-loss padding examples.
Sixteen raw shards (1.52 GB) are staged in `us-east5`; no training run has
started.
