# Summary slides — exp: generate think-augmented contacts-v1 corpus at scale (local) + publish

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide.
     Keep this current as the experiment progresses. -->

## What we're doing

Generate the **think-augmented** contacts-v1 training corpus at scale from afdb-24M — a `<think>`-pause-token twin of the exp53 [`contacts-v1`](https://huggingface.co/buckets/open-athena/MarinFold) corpus over the **same** entries / rounds / splits — and publish it to the `open-athena/MarinFold` HF bucket, so it can be used as a drop-in dataset for #124.

## Why

Pause tokens ([Goyal et al. 2023](https://arxiv.org/abs/2310.02226)) give a
model free compute before it commits to a prediction. To test that on
contacts-v1 (#124 + the inference-time eval), we first need a training corpus
that contains `<think>` tokens — this experiment produces it, aligned 1:1 with
the exp53 corpus.

## Results

**4,213,203 documents, 0 generation drops** — per-split/round counts (and the
245 budget-truncated train docs) match the exp53 non-think corpus exactly, so
it's a true 1:1 think twin. ~4.81 B tokens; ~84 % of docs carry a `<think>` run
(mean ~8.7 think tokens/doc, ~0.8 % of tokens). Every doc ≤ 8192, ends `<end>`,
`<think>` only between `<contact>` statements (0 invariant violations).
Generated locally (48 procs, ~20h), no vocab/tokenizer change.

## Published

`open-athena/MarinFold` → `data/document_structures/contacts_v1_think/`
(train/val/test = 2067/22/22 shards, round-descending) + README + tokenizer.
Ready for #124.
