# Summary slides — exp: generate sequence-only contacts-v1 dataset

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide.
     Keep this current as the experiment progresses. -->

## What we're doing

Generating a **sequence-only** variant of the contacts-v1 document corpus from
UniRef50: one `<contacts-v1.sequence_only>` document per sequence — the exact
contacts-v1 sequence section (random wrap-around `<pX> <AA>` statements +
`<n-term>`/`<c-term>`, shuffled) with **no structure section**. Published as
sharded parquet + the unified tokenizer, to later test whether mixing it into
training improves the contacts-v1 eval.

## Why

contacts-v1 documents are expensive (pyconfind side-chain contacts from
structures). Sequences are cheap and abundant (UniRef50 ≈ 60 M). If the model
benefits from seeing the same sequence-section representation on a much larger,
more diverse set of sequences, the contacts-v1 eval should improve. This
experiment only generates the data.

## How

- **Library:** one appended token (`<contacts-v1.sequence_only>`, ids
  preserved) + a `sequence_only` builder branch + `generate_sequence_only_document`
  (no pyconfind). The sequence section is byte-identical to contacts-v1's.
- **Driver:** stream UniRef50 `*.fasta.zst`, drop `>2000`-residue sequences
  (unindexable), split arbitrarily by `sha1(entry_id)` (~99/0.5/0.5), write
  typed sharded parquet — one worker per shard, local, resumable.
- **Source quirk:** UniRef50 is globally length-sorted; the dropped giants sit
  in shard 0, so the keep rate is high and `entry_id`-hashed splits stay
  length-balanced.

## Results so far

Library + driver implemented and unit-tested; a streamed real-data sample
validated the end-to-end path (well-formed docs, 0% `<UNK>`). Full local
generation + HuggingFace upload pending go-ahead.
