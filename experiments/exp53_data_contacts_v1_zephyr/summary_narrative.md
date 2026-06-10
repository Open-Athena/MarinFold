# Summary slides — exp53 generate contacts-v1 dataset on zephyr

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide.
     Keep this current as the experiment progresses. -->

## What we're doing

Generating the `contacts-v1` training-document corpus from afdb-24M at scale
on the marin Iris cluster (Zephyr), and publishing it to the
`open-athena/MarinFold` HF bucket + GCS. Up to 5 examples per structural
cluster, organised into pLDDT "rounds". Calls into the existing contacts-v1
generator — no re-implementation.

## Why

This is the input corpus for training contacts-v1 models. The round structure
(round-0 = highest-pLDDT representative per cluster) lets us curriculum the
data; exp53 keeps only clusters with ≥3 members and writes rounds in reverse
(round-4 first, round-0 last) so the highest-quality data is trained last.

## Approach

Two stages: (A) a cheap, structure-free **selection** in DuckDB — group
afdb-24M by structural cluster, rank by pLDDT, assign rounds, drop `<3`-member
clusters; (B) an embarrassingly parallel **generation** on Iris — fetch each
selected structure from GCS and call `generate_document`. A byte-identity test
pins Stage B to the library generator.

## Results so far

_(Fill in as results come in: selection counts, per-round/per-split totals,
generation wall-clock + worker count, output paths.)_
