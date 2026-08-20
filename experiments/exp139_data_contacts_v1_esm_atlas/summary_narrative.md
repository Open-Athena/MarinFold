# Summary slides — exp: generate contacts-v1 corpus from the ESMFold2 Atlas distillation set (67M) + save reusable pyconfind contacts

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide.
     Keep this current as the experiment progresses. -->

## What we're doing

Generate the [`contacts-v1`](https://github.com/Open-Athena/MarinFold/blob/main/marinfold/marinfold/document_structures/contacts_v1/SPEC.md) training corpus **at scale from the ESMFold2 Atlas distillation set** ([`open-athena/esm-atlas-esmfold2-distill`](https://huggingface.co/buckets/open-athena/esm-atlas-esmfold2-distill), the #91 curation — 66.76M single-chain monomers, ~2.08 TB) and publish it to the [`open-athena/MarinFold`](https://huggingface.co/buckets/open-athena/MarinFold) HF bucket — calling straight into the format's generator, no re-implementation. In the **same pyconfind pass**, also publish the **raw pyconfind contacts** as a reusable, doc-format-agnostic intermediate so future document types (e.g. `contacts-and-crops-v1`) can be generated from this source **without re-running pyconfind**.

This is the "67M proteins instead of 4M" training-corpus expansion flagged in `UPDATES.md` (#91) — ~16× the AFDB `contacts-v1` corpus (exp53).

## Why

_(Copy from the issue.)_

## Results so far

_(Fill in as results come in.)_
