# Summary slides — exp: does contact accuracy hold up at low MSA depth? stratify the natural eval set by ColabFold depth for the #232 training checkpoint

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide.
     Keep this current as the experiment progresses. -->

## What we're doing

**Does the best decontaminated checkpoint we have hold its contact accuracy where MSA-based methods lose theirs — at low MSA depth — and how does that break down inside vs. outside FoldBench?**

[PR #257](https://github.com/Open-Athena/MarinFold/pull/257) evaluated the [#232](https://github.com/Open-Athena/MarinFold/issues/232) `m2-p06` **training** checkpoint (step 363,000, decontaminated corpus) on the legacy 554, `eval-val`, and `eval-denovo`. `eval-test` was deliberately not scored. This experiment finishes that read and adds the axis the whole single-sequence thesis rests on: **MSA depth**.

Two deliverables:

1. **The usual numbers, completed.** All-range and long-range R-precision on `eval-val` (97), `eval-test` (217), and `eval-denovo` (19), plus the legacy 554 for continuity with every published MarinFold number.
2. **A depth-stratified table over every natural protein in our eval universe** (372 = 314 natural FoldBench monomers + 58 CAMEO-hard / CASP-FM targets), in tiers `<10`, `10–100`, `100–1000`, `≥1000` sequences, reported for three subsets:
   - all natural proteins,
   - FoldBench natural (314),
   - non-FoldBench natural (58).

"MSA depth" here is the depth of the ColabFold MSA that Protenix's `+MSA` arm actually ran with — the same pipeline (`runner.msa_search.update_seq_msa(..., mode="colabfold")`) for both halves, so the two subsets are measured on one ruler. MarinFold never sees the MSA; the depth is a property of the protein, and it is what an MSA-based competitor would have had to work with.

## Why

- **H1.** MarinFold's R-precision is roughly flat in MSA depth, because it has no coevolution signal to lose. Protenix-v2 `+MSA` falls off sharply below ~100 sequences. The MSA-free-vs-MSA-based gap therefore narrows, and may invert, in the shallow tiers.
- **H2.** The non-FoldBench natural set (CAMEO hard targets + CASP free-modeling domains, curated in [#65](https://github.com/Open-Athena/MarinFold/issues/65) precisely for this regime) is shallower than FoldBench natural and carries most of the low-depth signal. FoldBench natural is a deep-MSA set: its median ColabFold depth is ~3,000 and only 5 of 314 proteins sit below 10 ([#247](https://github.com/Open-Athena/MarinFold/issues/247)).
- **H3.** The `<10` bin is small enough that its interval will span most of the range between the neighbouring tiers. Neff (redundancy-weighted depth, the [#74](https://github.com/Open-Athena/MarinFold/issues/74)/[#65](https://github.com/Open-Athena/MarinFold/issues/65) `neff_tier` convention) populates the shallow end far better and is reported alongside raw depth as the better-powered axis.

## Results so far

_(Fill in as results come in.)_
