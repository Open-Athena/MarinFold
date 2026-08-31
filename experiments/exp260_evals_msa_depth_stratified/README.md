---
marinfold_experiment:
  issue: 260
  title: 'exp: does contact accuracy hold up at low MSA depth? stratify the natural eval set by ColabFold depth for the #232 training checkpoint'
  kind: evals
  branch: exp/260-msa-depth
---

# exp: does contact accuracy hold up at low MSA depth? stratify the natural eval set by ColabFold depth for the #232 training checkpoint

**Issue:** [#260](https://github.com/Open-Athena/MarinFold/issues/260) · **Kind:** `evals` · **Branch:** `exp/260-msa-depth`

## Question

**Does the best decontaminated checkpoint we have hold its contact accuracy where MSA-based methods lose theirs — at low MSA depth — and how does that break down inside vs. outside FoldBench?**

[PR #257](https://github.com/Open-Athena/MarinFold/pull/257) evaluated the [#232](https://github.com/Open-Athena/MarinFold/issues/232) `m2-p06` **training** checkpoint (step 363,000, decontaminated corpus) on the legacy 554, `eval-val`, and `eval-denovo`. `eval-test` was deliberately not scored. This experiment finishes that read and adds the axis the whole single-sequence thesis rests on: **MSA depth**.

Two deliverables:

1. **The usual numbers, completed.** All-range and long-range R-precision on `eval-val` (97), `eval-test` (217), and `eval-denovo` (19), plus the legacy 554 for continuity with every published MarinFold number.
2. **A depth-stratified table over every natural protein in our eval universe** (372 = 314 natural FoldBench monomers + 58 CAMEO-hard / CASP-FM targets), in tiers `<10`, `10–100`, `100–1000`, `≥1000` sequences, reported for three subsets:
   - all natural proteins,
   - FoldBench natural (314),
   - non-FoldBench natural (58).

"MSA depth" here is the depth of the ColabFold MSA that Protenix's `+MSA` arm actually ran with — the same pipeline (`runner.msa_search.update_seq_msa(..., mode="colabfold")`) for both halves, so the two subsets are measured on one ruler. MarinFold never sees the MSA; the depth is a property of the protein, and it is what an MSA-based competitor would have had to work with.

## Hypothesis

- **H1.** MarinFold's R-precision is roughly flat in MSA depth, because it has no coevolution signal to lose. Protenix-v2 `+MSA` falls off sharply below ~100 sequences. The MSA-free-vs-MSA-based gap therefore narrows, and may invert, in the shallow tiers.
- **H2.** The non-FoldBench natural set (CAMEO hard targets + CASP free-modeling domains, curated in [#65](https://github.com/Open-Athena/MarinFold/issues/65) precisely for this regime) is shallower than FoldBench natural and carries most of the low-depth signal. FoldBench natural is a deep-MSA set: its median ColabFold depth is ~3,000 and only 5 of 314 proteins sit below 10 ([#247](https://github.com/Open-Athena/MarinFold/issues/247)).
- **H3.** The `<10` bin is small enough that its interval will span most of the range between the neighbouring tiers. Neff (redundancy-weighted depth, the [#74](https://github.com/Open-Athena/MarinFold/issues/74)/[#65](https://github.com/Open-Athena/MarinFold/issues/65) `neff_tier` convention) populates the shallow end far better and is reported alongside raw depth as the better-powered axis.

## Background

- [PR #257](https://github.com/Open-Athena/MarinFold/pull/257) / [#232](https://github.com/Open-Athena/MarinFold/issues/232): the checkpoint under test. R (all) 0.605 on the legacy 554, 0.552 on `eval-val`, 0.610 on `eval-denovo`; contacts-v1 eval loss 2.968. The HF export lives only in CoreWeave `us-east-02a` S3.
- [#245](https://github.com/Open-Athena/MarinFold/issues/245): the `eval-val` / `eval-test` / `eval-denovo` cut and the read-budget policy for `eval-test`. This read is recorded in `data/eval_test_reads.md`.
- [#247](https://github.com/Open-Athena/MarinFold/issues/247): already carries ColabFold depth for the 314 natural FoldBench monomers, read from the `protenix-foldbench-msa` Modal volume.
- [#74](https://github.com/Open-Athena/MarinFold/issues/74) / [#65](https://github.com/Open-Athena/MarinFold/issues/65): the Neff definition (`msa_depth.py`) and the `protenix-exp74-msa` volume covering the 454 legacy non-FoldBench units, of which 58 are natural.
- [#82](https://github.com/Open-Athena/MarinFold/issues/82) / [#89](https://github.com/Open-Athena/MarinFold/issues/89): the fixed rollout+resample recipe and metric implementation. Nothing about scoring changes here.

## Approach

1. **Score.** Reuse the PR #257 harness with `eval-test` added: 887 `(dataset, stem)` units (legacy 554 + all 333 FoldBench monomers), one checkpoint, 100 rollouts each, 12 single-H100 shards at batch priority on `cw-us-east-02a` where the checkpoint already lives. Re-scoring `eval-val` / `eval-denovo` / legacy-554 alongside the new `eval-test` makes PR #257's published numbers a built-in reproduction gate.
2. **Measure depth.** Recompute raw depth and Neff for all 372 natural proteins from the two Modal volumes through one code path, on Modal CPU (no a3m egress).
3. **Join and report.** Per-protein R-precision × depth tier × subset, with bootstrap CIs, beside the Protenix-v2 `+MSA` / single-seq and ESMFold2 baselines that already exist for these proteins.

## Success criteria

- 887 units complete, 88,700 usable rollouts, zero unfinished.
- `eval-val`, `eval-denovo`, and legacy-554 reproduce PR #257 within 0.005.
- Depth for all 372 natural proteins from a single pipeline, with the 100-protein FoldBench-100 overlap used to confirm the two volumes agree.
- A tiered table + plot, and an explicit statement of which cells are too thin to carry a claim.

## Results

_(Fill in after the run completes.)_

## Conclusion

_(Fill in after results are in.)_
