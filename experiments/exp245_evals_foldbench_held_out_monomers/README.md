---
marinfold_experiment:
  issue: 245
  title: 'exp: FoldBench held-out monomer eval sets (eval-val / eval-test / eval-denovo) for the decontaminated #232 checkpoints'
  kind: evals
  branch: main
---

# exp: FoldBench held-out monomer eval sets (eval-val / eval-test / eval-denovo) for the decontaminated #232 checkpoints

**Issue:** [#245](https://github.com/Open-Athena/MarinFold/issues/245) · **Kind:** `evals` · **Branch:** `main`

## Question

**How do the decontaminated #232 checkpoints score on FoldBench's natural monomers — held-out ones we have never scored — and how does that compare to the structure-prediction baselines on the same proteins?**

Every contact number we publish today comes from a 554-protein set whose FoldBench half is the *first 100 rows* of `monomer_protein.csv`, chosen before we understood the training-set overlap ([#213](https://github.com/Open-Athena/MarinFold/issues/213), [#225](https://github.com/Open-Athena/MarinFold/issues/225)). [#232](https://github.com/Open-Athena/MarinFold/issues/232) trained models on corpora decontaminated against **all** of FoldBench, so for the first time the other 234 monomers are a legitimate held-out test set rather than an untested slice of training data.

This experiment cuts FoldBench's 334 monomers into three sets and scores them:

| set | definition | n |
|---|---|---:|
| **eval-val** | the natural monomers inside the historical FoldBench-100 — what every previous eval reported on | 97 |
| **eval-test** | every other natural FoldBench monomer — never scored by anything here | 218 |
| **eval-denovo** | every de novo designed FoldBench monomer | 19 |

Each protein carries a viral / non-viral flag so results can be stratified ([#241](https://github.com/Open-Athena/MarinFold/issues/241) found MarinFold ties ESMFold on viral proteins and loses badly on non-viral ones, so the split is not cosmetic).

## Hypothesis

- **H1.** eval-test and eval-val agree within noise for the #232 checkpoints. Both sets are decontaminated at the same rule, so a gap between them would mean the historical 100 is unrepresentative of FoldBench for reasons other than leakage.
- **H2.** The contaminated reference model (#199 CoreWeave cooldown, the current default) drops more from eval-val to eval-test than the #232 checkpoints do — its training data was never filtered against the 234, so eval-val is partly memorised for it and eval-test is not.
- **H3.** Baselines (Protenix-v2 single-seq/+MSA, ESMFold, ESMFold2) move little between eval-val and eval-test: their training data is unchanged by any of this, and eval-test is not novel to them.

## Background

- [#225](https://github.com/Open-Athena/MarinFold/issues/225) published both decontaminated corpora. Rule: **≥30 % identity over ≥50 % of the shorter sequence**, reference = the 554 eval proteins ∪ all 1,940 FoldBench protein chains, no E-value arm. AFDB 4,129,682 → 3,963,003; ESM-Atlas 66,759,922 → 65,553,178.
- [#232](https://github.com/Open-Athena/MarinFold/issues/232) trained the #199 recipe from scratch on exactly those corpora; [#244](https://github.com/Open-Athena/MarinFold/pull/244) evaluated the two best checkpoints (`m2-p06` 0.592 / `m1-p02` 0.579 on the legacy 554).
- [#226](https://github.com/Open-Athena/MarinFold/issues/226) established the FoldBench monomer universe (334), the RCSB chain-resolution recipe, and ground truth for 23 of the unused ones.
- [#241](https://github.com/Open-Athena/MarinFold/issues/241) supplied the designed-vs-natural audit and the kingdom/viral annotation for all 776 eval proteins.

## Approach

1. **Confirm decontamination** — prove, not assume, that the two #244 checkpoints never saw any of these 334 proteins under the 30 % rule: every monomer sequence present in the #225 reference; every ≥30 %/≥50 %-shorter alignment into the training corpora present in the applied drop list; published corpus row counts and the #232 tokenizer/trainer pins matching end to end.
2. **Build the three sets** — partition the 334 monomers on the #241 designed verdict (synthetic-taxon ∪ `DE NOVO PROTEIN` keyword; the two signals agree 19/19), annotate viral status from the NCBI lineage, and carry the residual-identity columns from step 1.
3. **Ground truth** — pyconfind contacts for the 199 monomers that have none, via #226's `build_gt_contacts.py` path (RCSB `-assembly1` mmCIF, auth chain preferred), with #226's 100/100 re-derivation control.
4. **Score MarinFold** — exp82 rollout+resample (100 rollouts, T=1.0, top-p 0.95, no top-k, 6L+128) + exp89 `compute_metrics.py`, over the 334 units, for three checkpoints: #232 `m2-p06`, #232 `m1-p02`, and #199 CoreWeave cooldown as the contaminated reference. CoreWeave fan-out, checkpoint-local.
5. **Baselines** — Protenix-v2 single-seq and +MSA (exp12/exp74), ESMFold and ESMFold2 (exp78), seq-KNN null (exp94) for the 199 new proteins; reuse the published predictions for the 135 already scored.
6. **Report** — R-precision (all / long) per set per predictor, paired deltas against the baselines, val→test deltas per checkpoint, and the viral / non-viral split.

## Success criteria

- Decontamination confirmed with zero surviving ≥30 %/≥50 %-shorter alignments into either training corpus, and the coverage-gate caveat quantified.
- All 334 units scored for all three checkpoints with 0 unfinished rollouts, and the #75 E8 gate reproduced on the legacy 554 within 0.005 to validate the path.
- A single table giving R-precision (all) for {3 MarinFold checkpoints + 5 baselines} × {eval-val, eval-test, eval-denovo}, with bootstrap CIs on the paired deltas and a viral / non-viral split.
- An answer to H2 stated as a number: the eval-val → eval-test drop for the contaminated model minus the same drop for the decontaminated ones.

## Results

_(Fill in after the run completes.)_

## Conclusion

_(Fill in after results are in.)_
