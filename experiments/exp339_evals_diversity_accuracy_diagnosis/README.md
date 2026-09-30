---
marinfold_experiment:
  issue: 339
  title: 'exp: diagnose prediction diversity and accuracy bottlenecks'
  kind: evals
  branch: research/diversity-diagnosis
---

# exp: diagnose prediction diversity and accuracy bottlenecks

**Issue:** [#339](https://github.com/Open-Athena/MarinFold/issues/339) · **Kind:** `evals` · **Branch:** `research/diversity-diagnosis`

**[Read the detailed research report](REPORT.md)** · **[Summary slides](plots/summary.pdf)**

## Question

Is MarinFold's accuracy limited mainly by insufficient inference diversity, training redundancy or memorization, or by sequence conditioning, sample quality, and representation?

## Hypothesis

Existing rollouts cover many correct contacts, but fail to rank and jointly concentrate them into accurate maps. Diversity alone is therefore insufficient. Training corpus redundancy and single-structure distillation may contribute without explaining the entire gap.

## Approach

Audit the current exp277 training recipe and existing experiments through September 26, 2026. Reanalyze exp321's existing eval-val rollouts with fixed-R, unordered-map, coverage, cardinality, and independent-pool contact ranking diagnostics. Reproduce published controls before drawing conclusions. Synthesize the evidence for memorization and redundancy, keeping historical and current checkpoints distinct. Use public primary literature to contextualize recommendations.

This is an exploratory diagnostic, not a fresh held-out confirmation. No new eval-test scoring or model training was performed. Existing raw rollouts were sufficient for the new experiments.

## Success criteria

A detailed report with reproducible local analyses, compact per-protein tables, plots and summary PDF; explicit uncertainty and causal limits; prioritized follow-up experiments with decision criteria, including when to pivot away from the present strategy.

## Results

Reanalyzed 29,100 existing exp277 rollouts across 97 natural eval-val proteins. All 194 public input files (35.9 MB) were downloaded anonymously and verified against recorded SHA-256 checksums. Consensus and oracle scores reproduced exp321 to floating-point precision on 324 protein/arm/range rows per metric. Three metric counterexample tests pass.

On the original 81-protein confirmation partition, every protein has 100 distinct maps. Valid-map union recall is 0.9362, but the best map with perfect internal truth ranking reaches only 0.5525, versus 0.5331 for the published emission-order oracle and 0.5625 for consensus. Independent-pool ranking improves the oracle to 0.5468 at an extra 100-map cost; a simple reference-blind selector still trails consensus. Bounded sequence guidance improves actual map quality despite reducing diversity.

The report also incorporates the completed second-epoch null, the latest candidate-limited native deduplication audit, 70% redesign token exposure, recent failed search interventions, and the large gap between contact-mode hits and structurally verified alternative folds. It separates measured effects from causal hypotheses and proposes controlled follow-ups with decision gates.

Reproduction commands and input provenance are in [REPORT.md](REPORT.md#limits-and-reproducibility). [Per-protein results](data/per_protein.csv), [paired intervals](data/paired_deltas.csv), and [runtime/provenance](data/provenance.json) are committed.

## Conclusion

The evidence does not support literal duplicate-map collapse or simple nearest-neighbor copying as the dominant explanation. It supports inadequate sequence-specific correctness, single-state teacher supervision, and a mismatch between generating one map, ranking an ensemble, and reconstructing a fold. Native redundancy is measurable, but the stronger current repetition is multiple redesigned sequences on the same backbones.

Keep MarinFold as a research platform, stop unchanged epochs and broad novelty-search loops, and run a bounded phase focused on sequence-aware scoring, controlled data exposure, and structure-aware objectives. If those cannot produce a replicated material natural-protein or blind alternative-state gain, pivot away from the current scratch-trained contact-string recipe as the main predictor.
