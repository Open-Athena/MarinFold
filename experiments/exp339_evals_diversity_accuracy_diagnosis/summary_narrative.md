# Summary slides — exp: diagnose prediction diversity and accuracy bottlenecks

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide.
     Keep this current as the experiment progresses. -->

## Recommendation

Keep MarinFold as a research platform, but stop treating more inference diversity as the main route to competitive folding.

Prioritize sequence-specific correctness, controlled native/redesign exposure, and structure-aware supervision.

Do not buy another unchanged epoch or broad novelty-penalty sweep. The detailed report is REPORT.md (September 26, 2026).

## New analysis: 29,100 existing rollouts

97 natural eval-val proteins; 200 iid and 100 guided maps each. No new GPU inference or eval-test scores.

All 194 public parquets verified by SHA-256. Published consensus/oracle metrics reproduced on 324 rows each to floating-point precision.

Main summaries use exp321's 81-protein confirmation partition. This is exploratory reuse of validation data, not a new held-out test.

## Correct contacts exist, but not together

Every protein has 100 distinct iid maps. Mean pairwise Jaccard is 0.285.

Mean sample R: 0.407. Best emission-ranked sample: 0.533. Perfect truth-ranked best-map ceiling: 0.553. Consensus: 0.563. Valid union recall: 0.936.

The union is not a realizable fold or a deployable predictor. Its gap to the best map measures distributed information, not an attainable promised score.

## Ranking helps modestly

Actual emission order barely beats a random permutation: +0.0011 mean fixed-R precision.

Ranking contacts using 100 independent extra iid maps raises oracle best-of-100 from 0.533 to 0.547. Even perfect internal ranking reaches only 0.553.

A reference-blind frequency selector scores 0.511, below consensus despite extra compute. Commonness and model likelihood are weak substitutes for correctness.

## Quality wins while diversity falls

Bounded native/polyalanine guidance raises oracle R from 0.533 to 0.555 and improves the equal-H100-time comparison.

It increases map similarity and reduces union recall. Pure-ratio decoding does the reverse: more coverage, worse accuracy and validity.

Contact clusters, whole-map medoids, beam search, novelty penalties and alanine masking have not provided a convincing alternative-mode search improvement.

## Data: redundancy and exposure differ

The native-corpus direct-witness audit removes 7.225% at 50% identity and 80% coverage of both chains. This is approximate-candidate/policy limited; full structural checks are unfinished.

Current exp277 exposure is 70.04% MPNN redesign and 84.00% ESM-backbone-derived. Redesigned sequences reuse existing backbones.

Redesign improves designs strongly but ties natural eval-val. A second epoch also ties. Matched-budget mixture and rare-cluster experiments are needed.

## Memorization is not established

Current exp277 natural R is 0.554, versus 0.407 for the tested native decontaminated KNN null.

That null excludes redesigns. Sequence decontamination also does not remove every remote homolog or familiar fold.

A causal matched-token removal/reweighting experiment, with sequence and structural neighbors separated, is required to settle template dependence.

## Contact diversity is not structural recovery

Exp304: at 1,000 iid maps per protein, both contact modes appear for 11/29 targets; a strict exploratory reconstruction screen finds both structures for only 2/29.

Both true-map reconstruction controls work for only 17/29, and the threshold is sensitive. Failures are partly inconclusive.

Exp311 natural GDT-TS: top-L contacts 0.502, confidence selection 0.528, oracle 0.624, ESMFold2 0.800. Selection alone leaves a large gap.

## Focused phase and stopping rule

First: a sequence-aware scorer. Then separate matched-budget redesign-dose and rare-cluster/backbone-weighting contrasts. Next: a stronger sequence encoder and pair/denoising objective. For diversity: actual multiple-state supervision.

Require a replicated material natural gain (roughly 0.02 R or 0.03 GDT-TS) before major scale-up. Diversity must improve blind, structurally validated recovery beyond equal-time iid.

If these fail, retain the tooling and contact interface but retire the current recipe as the main general-purpose predictor.
