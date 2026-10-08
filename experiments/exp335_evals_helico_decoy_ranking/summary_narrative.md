# Summary slides — exp: Helico confidence for Rosetta decoy ranking

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide.
     Keep this current as the experiment progresses. -->

## What we're doing

Can Helico's own confidence rank native and near-native protein structures above Rosetta decoys when each candidate is converted to Helico's contact-map representation, and how does it compare with AF2Rank, DeepAccNet, and the Rosetta energy function on the exact AF2Rank benchmark?

## Why

A candidate-derived contact map that is geometrically compatible with the target sequence should let contact-conditioned Helico produce a confident structure, whereas inconsistent decoy contacts should reduce Helico pTM. We therefore expect target-wise Helico pTM to correlate positively with candidate TM-score and recover high-quality candidates substantially better than Rosetta energy, but probably below AF2Rank because the contact map discards most of the template geometry available to AlphaFold. The native-versus-decoy discrimination question and the standard decoy-quality ranking question will both be reported.

## Exact benchmark and execution

The exact public AF2Rank benchmark joins cleanly: 133 targets, 180,079 decoys,
133 natives, and corrected baseline outputs with no missing key. All 180,212
candidates ran on 96 independent CoreWeave H100 jobs; all jobs succeeded with
no production retry. Wall time was 3 hours 35 minutes, with 314.45 inference
H100-hours and 320.92 accounted H100-hours.

All candidates are retained. A systematic extra N-terminal lysine in 1,000
`1iib` decoys is uniquely mapped and explicitly removed rather than silently
shifting indices or excluding candidates.

## Complete decoy-ranking result

Pure Helico pTM reaches 0.841 mean target-wise Spearman correlation with
candidate TM-score, above Rosetta at 0.759 but below AF2Rank at 0.925. Its mean
top-1 TM-score is 0.879, below Rosetta's 0.901.

The secondary Helico composite (pTM times candidate/output TM-score) reaches
0.873 correlation and 0.936 mean top-1 TM-score. The latter is slightly above
AF2Rank's 0.933 point estimate, but the paired 95% interval on the difference
[-0.0035, 0.0093] includes zero.

## Complete native-versus-decoy result

Pure Helico pTM ranks the exact native first for only 4/133 targets and gives a
mean native rank of 130.3. Confidence alone is therefore not an effective exact-
native selector despite tracking overall decoy quality.

The Helico composite selects 51/133 natives and gives mean rank 26.5. AF2Rank
selects 52/133 and gives mean rank 31.1. Paired target-bootstrap intervals for
top-1 recovery, mean rank, and AUROC all include zero difference. DeepAccNet and
Rosetta cannot be evaluated for native recovery because their native rows use
`-1` sentinel scores.

## Bottom line

The confidence-only hypothesis is only partially supported: Helico pTM ranks
quality well but does not reliably identify the true native. Combining pTM with
candidate/output structural agreement is competitive with AF2Rank for selecting
the best candidate and the exact native, though it remains weaker for ranking
the full decoy ensemble. Follow-up should validate the composite on independent
decoy sets.
