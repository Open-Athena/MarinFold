# exp254: audited seeding and candidate-selection results

## Decision target: three percentage points

The practical requirement is approximately +0.03 absolute R-precision over pooled consensus: 0.5217 to about 0.5517. None of the tested usable methods meets it. Small positive effects are diagnostic findings, not practical successes.

This does not establish that MarinFold ignores prompt contacts. Accuracy can stay unchanged even when predictions respond substantially, and pooling differently conditioned rollouts can hide seed-specific changes.

The current experiment supplies only one predicted contact per rollout. The central mechanistic question is how this checkpoint responds to a substantial supplied contact set.

## Consensus differences and uncertainty

Iid consensus is 0.5217, reproducing the published m2-p06 control within 0.005.

Top-100 seeded consensus gains 0.0017, 95% paired interval [-0.0011, +0.0046]. Long-only gains 0.0028 [-0.0018, +0.0076]; equal thirds gains 0.0030 [-0.0003, +0.0066].

Only the top-100 interval is wholly inside the preselected +/-0.005 practical band. All three consensus upper bounds are far below the current +0.03 objective. All point estimates are positive, so the original literal prediction of a nonpositive difference was not confirmed.

## Overall maps change little in this one-contact experiment

The seeded 100-rollout consensuses retain about 89-90% of the iid top-R map after forced seed rows are removed.

In equal-size comparisons of 50-rollout maps, cross-arm turnover is 18.08-18.22%, versus 18.30% between two iid halves. These are 40 resampled splits of existing samples per protein, not independent reruns or an equivalence test.

The aggregate similarity does not establish general prompt insensitivity: each rollout receives only one seed, seed-specific responses may cancel, and local conditional effects are not measured by whole-map overlap.

## Correct the oracle metric before comparing scores

The historical individual-rollout score divides true positives by min(R, emitted pairs). Roughly half the rollouts emit fewer than R pairs, so this is not comparable to consensus R-precision.

With fixed R, missing predictions count as misses. The iid oracle scores 0.4892, versus 0.4970 for top-100 seeding: gain 0.0078 [+0.0027, +0.0135].

Removing the forced seed before scoring leaves 0.4952 and a gain 0.0060 [+0.0007, +0.0121]. This is heterogeneous, exploratory evidence of better available candidates, well below the practical target. An oracle still requires ground truth to choose them.

## Seed correctness is an association, not a causal explanation

The original comparison credited the injected true contact in the rollout score. Excluding that seed reduces the top-100 within-protein true-versus-false contrast from 0.0124 to 0.0055 [+0.0017, +0.0095].

The corrected continuation score retains the historical P@min(R, emitted) definition. True and false seeds differ in identity and rank; even within-protein comparisons are observational.

This cannot establish that one contact carries little information, that wrong seeds are harmless, or why consensus changes little.

## Ranking remains an open problem

The iid pool covers 92.28% of true contacts, while vote ranking recovers 52.17% at R. This describes ranking headroom in the existing pool; more sampling could also improve ranking among existing pairs.

89.66% of resolved candidate pairs are unvoted negatives, not the previously stated 99.9%.

The tested cross-validated logistic features and quality weighting give at most 0.0015 observed R-precision gain. Their loss functions and features do not exhaust pointwise ranking or prove an unavoidable AUC/R tradeoff.

## Cluster candidates have measurable headroom

The K=10 k-means oracle scores 0.5375 versus 0.5217 pooled: gain 0.0158 [+0.0081, +0.0248]. Including the pooled map as another candidate raises the oracle gain to 0.0190 [+0.0119, +0.0277].

Even this fixed K=10 oracle result falls below the +0.03 practical target. A successful selector would not need to outperform the oracle to improve at all, but a small improvement is insufficient. These are contact-metric ceilings for the tested candidate sets, not limits on alternative clustering or downstream structural accuracy.

## The tested geometric selector loses

At K=5, geometric selection scores 0.4797: 0.0233 above blind selection [+0.0089, +0.0374], but 0.0420 below pooled consensus [-0.0588, -0.0268]. Including pooled as a fallback still loses 0.0199.

The residual needs no reference labels, but this diagnostic constructs candidate maps with ground-truth R and a resolved-residue mask. It is not yet a deployable selection pipeline.

A weak selector does not establish that folding confidence or another selector cannot capture the positive oracle headroom. Comparing correlations across experiments does not establish a twelvefold predictive improvement.

## Reproducibility and next decision

The public archive is pinned by data/artifacts.json, with SHA-256 for every file. CPU commands in the README reproduce the metrics and audit from an anonymous download. Regression tests cover seed removal, short predictions, and invalid comparisons.

Protein bootstrap intervals condition on one saved sampling draw. Exploratory contrasts are unadjusted for multiple comparisons. The next informative control is a matched intervention with substantial true versus false contact sets on this checkpoint, measuring both prediction changes and remaining-contact accuracy.

Exclude supplied pairs from both scoring universes and compare against repeated no-contact runs. A deployable method must gain about +0.03 at comparable inference cost. The Helico cut sweep tests noisy pair density, not folding multiple coherent cluster maps; its negative union result does not close that question.
