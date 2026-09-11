# exp254: controlled contact conditioning and audited inference results

## Controlled result: the model uses contact prompts

Same m2-p06 checkpoint, 97 eval-val proteins, two replicates, eight arms, 100 rollouts per arm: 155,200 completions. No eval-test or eval-denovo scoring.

Remove the union of all supplied contact sets from every scoring universe and remaining R. Large true contexts improve remaining-contact R-precision by +0.2724, 95% interval [+0.2359, +0.3098]. Matched false contexts reduce it by 0.3189 [0.2816, 0.3559]. Copying supplied contacts cannot earn this credit.

Predicted large contexts also change predictions: 35.53% withheld top-R turnover versus 21.86% between iid repeats. The earlier broad prompt-insensitivity interpretation is contradicted by this intervention.

## Practical decision: predicted contexts fail the +3pp target

The predeclared recipe uses 100 archived iid rollouts to choose floor(L/3) contacts, then 100 conditioned rollouts. Its complete final map scores 0.5077 versus 0.5247 for the same archived source plus 100 fresh iid rollouts.

Difference: −0.0170, 95% paired protein interval [−0.0246, −0.0099]. The +0.03 absolute R-precision target is ruled out for this recipe and saved run. A 10-contact predicted context also loses 0.0053 [0.0012, 0.0099].

Oracle controls establish context use; they are not deployable accuracy. The primary uses the conditioned final map and equal rollout counts, not equal measured compute. Other context selection or score aggregation requires its own labeled test.

## Exploratory blends do not rescue the practical gain

The primary final-map readout discards most first-pass score information. Four post-hoc, equal-weight blends retain archived scores and add either conditioned continuation votes or complete-document votes, at both context sizes. No new inference or weight tuning.

Large-context complete-document blending improves the primary −0.0170 delta to −0.0025, 95% interval [−0.0074, +0.0018]. Small-context blending gives −0.0009 [−0.0043, +0.0019]. The continuation-only blends are also nonpositive.

Retaining source information removes most of the original loss, but every interval upper bound remains below +0.002. None approaches the +0.03 practical target. These are exploratory checks, separate from the predeclared primary.

## Recovery checks and validation

Full raw-text verification rebuilds all 155,200 completions' contact votes. A separate implementation reproduces all 3,880 R-precision scores. 51 tests pass across exp254 and exp256.

A post-launch amendment retains exact 6L+128 budget terminations instead of dropping affected proteins. 155/19,400 large-false outputs are capped; every other arm has zero. Initial failure groups and both worker sources are preserved publicly.

Replaying the five original failed groups leaves mechanistic scores unchanged. Excluding all 21 resumed-worker proteins still gives −0.0170 [−0.0265, −0.0081]. These sensitivities do not replace the full 97-protein primary. Identical seeds did not guarantee identical recovered text.

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

## Reproducibility and scope

The controlled archive is pinned by data/conditioning_artifacts.json; the original audit inputs by data/artifacts.json. Anonymous downloads verify every file by SHA-256. Raw completions, source votes, plans, matrices, timing records, and recovery groups are public. CPU commands reproduce scores and plots.

Protein-bootstrap intervals average the two new replicates within protein and condition on saved draws and the archived source. Secondary comparisons are exploratory. The fixed-budget true/false effects establish strong context sensitivity, while the tested predicted-contact recipe fails the practical target.

The historical ranking, clustering, and Helico results above constrain their tested variants. They do not establish family-wide impossibility or close the separate cluster-and-fold question.
