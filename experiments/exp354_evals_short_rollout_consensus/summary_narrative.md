# Short-rollout consensus

## Result: full rollouts retain better precision

Current default: exp277, step 266344. Eval-val: 97 natural proteins.
100 full rollouts: R-precision 0.55743 all / 0.54149 long.
Best short arm: 1,000 rollouts capped at floor(L/2): 0.55003 / 0.53534.
All-range delta: -0.00740; paired 95% CI [-0.01094, -0.00396].
Every tested short cap loses precision. No eval-test scoring.

## What was compared

Five caps: 5, 10, 20, floor(L/5), floor(L/2) emitted contact statements.
1,000 independently resampled prompts; smaller caps reuse trajectory prefixes.
100-short subsets control for sample count. A fresh 100-full baseline.
Temperature 1, top-p 0.95, top-k off. Same candidate universe and exp89 metrics.
97,000 short and 9,700 full samples; zero unfinished full rollouts.
All 1,067 vote matrices reconstructed from raw samples and verified.

## AUC improves, at higher cost for the largest cap

At L/2: all-range AUC 0.93921 -> 0.95565; long AUC 0.92872 -> 0.94849.
Global ranking improves while highest-ranked-contact precision worsens.
Shared short generation: 25.0 minutes; full baseline: 6.9 minutes (3.60x).
L/2 uses 4.98x the output tokens and 10x the prompt tokens.
Smaller-cap standalone runtimes were not measured; their prefixes share generation.

## Scope and recommendation

Keep 100 full rollouts for contact precision under this tested recipe.
The fresh baseline is within 0.005 of the published exp277 baseline.
KNN reference: 0.40715 all / 0.39211 long; all 1,000-short arms clear it.
Six viral proteins: same direction, small subgroup. Low-MSA cut: zero coverage.
One sampling seed on eval-val; no held-out generalization claim.
Raw samples, vote matrices, tables, timings, and runtime provenance are public.
