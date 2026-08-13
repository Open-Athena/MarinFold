# Why arm S lost: vote coverage, not contact quality

**Issue #208 · paired analysis over 554 eval proteins × 4 checkpoints**

**Summary.** Arm S doubled its single-rollout precision and lost 0.021 R-precision
and 0.051 AUC. The mechanism is not that its contacts got worse — they got better.
It is that it emitted **53% fewer distinct pairs**, and a pair that receives no
votes cannot be ranked at all. Across proteins, how much coverage a checkpoint
lost predicts how much AUC it lost with ρ = **+0.781** (p = 6e-115).

---

## The measurement

Consensus scoring votes 100 rollouts into an `[L, L]` matrix and ranks pairs by
vote count. Two things can change when a policy is updated: *which* pairs get
votes (coverage) and *how well ordered* they are (quality). Counting the pairs in
the candidate band (separation ≥ 6) that receive at least one vote separates them.

| checkpoint | distinct pairs voted | vs baseline | contacts emitted / 100 rollouts | R-precision | AUC |
|---|---|---|---|---|---|
| baseline exp199 | 2267 | — | 16,191 | 0.6111 | 0.9487 |
| arm S step 40 | 1910 | −12.8% | 14,659 | 0.6087 | 0.9436 |
| arm S step 125 | **788** | **−52.9%** | 8,438 | 0.5898 | 0.8977 |
| arm D step 60 | 2164 | −3.5% | 16,058 | 0.6099 | 0.9481 |

## The mechanism, on paired data

Cross-protein correlations between rollout quality and consensus score are
confounded by difficulty — easy proteins score well on everything — so they cannot
establish this. The four checkpoints scored on the *same* 554 proteins can: for
each protein, compare a checkpoint's coverage change against its metric change
relative to the shared baseline.

| checkpoint | ρ(Δcoverage, ΔAUC) | ρ(Δcoverage, ΔR-precision) |
|---|---|---|
| arm S step 125 | **+0.781** (p = 6e-115) | +0.128 (p = 0.0025) |
| arm S step 40 | +0.142 (p = 0.0008) | −0.050 (p = 0.24) |
| arm D step 60 | −0.013 (p = 0.76) | +0.015 (p = 0.72) |

The proteins that lost the most coverage are the proteins that lost the most AUC.
Arm D shows no relationship because its coverage did not move.

The asymmetry between the two columns is the point. **AUC ranks every candidate
pair**, so an unvoted pair is an unranked pair and coverage loss hits it directly
(ρ = +0.78). **R-precision reads only the top R**, where the surviving contacts
genuinely are more precise — so it degrades far less (−0.021 vs −0.051) and
correlates weakly with coverage. A reward that trades recall for precision
therefore looks *almost* harmless on R-precision while gutting the ranking
underneath it.

## Why this was predictable from the reward, in hindsight

Arm S's reward is `n_scored · (precision − p̄)`: a per-contact quantity, where the
policy's lever is *which contacts to emit*. Raising per-rollout precision by being
more selective is a straightforward way to increase it.

Consensus R-precision is not a per-rollout quantity. It is a property of the
**vote distribution over 100 rollouts**, and it needs the marginal, uncertain
contact to be emitted *sometimes* — that is what a vote count is for. Selectivity
destroys exactly the signal the metric is built on. The two objectives are not
merely different; over the recall axis they point in opposite directions.

Arm D's F1 reward prices recall explicitly, which is why its coverage held at
−3.5% and its metrics held with it.

## What this does not show

Arm D preserved coverage and preserved the score. It did **not** improve the score
(R −0.0012, p = 0.39 at step 60). Not damaging the model is a necessary condition,
not a result. The open question after both arms is whether *any* per-rollout
objective can move a consensus metric, or whether the target has to be the vote
distribution itself — which would mean scoring a whole group of rollouts jointly
rather than each rollout against a baseline.
