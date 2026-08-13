# exp208 results: RL post-training on contacts-v1

**Issue [#208](https://github.com/Open-Athena/MarinFold/issues/208) · five arms, 554-protein held-out eval, all on 8×A100**

**The one-paragraph version.** A dense per-contact reward trains the model hard and
makes it **worse** — R-precision 0.5898 against the warm start's 0.6111
(p = 5.7e-19). The cause is not bad contacts but missing ones: it learns to be
selective, emits 65% fewer distinct pairs, and consensus scoring cannot rank a pair
that never receives a vote. Adding the leave-one-out consensus marginal
significantly repairs this (+0.0048 R, +0.0110 AUC over the dense reward, both
p < 1e-4) without closing the gap. Removing the stepwise term entirely — the
consensus signal alone — removes the shrinkage completely and is the most
promising configuration found; its held-out number is still running. A purely
document-level F1 reward neither helped nor hurt, because its policy never moved.

| arm | reward | policy moved? | R-precision | Δ warm start | AUC | vote coverage |
|---|---|---|---|---|---|---|
| — | baseline `exp199` | — | **0.6111** | — | **0.9487** | 2267 pairs |
| **D** | document F1 only, GRPO | **no** (KL 0.0014) | 0.6109 | −0.0001 (p = 0.93) | 0.9467 | −11.6% |
| **S** | dense per-contact | yes (KL 0.098) | 0.5898 | **−0.0213** (p = 5.7e-19) | 0.8976 | **−65.2%** |
| **B v1** | dense + consensus (`lam_doc` 4.5) | yes | 0.5879 | −0.0232 | 0.8986 | −65.3% |
| **B v2** | dense + consensus (`lam_doc` 1067) | yes | 0.5946 | −0.0165 | **0.9087** | −60.1% |
| **C** | consensus **only** (`lam_step` 0) | yes | _running_ | — | — | — |

Detailed write-ups: [ARM_S_RESULTS.md](ARM_S_RESULTS.md) ·
[ARM_B_RESULTS.md](ARM_B_RESULTS.md) · [ARM_D_RESULTS.md](ARM_D_RESULTS.md) ·
[VOTE_COVERAGE.md](VOTE_COVERAGE.md) · [ERR_DECAY_ANALYSIS.md](ERR_DECAY_ANALYSIS.md)

---

## 1. The finding: a per-rollout objective is wrong for a consensus metric

Arm S doubled single-rollout precision, 0.252 → 0.473, and lost 0.021 R-precision
and 0.051 AUC. Both statements are true simultaneously, and only the second one
matters.

The reward pays `n_scored · (precision − p̄)`. That is a **per-contact** quantity
whose lever is *which contacts to emit*, so the cheapest way to raise it is to emit
only confident ones. Consensus R-precision is a property of the **vote distribution
over 100 rollouts**, and it needs the marginal, uncertain contact emitted
*sometimes* — that is what a vote count is for. Over the recall axis the two
objectives point in opposite directions.

Measured on paired within-protein data across four checkpoints scored on the same
554 proteins:

| checkpoint | distinct pairs voted | ρ(Δcoverage, ΔAUC) |
|---|---|---|
| arm S step 40 | −15.8% | +0.142 (p = 8e-4) |
| arm S step 125 | **−65.2%** | **+0.781** (p = 6e-115) |
| arm D step 125 | −11.6% | −0.013 (p = 0.76, no relationship) |

The proteins that lost the most coverage lost the most AUC. **AUC absorbs the
damage and R-precision hides it**: AUC ranks every candidate pair, so an unvoted
pair is unrankable (arm S is worse on 98.4% of proteins), while R-precision reads
only the top R, where the surviving contacts genuinely *are* better. A reward that
trades recall for precision therefore looks almost harmless on the headline number
while gutting the ranking beneath it.

This confirms the pre-registered prediction 2, mechanism included: *"the step-only
arm raises precision and moves consensus R-precision by ≤ 0 — union coverage
shrinks and votes concentrate."*

At its own internal peak (step 40, precision and recall both still rising) arm S is
−0.0023, p = 0.12 — exactly the 0.0023 reproducibility floor. **It never beat the
warm start at any checkpoint measured.**

## 2. Did a purely document-level reward work?

Two were tried, and they behave completely differently — which turns out to be the
more useful result.

**Arm D — one scalar per rollout (section F1), GRPO group baseline, no per-token
term at all.** Every pathology disappears: precision and recall move together,
coverage holds, the collapse guard never fires, and the held-out number is
statistically identical to the warm start (−0.0001, p = 0.93).

**But this did not test the reward.** Its policy barely moved: mean KL 0.00069
against arm S's 0.0318, and 0.00135 vs 0.09763 over the last 25 steps — a factor of
72. Its own training reward is flat across all 125 steps (trend p = 0.87). A single
group-normalised scalar, spread thin across ~600 response tokens, does not deliver
enough gradient at lr 1e-6 to change the model. The Background section of the plan
flags this exact trap for #200 (KL 0.00051, "the policy barely moved, which bounds
how much of the flat result is attributable to the reward design at all"); arm D is
the same order. **It needs re-running at a higher LR before it means anything.**

**Arm C — the consensus marginal alone (`lam_step = 0`).** Also purely
document-level, but it *does* move the policy, because `lam_doc` is calibrated so
the term carries the same spread the stepwise term had. Over 100 steps:

| block | precision | recall | pred/gt | arm S pred/gt |
|---|---|---|---|---|
| 0–25 | 0.2697 | 0.2955 | **1.0967** | 1.0750 |
| 25–50 | 0.2929 | 0.3161 | **1.0811** | 0.9913 |
| 50–75 | 0.2957 | 0.3220 | **1.0919** | 0.8717 |
| 75–100 | 0.2829 | 0.3042 | **1.0784** | 0.7139 |

**No shrinkage whatsoever**, where arm S has fallen to 0.71 by the same point, and
precision and recall both rise. This is the causal confirmation that the *stepwise*
term is what destroys coverage — remove it and the collapse does not happen. Its
held-out score is still running.

The contrast between D and C is the lesson: "document-level" is not one thing. A
scalar reward's *effective* gradient depends entirely on how it is scaled and
delivered, and two document-level rewards here differ by ~70× in how far they moved
the policy.

## 3. The consensus marginal is doing real work

Arm B (dense + consensus) beats arm S on identical data and LR:

| | Δ vs arm S | p | better on |
|---|---|---|---|
| R-precision | **+0.0048** | 3.8e-05 | 47.3% |
| AUC | **+0.0110** | 3.3e-28 | 69.9% |

It recovers ~22% of arm S's deficit. Notably it recovers only **7.9%** of the
coverage deficit while recovering **22%** of the metric deficit, so it improves the
*ordering* of votes as well as their spread — which is what a marginal contribution
to the group's **correct** consensus should do, and the first evidence here that a
group-level objective does something a per-rollout one cannot.

Prediction 3 is therefore **confirmed in direction, refuted in magnitude** at this
setting: the consensus term helps significantly and does not reach baseline.

## 4. Three configuration failures, all caught by checking the run against its claim

None of these were visible in the eval numbers. Each was found by asking whether
the run did what its config said.

1. **`err_decay = 0.5` deleted the reward's baseline.** The k-th error in a section
   costs `p̄·δ^k`, so on real rollouts the *next* wrong contact costs a median of
   **exactly 0.000000** against +0.745 for a correct one — 2.3% of the reward's
   magnitude, for 92.8% of rollouts under 1% of a correct contact. No intermediate
   value helps (ranking quality is flat over δ ∈ [0, 0.9]). Now 1.0.
   [ERR_DECAY_ANALYSIS.md](ERR_DECAY_ANALYSIS.md)
2. **`p̄` drifted above true precision.** It was an unweighted mean over rollouts of
   per-rollout precision *ratios*, not the count-weighted aggregate, so a 3-contact
   rollout moved it as hard as a 200-contact one. It ended at 0.5501 against a true
   0.4733, which pays the policy to go quiet. Now count-weighted, and
   `INITIAL_PRECISION` corrected 0.45 → 0.26 against five direct measurements of the
   training pool.
3. **`lam_doc = 4.5` made the document term inert.** Its total contribution is
   exactly `lam_doc · marginal`; measured spreads are 18.6 (stepwise) against 0.0174
   (marginal), so 4.5 carried **0.42%**. Arm B v1 reproduced arm S to three decimals
   and to **within one vote-pair in 2267**. Now 1067, with the identity pinned by a
   test — this constant has been wrong twice, in both directions.

Plus one infrastructure failure worth recording: **SkyRL's FSDP2 policy sharding
silently destroys the policy** at any shard count, via a weight sync that pushes a
divergent copy into the inference engines (trainer/engine logprob gap 1.33 nats at
step 0 sharded, 0.017 unsharded). Every run here is unsharded. A zero-LR control
proved it was the sync and not the gradient.

## 5. What to run next

1. **Finish arm C and score it.** Consensus-only is the first configuration that
   preserves coverage while moving the policy. If it beats baseline, that is the
   experiment's answer.
2. **Re-run arm D at ~10× LR** until its KL is comparable to arm S's. Until then
   the document-F1 arm is untested, not ineffective.
3. **Sweep `lam_doc`.** The term demonstrably has leverage on exactly the variable
   that determines the metric; the open question is how much of it is needed.

## 6. Honest limitations

- One epoch over 2,000 prompts per arm, lr 1e-6, single seed. Arm-to-arm
  differences are paired over 554 proteins and well-powered; absolute claims about
  what RL can achieve here are not.
- The eval pipeline was validated by reproducing the baseline's published 0.6103 at
  **0.6111** with exp82's worker and exp89's metric implementation, so the numbers
  are comparable to the record. All scoring uses the published scripts, not
  re-derivations.
- Arm C's held-out score was not available at the time of writing.
