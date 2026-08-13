# Arm B: the consensus marginal works, and is not strong enough

**Issue #208 · two runs, 125 steps each, identical to arm S but for the document term**

**Summary.** The leave-one-out consensus marginal **significantly improves on the
pure dense reward**: R-precision +0.0048 (p = 3.8e-05) and AUC +0.0110
(p = 3.3e-28, better on 69.9% of proteins) against arm S, trained on identical
data at an identical learning rate. It recovers ~22% of arm S's deficit and does
not close it — arm B still finishes 0.0165 below its own warm start. Prediction 3's
**direction** is confirmed; its **magnitude** is not.

Getting there required catching a dead run first: at the shipped `lam_doc = 4.5`
the document term carried **0.42%** of the stepwise term's spread, and arm B
reproduced arm S to three decimals across every training metric and to within
p = 0.096 on the held-out set.

---

## 1. Two runs

| | arm B v1 | arm B v2 |
|---|---|---|
| `lam_doc` | 4.5 (shipped default) | **1067** |
| document term / stepwise spread | 0.42% | 100% |
| outcome | bit-for-bit arm S | a real, measurable effect |

`lam_doc` is calibrated, not chosen. The document term's total contribution to a
rollout's summed reward is exactly `lam_doc · marginal` — the per-token share is
`lam_doc·marg/len(response)`, so summing over the response recovers it. Measured on
the Phase 0 rollouts: per-rollout stepwise total `n_pred·(precision − p̄)` has
sd 18.6, group-of-16 centred marginals have sd 0.0174, so equal spread needs
`lam_doc = 18.6/0.0174 ≈ 1067`. That identity is now pinned by a test, because
this constant has been wrong twice, in both directions.

## 2. Results

| model | R-precision | Δ baseline | AUC | Δ baseline | pairs voted |
|---|---|---|---|---|---|
| baseline exp199 | **0.6111** | — | **0.9487** | — | 2267 |
| arm D step 125 (untrained) | 0.6109 | −0.0001 (p = 0.93) | 0.9467 | −0.0020 | −11.6% |
| arm S step 125 | 0.5898 | −0.0213 | 0.8976 | −0.0511 | −65.2% |
| arm B **v1** step 125 | 0.5879 | −0.0232 | 0.8986 | −0.0501 | −65.3% |
| arm B **v2** step 125 | **0.5946** | −0.0165 | **0.9087** | −0.0401 | −60.1% |

Against arm S, paired over 554 proteins:

| | Δ vs arm S | p | better on |
|---|---|---|---|
| arm B v1 | −0.0019 R / +0.0010 AUC | 0.096 / 0.266 | — (null) |
| **arm B v2** | **+0.0048 R** / **+0.0110 AUC** | **3.8e-05** / **3.3e-28** | 47.3% / 69.9% |

Arm B v1's coverage of 787.2 against arm S's 788.1 — one pair apart in 2267 — is
the cleanest possible statement that a document term at `lam_doc = 4.5` does
nothing at all.

## 3. What the effect is made of

The consensus term slows the shrinkage that destroys arm S, without stopping it:

| block | arm B v2 pred/gt | arm S pred/gt |
|---|---|---|
| 50–75 | 0.8872 | 0.8717 |
| 75–100 | 0.7483 | 0.7139 |
| 100–125 | **0.6186** | 0.5673 |

But coverage explains only part of the gain. Arm B v2 recovers **7.9%** of arm S's
coverage deficit while recovering **22%** of its R-precision deficit and **21.5%**
of its AUC deficit. The consensus marginal is therefore improving the *ordering* of
the votes as well as their spread — which is what it was specified to do, since a
rollout's marginal is its contribution to the group's **correct** consensus, not
its contribution to coverage.

## 4. Why it is not enough, and what follows

Arm B is `lam_step · r_step + lam_doc · A_consensus` with both terms at equal
spread. The two point in opposite directions on the recall axis: the stepwise term
pays for selectivity (raising per-rollout precision), the consensus term pays for
contributing to a correct group vote (which needs the marginal contact emitted
*sometimes*). At parity the stepwise term still wins — coverage ends at −60%
instead of −65%.

The obvious experiment is therefore **`lam_step = 0`: the consensus term alone.**
The evidence now says the stepwise term is the component that causes the damage and
the consensus term is the component that repairs it, so running the repair without
the damage is the natural next arm, and it is a one-flag change. A `lam_doc` sweep
is the second, since the term demonstrably has leverage on exactly the variable
that determines the metric.

Neither should be confused with a fix for arm D, which is a separate open thread:
that run's policy barely moved (KL 0.0014 against arm S's 0.098) and needs a
higher learning rate before it has been tested at all.
