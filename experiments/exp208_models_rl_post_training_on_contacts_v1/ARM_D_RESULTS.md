# Arm D: a document-level F1 reward is safe, and does not help

**Issue #208 · 125 steps, same data/budget/LR as arm S, unsharded on 6×A100**

**Summary.** Replacing arm S's dense per-contact reward with **one scalar per
rollout — the section F1 — and a group baseline (GRPO)** removes every pathology
arm S showed: vote coverage holds, precision and recall move together, the
collapse guard never fires, and the eval metrics land statistically identical to
the warm start (R −0.0001, p = 0.93).

**But this run does not evaluate the reward, because the policy barely moved.**
Mean KL from the reference is **0.00069** against arm S's **0.0318**, and by the
last 25 steps the gap is 0.00135 vs 0.09763 — a factor of 72. Arm D's own training
reward is statistically flat across all 125 steps (linear trend p = 0.87; first 40
steps vs remaining 85, p = 0.94). Nothing happened. The correct reading is not
"the document-level reward does not help" but "at this learning rate a single
scalar per rollout does not train the model at all", and the eval numbers below
are measuring a checkpoint that is nearly its own warm start.

---

## 1. What changed from arm S

| | arm S | arm D |
|---|---|---|
| reward | per-token, per contact: `+(1−p̄)` correct, `−p̄` wrong | one scalar per rollout: section F1 |
| baseline | `p̄`, an EMA of the policy's own precision | the group of 16 sibling rollouts |
| advantage | `contacts_dense` (per-token pass-through) | `grpo` |
| prices recall? | only through the contact count | directly |

Everything else is identical: same warm start, same 2,000 AFDB prompts, same 16
samples per prompt, same LR, same 125 steps, same unsharded placement.

A `check_reward_mode()` guard refuses the two mismatched pairings. `document_f1` +
`contacts_dense` fails at the first optimiser step with an error about a missing
dense signal, which describes a different problem; `dense` + `grpo` silently
reduces the per-token reward to one number per rollout and discards the signal
#208 exists to test. The silent direction is the dangerous one.

## 2. Training

| block | doc F1 | precision | recall | pred/gt | correct/rollout | arm S pred/gt |
|---|---|---|---|---|---|---|
| 0–25 | 0.3011 | 0.272 | 0.296 | 1.090 | 38.8 | 1.075 |
| 25–50 | **0.3281** | 0.298 | 0.319 | 1.074 | **46.4** | 0.991 |
| 50–75 | 0.3216 | 0.308 | 0.329 | 1.073 | 43.9 | 0.872 |
| 75–100 | 0.3074 | 0.298 | 0.312 | 1.050 | 44.4 | 0.714 |
| 100–125 | 0.3056 | 0.283 | 0.302 | 1.071 | 40.9 | **0.567** |

Precision and recall move **together** — 0.272/0.296 → 0.308/0.329 at the peak —
which is what a reward that prices both should do, and what arm S could not do.
`pred/gt` never leaves 1.05–1.09. There is no shrinkage to detect, and the collapse
tripwire never fires.

The block means suggest a peak at 25–50 and a decline after. **They do not survive
a test.** Per-step doc F1 has sd 0.0556, which swamps the block differences: peak
block vs final block p = 0.196, linear trend over 125 steps −0.000023/step
(p = 0.87), first 40 steps vs remaining 85 p = 0.94. Arm D is flat, not peaked, and
an earlier draft of this document read a shape into that noise and then explained
it with an exhausted prompt pool — which is doubly wrong, since at step 40 only 640
of 2,000 prompts have been seen.

## 3. Held-out evaluation

Consensus R-precision at n=100 over the 554-protein eval set, same pipeline as
arm S (exp82's `score_rollout_worker.py`, scored by the published
`build_rollout_rows.py`), paired per protein against the shared warm start:

| checkpoint | R-precision | Δ | p | AUC | Δ | p | coverage |
|---|---|---|---|---|---|---|---|
| baseline exp199 | **0.6111** | — | — | **0.9487** | — | — | 2267 |
| arm D step 60 | 0.6099 | −0.0012 | 0.392 | 0.9481 | −0.0006 | 0.293 | −4.6% |
| **arm D step 125** | **0.6109** | **−0.0001** | **0.925** | 0.9467 | −0.0020 | 0.0019 | −11.6% |
| arm S step 125 | 0.5898 | −0.0213 | 5.7e-19 | 0.8977 | −0.0511 | 2.4e-85 | −65.2% |

**Arm D finishes statistically identical to its warm start.** −0.0001 on
R-precision at p = 0.93 is as close to "no change" as this metric reports, against
arm S's −0.0213 at p = 5.7e-19 from the same budget on the same data.

AUC is the one place arm D moves at all: −0.0020, small but significant
(p = 0.0019), tracking its −11.6% coverage. That is the same mechanism as arm S,
three-and-a-half times smaller — arm D drifts slightly toward selectivity late in
the run rather than racing there. Coverage at step 60 (−4.6%) is better than at
step 125 (−11.6%), consistent with its own reward peaking at step ~40.

A full epoch of RL, 125 steps and 32,000 rollouts, moved the reported metric by
one ten-thousandth. That is the result.

## 4. Reading this against arm S

Arm S is the more interesting failure and arm D is the control that isolates it.
Arm S's reward is `n_scored·(precision − p̄)`, a per-contact quantity whose lever
is *which contacts to emit*; it raised per-rollout precision by becoming selective
and halved its vote coverage doing so, and consensus scoring cannot rank a pair
that never gets a vote ([VOTE_COVERAGE.md](VOTE_COVERAGE.md): ρ(Δcoverage, ΔAUC) =
+0.781). Arm D prices recall, keeps coverage, and keeps the score.

So the document-level reward wins the comparison #208 poses — but by not losing.
Neither arm beat the warm start on consensus R-precision at any checkpoint
measured. The honest summary is that this experiment has shown what *breaks* a
consensus metric far more clearly than what improves one.

## 5. What to try next, in order of what the evidence supports

1. **Re-run arm D at a learning rate that actually moves the policy.** This is
   first because until it is done, arm D has not been tested. Target a terminal KL
   comparable to arm S's ~0.098; at lr 1e-6 arm D reached 0.00135. Comparing two
   rewards at matched *nominal* LR compares them at wildly different effective step
   sizes, which is not a comparison of rewards.
2. **Score the group, not the rollout.** Both arms optimise a per-rollout
   quantity, and the metric is a property of the vote distribution over 100
   rollouts. The consensus leave-one-out marginal (arm B) targets it directly,
   is implemented and tested, and Phase 0 measured its correlation with
   per-rollout precision at ρ = 0.22 — it is *not* precision in disguise. It is
   also a dense reward, so it should move the policy like arm S rather than
   stalling like arm D.
3. **Then revisit the dense reward with the fixed p̄.** `p_bar_count_weighted`
   removes the drift that destroyed arm S's second half. It does not make the
   objective right, so this is third, not first.

Note what is *not* on this list: enlarging the prompt pool. An earlier draft
recommended it on the strength of a decline that turns out to be noise.
