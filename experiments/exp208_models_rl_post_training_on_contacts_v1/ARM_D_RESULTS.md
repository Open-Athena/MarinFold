# Arm D: a document-level F1 reward is safe, and does not help

**Issue #208 · 125 steps, same data/budget/LR as arm S, unsharded on 6×A100**

**Summary.** Replacing arm S's dense per-contact reward with **one scalar per
rollout — the section F1 — and a group baseline (GRPO)** removes every pathology
arm S showed. Vote coverage holds at −3.5% instead of −53%, the eval metrics are
statistically indistinguishable from the warm start rather than significantly
below it, and precision and recall rise together instead of trading off. It also
does not improve the model. After 125 steps arm D is back where it started on its
own reward (0.3056 vs 0.3011).

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

Both arms peak in the same block (25–50) and decline afterwards. That is worth
noting precisely because the two rewards are so different: a common peak points at
something shared — 2,000 prompts exhausted in one epoch, or the LR — rather than
at either reward's shape.

## 3. Held-out evaluation

<!--EVALD-->

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

1. **Score the group, not the rollout.** Both arms optimise a per-rollout
   quantity, and the metric is a property of the vote distribution over 100
   rollouts. The consensus leave-one-out marginal (arm B) is the term that
   actually targets it, is implemented and tested, and was never run. Phase 0
   measured its correlation with per-rollout precision at ρ = 0.22 — it is
   *not* precision in disguise, which is the whole reason it was specified.
2. **More data before more steps.** Both arms peak at step ~40 of a single epoch
   over 2,000 prompts and then decline. That is the shape of exhausting the
   prompt pool, and the pool can be enlarged cheaply — exp200's is 10,000.
3. **Then revisit the dense reward with the fixed p̄.** `p_bar_count_weighted`
   removes the drift that destroyed arm S's second half. It does not make the
   objective right, so this is third, not first.
