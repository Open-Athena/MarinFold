# Arm S: the dense per-contact reward trains, and then eats itself

**Issue #208 · 125 steps, 2,000 AFDB prompts × 16 rollouts, unsharded on 8×A100**

**Summary.** Single-rollout precision nearly doubled (0.252 → 0.473) and the model
got **worse**. On the metric of record — consensus R-precision over the 554-protein
eval set — the finished checkpoint scores 0.5898 against its warm start's 0.6111
(paired p = 5.7e-19), and its AUC is worse on 98.4% of proteins. At its internal
peak (step 40) it is a wash: −0.0023, p = 0.12, exactly the noise floor. So the
dense per-contact reward never beat the warm start at any checkpoint measured, and
the run's most impressive-looking number is the one that misleads.

In more detail. While the
p̄ baseline sat below the policy's true precision, the run improved on both axes
at once: precision 0.275 → 0.322 *and* correct contacts per rollout 38.8 → 46.3.
Then p̄ crossed above precision, every contact became net-negative, and the policy
spent the next 75 steps emitting less: by the end it produced 77 contacts at
0.435 precision, which is **fewer correct contacts than it started with** (33.8 vs
38.8). The reward is sound; its baseline is not self-stabilising.

---

## 1. Configuration

| | |
|---|---|
| warm start | `timodonnell/marinfold-contacts-v1-exp199-1_5b-step145199` |
| reward | dense per-contact, `err_decay=1.0`, `p̄₀=0.26` (both set by [ERR_DECAY_ANALYSIS.md](ERR_DECAY_ANALYSIS.md)) |
| advantage | `contacts_dense` (per-token pass-through, no group baseline) |
| data | 2,000 AFDB round-0 prompts, 16 samples each = 125 steps × 256 rollouts |
| placement | policy+ref on 1 GPU, 7 vLLM engines — **unsharded**, see below |
| wall clock | ~3.5 h |

Sharding is not optional here. SkyRL's FSDP2 policy diverges from the inference
engines and the first weight sync destroys the policy; the run above stays healthy
precisely because it is unsharded, and `rollout_train_logprobs_abs_diff_mean` held
0.0152 → 0.0145 across all 125 steps (a sharded run reads 1.33 at step 0). That
investigation is in [README.md](README.md).

## 2. What happened

| steps | precision | recall proxy (pred/gt) | p̄ − precision | contacts/rollout | **correct/rollout** |
|---|---|---|---|---|---|
| 0–25 | 0.275 | 1.075 | **−0.022** | 142.2 | 38.8 |
| 25–50 | 0.322 | 0.991 | **−0.028** | 143.9 | **46.3** |
| 50–75 | 0.368 | 0.872 | +0.038 | 115.6 | 42.8 |
| 75–100 | 0.406 | 0.714 | +0.020 | 101.5 | 41.5 |
| 100–125 | 0.435 | 0.567 | +0.042 | 77.0 | 33.8 |

Read the last column, not the first. Precision rises monotonically and looks like
unambiguous progress; the count of *correct* contacts peaks at step ~50 and then
falls below its starting value. Reported alone, "precision 0.25 → 0.47" would have
been a real result and a misleading one — which is exactly the failure the
experiment plan anticipated when it required precision and recall be reported
separately.

The sign of `p̄ − precision` is the mechanism, and it flips between blocks 2 and 3.

## 3. Why p̄ overshoots

The reward pays `p − p̄` per contact, so p̄ has to estimate the policy's precision.
It is updated as an EMA over rollouts of each rollout's own precision ratio:

```
p̄ ← 0.9·p̄ + 0.1·(correct_in_this_rollout / scored_in_this_rollout)
```

That is an unweighted mean of per-rollout ratios. The quantity the reward needs is
the **count-weighted** precision — total correct over total scored. The two differ
whenever rollouts vary in length, and they differ in a consistent direction: a
short rollout that emits 3 contacts and gets 2 right contributes 0.67 with the
same weight as a 200-contact rollout at 0.25.

So p̄ drifts above the aggregate, contacts become net-negative, the policy emits
fewer, rollouts get shorter — and shorter rollouts have noisier, higher per-rollout
ratios, which pushes p̄ higher still. The feedback is positive in the wrong
direction. `pred/gt` was still falling at −0.006/step at step 125 and extrapolates
to zero in ~89 more steps.

**The fix is one line** — accumulate `correct` and `scored` over the batch and
update p̄ from the ratio of sums — but it changes the reward mid-experiment, so it
is not applied to this run's numbers. It should be in place before the next dense
run.

## 4. Held-out evaluation

**Arm S is worse on the metric of record**, despite doubling single-rollout
precision. Consensus R-precision at n=100 rollouts on the 554-protein eval set,
generated with exp82's `score_rollout_worker.py` and scored with the published
`build_rollout_rows.py` (exp89's metric implementation, not a re-derivation):

| model | R-precision (all) | AUC (all) | R (long) | R (short) |
|---|---|---|---|---|
| baseline exp199 | **0.6111** | **0.9487** | 0.5637 | 0.6814 |
| arm S step 40 (internal peak) | 0.6087 | 0.9436 | 0.5597 | 0.6812 |
| arm S step 125 (final) | 0.5898 | 0.8977 | 0.5354 | 0.6649 |

Step 40 is included because it is where the training metrics peak — precision and
correct-contacts both still rising, p̄ still below true precision. It is the best
this reward ever looked from the inside.

The baseline's 0.6111 reproduces the 0.6103 on record for this checkpoint under
exp82's worker, which is the check that the pipeline above is measuring the right
thing.

Paired over the 554 proteins:

| checkpoint | metric | mean Δ | t | p | arm S better on |
|---|---|---|---|---|---|
| step 40 | R-precision | **−0.0023** | −1.56 | **0.119** (n.s.) | 35.7% |
| step 40 | AUC | −0.0052 | −6.10 | 2.0e-09 | 32.9% |
| step 125 | R-precision | −0.0213 | −9.23 | 5.7e-19 | 18.8% |
| step 125 | AUC | −0.0511 | −23.53 | 2.4e-85 | **1.4%** |

**At its internal peak the reward is not an improvement — it is a wash.** −0.0023
is indistinguishable from zero (p = 0.12) and sits exactly on this experiment's
measured noise floor of 0.0023 (#204). So the story is not "it worked and then the
baseline drifted". It is: the dense reward never beat the warm start at any
checkpoint measured, and after p̄ crossed it actively destroyed the model. Even at
step 40 the AUC is already down a small but unambiguous −0.0052, which is the
shrinkage beginning.

The AUC column is the one that explains the mechanism. AUC ranks *every* candidate
pair, and arm S is worse on 98.4% of proteins — nearly universal, not a
distributional shift. A pair the policy never emits gets zero votes and cannot be
ranked at all, so shrinking the output degrades the ranking everywhere at once.
R-precision falls less (−0.021) because it only looks at the top R, where the
surviving contacts really are more precise.

This is the precision/recall trade the plan warned about, and it is not a wash:
paid out on the metric the experiment is judged by, it is a loss.

## 5. What this says about #208's question

The experiment asks whether a dense per-contact reward beats a document-level one.
Arm S's answer is that the dense reward, as specified, does not beat *doing
nothing*: neutral at its peak, harmful by the end.

Two distinct things went wrong, and they should not be conflated.

**The baseline drifts.** p̄ is an EMA of the policy's own behaviour, and it is
biased upward by the unweighted average over rollouts. That is a genuine bug with
a one-line fix (now applied), and it explains the second half of the run.

**The reward optimises the wrong functional even when its baseline is honest.**
This is the more interesting failure, and the step-40 number is what exposes it.
Over steps 0–50, with p̄ still below true precision, the policy improved on both
internal axes — and bought nothing on the eval metric. The reward pays
`n_scored·(precision − p̄)`, which is a per-contact quantity; consensus
R-precision is a property of the *vote distribution over 100 rollouts*. Raising
per-rollout precision by becoming more selective is exactly the wrong move for a
consensus metric, because a pair that is never emitted gets zero votes and cannot
be ranked. Fixing p̄ would make the run stop degrading; it would not make the
objective the right one.

That reframes arm D. It is not merely the control for a drifting baseline — it
optimises F1, which prices recall directly and therefore does not reward
selectivity. Whether that is enough to move a *consensus* metric is a separate
question, and the honest prior after arm S is that a document-level scalar may run
into the same gap between "better rollouts" and "better vote distribution".
