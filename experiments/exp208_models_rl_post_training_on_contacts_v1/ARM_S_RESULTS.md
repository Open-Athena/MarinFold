# Arm S: the dense per-contact reward trains, and then eats itself

**Issue #208 · 125 steps, 2,000 AFDB prompts × 16 rollouts, unsharded on 8×A100**

**Summary.** The dense per-contact reward works — for about 50 steps. While the
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

<!--EVAL-->

## 5. What this says about #208's question

The experiment asks whether a dense per-contact reward beats a document-level one.
Arm S answers half of it: the dense signal *does* drive learning — 50 steps of
simultaneous precision and recall improvement is not noise, and it came from a
per-token reward with no group baseline at all. But its baseline is an EMA of the
policy's own behaviour, and that is a feedback loop the reward does not control.

A document-level F1 reward (arm D) has no p̄: the baseline comes from the group of
16 sibling rollouts, which is an unbiased comparison by construction, and F1 prices
recall directly rather than reaching it through a contact count. Arm D is the
natural control for exactly the failure documented here, and it is running.
