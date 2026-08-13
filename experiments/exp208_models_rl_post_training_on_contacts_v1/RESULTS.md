# exp208 results: RL post-training on contacts-v1

**Issue [#208](https://github.com/Open-Athena/MarinFold/issues/208) · five arms × 125 steps · 554-protein held-out eval · 8×A100**

**The one-paragraph version.** Every arm that actually trained the model made it
**worse**, and every arm that left the metric intact did so by not training. A
dense per-contact reward moves the policy hard and costs 0.021 R-precision
(p = 5.7e-19) — not through bad contacts but through missing ones: it learns to be
selective, emits **65% fewer distinct pairs**, and consensus scoring cannot rank a
pair that never receives a vote. Adding the leave-one-out consensus marginal
significantly repairs part of that damage (+0.0048 R, +0.0110 AUC over the dense
reward, both p < 1e-4). Two purely document-level rewards — a per-rollout F1 scalar
and the consensus marginal alone — left the model statistically unchanged, but
their KL says they never moved it, so they are **untested rather than ineffective**.

| arm | reward | terminal KL | R-precision | Δ warm start | AUC | vote coverage |
|---|---|---|---|---|---|---|
| — | baseline `exp199` | — | **0.6111** | — | **0.9487** | 2267 pairs |
| **C** | consensus marginal **only** (`lam_step=0`) | **0.00036** | 0.6116 | +0.0005 (p = 0.74) | 0.9484 | −0.5% |
| **D** | document F1 only, GRPO | **0.00135** | 0.6109 | −0.0001 (p = 0.93) | 0.9467 | −11.6% |
| **S** | dense per-contact | 0.09763 | 0.5898 | **−0.0213** (p = 5.7e-19) | 0.8976 | **−65.2%** |
| **B v1** | dense + consensus (`lam_doc` 4.5) | 0.09 | 0.5879 | −0.0232 | 0.8986 | −65.3% |
| **B v2** | dense + consensus (`lam_doc` 1067) | 0.07308 | **0.5946** | −0.0165 | **0.9087** | −60.1% |

Read the KL column alongside the score column. **The two arms with a moving policy
both lost; the two that scored at baseline had terminal KLs 50–200× smaller.** No
configuration tested here improved the metric.

Details: [ARM_S_RESULTS.md](ARM_S_RESULTS.md) · [ARM_B_RESULTS.md](ARM_B_RESULTS.md) ·
[ARM_D_RESULTS.md](ARM_D_RESULTS.md) · [VOTE_COVERAGE.md](VOTE_COVERAGE.md) ·
[ERR_DECAY_ANALYSIS.md](ERR_DECAY_ANALYSIS.md)

---

## 1. The finding: a per-rollout objective is wrong for a consensus metric

Arm S doubled single-rollout precision, 0.252 → 0.473, and lost 0.021 R-precision
and 0.051 AUC. Both are true at once, and only the second matters.

Its reward pays `n_scored · (precision − p̄)` — a **per-contact** quantity whose
lever is *which contacts to emit*, so the cheapest way to raise it is to emit only
confident ones. Consensus R-precision is a property of the **vote distribution over
100 rollouts**, and it needs the marginal, uncertain contact emitted *sometimes*;
that is what a vote count is for. Over the recall axis the two objectives point in
opposite directions.

Measured on paired within-protein data, four checkpoints on the same 554 proteins:

| checkpoint | pairs voted | ρ(Δcoverage, ΔAUC) |
|---|---|---|
| arm S step 40 | −15.8% | +0.142 (p = 8e-4) |
| arm S step 125 | **−65.2%** | **+0.781** (p = 6e-115) |
| arm D step 125 | −11.6% | −0.013 (p = 0.76) |

The proteins that lost the most coverage lost the most AUC. **AUC absorbs the
damage and R-precision hides it**: AUC ranks every candidate pair, so an unvoted
pair is unrankable (arm S is worse on 98.4% of proteins), while R-precision reads
only the top R, where the surviving contacts genuinely *are* better. A reward that
trades recall for precision looks nearly harmless on the headline number while
gutting the ranking beneath it.

This confirms pre-registered prediction 2 and its mechanism: *"the step-only arm
raises precision and moves consensus R-precision by ≤ 0 — union coverage shrinks
and votes concentrate."* At its own internal peak (step 40) arm S is −0.0023,
p = 0.12 — the 0.0023 reproducibility floor. **It never beat the warm start.**

## 2. Was a purely document-level reward tried?

Yes, twice, and neither trained the model.

**Arm D — one scalar per rollout (section F1), GRPO group baseline, no per-token
term.** Scored 0.6109 (−0.0001, p = 0.93). Mean KL 0.00069 against arm S's 0.0318;
0.00135 vs 0.09763 over the last 25 steps. Its own training reward is flat over all
125 steps (trend p = 0.87).

**Arm C — the consensus marginal alone (`lam_step = 0`).** Scored 0.6116 (+0.0005,
p = 0.74), coverage −0.5%, every training metric statistically flat (trend
p = 0.28–0.63). Terminal KL **0.00036** — the least movement of any arm, less even
than arm D.

Arm C's flat `pred/gt` (1.09–1.11 throughout, where arm S falls to 0.57) is
therefore **not** evidence that the consensus term prevents shrinkage. A policy
that does not move does not shrink. An earlier draft of this document claimed the
causal reading; it does not hold.

Two reasons these signals are so weak, and they compound:

- **Dilution.** A per-rollout scalar spread over ~600 response tokens gives each
  token ~0.03, against the dense reward's 0.248 concentrated on the three tokens of
  each contact. Equalising the *summed* contribution (which `lam_doc = 1067` does)
  does not equalise the per-token gradient.
- **No normalisation, for arm C specifically.** With `lam_step = 0` the advantage is
  the raw consensus marginal under a pass-through estimator, whereas arm D's GRPO
  normalises to unit variance. Arm C had both the thinnest and the least-scaled
  signal, and moved 4× less than arm D.

So "document-level" is not one thing: two document-level rewards here differ by 4×
in how far they moved the policy, and both are 50–200× below the dense arms.
**Neither has been tested at a learning rate that trains the model.**

## 2b. The re-runs at lr 1e-5

Both document-level arms were re-run at **10× the learning rate**, the change their
KL numbers called for. Both now train.

![arm D](plots/exp208_armD_lr.png)

![arm C](plots/exp208_armC_lr.png)

Faint traces are per-step values; bold is an 11-step rolling mean. Per-step noise
has sd ~0.056, which is why the block means in earlier drafts of these documents
looked like structure that was not there — the rolling mean is what makes the
comparison readable.

| arm | reward | lr 1e-6, last 20 steps | lr 1e-5, last 20 steps |
|---|---|---|---|
| D | document F1 | 0.3171 | **0.3967** (at step 78) |
| C | consensus marginal | 0.3118 | **0.3334** (at step 71) |

**Arm D separates clearly**; arm C's improvement is real but much smaller, matching
their KL ratios (11.5× vs 2.5× over their originals). Both re-runs raise precision
*and* recall *and* correct-contacts-per-rollout while `pred/gt` stays near 1.0 —
the opposite of arm S, which bought precision by emitting less.

These are **training-set** curves on **partial runs**, and both arms have previously
looked like they were improving on this axis before the held-out number said
otherwise. Only the 554-protein consensus score settles it.

## 3. The consensus marginal does real work — inside a dense reward

Arm B (dense + consensus) beats arm S on identical data and LR:

| | Δ vs arm S | p | better on |
|---|---|---|---|
| R-precision | **+0.0048** | 3.8e-05 | 47.3% |
| AUC | **+0.0110** | 3.3e-28 | 69.9% |

This is the only significant improvement anywhere in the experiment. It recovers
~22% of arm S's deficit — while recovering only **7.9%** of its coverage deficit,
so the consensus term improves the *ordering* of votes as well as their spread.
That is what a marginal contribution to the group's **correct** consensus should
do, and it is the one piece of evidence that a group-level objective does something
a per-rollout objective cannot.

Prediction 3 is **confirmed in direction, refuted in magnitude**: significant, and
nowhere near enough to reach baseline.

## 4. Four instrumentation failures, none visible in the eval numbers

Each was found by asking whether a run did what its config claimed — not by reading
its score.

1. **`err_decay = 0.5` deleted the reward's baseline.** The k-th error in a section
   costs `p̄·δ^k`, so on real rollouts the *next* wrong contact costs a median of
   **exactly 0.000000** against +0.745 for a correct one — 2.3% of the reward's
   magnitude. No intermediate value helps (ranking quality flat over δ ∈ [0, 0.9]).
   Now 1.0. [ERR_DECAY_ANALYSIS.md](ERR_DECAY_ANALYSIS.md)
2. **`p̄` drifted above true precision.** An unweighted mean over rollouts of
   per-rollout precision *ratios*, not the count-weighted aggregate: a 3-contact
   rollout moved it as hard as a 200-contact one. Ended at 0.5501 against a true
   0.4733, which pays the policy to go quiet. Now count-weighted;
   `INITIAL_PRECISION` 0.45 → 0.26 against five direct measurements of the pool.
3. **`lam_doc = 4.5` made the document term inert.** Its total contribution is
   exactly `lam_doc · marginal`; measured spreads are 18.6 (stepwise) vs 0.0174
   (marginal), so 4.5 carried **0.42%**. Arm B v1 reproduced arm S to three decimals
   and to **within one vote-pair in 2267**. Now 1067, identity pinned by a test.
4. **The constant-advantage guard could not fire.** It took `std` across the full
   padded row; `advantages * response_mask` zeroes the padding, so a constant
   per-token advantage still read as varying. It could only ever trip on a batch
   with no padding — and arm C, whose advantage is constant by construction, trained
   125 steps without it firing. Now computed over response tokens only.

Plus one infrastructure failure: **SkyRL's FSDP2 policy sharding silently destroys
the policy** at any shard count, via a weight sync that pushes a divergent copy into
the inference engines (trainer/engine logprob gap 1.33 nats sharded vs 0.017
unsharded). A zero-LR control proved it was the sync, not the gradient. Every run
here is unsharded.

## 5. What to run next

1. **Re-run arms C and D at 10–100× the learning rate**, targeting a terminal KL
   near arm S's ~0.098. Both document-level arms are currently untested, and this is
   the cheapest way to convert two null results into real ones.
2. **Sweep `lam_doc` in the dense+consensus arm.** It is the only lever shown to
   improve anything, and it demonstrably acts on the variable that sets the metric.
3. **Normalise the advantage for sequence-level rewards.** Arm C's pass-through
   estimator left its signal unnormalised; GRPO-style normalisation is the obvious
   fix and explains part of the 4× gap to arm D.

## 6. Limitations

- One epoch over 2,000 prompts per arm, lr 1e-6, single seed. Arm-to-arm
  differences are paired over 554 proteins and well-powered; absolute claims about
  what RL can achieve on this task are not supported by this budget.
- The eval pipeline reproduces the baseline's published 0.6103 at **0.6111** using
  exp82's worker and exp89's metric implementation, so the numbers are comparable to
  the record. All scoring uses the published scripts, not re-derivations.
- No arm improved the metric. The honest headline is a negative result plus one
  significant intra-arm effect (consensus > no consensus, inside a dense reward).
