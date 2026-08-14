# exp208 results: RL post-training on contacts-v1

**Issue [#208](https://github.com/Open-Athena/MarinFold/issues/208) · eleven scored runs · five reward designs · 554-protein held-out eval · 8×A100**

## The result

**No reward tested here improves consensus R-precision, the metric #208 is judged
by.** One — the leave-one-out consensus marginal at lr 4e-5 — significantly improves
the *ranking* metric (AUC +0.0032, p = 4e-05, better on 61% of proteins) while
leaving R-precision flat, and it is the only arm that does not damage the model.

Everything is explained by one mechanism: **the eval is a vote over 100 rollouts,
and a reward that makes each rollout individually better makes the hundred
redundant.** Consensus scoring cannot rank a pair that no rollout emits, so a
policy that becomes more selective — the natural response to almost any
per-contact reward — destroys the ranking underneath its own improving precision.

| arm | reward | lr | terminal KL | R-precision | Δ warm start | AUC | vote coverage |
|---|---|---|---|---|---|---|---|
| — | baseline `exp199` | — | — | **0.6111** | — | **0.9487** | 2267 pairs |
| **C v1** | consensus marginal only | 1e-6 | 0.0004 | 0.6116 | +0.0005 (p = 0.74) | 0.9484 | −0.5% |
| **C v2** | consensus marginal only | 1e-5 | 0.012 | 0.6112 | +0.0001 (p = 0.95) | 0.9445 | −24.6% |
| **C v3** | consensus marginal only | 4e-5 | 0.168 | 0.6099 | −0.0012 (p = 0.53) | **0.9519** | **−6.4%** |
| **D v1** | document F1 only (GRPO) | 1e-6 | 0.0014 | 0.6109 | −0.0001 (p = 0.93) | 0.9467 | −11.6% |
| **D v2** | document F1 only (GRPO) | 1e-5 | 0.084 | 0.6027 | −0.0083 (p = 1.6e-05) | 0.9184 | −61.0% |
| **C v4** | consensus only, 10k prompts | 4e-5 | **3.96** ⚠ | 0.5959 | −0.0151 (p = 5.7e-07) | 0.9405 | −18.7% |
| **N2** | novelty-weighted, normalised | 1e-6 | 0.063 | 0.5991 | −0.0119 (p = 1.6e-09) | 0.9161 | −55.5% |
| **N** | novelty-weighted, unnormalised | 1e-6 | 0.094 | 0.5954 | −0.0157 (p = 9.3e-15) | 0.9104 | −59.4% |
| **B v2** | dense + consensus (`lam_doc` 1067) | 1e-6 | 0.098 | 0.5946 | −0.0165 (p = 1.9e-17) | 0.9087 | −60.1% |
| **S** | dense per-contact | 1e-6 | 0.098 | 0.5898 | −0.0213 (p = 5.7e-19) | 0.8976 | −65.2% |
| **B v1** | dense + consensus (`lam_doc` 4.5 — inert) | 1e-6 | 0.098 | 0.5879 | −0.0232 (p = 1.3e-24) | 0.8986 | −65.3% |

Ordered by R-precision. **Vote coverage orders the table almost perfectly**, which
is the whole finding in one column. ⚠ C v4 diverged and was stopped at step 270;
its row is the step-200 checkpoint.

## How to read this document

Three things are worth taking away, in descending order of how portable they are:

1. **§1 and §3f — the mechanism.** Per-contact rewards sharpen; only a
   group-scored objective resists it. This is the finding.
2. **§4 and §3e — the reward-design invariant.** Three separate modifications broke
   `E[r] = p − p̄` by weighting one side of a centred reward, each costing a full
   training run and each catchable by a five-line calculation beforehand.
3. **§3b — a refuted analysis, kept deliberately.** A tempting fit (R² = 0.95) that
   said diversity loss depends only on how far the policy moves. Arm C v3 refuted
   it. It is retained with its refutation because the confound it fell into is easy
   to repeat.

Detailed write-ups: [ARM_S_RESULTS.md](ARM_S_RESULTS.md) ·
[ARM_B_RESULTS.md](ARM_B_RESULTS.md) · [ARM_D_RESULTS.md](ARM_D_RESULTS.md) ·
[VOTE_COVERAGE.md](VOTE_COVERAGE.md) · [ERR_DECAY_ANALYSIS.md](ERR_DECAY_ANALYSIS.md)

All numbers come from exp82's `score_rollout_worker.py` and exp89's metric
implementation via `build_rollout_rows.py` — the published scripts, not
re-derivations. The pipeline reproduces the baseline's recorded 0.6103 at 0.6111.

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
**Neither had, at this point, been tested at a learning rate that trains the
model.** Both were re-run at higher LR (§2b, §3c, §3d), which is what turned these
two null results into real ones.

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
their KL ratios. Both re-runs raise precision *and* recall *and*
correct-contacts-per-rollout while `pred/gt` stays near 1.0 — the opposite of arm S,
which bought precision by emitting less.

**Read these two charts as a cautionary tale, not a result.** They are training-set
curves, and the held-out score below contradicts them: the arm whose curve rises
most is the arm that lost the most on the metric of record.

**The training gains did not transfer, and arm D v2 is the cleanest demonstration
in this experiment of why.**

| arm | R-precision | Δ baseline | AUC | union pairs | total votes | votes/pair |
|---|---|---|---|---|---|---|
| baseline `exp199` | **0.6111** | — | 0.9487 | 2267 | 16,191 | 7.14 |
| arm D v1 (lr 1e-6) | 0.6109 | −0.0001 (p = 0.93) | 0.9467 | 2005 | 15,797 | 7.88 |
| **arm D v2 (lr 1e-5)** | 0.6027 | **−0.0083** (p = 1.6e-05) | 0.9184 | **885** | 14,519 | **16.41** |
| arm C v1 (lr 1e-6) | 0.6116 | +0.0005 (p = 0.74) | 0.9484 | 2256 | — | — |
| **arm C v2 (lr 1e-5)** | 0.6112 | +0.0001 (p = 0.95) | 0.9445 | 1709 | 15,668 | 9.17 |
| arm S (dense) | 0.5898 | −0.0213 | 0.8976 | 788 | 8,438 | 10.71 |

Arm D v2 trained beautifully by every per-rollout measure — precision 0.4125 against
arm S's 0.4354, **50.8** correct contacts per rollout against arm S's 33.8, `pred/gt`
a healthy 0.90 — and it is **significantly worse than its warm start**.

The last three columns say why, and they identify a **second, distinct failure mode**:

- **Arm S collapsed volume.** Total votes fell 16,191 → 8,438 (−48%). It emitted
  fewer contacts.
- **Arm D v2 collapsed diversity.** Total votes barely moved (−10%), but the union
  fell 2267 → 885 and votes-per-pair went **7.1 → 16.4**. It emitted just as many
  contacts — *the same ones every time*.

That is sharpening, and it is exactly what the hypothesis section predicted before
any of this ran: *"a precision-only reward is a sharpening operator by
construction: it pushes every rollout toward the model's single best guess. So it
can raise per-rollout precision and lower consensus R-precision."* A document-level
F1 reward turns out to be a precision-only reward in this sense; optimising each
rollout to be individually excellent makes the 100 rollouts redundant, and a
consensus over redundant rollouts carries less information than a consensus over
diverse ones.

**Arm C v2 resists it**, which is the one place the consensus marginal's design
shows through: votes/pair 9.17 against arm D v2's 16.41, coverage −24.6% against
−61.0%, and R-precision holding at baseline. Rewarding a rollout for what it adds
to the group's vote is structurally an anti-sharpening objective. This is
suggestive rather than conclusive — arm C v2's terminal KL is 0.0123 against arm D
v2's 0.0836, so it also simply moved less, and the two are not matched.

**This invalidates the tripwire built earlier in this experiment.** It watches
`pred/gt`, which reads 0.90 — perfectly healthy — for a run whose eval coverage fell
61%. Per-rollout statistics cannot see diversity collapse. `contacts/votes_per_pair`
is now logged for exactly this: total emitted contacts over distinct pairs within a
rollout group, 1.0 = disjoint proposals, G = identical ones.

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

## 3b. ~~Coverage loss tracks distance moved, not which reward~~ (REFUTED — see §3c)

Plotting every arm's vote coverage against how far its policy actually moved:

![coverage vs KL](plots/exp208_coverage_vs_kl.png)

| arm | terminal KL | coverage | fit | residual |
|---|---|---|---|---|
| arm C v1 | 0.00036 | −0.5% | +3.9 | −4.4 |
| arm D v1 | 0.00135 | −11.6% | −11.5 | −0.1 |
| **arm C v2** | 0.0123 | −24.6% | −37.1 | **+12.5** |
| arm B2 | 0.07308 | −60.1% | −57.8 | −2.3 |
| arm D v2 | 0.0836 | −61.0% | −59.3 | −1.7 |
| arm S | 0.09763 | −65.2% | −61.1 | −4.1 |

`coverage% ≈ −26.7·log10(KL) − 88.1`, **R² = 0.95**. Six arms with four different
reward designs — dense per-contact, dense+consensus, document F1, consensus-only —
lie on one line. To a first approximation the reward does not determine how much
diversity is lost; *how far the policy moved* does.

If that holds, it reframes every result above. The arms did not fail because their
objectives were wrong in kind; they failed because training this policy at all
costs vote diversity, and the metric is a vote. It also predicts that **more data
will not help**: the runs already use `epochs=1` over 2,000 prompts with
`train_batch_size=16`, i.e. exactly 125 steps and no prompt seen twice, so there is
no repetition to relieve. More prompts buys more steps, more steps means more
movement, and more movement is further down this line.

The one crack in it is **arm C v2's +12.5 residual** — the only arm that lost
meaningfully less coverage than its KL predicts, and the only one whose reward
explicitly pays for what a rollout *adds* to the group. That is consistent with the
consensus marginal being structurally anti-sharpening, and it is one point.

**The decisive test is arm C at arm S's step size** (lr 4e-5, targeting terminal
KL ≈ 0.098 from arm C's own scaling KL ∝ lr^1.53). Two outcomes, both worth having:

- lands at ~−60% → the fit is the whole story, the reward is irrelevant to
  diversity, and it has to be targeted **explicitly**.
- holds near −25% → the consensus marginal genuinely resists sharpening.

**It landed at −6.4%** (§3c), refuting the fit outright. Every high-KL point in the
fit contained the stepwise term, so "distance moved destroys diversity" was really
"the stepwise term destroys diversity, and only the stepwise arms had moved far."
This section is kept because the confound is an easy one to repeat: five points
across four reward designs, R² = 0.95, and still wrong about the causal variable.

## 3c. Arm C at arm S's step size: the only arm to beat baseline on anything

The decisive run above — arm C's reward at lr 4e-5, chosen to reach arm S's
terminal KL — overshot to **KL 0.168**, the largest policy movement of any arm, and
did *not* collapse:

| arm | terminal KL | R-precision | Δ base | AUC | Δ base | union pairs | total votes | votes/pair |
|---|---|---|---|---|---|---|---|---|
| baseline `exp199` | — | **0.6111** | — | 0.9487 | — | 2267 | 16,191 | 7.14 |
| arm C v2 (lr 1e-5) | 0.0123 | 0.6112 | +0.0001 (p = 0.95) | 0.9445 | −0.0043 | −24.6% | 15,668 | 9.17 |
| **arm C v3 (lr 4e-5)** | **0.168** | 0.6099 | −0.0012 (p = 0.53) | **0.9519** | **+0.0032** (p = 4e-05) | **−6.4%** | **19,871** | 9.36 |
| arm D v2 (lr 1e-5) | 0.0836 | 0.6027 | −0.0083 | 0.9184 | −0.0303 | −61.0% | 14,519 | 16.41 |
| arm S (dense) | 0.0976 | 0.5898 | −0.0213 | 0.8976 | −0.0511 | −65.2% | 8,438 | 10.71 |

**AUC +0.0032, p = 4e-05, better on 61% of proteins — the first statistically
significant improvement over the warm start anywhere in this experiment.**
R-precision is unchanged (−0.0012, p = 0.53).

**But it is a redistribution, not a uniform gain, and the loss is more significant
than the win.** By separation band and cut:

| band | cut | baseline | arm C v3 | Δ | p |
|---|---|---|---|---|---|
| all | AUC | 0.9487 | 0.9519 | **+0.0032** | 4e-05 |
| **long** | **AUC** | 0.9340 | 0.9398 | **+0.0058** | 7.1e-07 |
| medium / short | AUC | — | — | +0.0006 / +0.0004 | 0.54 / 0.66 |
| all | **L/5** | 0.8189 | 0.8051 | **−0.0138** | **1.9e-07** |
| long | L/5 | 0.7105 | 0.6991 | −0.0113 | 0.00023 |
| short | R | 0.6814 | 0.6746 | −0.0068 | 0.022 |
| all | R | 0.6111 | 0.6099 | −0.0012 | 0.53 |

The gain is **concentrated in long-range AUC** (+0.0058, the hardest and most
valuable band) and the loss is **at the top of the ranking** (L/5 precision −0.0138,
a smaller p-value than the AUC gain). That is exactly what wider vote coverage
should do: spreading mass over more candidates improves the global ordering,
especially where the model was previously blind, and dilutes the very top of the
list. Reporting only the AUC line would be choosing the favourable metric.

Whether longer training converts this into a net win, or simply deepens both sides
of the trade, is the question the 10k run was used to answer — see §3d. It does
neither: it diverges.

**This refutes §3b.** The R² = 0.95 fit predicted ~−67% coverage at KL 0.168; the
measured value is −6.4%, and total votes went *up* 23% (16,191 → 19,871). The fit
was confounded: every high-KL point in it contained the stepwise term, so what
looked like "distance moved destroys diversity" was really "the stepwise term
destroys diversity, and only the stepwise arms had moved far." The consensus
marginal differs in kind, not merely in degree.

Its training trace shows the same thing from the inside — `pred/gt` **rises**
(1.16 → 1.21) and contacts per rollout climb 153.6 → 162.8, where every other arm
at every learning rate shrank. A reward that pays a rollout for what it *adds* to
the group's vote makes the policy emit more, and more varied, candidates.

The honest reading: this moves the ranking metric (AUC) and not the headline one
(R-precision). AUC rewards getting the whole candidate ordering right, which is
what better vote coverage buys; R-precision reads only the top R, where the extra
coverage does not yet help. Whether more training converts the AUC gain into an
R-precision gain is exactly the open question — so the next run is this
configuration on **10,000 prompts (625 steps, 5×)**, checkpointing every 50 steps.

## 3d. Training longer at the same LR diverges

Arm C v3's configuration on 10,000 prompts (625 steps planned, 5×), checkpointing
every 50, **stopped at step 270**:

| checkpoint | R-precision | Δ base | AUC | Δ base | AUC (long) | coverage | KL |
|---|---|---|---|---|---|---|---|
| baseline `exp199` | **0.6111** | — | 0.9487 | — | 0.9340 | — | — |
| arm C4 step 50 | 0.6036 | −0.0074 | 0.9391 | −0.0097 | 0.9199 | −32.5% | ~0.02 |
| **arm C3 step 125** (2k) | 0.6099 | −0.0012 | **0.9519** | **+0.0032** | **0.9398** | −6.4% | 0.168 |
| arm C4 step 200 | 0.5959 | **−0.0151** (p = 5.7e-07) | 0.9405 | −0.0082 | 0.9217 | −18.7% | ~0.7 |

**It did not continue improving.** R-precision degrades monotonically
(−0.0074 → −0.0151) and arm C3's AUC gain is gone (+0.0032 → −0.0082).

The cause is **KL runaway, not the data**: 0.064 at step 60 → 0.77 at step 213 →
**3.96 at step 270**. At lr 4e-5 the policy does not converge to a new operating
point, it diverges, and the training metrics show it — precision falls to 0.265
(below the base model's ~0.27) while `pred/gt` climbs to 1.44 and contacts/rollout
to 200. That is over-emission: the consensus marginal pays a rollout for adding
candidates to the group vote, and with the stepwise term at zero there is nothing
opposing it.

So the two failure modes now bracket the reward design:

- **the stepwise per-contact term alone** sharpens — precision up, coverage −65%,
  the model emits less and the committee becomes redundant;
- **the consensus marginal alone** over-emits — coverage held, but at high KL the
  model emits ever more, precision falls below baseline, and the run diverges.

Arm C3's +0.0032 sits between them, at KL ≈ 0.168. That is a **narrow window**, not
a direction that scales with more training.

**One thing this does not settle**: arm C4 changed the prompt pool *and* ran past
arm C3's KL, so "does more data help" is still open — it would need a lower LR (say
1e-5) over 625 steps to land near KL 0.168 gradually. The question the run *was*
asked, "does it keep improving with more training at this setting", is answered no.

## 3e. Novelty weighting, and a reward-design rule this experiment keeps proving

Arm N pays a correct contact by how few siblings also found it —
`r = (1-p̄)·(floor + (1-floor)·novelty)`, wrong contacts still `-p̄` — intended as
the synthesis of the two bracketing failures. At a **matched** terminal KL
(0.0936 against arm S's 0.0976):

| arm | R-precision | Δ base | AUC | union pairs | votes/pair |
|---|---|---|---|---|---|
| baseline `exp199` | **0.6111** | — | 0.9487 | 2267 | 7.14 |
| arm C3 (consensus, lr 4e-5) | 0.6099 | −0.0012 | **0.9519** | −6.4% | 9.36 |
| **arm N (novelty, unnormalised)** | 0.5954 | **−0.0157** (p = 9.3e-15) | 0.9104 | **−59.4%** | 10.46 |
| arm S (dense stepwise) | 0.5898 | −0.0213 | 0.8976 | −65.2% | 10.71 |

**Arm N landed next to arm S, not next to arm C3.** It is significantly better than
arm S (+0.0056, p = 4.5e-06) and still far below baseline: the same sharpening
failure, slightly softened.

The cause is algebra that should have been checked before the GPU time. Novelty
scales the **positive** term down — a redundant correct contact pays `floor` = 0.25
instead of 1 — while the penalty stays at a full `-p̄`. So

    E[emit] = p·(1-p̄)·w̄ - (1-p)·p̄     with mean weight w̄ < 1

which is strictly **more negative** than the `p - p̄` the centring is built on.
Scaling the reward for correct contacts *down* strengthens the pressure to emit
fewer of them: the design pushed in the opposite direction to its intent, and the
training trace shows it plainly (pred/gt 1.08 → 0.64 against arm S's 1.08 → 0.57).

**The rule this experiment keeps re-proving.** Three separate reward modifications
have now broken the same invariant:

| change | what it did | effect |
|---|---|---|
| `err_decay = 0.5` | discounted the k-th error | penalty → 2.3% of the positive term; baseline gone |
| `p̄` unweighted mean | over-counted short rollouts | p̄ drifted above true precision; paid for silence |
| novelty, unnormalised | scaled the positive term down | E[emit] below `p - p̄`; strengthened the shrink |

Every one is a weight applied to **one side** of a centred reward. The invariant is
`E[r] = p - p̄`, and any modification must be checked against it *before* it runs —
it is a five-line calculation and each of these cost a full training run.

Fix: normalise the novelty weights to mean 1 over the group's correct contacts, so
the term redistributes (novel pays more, redundant less) with the average — and
therefore the baseline — unchanged. Pinned by a test that asserts mean reward per
correct contact equals `(1-p̄)` whatever the novelty distribution.
`novelty_normalize=false` reproduces arm N. **Result in §3f: it helped
significantly and not nearly enough** — R −0.0119 against arm N's −0.0157, coverage
−55.5% against −59.4%, still far below baseline.

## 3f. The per-contact family forms a ladder, and none of it escapes

Normalising the novelty weights (§3e) fixed the algebra and helped — significantly,
and not nearly enough:

| arm | reward | R-precision | Δ base | AUC | coverage | final pred/gt |
|---|---|---|---|---|---|---|
| baseline `exp199` | — | **0.6111** | — | 0.9487 | — | — |
| **arm C3** | consensus marginal | 0.6099 | −0.0012 (p = 0.53) | **0.9519** | **−6.4%** | 1.21 |
| arm N2 | novelty, **normalised** | 0.5991 | −0.0119 | 0.9161 | −55.5% | 0.69 |
| arm N | novelty, unnormalised | 0.5954 | −0.0157 | 0.9104 | −59.4% | 0.64 |
| arm S | plain per-contact | 0.5898 | −0.0213 | 0.8976 | −65.2% | 0.57 |

Every step of diversity-awareness helps by a significant margin — N2 beats N by
+0.0038 (p = 8e-04), N2 beats S by +0.0094 (p = 1.2e-14) — and **vote coverage
tracks the ordering exactly**, −65.2% → −59.4% → −55.5%. The mechanism is
consistent throughout.

But the whole per-contact family sits 0.012–0.021 below baseline, and the gap to
arm C3 (−0.0012, coverage −6.4%) is far larger than the gaps within it. Making a
per-contact reward diversity-aware mitigates the damage; it does not remove it.

**Why, and this is the experiment's most general conclusion.** A p̄-centred
per-contact reward is *intrinsically a sharpening operator*. Emitting a contact has
expected value `p - p̄`, so any candidate below the policy's current precision is
net-negative, and the optimal move — as the model gets better at ranking its own
candidates — is to emit fewer and better ones. That is a **first-order** pressure
built into the reward's definition. Novelty weighting is a **second-order**
redistribution among contacts that are already being emitted, and it cannot
overcome the term that decides whether to emit at all.

The consensus marginal escapes because it never scores a contact for being correct.
It scores a *rollout* for what it contributes to the group's vote, so a rollout that
withholds an uncertain-but-true contact is penalised by the group's worse consensus
— exactly the pressure the per-contact family lacks. That is why it is the only
reward here that raised coverage-dependent metrics, and why it is the only one
whose training trace shows `pred/gt` going **up**.

## 4. Instrumentation failures, none visible in the eval numbers

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

Everything below is what the evidence supports, in order.

1. **Group-scored objectives, not per-contact ones.** This is the conclusion of
   §3f. A p̄-centred per-contact reward is intrinsically a sharpening operator, and
   the four arms in that family form a ladder (S → N → N2 → B v2) where every
   increment of diversity-awareness helps and none escapes. The consensus marginal
   is the only tested reward that scores a *rollout* for its contribution to the
   group rather than a *contact* for being correct, and it is the only one that
   preserved coverage. Variants worth trying are all of that shape: the marginal
   against a larger group, a coverage-explicit group reward, or a marginal computed
   on the eval's own 100-rollout scale rather than the training group of 16.
2. **Settle whether more data helps, at a matched KL.** Arm C v4 confounded pool
   size with KL runaway — it diverged (KL 3.96) rather than testing the 10,000-prompt
   pool. The clean version is lr 1e-5 over 625 steps, landing near arm C v3's
   KL ≈ 0.168 gradually. The dataset is built and staged
   (`data/skyrl_train_10k.parquet`).
3. **Find the KL sweet spot.** Arm C's three learning rates trace a curve — 0.0004
   (nothing), 0.012 (nothing), 0.168 (AUC +0.0032), 3.96 (divergence). The useful
   window is narrow and only three points wide; 2–3 more runs would locate it.

Explicitly **not** recommended: more per-contact reward shaping. Four attempts, a
clean monotone ladder, and none of them reaches baseline.

## 6. Limitations

- **One epoch, one seed, 2,000 prompts per arm** (except C v4). Arm-to-arm
  differences are paired over 554 proteins and well-powered; absolute claims about
  what RL can achieve on this task are not supported by this budget.
- **The AUC gain is one run.** Arm C v3's +0.0032 has not been replicated, and the
  two attempts to extend it (more data, more steps) both failed. Treat it as a
  located effect, not a robust one.
- **Arm C v3 is a redistribution, not a uniform gain** (§3c): +0.0058 long-range
  AUC against −0.0138 L/5 precision, the latter with a smaller p-value.
- **No run used a KL penalty.** `init_kl_coef` was never set, which is why C v4
  could diverge to 3.96 unchecked. A KL-regularised re-run of any of these arms
  might behave quite differently and none was tried.
- **Every number is from the published scoring path** — exp82's
  `score_rollout_worker.py`, exp89's metric via `build_rollout_rows.py` — and the
  pipeline reproduces the baseline's recorded 0.6103 at 0.6111. Nothing here is
  scored by code written for this experiment.

## 7. Conclusions that were wrong along the way

Recorded because the corrections were more informative than the claims, and because
each was caught by measurement rather than review.

| claim | why it looked right | what refuted it |
|---|---|---|
| Arm S was length-gaming its reward | reward rose while response length hit the cap | contact tallies: it emitted *fewer* contacts (pred/gt 1.11 → 0.006); the length was a policy collapse from FSDP2 sharding |
| Diversity loss depends only on how far the policy moves (R² = 0.95) | six arms, four reward designs, one clean line | arm C v3 at the largest KL of any run lost 6.4%, not the predicted 67% — every high-KL point in the fit shared the stepwise term |
| Arm C v3 "beats baseline" | AUC +0.0032, p = 4e-05 | the band breakdown: L/5 precision −0.0138 at a *smaller* p-value; it redistributes |
| Arm C's flat pred/gt showed the consensus term prevents shrinkage | pred/gt held 1.09–1.11 where arm S fell to 0.57 | terminal KL 0.00036 — the policy never moved. A policy that does not move does not shrink |
| Novelty weighting would oppose sharpening | it pays more for contacts the group missed | it scaled the *positive* term down while the penalty stayed fixed, making `E[emit]` more negative and *strengthening* the shrink |
