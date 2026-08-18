# Summary slides — exp237: RL on the `<contacts-v1.multi>` model from #230

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide. -->

## What we did

RL on #230's `<contacts-v1.multi>` checkpoint, with the reward computed over the
**sections of a single rollout** rather than over the rollouts of a group.

Five reward designs, ~25 scored checkpoints, all on 8 × A100 via SkyRL:
**M-C** each section's contribution to its own rollout's consensus · **M-F** the
last section's F1 · **M-B** the best section's F1 (oracle) · **M-BC** best +
consensus · **M-FC** last + consensus · **M-0** a zero-LR control.

## Why

#208 ran eleven runs across five rewards and none improved consensus
R-precision. The mechanism was a **unit mismatch**: the reward scored one
rollout, while the metric votes over 100 *independent* rollouts — an object no
rollout can see.

Under `<contacts-v1.multi>` one rollout emits ~22 contact sets in a single
generation, so a reward on those is computed on the kind of object the metric
scores, with credit assignment inside the sequence.

## The result

**Three arms beat the warm start on every mode**, all CIs excluding zero. Best
consensus **0.5775** (M-B), best oracle **0.5663** (M-B), best final section
**0.5267** (M-C).

**The primary criterion is not met.** 22 independent plain rollouts read 0.5896.
At a larger budget the two draw level — 0.6054 against plain-100's 0.6058. So RL
**closed the gap between the multi format and ordinary sampling; it did not open
one.**

## #208's negative result is a dose-response, not a verdict

Consensus against distance moved: 0.5673 at KL 0, **0.5775 at 0.009**, 0.5529 at
0.031, 0.3969 at 0.486. Every reward helps at small KL and damages at large.

#208 ran its arms at KL 0.06–0.10 and to 3.96 — past the peak on all of them —
and its two arms under 0.0015 never moved. **The window it needed lay between the
two learning rates it tried.** Two runs at a 3.3× different rate agree to 0.002 at
matched KL, so this is distance, not schedule.

## Reward design, in three parts

**`E[r] = 0` is necessary and not sufficient.** M-C's per-section marginal is
**366× larger at 1 section than at 22** — it paid the policy to emit a worse
answer — while `E[A] = 0` held exactly, because centring is computed *inside* the
quantity being gamed.

**Every arm moved its candidate count in the direction its own reward pointed.**
M-C's gradient is strongly negative (collapsed to 1.1 sections); M-B's strongly
positive and self-paying (grew to ~26 and held); M-F's nearly flat — and **a weak
gradient is not a safe one, it is an unconstrained one**: M-F ran to 259 sections
carrying 1.4 contacts each.

**Gates written against the failure you have seen will miss the next one.** All
three original criteria are one-sided; M-F's failure pushed every one *away* from
its threshold and none fired.

## Two results that outlive the experiment

**Selection is dominated by aggregation.** An ORACLE selector of one draft reads
0.5646 where simply voting the same drafts reads 0.5750. A final section should
synthesise its predecessors, not pick among them.

**The spread is the resource.** Making candidates more uniform raises mean
section F1 from 0.432 to 0.532 and *lowers* best-of-22 — the same trade that
defeats every sharpening reward here, appearing in the corpus's own size law.

## What next

**Fix M-C's scale bug and re-run it** — it is the best final-section arm *despite*
the bug. The fix is a rollout-level `GRPO(C_i(all))` base (measured scale-correct)
plus a **zero-sum** shaping term. The obvious repair — scoring against the causal
prefix — was tested and **refuted**.

**Nothing further on M-B.** Two rates, 13 checkpoints, a smooth peak, agreement to
0.002.

**The diversity gap is a corpus question.** One rollout's sections cover 658
distinct pairs against 1,065 for 22 independent rollouts.

## Figures

`dose_response.png` — 20 checkpoints, five arms, R-precision against distance.

`reward_vs_section_count.png` — each reward's gradient in the count direction.

`section_count_incentive.png` — M-C's scale pathology, 366× at one section.

`gates_over_training.png` · `reward_curves.png` · `reward_shape.png` ·
`coverage_vs_kl.png`
