# Summary slides — exp237: RL on the `<contacts-v1.multi>` model from #230

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide. -->

## What we did

RL on #230's `<contacts-v1.multi>` checkpoint, with the reward computed over the
**sections of a single rollout** rather than over the rollouts of a group.

Six reward designs, seven runs, **37 scored checkpoints**, all on 8 × A100 via
SkyRL: **M-C** each section's contribution to its own rollout's consensus ·
**M-F** the last section's F1 · **M-B** the best section's F1 (oracle) ·
**M-BC** best + consensus · **M-FC** last + consensus · **M-K** the rollout's own
consensus R-precision · **M-0** a zero-LR control.

## Why

#208 ran eleven runs across five rewards and none improved consensus
R-precision. The mechanism was a **unit mismatch**: the reward scored one
rollout, while the metric votes over 100 *independent* rollouts — an object no
rollout can see.

Under `<contacts-v1.multi>` one rollout emits ~22 contact sets in a single
generation, so a reward on those is computed on the kind of object the metric
scores, with credit assignment inside the sequence.

## The result

**Best consensus 0.5806 (M-K)**, best oracle 0.5663 (M-B), best final section
0.5267 (M-C) — three criteria, three different arms. M-K is the only arm to
improve **all four** aggregation modes with every CI excluding zero, on every cut.

**The primary criterion is not met.** 22 independent plain rollouts read 0.5896;
M-K is 0.0090 short. At a larger budget the two draw level — M-K pooled 0.6098
against plain-100's 0.6058, a paired CI that includes zero. So RL **closed the gap
between the multi format and ordinary sampling; it did not clearly open one.**

## #208's negative result is a dose-response, not a verdict

Consensus against distance moved: 0.5673 at KL 0, **0.5806 at 0.016**, 0.5529 at
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

## The arm that won was designed from a failure

M-C's collapse was traced to a scale bug: its per-section marginal is 366× larger
at one section than at 22, so it paid the policy to emit fewer candidates. The
diagnosis named the repair — a **rollout-level** base that *falls* when sections
are dropped.

**M-K is that base with nothing else on top:** reward = the rollout's own
consensus R-precision, GRPO-centred. Section count then held flat at ~22 for 48
steps, and it produced the experiment's best consensus. It is also the only arm
neither killed by a gate nor diverged — it ran out of scheduled steps.

**Within-sequence credit assignment was the hypothesis's mechanism, and it turned
out not to be needed.** What was needed was for the reward to be computable on
the object the metric scores — which is what the multi format makes possible.

## Two results that outlive the experiment

**Selection is dominated by aggregation.** An ORACLE selector of one draft reads
0.5646 where simply voting the same drafts reads 0.5750. A final section should
synthesise its predecessors, not pick among them.

**The spread is the resource.** Making candidates more uniform raises mean
section F1 from 0.432 to 0.532 and *lowers* best-of-22 — the same trade that
defeats every sharpening reward here, appearing in the corpus's own size law.

## What next

**Train M-K further.** It is the only arm that stopped for its schedule rather
than for a reason. Every other arm answers "how long can you train?" with "not
long"; M-K has not been asked.

**Then turn on the shaping and blend terms.** M-K is the `beta = lam = 0` corner
of the derived arm. The base alone beat every shaped and blended arm, so shaping
is now a hypothesis to test, not a fix to apply.

**Nothing further on M-B at lr 1e-5.** Two rates, 10 checkpoints, a smooth peak,
agreement to 0.002.

**The diversity gap is a corpus question.** One rollout's sections cover 658
distinct pairs against 1,065 for 22 independent rollouts.

## Figures

`curves_accuracy.png` — all 37 scored checkpoints against training step.

`curves_reward.png` — each arm against its own reward.

`dose_response.png` — R-precision against distance moved.

`reward_vs_section_count.png` — each reward's gradient in the count direction.

`section_count_incentive.png` — M-C's scale pathology, 366× at one section.

`gates_over_training.png` · `reward_shape.png` · `coverage_vs_kl.png`
