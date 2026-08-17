# Summary slides — exp237: RL on the `<contacts-v1.multi>` model from #230

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide.
     Keep this current as the experiment progresses. -->

## What we're doing

RL on #230's `<contacts-v1.multi>` checkpoint, with the reward computed over
the **sections of a single rollout** rather than over the rollouts of a group.

Four arms, all warm-started from `plm-exp230-cv1-multi-1_5b-.../step-1988`:
**M-C** rewards each section by its leave-one-out contribution to its own
rollout's consensus; **M-F** rewards the last section's F1; **M-B** rewards the
best section's F1 (an oracle ceiling); **M-0** is a zero-LR control.

## Why

#208 ran eleven scored runs across five reward designs and none improved
consensus R-precision. The mechanism was a **unit mismatch**: the reward acted
on one rollout, while the metric scored a vote over 100 *independent* rollouts —
an object no single rollout can see. Making each rollout individually better
made the hundred redundant, and consensus scoring cannot rank a pair that no
rollout emits.

Under `<contacts-v1.multi>` the candidate set lives **inside one rollout** —
~22 contact sets in one sequence. A reward on the aggregate of one rollout's
sections is therefore computed on the same kind of object the metric scores, and
its credit assignment is within the sequence, where the gradient can reach it.

That is the only reason to expect a different outcome from #208, and it is the
one thing this experiment tests.

## The risk, preregistered

#230's checkpoint already sits at mean pairwise Jaccard **0.304** between its
sections, past exp200's 0.30 diversity-collapse criterion **before any RL**.
#208's dominant failure was RL collapsing diversity. So union coverage, total
votes and votes-per-pair are reported every batch, and three diversity gates are
kill criteria rather than diagnostics.

## Results — the hypothesis is half right

Moving the reward's unit from *rollouts of a group* to *sections of one rollout*
produces what #208 could not: an RL checkpoint that **improves consensus
R-precision**. Arm M-C step-18 is the best on every aggregation mode —
consensus 0.5750 (+0.0077), oracle-best 0.5578 (+0.0235), last 0.5267 (+0.0701).
On eval2-natural, last goes 0.1696 → 0.2421.

It does not close the gap to independent sampling. 22 plain rollouts read 0.5896
against 0.5750 — the closest a multi-mode number has come, and 0.0146 short.

## #208's result is a dose-response, not a verdict

Consensus against distance moved: 0.5673 at KL 0, **0.5750 at 0.007**, 0.5741 at
0.016, 0.5529 at 0.031, 0.3969 at 0.486. Every reward here helps at small KL and
damages at large. #208 ran its arms at KL 0.06–0.10 and to 3.96 — past the peak
on all of them — and its two arms under 0.0015 never moved at all. The window it
needed lay **between the two learning rates it tried**.

## Reward shape decides how a run fails, not whether

M-C and M-F halve the contacts emitted while becoming *more* diverse (Jaccard
0.60× and 0.44×). M-B holds the volume and emits the same contacts 1.7× as often.
Both routes end at the same coverage floor. #208 found these two modes across
different reward families; here they are produced deliberately by reward shape,
on one model and one data order.

## Why M-C collapsed: the reward is not scale-free

`m_k = C(all) − C(all \ {k})` grows as a rollout emits *fewer* sections, because
each survivor is then more load-bearing. So a rollout can raise the reward on
every one of its sections by making its own consensus worse.

Measured by truncating real rollouts: the reward per section is **366× larger at
1 section than at 22**, while the rollout's own consensus falls 0.543 → 0.341. In
groups differing in nothing but section count, a one-section rollout gets **+4.80**
advantage and a 22-section rollout **−0.22**.

`E[A] = 0` holds exactly throughout — centring is computed *within* the quantity
being gamed. This explains the accelerating collapse (13.7 → 1.1 sections over
five steps) and why M-F and M-B, which have no such term, kept or grew theirs.

**The clause for #208's rule:** `E[r] = 0` constrains the mean over the candidates
the policy emitted; it says nothing about whether the reward's *scale* depends on
how many that was.

## The other lesson: gate on `union/R`, and on precision

The preregistered coverage gate stopped all three arms; union/R never left 2.8–4.6
in any of them, including the collapsed one, whose union/R was *higher* than the
warm start's. Coverage was never binding. Precision (0.50 → 0.14) was.

## Figures

`plots/dose_response.png` — R-precision against distance moved, three
aggregation modes, with the budget-matched bar drawn on.

`plots/gates_over_training.png` — the diversity gates per batch, rolling median.

`plots/reward_shape.png` — arm M-C's marginal distribution and its
group-centred advantage, i.e. the mechanism this run proposed and then refuted.

`plots/section_count_incentive.png` — the scale pathology: 366x the reward for
a worse answer.

`plots/coverage_vs_kl.png` — coverage against distance, #208's question re-asked.
