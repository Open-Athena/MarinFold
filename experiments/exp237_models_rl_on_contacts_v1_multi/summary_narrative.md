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

## Two lessons for the next reward

**`E[r] = p − p̄` is necessary and not sufficient.** M-C's advantage is centred so
`E[A] = 0` holds exactly — and it still shrank the policy, because 45 % of section
marginals are an atom at exactly zero, so the *median* section is negative while
the mean is zero. Checkable from a histogram before the run.

**Gate on `union/R`, and on precision.** The preregistered coverage gate stopped
all three arms; union/R never left 2.8–4.6 in any of them, including the collapsed
one, whose union/R was *higher* than the warm start's. Coverage was never binding.
Precision (0.50 → 0.14) was.
