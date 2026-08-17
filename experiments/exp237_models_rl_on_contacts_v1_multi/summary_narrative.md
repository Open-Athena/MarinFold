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

## Results so far

_(Fill in as results come in.)_
