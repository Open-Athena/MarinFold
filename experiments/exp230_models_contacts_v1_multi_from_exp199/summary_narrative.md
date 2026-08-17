# Summary slides — exp: fine-tune contacts-v1-exp199-1.5B into a clean <contacts-v1.multi> multi-draft model (on-policy drafts, PDB + AFDB, 30%-decontaminated)

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide.
     Keep this current as the experiment progresses. -->

## What we're doing

Can we port [#163](https://github.com/Open-Athena/MarinFold/issues/163)'s `<contacts-v1.multi>` multi-draft format onto the current best base model (`contacts-v1-exp199-1.5B`) — with **on-policy** drafts, an **experimental-PDB** data component, and a protein pool **decontaminated at 30% identity** against the eval set — such that the resulting checkpoint (a) writes many diverse candidate contact maps under `<contacts-v1.multi>`, (b) is a **clean single-document decoder** under plain `<contacts-v1>`, and (c) gives up nothing on plain-mode R-precision?

This is the SFT starting point for best-of-N RL ([#200](https://github.com/Open-Athena/MarinFold/issues/200) / [#208](https://github.com/Open-Athena/MarinFold/issues/208)), which is a **separate** experiment and explicitly out of scope here.

## Why

**H1 — the format transfers to a stronger base.** #163 established that multi-draft generation is a loss-weight-profile question, not a capacity question: `w_draft` has to compete with `w_final` for the section-boundary transition, and arm **F** (header 0.1 / draft 1.0 / final 1.0) plus a **50% plain-rehearsal mix** is the configuration that both emitted ~15 candidates and held the base task (R-prec 0.3374 vs base 0.3357 — a tie). Nothing in that mechanism is specific to E8, so it should reproduce from exp199.

**H2 — the mode leak is an under-training artifact, and steps are the lever.** #163's arm F emits **~2.94 sections under the plain `<contacts-v1>` sentinel** — the leak this issue is chartered to fix. It was trained for **405 steps**. [#175](https://github.com/Open-Athena/MarinFold/issues/175) got a *completely* clean token-0 mode switch out of the same kind of marker on the same kind of 50:50 marked mixture — **0.1 vs 42.0 retracts per rollout on one checkpoint, prompt-selected** — after **2,070 steps**. The two runs differ by ~5x in optimization, not in mechanism. Predicted: mean sections in plain mode falls to ~1.0 somewhere between 405 and ~2,000 steps. **Intermediate checkpoints make this falsifiable rather than assumed** (see Approach, Phase 3).

**H3 — drafts must be on-policy, and the risk is diversity, not quality.** #163's drafts were E8 rollouts at ~12% per-contact precision. exp199 is a far stronger sampler (R-prec **0.6103** under exp82's reference worker, [#209](https://github.com/Open-Athena/MarinFold/issues/209)), and RL will sample from *this* model, so the training drafts must come from it. The preregistered risk is the opposite of the obvious one: a stronger sampler is a *more consistent* one, so candidate diversity may shrink below #163's mean Jaccard of 0.071 — and best-of-N reward is paid for by spread, not by mean quality. **Jaccard is a reported gate, not an afterthought.**

**H4 — experimental PDB is worth including here.** #222's corpora are the first contacts-v1 documents whose `<begin_statements>` section is a *measurement* rather than a prediction. The multi-draft document is exactly the place that distinction lands: drafts are model opinion, the final section is truth, and the model is being taught to tell them apart.

## Result

Both gates pass. The `<contacts-v1.multi>` format ports onto exp199 at
essentially no cost to plain-mode accuracy, and the token-0 marker is a clean
mode switch.

Gate A, plain contacts-v1 R-precision, base and fine-tune scored in one run:
legacy-554 goes 0.6083 to 0.6058, a change of -0.0025, inside the preregistered
-0.005 tolerance. On eval2-natural, the honest low-homology cut, it improves:
0.3354 to 0.3381.

Gate B: 1.008 contact sets per rollout under the plain sentinel, with 99.8% of
rollouts emitting exactly one. #163's arm F read 2.94 here.

The base reproduces its own published reference to within 0.003 on all three
cuts, which is what makes these numbers trustworthy rather than merely
self-consistent.

## The leak never opened

H2 framed the mode leak as an under-training artifact that more steps would
close, somewhere between 405 and about 2,000. Measuring every checkpoint shows it
never opened at all: plain mode sits at roughly 1.0 contact sets from step 250,
peaks at 1.15 near step 750, and is exactly 1.000 at steps 1500 and 1750. Multi
mode climbs from 16.8 to 22.0 sections over the same span.

Step 0 is measured for both modes rather than left blank. Vocab id 7 is renamed
in place, so the base model handed the multi marker sees the identical integer;
pairing exp199's weights with the renamed tokenizer is the honest reading of
what that token already meant. The base emits 0.999 sets under it -- a single
document decoder under either marker. So the fine-tune moved exactly one token's
behaviour from 1 to 22, and left the other one alone.

So at a 50% plain-rehearsal mix the leak simply does not appear, and the extra
optimization bought stability rather than the fix. A future run could probably
spend far fewer than 1,989 steps.

Both curves have to be read together. Plain falling to 1.0 on its own would be
equally consistent with the model having LOST the format; multi staying high on
its own says nothing about the leak. Only the pair distinguishes a mode switch
from a mode collapse.

## The negative result

At matched sampling budget, the multi format is not an accuracy win.

Gate A's plain number votes 100 rollouts, so comparing it to a single multi
rollout is unfair in the multi format's favour on cost and against it on
accuracy. Re-running plain at 22 rollouts -- matching the ~22 sections a multi
rollout emits -- and applying the same aggregation rules gives the honest
comparison.

On the legacy 554: consensus across 22 independent plain rollouts reads 0.5896,
against 0.5673 for consensus across one multi rollout's 22 sections. Oracle-best
is 0.5680 against 0.5342. Independent sampling wins on every cut.

The reason is measurable directly. The union of 22 independent rollouts covers
1,065 distinct contact pairs; 22 sections of one rollout cover 658. Independent
sampling explores about 62% more of the space.

## Consensus beats the oracle

In BOTH regimes, voting across candidates beats picking the single best candidate
even with ground-truth selection: 0.5896 against 0.5680 for independent rollouts,
0.5673 against 0.5342 for multi sections.

That means the candidate sets carry complementary information rather than being
noisy copies of one guess. For the RL hand-off it says a reward that can exploit
combinations has more to work with than best-of-N over individual candidates.

A multi rollout's last section also beats a single plain rollout, 0.4566 against
0.4454, and beats its own second-to-last section, 0.4284. The ordering is
meaningful: the model treats its final section as a commitment, which is what the
training format taught even though drafts were shown in random order.

## H3's risk materialized

Mean pairwise Jaccard between a multi rollout's sections is 0.304, against 0.071
for #163's arm F -- 4.3x more similar, and just past exp200's 0.30
diversity-collapse criterion.

This was preregistered: exp199 is a stronger and therefore more consistent
sampler, and best-of-N is paid for by spread rather than by mean quality.

The spread that matters is still there and is not subtle. Best-section F1 exceeds
last-section F1 by 0.0845 plus or minus 0.0039, over twenty sigma. Lower
diversity than #163, still productive diversity -- but if an RL loop can afford
independent rollouts, the budget-matched result says it should prefer them.

## Published

The checkpoint is at
checkpoints/plm-exp230-cv1-multi-1_5b-lr1e-4-e1-cos-a100/hf/step-1988 on the
public bucket, 5.89 GB, tokenizer co-located and rope repaired for transformers
4.x readers.

The corpus is at data/document_structures/contacts_v1_multi_exp230, 6.26 GB:
519,998 documents, the tokenized form carrying the profile-F loss weights, and
the renamed tokenizer. Both verified anonymously readable.
