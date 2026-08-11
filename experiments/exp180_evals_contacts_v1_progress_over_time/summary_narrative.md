# Summary slides — exp: track contacts-v1 R-precision and validation loss over time

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide.
     Keep this current as the experiment progresses. -->

## What we're doing

How good was the best contacts-v1 model we had on any given date — on contact
R-precision and on held-out validation loss — and how tightly do those two
track each other?

Every number exists somewhere already (#67, #75, #82, #89, #108, #117, #120,
#146, #155, #160, #166, #169, #199/#204), but they are scattered across issue
comments, per-experiment CSVs and two W&B projects, with three different
inference recipes and two different validation-loss objectives mixed in. This
experiment collects them into one dataset and three standing figures, and keeps
them refreshable as new models land.

## Why

Not a hypothesis-testing experiment — it is a **standing tracker**. The two
things it is built to make visible, both of which prior issues assert
piecemeal:

1. Progress on contact accuracy has come almost entirely from the **base
   model**, not from inference or post-training.
2. Validation loss predicts contact accuracy **across training generations**
   and stops predicting it **within a generation** (#169's finding), so the
   loss frontier and the accuracy frontier are not the same curve.

## Results so far

The accuracy frontier moved in four jumps, all from the base model, none
from inference or post-training: ~0.03 to 0.425 when #75's E8 rung finished
(2026-06-21), 0.436 to 0.534 when #117's 16-epoch bs256 run finished
(2026-07-22), 0.534 to 0.562 when #166's amino-acid augmentation
continue-train of #117 finished (2026-07-31), and 0.562 to 0.587 when #199's
AFDB + ESM-Atlas sweep finished (2026-08-09/10).

The last two are both *data* changes rather than hyperparameter sweeps or
epoch counts, and they are the two largest gains since #117. Between the first
two jumps, five weeks of post-training and inference work moved the frontier
by +0.011 (#120's re-epoch), and #160's backtracking fine-tune moved it -0.020.

Structure predictors on the same 554 proteins: Protenix-v2 single-seq 0.603,
ESMFold 0.755, ESMFold2 0.786, Protenix-v2 + MSA 0.812.

## The single-sequence gap has effectively closed

MarinFold #199 CoreWeave p06-aug scores 0.587 against Protenix-v2
single-sequence's 0.603 — a paired difference of -0.016 with a 95% CI of
[-0.041, +0.009] over the 554 proteins.

Protenix still has the higher mean. This is not a claim to have passed it; it
is a claim that this benchmark can no longer separate them. Two months ago the
same comparison read 0.029 vs 0.603.

ESMFold2 (0.786) and MSA Protenix (0.812) are untouched by any of this, and
the MSA gap has barely moved. That is where the remaining distance is.

## #199 in context

#199 trains on AFDB plus 71.4B tokens of ESM-Atlas, and its best checkpoint is
the first frontier point trained from scratch on CoreWeave H100s rather than
continued from a #117 TPU checkpoint. It is +0.025 over #166 and +0.054 over
the #117 control re-scored in the same batch.

Two things it does not show. The CoreWeave/TRC gap (0.587 vs 0.524 at the same
hyperparameter point) confounds initialization, schedule and step count, so it
is not a platform result. And #199 has never been run against #155, the other
ESM-Atlas result (0.554).

#204 also re-evaluated the unchanged #117 checkpoint four times: they span
0.0023. That is this tracker's first noise estimate, and it makes #199's
p03-base vs p03-aug difference (0.0036) a tie rather than a result.

## Loss tracks accuracy across generations, and less each time

The exchange rate is collapsing. #75 E8 to #117 E16 bought 2.06 R-precision
per nat; #117 to #166 bought 0.71; #166 to #199 CW bought about 0.33.

Inside one run it buys nothing at all: the 0.008-nat gap between #117's
early-stop and final checkpoints is worth zero, and #146's 3B is 0.0012 better
on loss and 0.023 worse on R-precision.

#204's sigmoid fit says the same thing from the other side — an upper asymptote
of 0.5955, which #199 CW is already at 98.6% of. Either the relationship
saturates near there, or the fit is extrapolating past its support.

So the loss frontier is useful for deciding which checkpoints are worth
scoring, and useless for picking between two checkpoints of one run.

## Two traps in these numbers

**Inference recipe.** The same weights score ~0.086 higher under exp82's
rollout recipe than under exp89's original pairwise scorer (#61/#75 E8: 0.339
to 0.425; #120: 0.350 to 0.436). The figures keep them visually separate.
Eval-checkpoint skill output is pairwise before 2026-08-11 and rollout from
that date on; anything from exp82's rollout workers or the exp169 dispatcher
is rollout. Never infer the recipe from the magnitude. A third recipe, oracle best-of-100, is a diagnostic upper
bound and never enters the frontier line.

**Loss scale.** marin #7209 changed the packed-LM objective to mask padding
targets, which raises the reported loss by ~0.38 nats. Everything through #166
is on the old scale; #199 is the first sweep on the new one. The figures plot
the historical axis and convert #199 onto it, in green with a tilde. The
conversion is good to about 0.025 nats — fine at the 0.1-nat scale, unreadable
at the 0.01-nat scale. R-precision is unaffected.

## Per-protein comparison with Protenix-v2

Over the 554 proteins, MarinFold #199 CW scores 0.587. Protenix-v2
single-sequence scores 0.603 (paired difference -0.016, MarinFold higher on
36%). Protenix-v2 with MSAs scores 0.812 (paired difference -0.224, MarinFold
higher on 9%).

Re-pointed #117 to #166 to #199 CW, the shape did not change, only the gap:
every length bin improved at each step and the single-sequence deficit went
-0.069 to -0.041 to -0.016.

## The two baselines trend opposite ways with length

Against single-sequence Protenix the gap narrows with length and now changes
sign in two bins, not one: -0.079 below 100 residues, +0.013 in the 200-400
bin (171 proteins), +0.183 above 400.

Against MSA Protenix it widens: -0.168 below 100 residues, -0.409 above 400.
MarinFold does not win a single protein above 400 residues in that comparison.

The reason is in the marginals. MarinFold declines with length (0.58 to 0.45)
and single-sequence Protenix declines much faster (0.66 to 0.27), while MSA
Protenix improves with length (0.75 to 0.86).

So "MarinFold holds up better on long proteins" is about the single-sequence
baseline only, it is a shallower decline rather than absolute strength, and
the > 400 bin holds 17 proteins either way.

## The unexplored lever

A diagnostic on #155's final checkpoint: scoring each of its 100 sampled
rollouts per protein on its own best contacts, instead of voting them
together, reads 0.595 — +0.041 of headroom that isn't reachable without ground
truth, but points at reranking/selection as an unexplored lever.

That +0.041 is now larger than the 0.016 still separating the best model from
single-sequence Protenix. The comparison crosses two checkpoints, so treat it
as a rough ordering rather than arithmetic — but the ordering is the point.

## Open gap

#108's 3B on CoreWeave held the loss frontier from 2026-07-11 to 07-16 at
2.7418 and was never contact-scored. Given #146 — a 3B at matched loss
under-performing the 1.5B — it is probably not a missed frontier point, but
that is an inference, not a measurement.

## Keeping this current

Two commands: build_dataset.py re-pulls W&B, plot_progress.py redraws (plus
plot_vs_protenix.py for the head-to-head pair). New benchmark scores are
hand-added to RPRECISION_ROWS with a source citation, an explicit inference
recipe and an explicit loss scale; a new W&B sweep tag goes in WANDB_SOURCES
with its scale — read that off the run's own requirements.txt, not its date.
When a new model takes the frontier, re-point MARINFOLD_MODEL and ROWS_CSV in
plot_vs_protenix.py — it needs published per-protein rows, not a summary. Full
procedure in the README.
