# Summary slides — exp: track contacts-v1 R-precision and validation loss over time

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide.
     Keep this current as the experiment progresses. -->

## What we're doing

How good was the best contacts-v1 model we had on any given date — on contact
R-precision and on held-out validation loss — and how tightly do those two
track each other?

Every number exists somewhere already (#67, #75, #82, #89, #108, #117, #120,
#146, #155, #160, #169), but they are scattered across issue comments,
per-experiment CSVs and two W&B projects, with three different inference
recipes mixed in. This experiment collects them into one dataset and three
standing figures, and keeps them refreshable as new models land.

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

The accuracy frontier moved in three jumps, all from the base model, none
from inference or post-training: ~0.03 to 0.425 when #75's E8 rung finished
(2026-06-21), 0.436 to 0.534 when #117's 16-epoch bs256 run finished
(2026-07-22), and 0.534 to 0.554 when #155's crops+contacts-v1+ESM-Atlas
3-way mixture restart finished training, 2026-08-01 (step 74793). #155 is
the first jump from a *data* change rather than a hyperparameter sweep or
epoch count.

Between the first two, five weeks of post-training and inference work moved
it by +0.011 (#120's re-epoch), and #160's backtracking fine-tune moved it by
-0.020.

A new diagnostic on #155's final checkpoint: scoring each of its 100 sampled
rollouts per protein on its own best contacts, instead of voting them
together, reads 0.595 — +0.041 of headroom that isn't reachable without
ground truth, but points at reranking/selection as an unexplored lever.

Structure predictors on the same 554 proteins: Protenix-v2 single-seq 0.603,
ESMFold 0.755, ESMFold2 0.786, Protenix-v2 + MSA 0.812. We are still below
all of them (the oracle ceiling alone would essentially close the gap to
Protenix-v2 single-seq, but it isn't a deliverable number).

## Loss tracks accuracy across generations, not inside one

The 0.053-nat #75 to #117 gap buys +0.109 R-precision — about 2 R-precision
per nat. The 0.008-nat gap between #117's early-stop and final checkpoints
buys nothing, and #146's 3B is 0.0012 better on loss and 0.023 worse on
R-precision.

So the loss frontier is useful for deciding which checkpoints are worth
scoring, and useless for picking between two checkpoints of one run.

## The trap in these numbers

The same weights score ~0.086 higher under exp82's rollout recipe than under
exp89's original pairwise scorer (#61/#75 E8: 0.339 to 0.425; #120: 0.350 to
0.436). That is comparable to two generations of model progress.

The figures keep pairwise and rollout visually separate. Anything from the
eval-checkpoint skill is pairwise; anything from exp82's rollout workers or
the exp169 dispatcher is rollout. Never infer the recipe from the magnitude.

A third recipe, oracle best-of-100, is a diagnostic upper bound, not a
deployable score (see Results so far) — it never enters the "best model
trained to date" frontier line or headline labelling, only its own marker.

## Open gap

#108's 3B on CoreWeave held the loss frontier from 2026-07-11 to 07-16 at
2.7418 and was never contact-scored. Given #146 — a 3B at matched loss
under-performing the 1.5B — it is probably not a missed frontier point, but
that is an inference, not a measurement.

## Keeping this current

Two commands: build_dataset.py re-pulls W&B, plot_progress.py redraws. New
benchmark scores are hand-added to RPRECISION_ROWS with a source citation and
an explicit inference recipe. Full procedure in the README.
