# Summary slides — exp: track contacts-v1 R-precision and validation loss over time

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide.
     Keep this current as the experiment progresses. -->

## What we're doing

How good was the best contacts-v1 model we had on any given date — on contact
R-precision and on held-out validation loss — and how tightly do those two
track each other?

Every number exists somewhere already (#67, #75, #82, #89, #108, #117, #120,
#146, #155, #160, #166, #169), but they are scattered across issue comments,
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
(2026-07-22), and 0.534 to 0.562 when #166's amino-acid augmentation
continue-train of #117 finished, 2026-07-31. #166 is the first jump from a
*data* change rather than a hyperparameter sweep or epoch count.

Between the first two, five weeks of post-training and inference work moved
it by +0.011 (#120's re-epoch), and #160's backtracking fine-tune moved it by
-0.020.

Two data-side results landed a day apart. #155's crops+contacts-v1+ESM-Atlas
3-way mixture restart finished 08-01 at 0.554 — 0.008 below #166 and a day
later, so it never appears as a step. They were not run against each other.

#166 is also the only frontier point with a within-run control: #190 re-scored
its own #117 initialization alongside it (0.5336 vs #169's 0.5344), making the
+0.0282 a paired result rather than a cross-harness subtraction.

A diagnostic on #155's final checkpoint: scoring each of its 100 sampled
rollouts per protein on its own best contacts, instead of voting them
together, reads 0.595 — +0.041 of headroom that isn't reachable without
ground truth, but points at reranking/selection as an unexplored lever.

Structure predictors on the same 554 proteins: Protenix-v2 single-seq 0.603,
ESMFold 0.755, ESMFold2 0.786, Protenix-v2 + MSA 0.812. We are still below
all of them, though the single-sequence gap is now 0.041.

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

## Per-protein comparison with Protenix-v2

Over the 554 proteins, MarinFold #166 scores 0.562. Protenix-v2 single-sequence
scores 0.603 (paired difference -0.041, MarinFold higher on 34%). Protenix-v2
with MSAs scores 0.812 (paired difference -0.250, MarinFold higher on 8%).

Re-pointed from #117 to #166 the shape did not change, only the gap: every
length bin improved and the single-sequence deficit halved.

## The two baselines trend opposite ways with length

Against single-sequence Protenix the gap narrows with length and changes sign
in the > 400 bin: -0.107 below 100 residues, +0.122 above 400. The 200-400 bin
is now -0.011, which on 171 proteins is a tie.

Against MSA Protenix it widens: -0.195 below 100 residues, -0.471 above 400.
MarinFold does not win a single protein above 400 residues in that comparison.

The reason is in the marginals. MarinFold declines with length (0.56 to 0.39)
and single-sequence Protenix declines faster (0.66 to 0.27), while MSA Protenix
improves with length (0.75 to 0.86).

So "MarinFold holds up better on long proteins" is about the single-sequence
baseline only, it is a shallower decline rather than absolute strength, and
that bin holds 17 proteins either way.

## Open gap

#108's 3B on CoreWeave held the loss frontier from 2026-07-11 to 07-16 at
2.7418 and was never contact-scored. Given #146 — a 3B at matched loss
under-performing the 1.5B — it is probably not a missed frontier point, but
that is an inference, not a measurement.

## Keeping this current

Two commands: build_dataset.py re-pulls W&B, plot_progress.py redraws (plus
plot_vs_protenix.py for the head-to-head pair). New benchmark scores are
hand-added to RPRECISION_ROWS with a source citation and an explicit inference
recipe; a new W&B sweep tag goes in WANDB_SOURCES. When a new model takes the
frontier, re-point MARINFOLD_MODEL and ROWS_CSV in plot_vs_protenix.py — it
needs published per-protein rows, not a summary. Full procedure in the README.
