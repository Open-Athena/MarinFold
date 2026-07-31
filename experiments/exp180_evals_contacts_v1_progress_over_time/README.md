---
marinfold_experiment:
  issue: 180
  title: 'exp: track contacts-v1 R-precision and validation loss over time'
  kind: evals
  branch: claude/rprecision-validation-loss-plots-750306
---

# exp: track contacts-v1 R-precision and validation loss over time

**Issue:** [#180](https://github.com/Open-Athena/MarinFold/issues/180) · **Kind:** `evals` · **Branch:** `claude/rprecision-validation-loss-plots-750306`

## Question

How good was the best contacts-v1 model we had on any given date — on contact
R-precision and on held-out validation loss — and how tightly do those two
track each other?

Every number exists somewhere already (#67, #75, #82, #89, #108, #117, #120,
#146, #155, #160, #169), but they are scattered across issue comments, per-experiment
CSVs and two W&B projects, with two different inference recipes mixed in. This
experiment collects them into one dataset and three standing figures, and keeps
them refreshable as new models land.

## Hypothesis

Not a hypothesis-testing experiment — it is a **standing tracker**. The two
things it is built to make visible, both of which prior issues assert
piecemeal:

1. Progress on contact accuracy has come almost entirely from the **base
   model**, not from inference or post-training.
2. Validation loss predicts contact accuracy **across training generations**
   and stops predicting it **within a generation** (#169's finding), so the
   loss frontier and the accuracy frontier are not the same curve.

## Background

- **#89** — the fixed 554-protein contact benchmark: ground-truth universe,
  candidate pairs, and `compute_metrics.py`. Every R-precision number here is
  computed by that implementation.
- **#82** — settled the best LM-only inference: n=100 rollouts + per-rollout
  document resampling + pairwise tie-break. **#142** then removed the
  `top_k=50` that was truncating rollouts.
- **#75 / #117 / #146** — Eric's LR/WD/epoch sweeps (`eric-czech/marin`, tags
  `exp75`, `exp117`, `exp146`, `exp153`); **#67 / #85 / #108 / #120 / #137 /
  #150 / #155** — MarinFold-side runs (`open-athena/MarinFold`).
- **#169** — showed val loss is not a usable checkpoint selector at the
  0.01-nat scale, and that matched loss does not mean matched accuracy across
  model sizes. This experiment is the longitudinal version of its panel A.
- **#74 / #78** — the structure-predictor baselines drawn as reference lines.

## Approach

Two tables and three figures, all code in the experiment dir.

- `build_dataset.py` — pulls every finished contacts-v1 training run reporting
  `eval/contacts-v1-val/loss` from both W&B projects, and carries the
  hand-curated benchmark scores (one source citation per row, since these live
  in issue comments and per-experiment CSVs rather than in W&B). Recomputes the
  structure-predictor lines from exp89's per-protein table.
- `plot_progress.py` — the three figures: R-precision vs date, val loss vs
  date, R-precision vs val loss. Both date figures draw a running-best
  staircase; both R-precision figures carry the structure predictors as dotted
  reference lines.

**The one methodological trap, handled explicitly:** the same weights score
**~0.086 higher** under exp82's rollout recipe than under exp89's original
pairwise scorer (#61/#75 E8: 0.339 → 0.425; #120: 0.350 → 0.436). The figures
keep the two recipes visually separate rather than pooling them.

Metric throughout: **R-precision, all ranges (seq-sep ≥ 6), mean over the 554
proteins**. Date = when the training run finished, not when it was evaluated.

## Success criteria

- One dataset with a traceable source for every plotted number, and three
  figures that a future reader can regenerate with two commands.
- A documented refresh procedure so the tracker stays current as #146, #150,
  #153, #155 and the post-training line produce new checkpoints.


## How to update this experiment

This is a **standing tracker**. Refreshing it after new models land is two
commands plus, usually, one hand-edit. Do it whenever a training sweep finishes
or a checkpoint gets scored on the #89 benchmark.

### 1. New training runs only (no new contact scores)

The validation-loss figure reads W&B directly, so it needs nothing but a
refresh:

```bash
cd experiments/exp180_evals_contacts_v1_progress_over_time
uv run python build_dataset.py     # re-pull W&B, rebuild the tables
uv run python plot_progress.py     # redraw the three figures
uv run python build_summary.py     # optional: refresh plots/summary.pdf
```

`build_dataset.py` re-pulls every run from `open-athena/MarinFold` and from
`eric-czech/marin` tags `exp75` / `exp117` / `exp146` / `exp153`. **If a new
sweep lands under a new tag, add it to `WANDB_SOURCES` first** — otherwise its
runs are silently absent and the loss frontier will look flat when it isn't.

Two other things to check when adding a source:

- **The loss key.** `LOSS_KEYS` is tried in order. A run that logs the
  contacts-v1 val loss under some new component name will be dropped, not
  crash. Grep the new runs' summary keys before trusting the output.
- **`EXCLUDE_SUBSTRINGS`.** Smoke tests, batch-calibration probes and profiling
  runs are excluded by name. A new naming convention may need a new substring —
  or, worse, may accidentally match an existing one and drop a real run. The
  printed run count (`wrote data/val_loss_runs.csv (N runs)`) is the cheap
  sanity check.

### 2. A checkpoint gets scored on the 554-protein benchmark

Add a row to `RPRECISION_ROWS` in `build_dataset.py` and re-run both commands.
Each row needs:

| field | what |
|---|---|
| `label` | short identifier used on the figures, e.g. `#146 3B E8` — keep it under ~20 chars |
| `model` | the W&B run name + step, so the row is traceable |
| `date` | when the **training run finished** (W&B `heartbeatAt`), not when it was scored |
| `params`, `issue` | 1.5B / 3B / …; the issue number the checkpoint came from |
| `val_loss`, `val_loss_key` | `None` / `""` if the model's vocab makes its loss non-comparable (see #160) |
| `r_precision` | **R-precision, all ranges**, mean over 554 — the `range=all, cut=R` cell |
| `inference` | `ROLLOUT` or `PAIRWISE` — **never guess this**, see below |
| `source` | the CSV or gist the number came from, precisely enough to re-find |

Then add a label offset for the new point in `RP_LABEL_OFFSET` (and
`S3_OFFSET`) in `plot_progress.py`. Offsets are hand-placed in points; the
figures are dense enough that a new point will collide with something. **Render
and look at the PNGs** — the collision check is visual, there is no test for it.

### 3. Getting the inference recipe right

This is the one thing that will quietly corrupt the figures. The same weights
score **~0.086 higher** under rollout than under pairwise, which is comparable
to two generations of model progress. If a new score is filed under the wrong
recipe it will look like a jump that never happened.

- The **`eval-checkpoint` skill** runs exp89's **pairwise** scorer. Anything
  produced by it is `PAIRWISE`.
- exp82's `score_rollout_*.py` workers and the exp169 dispatcher produce
  `ROLLOUT` (n=100, resampled, top-k off).
- If a score came with `top_k=50` (anything predating #142), it is a **third**
  realisation — record it in `FOOTNOTE_ROWS`, not as a checkpoint.

When in doubt, ask where the score came from rather than inferring it from the
magnitude.

### 4. Keep the caveats honest

`Caveats worth carrying` below is part of the deliverable, not decoration. When
one is resolved — e.g. someone proves `contacts-v1-val-orig` and
`contacts-v1-val` are the same split — delete it and the corresponding figure
footnote. When a new one appears (a new tokenizer, a changed eval set, a
different val split), add it before the figures get quoted elsewhere.

If the **554-protein eval set itself** ever changes, this experiment's whole
y-axis changes with it. That is a rebuild, not a refresh: every historical
number would have to be re-scored, and the honest interim move is to freeze
these figures and start a second set.

### 5. Cross-check

`data/rprecision_footnotes.csv` holds measurements that are alternate
realisations of a checkpoint rather than new checkpoints. Nothing in it should
ever appear on a figure; it exists so a number found in an old issue comment
can be identified as "already known, different recipe" instead of being added
as a new point.

## Results

Three figures. All R-precision values are **all ranges (seq-sep ≥ 6), mean over
the 554-protein eval set**, computed by exp89's `compute_metrics.py`.

![R-precision over time](plots/rprecision_frontier.png)

![Validation loss over time](plots/val_loss_frontier.png)

![R-precision vs validation loss](plots/rprecision_vs_val_loss.png)

### The accuracy frontier

| date | model | val loss | R-precision (all) | recipe |
|---|---|---:|---:|---|
| 2026-06-14 | #67 quick 1.5B | 2.9800 | 0.029 | pairwise |
| 2026-06-14 | #75 E1 | 3.0458 | 0.028 | pairwise |
| 2026-06-20 | #75 E2 | 2.9421 | 0.029 | pairwise |
| 2026-06-21 | #75 E4 | 2.9238 | 0.031 | pairwise |
| **2026-06-21** | **#61/#75 E8** | **2.7566** | 0.339 / **0.425** | pairwise / rollout |
| **2026-07-16** | **#120 re-epoch** | **2.7213** | 0.350 / **0.436** | pairwise / rollout |
| 2026-07-19 | #117 E8 bs64 | 2.7131 | 0.419 | pairwise |
| 2026-07-22 | #117 E16 early stop | 2.6961 | 0.532 | rollout |
| **2026-07-22** | **#117 E16 final** | **2.7037** | **0.534** | rollout |
| 2026-07-27 | #146 3B E8 | 2.7025 | 0.512 | rollout |
| 2026-07-28 | #160 backtracking | — | 0.416 | rollout |
| **2026-07-31** | **#155 3-way restart** (step 60000, in flight) | — | **0.553** | rollout |

Structure predictors on the same 554 proteins and the same metric:
Protenix-v2 single-seq **0.603**, ESMFold **0.755**, ESMFold2 **0.786**,
Protenix-v2 + MSA **0.812**.

### The loss frontier

3.15 → 2.98 → **2.7566** (#61/#75 E8, 06-21) → **2.7418** (#108's 3B on
CoreWeave H100s, 07-11) → **2.7213** (#120, 07-16) → 2.7131 → 2.7112 →
**2.7037** (#117 E16 final, 07-22) → **2.7025** (#146 3B, 07-27), over 155
finished runs.

### What the figures show

- **The accuracy frontier moved in three jumps**, all from the base model,
  none from inference or post-training: ~0.03 → **0.425** when #75's
  E8 rung finished (2026-06-21), 0.436 → **0.534** when #117's 16-epoch
  bs256 run finished (2026-07-22), and 0.534 → **0.553** at #155's 3-way
  mixture restart's step 60000 (2026-07-31). Between the first two, five weeks
  of post-training and inference work moved it by +0.011 (#120's re-epoch).
  (#75's E4 winner, 0.031, landed the same day as E8, so the pre-jump frontier
  reads 0.029 — #67's.) The third jump is the odd one out: unlike every other
  row, that run has not finished training (target step 74800) and its
  checkpoint's vocab is a superset, so it has no comparable val loss — see the
  caveat below.
- **Loss and accuracy agree across generations and stop agreeing inside one.**
  The 0.053-nat #75→#117 gap buys +0.109 R-precision (~2 R-precision per nat).
  The 0.008-nat gap between #117's early-stop and final checkpoints buys
  nothing (paired Δ +0.0026 in the *final*'s favour, CI crosses zero), and
  #146's 3B is 0.0012 *better* on loss and 0.023 *worse* on R-precision.
- **#108's 3B on CoreWeave held the loss frontier from 2026-07-11 to 07-16 at
  2.7418** and was never contact-scored. Given #146's result — a 3B at matched
  loss under-performing the 1.5B — it is probably not a missed frontier point
  on accuracy, but that is an inference, not a measurement. It is the one real
  gap in the accuracy figure.
- **#85's LR re-heat did not lower loss.** It finished at 2.9801 against #67's
  2.9800. The Week-of-June-22 `UPDATES.md` entry says it "improved eval loss
  somewhat"; the W&B history does not support that (the run's five eval points
  run 2.9825 / 2.9828 / 2.9843 / 2.9820 / 2.9801).
- **#67 never held the loss frontier.** It finished 2026-06-14 15:36 at 2.9800,
  about two hours after `prot-exp75-cv1-1_5b-e2-lr7e-4-wd0p05-v1` reached
  2.9787.
- **All of this is still below single-sequence Protenix-v2** (0.553 vs 0.603),
  and well below ESMFold2 (0.786).

## Conclusion

Both success criteria are met: `data/rprecision_checkpoints.csv` carries a
source citation per number, and the figures regenerate from two commands.

The substantive read is that **contact accuracy has come from the base model
and essentially nowhere else**. Three training results account for the entire
frontier; the settled inference recipe is worth a large constant (+0.086) but
was banked once in June and has not moved since; and the post-training line
(#120, #160) has produced +0.011 and −0.020 respectively. The third jump —
#155's crops+contacts-v1+ESM-Atlas 3-way mixture, still mid-training — is the
first frontier point to come from a *data* change rather than a hyperparameter
sweep or an epoch count.

Validation loss remains a good *cross-generation* proxy — ~2 R-precision per
nat over the measured range — and a useless *within-generation* one. Since this
experiment costs nothing to keep current and a benchmark run costs ~10 min on 4
TPU slices, the practical recommendation from #169 stands: select checkpoints
on the contact metric, and use the loss frontier only to decide which
checkpoints are worth scoring.

## Method and provenance

### What is being measured

**R-precision, all ranges** — precision at the top *R* ranked residue pairs,
where *R* is that protein's true-contact count (seq-sep ≥ 6), meaned over the
**554-protein eval set** (#74/#78 + #41/#65 curation), scored by exp89's
`compute_metrics.py` against the published exp89 ground-truth universe. This is
the only accuracy number on these plots; long/medium/short-range cuts, AUC and
contacts@L exist in the source CSVs but are not plotted.

**Validation loss** — `eval/contacts-v1-val/loss` (Eric's runs log the same
quantity as `eval/tokenized/contacts-v1-val/loss`), the full held-out
contacts-v1 split: 41,954 documents / 47,821,958 tokens.

**Date** — when the training run *finished* (W&B `heartbeatAt`), i.e. when the
checkpoint came into existence. Not when it was evaluated; several checkpoints
were scored weeks later.

### The two inference recipes

The same weights score **~0.086 higher** under the settled rollout recipe than
under exp89's original pairwise scorer, on both checkpoints measured under both:

| checkpoint | pairwise | rollout | Δ |
|---|---:|---:|---:|
| #61/#75 E8 | 0.3389 | 0.4245 | +0.0856 |
| #120 re-epoch | 0.3495 | 0.4357 | +0.0862 |

So the figures never merge them — marker shape and colour carry the recipe, and
a checkpoint measured both ways shows both points. Recipes:

- **pairwise** — autoregressive `P(<contact> <pi> <pj>)`, symmetrised. exp89's
  original scorer; still what the `eval-checkpoint` skill runs.
- **rollout** — n=100 sampled rollouts + per-rollout document resampling +
  pairwise tie-break, **top-k off**. Settled in exp82; `top_k=50` was removed
  in #142 (it cost ~0.007–0.012 R-precision by truncating long rollouts).

Consequence for the accuracy frontier: it is the running max over each
checkpoint's *best available* measurement. Every step of it happens to be a
rollout number, and the pairwise-only points (#67, #75 E1/E2/E4, #117 E8 bs64)
never touch it, so the mixture does not change the staircase. If a rollout
number is ever produced for #117 E8 bs64 it would land near 0.50 and still sit
under the #117 E16 step.

### Structure-predictor reference lines

Dotted lines on both R-precision figures, same 554 proteins and same metric,
recomputed from `../exp89_evals_contacts_v1_model_on_eval_set/data/contact_precision_all.csv`.
These are the `predictor=structure` rows. Protenix's *distogram* readouts
(0.380 single-seq, 0.465 MSA) are in the same CSV but are not drawn — the
structure readout is the one the project quotes.

### What is in, what is out

**In the accuracy figure** — the 12 checkpoints with a benchmark score
(`data/rprecision_checkpoints.csv`, one citation per row).

**In the loss figure** — every *finished* W&B run reporting a contacts-v1 val
loss across `open-athena/MarinFold` and `eric-czech/marin` tags exp75 / exp117 /
exp146 / exp153 (n=155 after exclusions).

**Excluded, and why:**

- **Crashed / preempted runs** (94 of them). Their last logged loss is
  mid-training, not a trained model. Several would otherwise punch spurious
  holes in the frontier — e.g. a preempted exp117 run sitting at 2.7309.
- **Smoke tests, batch-calibration probes, profiling runs, the NeMo #112
  throughput runs, and the bio2token vetting run** — matched on name
  (`smoke`, `probe`, `profile`, `-prof`, `vet-`, `nemo`).
- **#160 backtracking** appears in the accuracy figure (R 0.4158) but not the
  loss one: it has a 3849-token superset vocabulary, so its val loss is not
  comparable to the 2845-vocab runs.
- **In-flight runs** are drawn as hollow diamonds on the loss figure and kept
  off *that* frontier — they have not finished training. Four such runs are
  live as of 2026-07-31 (the original #155 3-way mix, its no-crops ablation,
  its restart, and an unrelated exp124 run); the best of them is #155's
  restart at `contacts-v1-val` 2.6843. **One exception:** #155's 3-way restart *does* appear on the
  **accuracy** frontier at step 60000 (R 0.553) — the #89 benchmark scores a
  specific checkpoint, not a finished run, so an in-flight run's intermediate
  checkpoint can still be scored and plotted like any other. It stays off the
  loss figure's frontier because that figure is specifically about *finished*
  runs.

### Caveats worth carrying

1. **#120's loss is `eval/contacts-v1-val-orig/loss`** (2.7213), everything
   else is `contacts-v1-val`. Almost certainly the same held-out split under a
   different cache name, but that has never been proven — #160 flagged the same
   thing. Treat #120's position on the loss axis as approximate.
2. **#146's 3B is confounded.** It differs from the #117 1.5B in epochs (8 vs
   16) and weight decay (0.4 vs 0.2) as well as parameter count, so "3B at
   equal loss is worse" indicts *this checkpoint*, not scale.
3. **The #117 early-stop checkpoint is step 33450 of the run that ends at
   35679.** At day resolution it plots on top of the final; both markers are
   drawn, one label carries both numbers.
4. **exp89's harness vs exp82's**: the #117 final reads 0.5344 through exp169's
   TPU worker and 0.5350 through exp82's — a 0.0006 backend difference, inside
   the ≤0.006 TPU-vs-CUDA agreement #89 established. Either is fine; the plots
   use exp169's for that checkpoint.
5. **#155's 3-way restart is an in-flight checkpoint, not a finished run** —
   step 60000 of a run targeting step 74800. Its position on the accuracy
   frontier (R 0.553) could still move, up or down, once the run finishes;
   unlike every other frontier point, it is not yet a settled result. It also
   has no val loss on these figures (3848-token superset crops tokenizer,
   same reason as #160).

## Files

| path | what |
|---|---|
| `build_dataset.py` | assembles both tables; W&B pull + the hand-curated benchmark rows |
| `plot_progress.py` | the three figures |
| `data/rprecision_checkpoints.csv` | 14 rows (12 checkpoints; 2 measured under both recipes), one source citation each |
| `data/structure_baselines.csv` | the four dotted lines, recomputed from exp89 |
| `data/val_loss_runs.csv` | 320 W&B runs with a contacts-v1 val loss, with state + exclusion flag |
| `data/rprecision_footnotes.csv` | measurements that are alternate realisations of a checkpoint, not new checkpoints |
| `data/exp155_3way_restart_step60000_rollout_summary.csv` | aggregate R-precision/AUC for #155's step-60000 checkpoint, this session's rollout eval |
| `data/exp155_3way_restart_step60000_rollout_rows.csv.gz` | per-protein rows behind that summary (554 × 20) |
| `plots/*.png.meta.json` | the numbers behind each figure |

Sources for every R-precision value are in the `source` column of
`data/rprecision_checkpoints.csv` — exp82 / exp89 / exp120 / exp155 / exp160 / exp169
data CSVs, plus [eric-czech's checkpoint gist](https://gist.github.com/eric-czech/bfa78571dcb8f673884bf70e6cc68e14)
for #117 E8 bs64.
