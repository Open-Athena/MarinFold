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
#146, #155, #160, #166, #169), but they are scattered across issue comments, per-experiment
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
- **#75 / #117 / #146 / #166** — Eric's LR/WD/epoch sweeps and the AA-augmentation
  run (`eric-czech/marin`, tags `exp75`, `exp117`, `exp146`, `exp153`, `exp166`); **#67 / #85 / #108 / #120 / #137 /
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
- `plot_progress.py` — three figures: R-precision vs date, val loss vs date,
  R-precision vs val loss. Both date figures draw a running-best staircase;
  both R-precision figures carry the structure predictors as dotted reference
  lines.
- `plot_vs_protenix.py` — two more figures: one pinned contacts-v1 checkpoint
  against Protenix-v2 single-sequence and against Protenix-v2 with MSAs, one
  point per eval protein. The frontier figures say how far along we are; these
  say *which proteins* the number is made of.

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
`eric-czech/marin` tags `exp75` / `exp117` / `exp146` / `exp153` / `exp166`.
The loss figure's source footnote is generated from those columns, so it can
never disagree with what was actually pulled. **If a new
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

### 5. The head-to-head figure pins a specific model

`plot_vs_protenix.py` hard-codes `MARINFOLD_MODEL = "exp166_aaaug_step35679"`
and reads its per-protein scores from `ROWS_CSV`, currently
`../exp166_models_contacts_v1_aa_augmentation/data/exp166_rows.csv.gz`. It emits
one figure per entry in `BASELINES` (currently Protenix-v2 single-seq and MSA);
`--baseline` restricts it to one.

**When a new model takes the accuracy frontier, re-point it**: set
`MARINFOLD_MODEL` / `MARINFOLD_LABEL` / `MARINFOLD_SHORT`, and point `ROWS_CSV`
at whichever experiment holds the new per-protein rows (the two move together —
the model string is a value *inside* that file). It needs *per-protein*
precision, not a summary — a mean is not enough to draw the scatter, so the
scoring run has to have published its rows CSV.

Adding another baseline is one entry in `BASELINES`. The rows available in
exp89's CSV are `(protenix-v2, single_seq|msa, structure|distogram)`,
`(esmfold, single_seq, structure)` and `(esmfold2, single_seq, structure)`.

**Do not write the commentary block by hand.** `describe()` generates it from
each figure's own numbers precisely because the two current baselines trend in
opposite directions with length — a written-once summary was wrong on one of
them, and would be again for any new baseline.

### 6. Cross-check

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

### Per-protein comparison with Protenix-v2

The frontier figures compress each model to one number. These unroll the same
comparison over the 554 proteins, for the current best model — **#166 AA aug**
(0.562) — against both Protenix-v2 configurations.

**Single-sequence** — the like-for-like baseline, since MarinFold also reads
sequence alone:

![MarinFold vs Protenix-v2 single-sequence](plots/marinfold_vs_protenix_ss.png)

**With MSAs** — not like-for-like, and much the stronger baseline:

![MarinFold vs Protenix-v2 with MSAs](plots/marinfold_vs_protenix_msa.png)

| | Protenix-v2 SS | Protenix-v2 + MSA |
|---|---:|---:|
| baseline mean | 0.603 | 0.812 |
| paired difference | −0.041 | −0.250 |
| 95% CI | [−0.065, −0.018] | [−0.270, −0.230] |
| MarinFold higher on | 34% | 8.3% |
| Spearman | 0.61 | 0.49 |

By length (MarinFold is 0.555 / 0.588 / 0.539 / 0.388 across the four bins):

| length | n | Δ vs SS | MF higher | Δ vs MSA | MF higher |
|---|---:|---:|---:|---:|---:|
| < 100 | 81 | −0.107 | 30% | −0.195 | 19% |
| 100–200 | 285 | −0.051 | 29% | −0.217 | 10% |
| 200–400 | 171 | −0.011 | 42% | −0.308 | 1% |
| > 400 | 17 | **+0.122** | 71% | **−0.471** | 0% |

**The two baselines trend in opposite directions with length, and that is the
main thing to take from this pair.** Against single-sequence Protenix the gap
narrows monotonically and changes sign in the longest bin. Against MSA Protenix
it widens monotonically, and MarinFold does not win a single protein above 400
residues.

The reason is visible in the marginals: MarinFold declines with length
(0.56 → 0.39) and so does single-sequence Protenix, only faster (0.66 → 0.27) —
whereas MSA Protenix *improves* with length (0.75 → 0.86), presumably because
longer chains have deeper, more informative alignments.

So "MarinFold holds up better on long proteins" is a statement about the
single-sequence baseline only. It is a shallower decline, not an absolute
strength: the > 400 bin is where MarinFold is weakest in absolute terms
(0.388), and it is also where the MSA gap is widest. That bin holds **17
proteins** either way, so both readings of it are weak estimates.

The Spearman values are worth noting too — 0.61 against single-sequence, 0.49
against MSA. MarinFold's notion of which proteins are hard tracks the
single-sequence predictor considerably better than the MSA-informed one.

**Both figures moved with the model, and the shape did not.** Re-pointed from
#117 (0.534) to #166 (0.562), every bin improved and the single-sequence gap
halved (−0.069 → −0.041), but the two baselines still trend in opposite
directions and the sign change is still confined to the same 17-protein bin.
The one qualitative change is the 200–400 bin, now −0.011 against
single-sequence: on 171 proteins that is a tie, not a deficit.

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
| **2026-07-31** | **#166 AA aug** (from #117 init) | **2.6642** | **0.562** | rollout |
| 2026-08-01 | #155 3-way restart final (step 74793, run complete) | — | 0.554 | rollout |
| 2026-08-01 | #155 3-way restart final, oracle best-of-100 | — | 0.595 | oracle (not deployable) |

Bold = took the accuracy frontier on its date. **#155's 3-way restart no longer
does**: it finished 08-01 at 0.554, one day after #166 reached 0.562, so it
lands 0.008 below a frontier that had already moved. It held the frontier for
exactly as long as it took to notice #166 had been scored.

Structure predictors on the same 554 proteins and the same metric:
Protenix-v2 single-seq **0.603**, ESMFold **0.755**, ESMFold2 **0.786**,
Protenix-v2 + MSA **0.812**.

### The loss frontier

3.15 → 2.98 → **2.7566** (#61/#75 E8, 06-21) → **2.7418** (#108's 3B on
CoreWeave H100s, 07-11) → **2.7213** (#120, 07-16) → 2.7131 → 2.7112 →
**2.7037** (#117 E16 final, 07-22) → **2.7025** (#146 3B, 07-27) →
**2.6642** (#166 AA aug, finished 07-31), over 173 finished runs. #155's 3-way
restart reached 2.6819 on 08-01 — below #146 but above #166, so it does not
take this frontier either.

### What the figures show

- **The accuracy frontier moved in three jumps**, all from the base model,
  none from inference or post-training: ~0.03 → **0.425** when #75's
  E8 rung finished (2026-06-21), 0.436 → **0.534** when #117's 16-epoch
  bs256 run finished (2026-07-22), and 0.534 → **0.562** when #166's
  AA-augmentation continue-train of #117 finished (2026-07-31). Between the
  first two, five weeks of post-training and inference work moved it by
  +0.011 (#120's re-epoch). (#75's E4 winner, 0.031, landed the same day as
  E8, so the pre-jump frontier reads 0.029 — #67's.)
- **Two data-side results landed a day apart, and only one of them is on the
  frontier.** #166 (AA augmentation on top of #117, 07-31, 0.562) and #155's
  3-way crops+contacts-v1+ESM-Atlas mixture restart (08-01, 0.554) are both
  *data* changes rather than hyperparameter sweeps — the first of their kind
  here — but #166 is 0.008 higher and one day earlier, so #155 never appears
  as a step. They are not alternatives that were run against each other:
  different data, different initialisation, different token budgets, and no
  paired comparison exists. Only the final, finished checkpoint is plotted for
  #155 — two earlier checkpoints from the same run were also scored while it
  was still training (step 60000: R 0.553, step 70000: R 0.556) but are
  intentionally left off the figures now that a settled result exists; see the
  caveats section.
- **#166 is the only frontier point with a within-run control.** #190 re-scored
  #117 alongside it on the same 554 proteins and the same inference path,
  reading 0.5336 against #169's 0.5344 — so its **+0.0282** (95% CI
  0.0226–0.0338, higher on 67% of proteins) is a paired result, not a
  subtraction across two harnesses. Every other gap in the table above is the
  latter.
- **The oracle best-of-100 diagnostic (new, #155 final checkpoint only):**
  scoring each of the 100 sampled rollouts per protein on its own first-R
  precision and taking the max, instead of voting them together, reads
  **0.595** — **+0.041** over the same checkpoint's deployable rollout+vote
  number (0.554). That is real headroom left on the table by the
  aggregation step, not by the base model: the *union* of information across
  100 samples ranks higher than any one of them (that's why voting beats a
  single rollout, per exp82), but picking a *specific* best-performing rollout
  per protein beats the vote too. Both can be true because voting and
  cherry-picking extract different structure from the sample — voting favors
  contacts many rollouts agree on, while the oracle rewards a rollout that
  happens to get the high-value ones right even if it disagrees elsewhere.
  0.595 is a **ceiling that assumes free ground truth** (you cannot know
  which of the 100 rollouts is best without it) — it bounds how much a better
  *selection* method could close the gap to Protenix-v2 single-seq's 0.603,
  not a number the current pipeline can bank without one.
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
- **All of this is still below single-sequence Protenix-v2** (0.562 deployable
  vs 0.603; #155's 0.595 oracle ceiling is a different checkpoint), and well
  below ESMFold2 (0.786). The remaining single-seq gap is now **0.041** — less
  than the 0.041 of headroom the oracle diagnostic found inside *one*
  checkpoint's own rollouts, which is a coincidence of magnitude rather than a
  result, but does put the two levers on the same scale.

## Conclusion

Both success criteria are met: `data/rprecision_checkpoints.csv` carries a
source citation per number, and the figures regenerate from two commands.

The substantive read is that **contact accuracy has come from the base model
and essentially nowhere else**. Three training results account for the entire
frontier; the settled inference recipe is worth a large constant (+0.086) but
was banked once in June and has not moved since; and the post-training line
(#120, #160) has produced +0.011 and −0.020 respectively. The third jump —
#166's amino-acid augmentation, which continued #117 for 8 epochs and finished
on 07-31 at R 0.562 — is the first frontier point to come from a *data* change
rather than a hyperparameter sweep or an epoch count, and the only one carrying
a within-run control (+0.0282 paired against its own #117 initialisation).

Two things about that jump are worth separating. It is **+0.028 for a
continue-train**, which is nearly three times what the post-training line has
ever returned and came from changing what the data looks like rather than how
long it is trained on. And it **arrived one day before #155's 3-way mixture
restart** (0.554), the other data-side result in flight — so the honest summary
is that two different data interventions landed within a day of each other at
0.562 and 0.554, neither was run against the other, and the frontier records
only the first. Whether AA augmentation and the ESM-Atlas mixture compose is an
open question this tracker cannot answer.

The new **oracle best-of-100** diagnostic adds a second, orthogonal lever:
+0.041 R-precision sits in the *inference* step even after the settled
rollout+vote recipe, unreachable without ground truth but a real signal that
better *selection among* already-sampled rollouts (not more sampling, not a
better base model) is worth investigating — e.g. a learned reranker, or a
cheaper proxy for "is this rollout one of the good ones."

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

### The three inference recipes

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
  in #142 (it cost ~0.007–0.012 R-precision by truncating long rollouts). This
  is the only one of the three that is actually **deployable** — the other two
  either require the ground truth being scored against (oracle) or are
  strictly worse (pairwise).
- **oracle_best_of_100** *(new)* — same 100 rollouts as `rollout`, but instead
  of voting them into one `[L,L]` matrix, each rollout is scored on its own
  first-*R* precision (its emitted contacts, in generation order, restricted
  to resolved pairs — same *R* definition as the standard "R" cut), and the
  reported number is the **max over the 100**. Requires ground truth to know
  *which* rollout was best, so it is a diagnostic upper bound, never a
  deployable score — kept in its own marker/colour, excluded from the "best
  model trained to date" frontier line and from headline-label selection
  (see `plot_rprecision_frontier`'s `deployable` filter). Currently computed
  for one checkpoint (#155's final, #155 issue) via
  `exp82_evals_contacts_v1_contact_prediction/score_rollout_worker_oracle.py`
  + `build_oracle_best_rollout.py`.

Consequence for the accuracy frontier: it is the running max over each
checkpoint's *best available deployable* measurement. Every step of it happens
to be a rollout number, and the pairwise-only points (#67, #75 E1/E2/E4, #117
E8 bs64) never touch it, so the mixture does not change the staircase. If a
rollout number is ever produced for #117 E8 bs64 it would land near 0.50 and
still sit under the #117 E16 step.

### Structure-predictor reference lines

Dotted lines on both R-precision figures, same 554 proteins and same metric,
recomputed from `../exp89_evals_contacts_v1_model_on_eval_set/data/contact_precision_all.csv`.
These are the `predictor=structure` rows. Protenix's *distogram* readouts
(0.380 single-seq, 0.465 MSA) are in the same CSV but are not drawn — the
structure readout is the one the project quotes.

### What is in, what is out

**In the accuracy figure** — the 12 checkpoints with a benchmark score
(`data/rprecision_checkpoints.csv`, 15 rows: 2 checkpoints measured under two
recipes, plus the oracle diagnostic row, one citation per row).

**In the loss figure** — every *finished* W&B run reporting a contacts-v1 val
loss across `open-athena/MarinFold` and `eric-czech/marin` tags exp75 / exp117 /
exp146 / exp153 (n=157 after exclusions).

**Excluded, and why:**

- **Crashed / preempted / failed runs** (94 crashed, 12 failed). Their last
  logged loss is mid-training, not a trained model. Several would otherwise
  punch spurious holes in the frontier — e.g. a preempted exp117 run sitting
  at 2.7309.
- **Smoke tests, batch-calibration probes, profiling runs, the NeMo #112
  throughput runs, and the bio2token vetting run** — matched on name
  (`smoke`, `probe`, `profile`, `-prof`, `vet-`, `nemo`).
- **#160 backtracking** appears in the accuracy figure (R 0.4158) but not the
  loss one: it has a 3849-token superset vocabulary, so its val loss is not
  comparable to the 2845-vocab runs.
- **In-flight runs** are drawn as hollow diamonds on the loss figure and kept
  off *that* frontier — they have not finished training. Five such runs are
  live as of 2026-08-02 (the original #155 3-way mix at 2.8080, its no-crops
  ablation at 2.7622, and three exp177 tokenization-variant runs — two
  `next_token` runs at 2.9340 and 3.4099, plus a `soft_target` run still far
  from converged at 13.91); the best of them is the no-crops ablation. The
  two runs above 3.26 nats sit off the top of this figure's axis, same as
  the finished-run outliers the caption already accounts for. #155's
  3-way *restart* — previously in this set — **finished training on 08-01 at
  2.6819** and is now a regular point on the finished-run frontier (see "The
  loss frontier" above), labelled `#155 3-way restart (finished)`. While it
  was still training, two of its intermediate checkpoints were also scored
  on the accuracy benchmark (step 60000: R 0.553, step 70000: R 0.556) —
  the #89 benchmark scores a specific checkpoint, not a finished run, so
  this was possible before the run finished. Now that the run has a settled
  final checkpoint (step 74793, R 0.554), only that final point is plotted;
  the two earlier ones are dropped from the figures, though the underlying
  eval outputs are still in `data/` (see Files table).

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
5. **#155's 3-way restart finished training on 08-01 at step 74793** (target
   74800), R-precision 0.5545 — a settled result, not an in-flight snapshot,
   and the only checkpoint from this run plotted on the accuracy figure. Two
   earlier checkpoints scored while the run was still training (step 60000:
   R 0.553, step 70000: R 0.556, briefly the higher of the two) are excluded
   from the figures now that a final result exists — the 0.002 gap between
   the 70k checkpoint and the final is itself a small, likely-noise dip, not
   a reason to keep the intermediate point around. Their raw eval outputs
   remain in `data/` if needed. It has no val loss on these figures
   (3848-token superset crops tokenizer, same reason as #160) — its
   loss-figure label reads `(finished)` to distinguish it from the still-live
   original 3-way mix and no-crops ablation runs of the same experiment
   family.
6. **#166 is a continue-train of #117, not an independent run.** Its 8 epochs
   start from #117's step-35679 weights, so it inherits that run's tokenizer
   (which is why its val loss *is* comparable, unlike #155's) and its
   R-precision is not independent evidence about the recipe — it is evidence
   about AA augmentation applied to that specific checkpoint. Its step number
   (35679) coincidentally equals #117's; they are different weights.
7. **#155 and #166 have never been compared to each other.** They differ in
   data, initialisation, epoch count and token budget, and their 0.008
   difference is a between-run gap read off two separate eval jobs, well inside
   the range where nothing can be concluded. Do not read the frontier as
   "AA augmentation beat the 3-way mixture".
8. **The oracle best-of-100 diagnostic is scored for one checkpoint only**
   (#155's final, step 74793). It shares the same 100 rollouts as that
   checkpoint's `rollout` row — same TPU eval run, same per-protein samples —
   scored a second way, not an independent measurement. Treat it as a
   headroom bound on *that specific checkpoint*, not yet established as a
   general property of the model family.

## Files

| path | what |
|---|---|
| `build_dataset.py` | assembles both tables; W&B pull + the hand-curated benchmark rows |
| `plot_progress.py` | the three progress figures |
| `plot_vs_protenix.py` | the two comparison scatters + their paired CSVs |
| `data/rprecision_checkpoints.csv` | 16 rows (13 checkpoints; 2 measured under both pairwise/rollout recipes, plus the oracle diagnostic row), one source citation each |
| `data/structure_baselines.csv` | the four dotted lines, recomputed from exp89 |
| `data/val_loss_runs.csv` | 368 W&B runs with a contacts-v1 val loss, with state + exclusion flag |
| `data/rprecision_footnotes.csv` | measurements that are alternate realisations of a checkpoint, not new checkpoints |
| `data/marinfold_vs_protenix_{ss,msa}.csv` | the 554 paired per-protein scores behind each comparison figure |
| `data/exp155_3way_restart_step60000_rollout_summary.csv` | aggregate R-precision/AUC for #155's step-60000 checkpoint, this session's rollout eval — not plotted, superseded by the run's final checkpoint (see caveat 5) |
| `data/exp155_3way_restart_step60000_rollout_rows.csv.gz` | per-protein rows behind that summary (554 × 20) |
| `data/exp155_3way_restart_step70000_rollout_summary.csv` | same, for the step-70000 checkpoint — also not plotted |
| `data/exp155_3way_restart_step70000_rollout_rows.csv.gz` | per-protein rows behind that summary (554 × 20) |
| `data/exp155_3way_restart_step74793_rollout_summary.csv` | same, for the final checkpoint (step 74793, run complete) — the one point plotted for this run |
| `data/exp155_3way_restart_step74793_rollout_rows.csv.gz` | per-protein rows behind that summary (554 × 20) |
| `data/exp155_3way_restart_step74793_oracle_best100_summary.csv` | oracle best-of-100 mean R-precision by range, same checkpoint and rollouts as the summary above |
| `data/exp155_3way_restart_step74793_oracle_best100_rows.csv.gz` | per-protein, per-range oracle rows behind that summary |
| `../exp166_models_contacts_v1_aa_augmentation/data/exp166_rows.csv.gz` | per-protein rows for the pinned head-to-head checkpoint — read, not copied, so the scatter can never drift from #190's published scores |
| `plots/*.png.meta.json` | the numbers behind each figure |
| `../exp82_evals_contacts_v1_contact_prediction/score_rollout_worker_oracle.py` | rollout-eval TPU worker; additive fork of exp82's `score_rollout_worker.py` that also writes a per-rollout, emission-order detail table |
| `../exp82_evals_contacts_v1_contact_prediction/build_oracle_best_rollout.py` | scores that detail table into the oracle best-of-100 summary/rows CSVs above |

Sources for every R-precision value are in the `source` column of
`data/rprecision_checkpoints.csv` — exp82 / exp89 / exp120 / exp155 / exp160 /
exp166 / exp169 data CSVs, plus
[eric-czech's checkpoint gist](https://gist.github.com/eric-czech/bfa78571dcb8f673884bf70e6cc68e14)
for #117 E8 bs64.
