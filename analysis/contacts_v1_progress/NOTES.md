# contacts-v1 progress over time

Three figures tracking the best contacts-v1 model we had at each date, on
contact accuracy and on held-out validation loss, plus the two against each
other.

* [`plots/rprecision_frontier.png`](plots/rprecision_frontier.png)
* [`plots/val_loss_frontier.png`](plots/val_loss_frontier.png)
* [`plots/rprecision_vs_val_loss.png`](plots/rprecision_vs_val_loss.png)

```bash
uv run --with wandb --with pandas python build_dataset.py   # refresh from W&B
uv run --with pandas --with matplotlib python plot_progress.py
```

## What is being measured

**R-precision, all ranges** — precision at the top *R* ranked residue pairs,
where *R* is that protein's true-contact count (seq-sep ≥ 6), meaned over the
**554-protein eval set** (#74/#78 + #41/#65 curation), scored by exp89's
`compute_metrics.py` against the published exp89 ground-truth universe. This is
the only accuracy number on these plots; long/medium/short-range cuts, AUC and
contacts@L exist in the source CSVs but are not plotted.

**Validation loss** — `eval/contacts-v1-val/loss` (Eric's runs log the same
quantity as `eval/tokenized/contacts-v1-val/loss`), the full held-out contacts-v1
split: 41,954 documents / 47,821,958 tokens.

**Date** — when the training run *finished* (W&B `heartbeatAt`), i.e. when the
checkpoint came into existence. Not when it was evaluated; several checkpoints
were scored weeks later.

## The one trap: two inference recipes

The same weights score **~0.086 higher** under the settled rollout recipe than
under exp89's original pairwise scorer, on both checkpoints measured under both:

| checkpoint | pairwise | rollout | Δ |
|---|---:|---:|---:|
| #61/#75 E8 | 0.3389 | 0.4245 | +0.0856 |
| #120 re-epoch | 0.3495 | 0.4357 | +0.0862 |

So the figures never merge them — marker shape and colour carry the recipe, and
a checkpoint measured both ways shows both points. Recipes:

* **pairwise** — autoregressive `P(<contact> <pi> <pj>)`, symmetrised. exp89's
  original scorer; still what the `eval-checkpoint` skill runs.
* **rollout** — n=100 sampled rollouts + per-rollout document resampling +
  pairwise tie-break, **top-k off**. Settled in exp82; top-k=50 was removed in
  #142 (it cost ~0.007–0.012 R-precision by truncating long rollouts).

Consequence for the accuracy frontier: it is the running max over each
checkpoint's *best available* measurement. Every step of it happens to be a
rollout number, and the pairwise-only points (#67, #75 E1/E2/E4, #117 E8 bs64)
never touch it, so the mixture does not change the staircase. If a rollout
number is ever produced for #117 E8 bs64 it would land near 0.50 and still sit
under the #117 E16 step.

## Structure-predictor reference lines

Dotted lines on both R-precision figures, same 554 proteins and same metric,
recomputed from `experiments/exp89_.../data/contact_precision_all.csv`:

| predictor | R-precision (all) |
|---|---:|
| Protenix-v2 single-seq | 0.603 |
| ESMFold | 0.755 |
| ESMFold2 | 0.786 |
| Protenix-v2 + MSA | 0.812 |

These are the `predictor=structure` rows. Protenix's *distogram* readouts
(0.380 single-seq, 0.465 MSA) are in the same CSV but are not drawn — the
structure readout is the one the project quotes.

## What is in, what is out

**In the accuracy figure** — the 11 checkpoints with a benchmark score
(`data/rprecision_checkpoints.csv`, one citation per row).

**In the loss figure** — every *finished* W&B run reporting a contacts-v1 val
loss across `open-athena/MarinFold` and `eric-czech/marin` tags exp75 / exp117 /
exp146 / exp153 (n=153 after exclusions).

**Excluded, and why:**

* **Crashed / preempted runs** (94 of them). Their last logged loss is
  mid-training, not a trained model. Several would otherwise punch spurious
  holes in the frontier — e.g. a preempted exp117 run sitting at 2.7309.
* **Smoke tests, batch-calibration probes, profiling runs, the NeMo #112
  throughput runs, and the bio2token vetting run** — matched on name
  (`smoke`, `probe`, `profile`, `-prof`, `vet-`, `nemo`).
* **#160 backtracking** appears in the accuracy figure (R 0.4158) but not the
  loss one: it has a 3849-token superset vocabulary, so its val loss is not
  comparable to the 2845-vocab runs.
* **In-flight runs** (#155 3-way mixture and siblings) are drawn as hollow
  diamonds but kept off the frontier — they have not finished. The best of them
  currently reads `contacts-v1-val` 2.7110.

## Caveats worth carrying

1. **#120's loss is `eval/contacts-v1-val-orig/loss`** (2.7213), everything else
   is `contacts-v1-val`. Almost certainly the same held-out split under a
   different cache name, but that has never been proven — #160 flagged the same
   thing. Treat #120's position on the loss axis as approximate.
2. **#146's 3B is confounded.** It differs from the #117 1.5B in epochs (8 vs
   16) and weight decay (0.4 vs 0.2) as well as parameter count, so "3B at equal
   loss is worse" indicts *this checkpoint*, not scale.
3. **The #117 early-stop checkpoint is step 33450 of the run that ends at
   35679.** At day resolution it plots on top of the final; both markers are
   drawn, one label carries both numbers.
4. **exp89's harness vs exp82's**: the #117 final reads 0.5344 through exp169's
   TPU worker and 0.5350 through exp82's — a 0.0006 backend difference, inside
   the ≤0.006 TPU-vs-CUDA agreement #89 established. Either is fine; the plots
   use exp169's for that checkpoint.

## Things the plots surface

* **The accuracy frontier moved in exactly two jumps**, both from the base
  model, neither from inference or post-training: ~0.03 → **0.425** when #75's
  E8 rung finished (2026-06-21), and 0.436 → **0.534** when #117's 16-epoch
  bs256 run finished (2026-07-22). Between them, five weeks of post-training and
  inference work moved it by +0.011 (#120's re-epoch). (#75's E4 winner, 0.031,
  landed the same day as E8, so the pre-jump frontier reads 0.029 — #67's.)
* **Loss and accuracy agree across generations and stop agreeing inside one.**
  The 0.053-nat #75→#117 gap buys +0.109 R-precision (~2 R-precision per nat).
  The 0.008-nat gap between #117's early-stop and final checkpoints buys nothing
  (paired Δ +0.0026 in the *final*'s favour, CI crosses zero), and #146's 3B is
  0.0012 *better* on loss and 0.023 *worse* on R-precision.
* **#108's 3B on CoreWeave H100s held the loss frontier from 2026-07-11 to
  07-16 at 2.7418** and was never contact-scored. Given #146's result — a 3B at
  matched loss under-performing the 1.5B — it is probably not a missed frontier
  point on accuracy, but that is an inference, not a measurement.
* **#85's LR re-heat did not lower loss.** It finished at 2.9801 against #67's
  2.9800. The Week-of-June-22 UPDATES.md entry says it "improved eval loss
  somewhat"; the W&B history does not support that (the run's five eval points
  run 2.9825 / 2.9828 / 2.9843 / 2.9820 / 2.9801).
* **#67 never held the loss frontier.** It finished 2026-06-14 15:36 at 2.9800,
  about two hours after `prot-exp75-cv1-1_5b-e2-lr7e-4-wd0p05-v1` reached 2.9787.
* **All of this is still below single-sequence Protenix-v2** (0.534 vs 0.603),
  and well below ESMFold2 (0.786).

## Files

| path | what |
|---|---|
| `build_dataset.py` | assembles both tables; W&B pull + the hand-curated benchmark rows |
| `plot_progress.py` | the three figures |
| `data/rprecision_checkpoints.csv` | 13 rows (11 checkpoints; 2 measured under both recipes), one source citation each |
| `data/structure_baselines.csv` | the four dotted lines, recomputed from exp89 |
| `data/val_loss_runs.csv` | 318 W&B runs with a contacts-v1 val loss, with state + exclusion flag |
| `data/rprecision_footnotes.csv` | measurements that are alternate realisations of a checkpoint, not new checkpoints |
| `plots/*.png.meta.json` | the numbers behind each figure |

Sources for every R-precision value are in the `source` column of
`rprecision_checkpoints.csv` — exp82/exp89/exp120/exp160/exp169 data CSVs, plus
[eric-czech's checkpoint gist](https://gist.github.com/eric-czech/bfa78571dcb8f673884bf70e6cc68e14)
for #117 E8 bs64.
