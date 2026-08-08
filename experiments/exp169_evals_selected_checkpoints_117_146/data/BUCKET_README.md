# contacts-v1 model eval — issue #169

Contact-prediction results for the three checkpoints selected in
[MarinFold #169](https://github.com/Open-Athena/MarinFold/issues/169): the final
and early-stop winners of
[#117](https://github.com/Open-Athena/MarinFold/issues/117) (1.5B) and the
[#146](https://github.com/Open-Athena/MarinFold/issues/146) 3B, all selected by
`eval/tokenized/contacts-v1-val/loss`.

Companion to `data/contacts-v1-model-eval-exp89/`, which holds the previous
generation of this eval (the #61/#75 checkpoint and the structure-predictor
baselines). Same ground truth, same candidate-pair universe, same metric code —
rows from the two prefixes concatenate directly.

## What was evaluated

| label (directory name) | model | step | `contacts-v1-val` loss | source checkpoint |
|---|---|---:|---:|---|
| `exp117_e16_final_step35679` | 1.5B Qwen3, 16 epochs | 35,679 | 2.703709 | `open-athena/marinfold-exp117` · `prot-exp117-cv1-s02-1_5b-e16-lr3p162e-3-wd0p2-bs256-europe-west4/hf/step-35679` |
| `exp117_e16_early_step33450` | 1.5B Qwen3, 16 epochs | 33,450 | 2.696074 | same run, `checkpoints/step-33450` (levanter; **no HF export existed** — converted for this eval) |
| `exp146_3b_e8_step17839` | 3B Qwen3, 8 epochs | 17,839 | 2.702478 | `open-athena/marinfold-exp146` · `prot-exp146-cv1-s01-3b-e8-lr3p162e-3-wd0p4-bs256-us-east1/hf/step-17839` |

## Method

- **Eval set** — 554 `(dataset, stem)` units over 552 unique proteins
  (FoldBench-100 + the exp65 low-MSA / novel-fold candidates), from MarinFold
  [#74](https://github.com/Open-Athena/MarinFold/issues/74) /
  [#78](https://github.com/Open-Athena/MarinFold/issues/78). Ground truth is
  pyconfind side-chain contacts on the experimental structure (`native_only`,
  degree ≥ 0.001, sequence separation ≥ 6) — the same definition the contacts-v1
  training documents use.
- **Inference** — the recipe settled in MarinFold
  [#82](https://github.com/Open-Athena/MarinFold/issues/82): 100 sampled
  contacts-v1 rollouts per protein, each from a freshly resampled document
  realization, voted into an `[L, L]` per-pair occurrence matrix. Sampling is
  `temperature 1.0`, `top_p 0.95`, **top-k off**, budget `6L + 128` tokens.
  vLLM 0.9.2, bf16, one H100 per shard.
- **Metrics** — MarinFold
  [#89](https://github.com/Open-Athena/MarinFold/issues/89)'s `compute_metrics`
  implementation, unmodified: precision at `L`, `L/2`, `L/5` and `R` (=number of
  true contacts) plus ROC AUC, over the `all` / `short` / `medium` / `long`
  separation ranges — 20 rows per protein per checkpoint.

## Files

| path | what |
|---|---|
| `exp169_rows.csv.gz` | per-protein metrics; 554 × 20 rows per checkpoint |
| `exp169_summary.csv` | aggregate means per (model, range, cut) |
| `exp169_paired.csv` | paired per-protein differences between every pair of checkpoints, with 95% CIs and win rates |
| `exp169_rollout_summary.csv` | the same aggregate as emitted by the shared exp82 metric driver |
| `gt_universe.jsonl` | the ground-truth universe exactly as scored (identical to exp89's) |
| `eval_targets.parquet` | the 554 prompts exactly as scored (`dataset`, `stem`, `L`, `input_seq`) |
| `scores/<label>/<dataset>__<stem>.npz` | the `[L, L]` float16 vote matrix per protein per checkpoint |
| `plots/*.png` | the two figures, with `.meta.json` sidecars carrying the plotted numbers |

## Training-trajectory extension

`trajectory/` contains the permanent-checkpoint comparison for the #146 3B E8,
#117 1.5B E8 BS64, and #117 1.5B E16 runs. All permanent checkpoints were used
for both E8 runs, while alternating E16 checkpoints give eight points over
twice the token span.

The BS64 run performs four optimizer updates per BS256 update at a fixed token
budget. Its shorter E8 schedule also changes the learning-rate trajectory, so
the results compare configurations rather than batch size in isolation.

| path | what |
|---|---|
| `trajectory/runs/<run>/step-<N>/` | immutable inputs, provenance, sparse rollout votes, and durable timing parts for one full evaluation |
| `trajectory/derived/<run>/step-<N>/` | validated per-protein metrics, summary, timings, and checksums for one checkpoint |
| `trajectory/summary/trajectory_checkpoint_metrics.csv` | plot-ready loss and all/short/medium/long R-precision by tokens and step |
| `trajectory/summary/trajectory_paired_changes.csv` | paired per-protein R-precision changes between adjacent evaluated checkpoints |
| `trajectory/summary/trajectory_matched_token_changes.csv` | paired differences between runs at shared nominal token budgets |
| `trajectory/summary/trajectory_metric_rows.csv.gz` | all 265,920 per-protein metric rows |
| `trajectory/summary/trajectory_timings.csv.gz` | completion and generation telemetry for all 13,296 checkpoint-protein evaluations |
| `trajectory/summary/checkpoint_trajectory.png` | loss and R-precision trajectory figure, with its provenance sidecar |

The raw trajectory uses sparse nonzero vote triplets rather than dense score
matrices. Durable parts allow a replacement worker to resume at the first
unfinished protein. This path was exercised when the 1.5B step-22300 worker was
reclaimed after 480 of 554 proteins and again when the BS64 step-44600 worker
was replaced after its first 16-protein part.

The score matrices are the expensive artifact — ~100 sampled rollouts × 554
proteins × 3 checkpoints of H100 time — and every table here is derivable from
them plus `gt_universe.jsonl`. They are published so the eval can be re-scored
under a different metric or candidate-pair definition without re-running
inference.

## Reproducing

Code: [`experiments/exp169_evals_selected_checkpoints_117_146/`](https://github.com/Open-Athena/MarinFold/tree/main/experiments/exp169_evals_selected_checkpoints_117_146)
in the MarinFold repo. The scoring worker and metric implementation live in
`exp82_evals_contacts_v1_contact_prediction/` and `exp89_evals_contacts_v1_model_on_eval_set/`
and are used unmodified.
