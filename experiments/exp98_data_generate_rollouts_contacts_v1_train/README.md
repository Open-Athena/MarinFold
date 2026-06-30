---
marinfold_experiment:
  issue: 98
  title: "exp: generate rollouts for subset of contacts-v1 training set"
  kind: data
  branch: claude/elastic-hamilton-a50052
---

# exp: generate rollouts for subset of contacts-v1 training set

**Issue:** [#98](https://github.com/Open-Athena/MarinFold/issues/98) · **Kind:** `data` · **Branch:** `claude/elastic-hamilton-a50052`

## Question

Ultimately: does fine-tuning on **high-accuracy rollouts** improve the base model
more than just re-epoching the data? This experiment is the **data-collection +
scale probe** step — collect rollouts from the model and learn what scale we can
generate them at on TPU. (The training comparison is a later experiment.)

## Background

Eric's tuned contacts-v1 1.5B (issue [#61](https://github.com/Open-Athena/MarinFold/issues/61)/[#75](https://github.com/Open-Athena/MarinFold/issues/75),
eval loss **2.7566**) is the model. exp82 ([#82](https://github.com/Open-Athena/MarinFold/issues/82))
settled the best LM-only contact-prediction recipe — **rollout + per-rollout
document resampling** — and exp89 ([#89](https://github.com/Open-Athena/MarinFold/issues/89))
established the vLLM/iris-TPU inference path (bf16 export, the gotchas). This
experiment is exp82's open follow-up #4 ("push the rollout+resample run to the
iris/vLLM-TPU path for scale"), on the **train** split, saving the rollouts
themselves.

## Approach

- **Targets** — [`select_targets.py`](select_targets.py): sample **1000** targets
  from the contacts-v1 **train** split (exp53 corpus,
  `…/exp53_contacts_v1_5x/documents/train/`), restricted to **round-0** (highest
  pLDDT, cleanest GT), `L ≤ 512`, not truncated, and `≥ 5` GT contacts
  (seq-sep ≥ 6). GT contacts are read straight from the document text — no
  pyconfind. → [`data/targets.parquet`](data/targets.parquet) (1000 rows; L mean
  191 / median 161 / max 510; mean 143 GT contacts; 1000 distinct structural
  clusters).
- **Recipe** — **rollout + resample** (exp82's settled best): for each target,
  **1000** rollouts, each from a *fresh* contacts-v1 document realization
  (resampled N-terminus start + statement order). T=1.0, top-p=0.95, top-k=50,
  budget `4·L+64` tokens. Prefixes are pre-built locally
  ([`gen_prompts.py`](gen_prompts.py)) so the TPU worker stays marinfold-free.
- **Inference** — [`gen_rollouts_worker_vllm_tpu.py`](gen_rollouts_worker_vllm_tpu.py):
  vLLM on **iris TPU v5p-8**, the **bf16** model
  (`…-bc3084/hf_bf16/step-35679`, the exp89 TPU artifact),
  **tensor_parallel_size=4** (uses all 4 chips of the v5p-8). Sharded across
  8× v5p-8 for the full run; resumes on restart (skips done targets).
- **Scoring** — [`rollout_metrics.py`](rollout_metrics.py) (pure Python,
  unit-tested): parse each rollout's contacts, compute precision / recall / F1 per
  separation band (all sep≥6 / short / medium / long) vs GT. Headline + "best"
  selection use **all sep≥6** (the document's own contact definition).
- **Saved per target** (GCS
  `…/exp98_rollouts_contacts_v1_train/runs/full/`):
  - `rollout_metrics/<entry>.parquet` — one row per rollout, all per-band metrics.
  - `best_rollouts/<entry>.json` — the **best-recall** and **best-F1** rollouts
    saved verbatim as complete contacts-v1 documents (token order preserved — a
    future experiment may train on these).
  - `timings/<entry>.csv` — per-target generation timing.
- **Aggregate** — [`aggregate_results.py`](aggregate_results.py): throughput +
  accuracy summaries, plots, CSVs.

## Success criteria

For each target we know the accuracy (precision/recall/F1) of **every** rollout,
and the best-recall + best-F1 rollouts are saved exactly. Plus a careful
timing/resource report (TPU type, sequence length, tokens, throughput, total
time) to inform scaling.

## Results

### Calibration (10 length-stratified targets, 1000 rollouts each, 1× v5p-8)

Validated the worker end-to-end and measured throughput in three configs
([`data/calib_throughput.csv`](data/calib_throughput.csv);
projections are to the full 1000×1000 = 1M rollouts, and *over-weight* the
one-time XLA compile warmup because the calib set is small):

| config | tok/s (agg) | steady-state tok/s | proj. 1× v5p-8 | proj. 8× v5p-8 |
|---|---|---|---|---|
| resample, tp=1 (1 of 4 chips) | 7,332 | ~9–10k | 10.7 h | 1.33 h |
| nsample, tp=1 | 8,559 | ~12k | 9.2 h | 1.15 h |
| **resample, tp=4 (all 4 chips)** | **11,317** | **~16–18k** | **6.9 h** | **0.86 h** |

- **tp=4 uses all 4 v5p-8 chips → ~1.8–2× over tp=1** (tp=1 left 75% of the chip
  idle). Same accuracy.
- **resample ≈ nsample on accuracy**; resample ~15% slower (it does 1000 distinct
  prefills vs nsample's one shared prefill — the gap is largest at large L). We
  keep resample for realization diversity (better future training data).
- **0% of rollouts hit the token cap** (every rollout ends with `<end>`), matching
  exp82; mean ~280 tokens/rollout.
- **Accuracy signal — best-of-1000 ≫ mean** (per-rollout, all sep≥6), strongly
  length-dependent:

  | L | n_gt | mean recall | best recall | best F1 |
  |---|---|---|---|---|
  | 30 | 8 | 0.16 | 1.00 | 0.84 |
  | 78 | 11 | 0.64 | 0.82 | 0.82 |
  | 218 | 117 | 0.10 | 0.38 | 0.40 |
  | 510 | 462 | 0.06 | 0.14 | 0.20 |

  Small proteins are nearly solved by the best rollout; large ones stay hard.
  Plots: [`plots/calib_gentime_vs_L.png`](plots/calib_gentime_vs_L.png),
  [`plots/calib_bestacc_vs_L.png`](plots/calib_bestacc_vs_L.png).

### Full run (1000 targets × 1000 rollouts = 1,000,000 rollouts)

resample, tp=4, **8× v5p-8** in us-east5-a. **0 failures** — all 1000 targets have
`rollout_metrics/`, `best_rollouts/`, `timings/` on GCS
(`…/exp98_rollouts_contacts_v1_train/runs/full/`). Consolidated per-target table:
[`data/full_per_target.csv`](data/full_per_target.csv) (+ `…/runs/full/per_target_summary.parquet`
on GCS for the Colab explorer); throughput: [`data/full_throughput.csv`](data/full_throughput.csv).

**Resources & throughput** ([`plots/full_throughput_vs_L.png`](plots/full_throughput_vs_L.png)):

| metric | value |
|---|---|
| rollouts | 1,000,000 (1000 targets × 1000) |
| generated tokens | 283.1 M (mean **283 tok/rollout**) |
| effective rate (1 v5p-8-equiv, tp=4) | **17,177 tok/s** |
| generation compute | **4.58 v5p-8-hours** (~34 min/shard) |
| wall-clock on 8× v5p-8 | **~41 min** (incl. ~5 min setup) |
| rollouts hitting token cap | **0%** (every rollout ends with `<end>`) |

So at this model/length mix, **one v5p-8 generates ~1M rollouts in ~4.6 h**, and
8× does it in **~40 min**. The earlier calibration (11.3k tok/s) under-estimated
because its 10-target set over-weighted the one-time XLA compile warmup; at full
scale the steady-state rate (~17k tok/s) dominates.

**Accuracy** — per-rollout precision/recall/F1 for **all 1M rollouts** is saved;
headline is **best-of-1000 per target** (all sep≥6). Mean per-rollout recall is
0.109, but the **best rollout per target is ~3× better**
([`plots/full_best_hist.png`](plots/full_best_hist.png)):

| best-of-1000 (all sep≥6) | mean | median | max |
|---|---|---|---|
| recall | **0.335** | 0.275 | 1.000 |
| F1 | **0.340** | 0.291 | 0.955 |

Strongly **length-dependent** ([`plots/full_bestacc_vs_L.png`](plots/full_bestacc_vs_L.png),
[`plots/full_bestf1_vs_ngt.png`](plots/full_bestf1_vs_ngt.png)):

| L tier | n | mean n_gt | best recall | best F1 | mean-rollout recall |
|---|---|---|---|---|---|
| ≤100 | 231 | 41 | 0.555 | 0.521 | 0.170 |
| 101–200 | 391 | 95 | 0.337 | 0.341 | 0.103 |
| 201–300 | 215 | 195 | 0.217 | 0.243 | 0.080 |
| 301–400 | 93 | 294 | 0.180 | 0.216 | 0.077 |
| 401–510 | 70 | 390 | 0.162 | 0.200 | 0.078 |

**"Solved" fractions:** 19.8% of targets reach best-F1 ≥ 0.5, 5.5% ≥ 0.7, 0.2% ≥
0.9 (best-recall: 20.5% / 8.2% / 1.1%). The best-recall and best-F1 rollouts are
saved verbatim as complete contacts-v1 documents in `best_rollouts/<entry>.json`.

**Interactive explorer:** [`explore_results.ipynb`](explore_results.ipynb) (open in
Colab) loads the per-target summary + drills into any target's 1000-rollout
distribution and its saved best rollouts from GCS.

## Conclusion

We can generate model rollouts for the contacts-v1 train set **cheaply and at
scale**: **1,000,000 rollouts** (1000 targets × 1000) cost **~4.6 v5p-8-hours**
(~40 min wall on 8× v5p-8), with an efficient vLLM/TPU path — bf16 model,
`tensor_parallel_size=4` to use all 4 chips of a v5p-8 (~1.8–2× over the
single-chip default), rollout+resample for document diversity, 0% truncation,
~283 tokens/rollout. Extrapolating, **10× more targets (10k) ≈ ~46 v5p-8-hours**
(~7.5 h on 8× v5p-8) — comfortably affordable.

For every target we have the precision/recall/F1 of **all 1000 rollouts**, and the
best-recall + best-F1 rollouts saved exactly (token order preserved) for a future
train-on-rollouts experiment. The accuracy signal is the encouraging part: while a
*typical* rollout recovers ~11% of contacts, the **best of 1000 recovers ~34% on
average and solves small proteins outright** (best recall up to 1.0; 20% of
targets reach F1 ≥ 0.5) — i.e. sampling many rollouts surfaces substantially
higher-quality contact sets than any single decode, which is exactly the raw
material the train-on-high-accuracy-rollouts idea needs. Quality falls off with
length (best F1 0.52 at L≤100 → 0.20 at L>400), so the high-value training
documents are concentrated in shorter proteins.

**Suggested follow-ups (human to decide):** (1) scale targets up (the cost is
affordable); (2) the actual fine-tuning experiment — train on the saved best
rollouts vs re-epoching; (3) for large proteins, more rollouts or a stronger model
to lift best-of-N; (4) `best_rollouts/` is a natural HuggingFace dataset to
publish for (2).
