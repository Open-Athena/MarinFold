# Summary slides — exp: generate rollouts for subset of contacts-v1 training set

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide.
     Keep this current as the experiment progresses. -->

## What we're doing

Collect rollouts from Eric's tuned contacts-v1 1.5B model (eval loss 2.7566) at
scale, and learn how fast we can generate them on TPU. Sample 1000 training
targets (round-0, L≤512, ≥5 contacts); for each, generate 1000 rollouts; score
every rollout's precision/recall/F1; save the best-recall and best-F1 rollouts
verbatim for a future "train on high-accuracy rollouts" experiment.

## Why

Groundwork for the real question: does fine-tuning on high-accuracy rollouts beat
re-epoching the data? First we need the rollouts and a feel for the generation
cost at scale. Builds on exp82 (rollout+resample is the best LM-only recipe) and
exp89 (the vLLM/iris-TPU path).

## How (inference)

vLLM on iris TPU v5p-8, bf16 model, **tensor_parallel_size=4** (all 4 chips).
rollout+resample recipe (each rollout a fresh document realization). Prefixes
pre-built locally; the TPU worker is marinfold-free and resumes on restart.

## Calibration — throughput

On one v5p-8 (10 length-stratified targets × 1000 rollouts): tp=4 gives
**~11.3k tok/s aggregate (~16–18k steady-state)**, ~1.8–2× over tp=1 (which left
3 of 4 chips idle). 0% of rollouts hit the token cap. resample ≈ nsample on
accuracy, ~15% slower. Projected full 1M-rollout run: ~0.86 h on 8× v5p-8.

## Calibration — accuracy

Per-rollout accuracy is low on average but **best-of-1000 ≫ mean**, and strongly
length-dependent: small proteins (L≈30–80) reach best recall 0.8–1.0, large ones
(L>250) stay near 0.15–0.2. This is the point of sampling many rollouts — the best
one is far better than the typical one.

## Full run — scale

1,000,000 rollouts (1000 targets × 1000) in **~41 min wall on 8× v5p-8**
(~4.6 v5p-8-hours of generation; 283 M tokens; 17.2k tok/s effective; 0%
truncated; 0 failures). 10× more targets ≈ ~7.5 h on 8× v5p-8 — affordable.

## Full run — accuracy

Best-of-1000 per target (all sep≥6): recall mean **0.335** (max 1.0), F1 mean
**0.340** (max 0.955) — ~3× the typical single rollout. Strongly length-dependent
(best F1 0.52 at L≤100 → 0.20 at L>400); 20% of targets reach F1 ≥ 0.5. Sampling
many rollouts surfaces far higher-quality contact sets than any single decode —
the raw material for the train-on-rollouts idea.

## Deliverables

Per target: precision/recall/F1 of all 1000 rollouts + the best-recall and best-F1
rollouts saved verbatim (complete contacts-v1 documents). Interactive explorer:
explore_results.ipynb (Colab).
