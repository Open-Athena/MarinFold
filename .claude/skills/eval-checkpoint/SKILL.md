---
name: eval-checkpoint
description: >-
  Run the MarinFold contacts-v1 contact-prediction eval set (554 proteins) on a
  model checkpoint and report R-precision — plus precision@{L,L/2,L/5} and ranking
  AUC — per sequence-separation range (all / short / medium / long), next to the
  Protenix-v2 / ESMFold / ESMFold2 baselines. Use this whenever someone wants to
  evaluate, score, or benchmark a contacts-v1 checkpoint on contact prediction:
  "eval this checkpoint", "what's the R-precision of <run>", "how does the new
  model do on the eval set", "score step-N on contacts", "benchmark it vs ESMFold",
  including brand-new checkpoints straight out of a training run (exp108, exp120,
  and successors). It handles the Levanter→HF export, pulling the prebuilt ground
  truth from the bucket, scoring, and the exact metric code. Reach for it before
  hand-rolling contact metrics or rebuilding ground truth — getting either wrong
  silently corrupts the numbers.
---

# eval-checkpoint — contacts-v1 checkpoint → R-precision

The canonical way to answer "how good is this checkpoint at predicting contacts?"
is the **exp89 harness** (`experiments/exp89_evals_contacts_v1_model_on_eval_set/`).
It scores a model on a fixed 554-protein eval set and reports the same metrics for
MarinFold and every structure-predictor baseline, over one shared candidate-pair
universe, so numbers are directly comparable.

**The exp89 README is the source of truth for exact commands and paths — read it
first (`experiments/exp89_evals_contacts_v1_model_on_eval_set/README.md`, the
"Running" section).** This skill is the orchestration and the guardrails; it does
not restate the full command block, because the README is maintained and this file
would drift. Your job is to run the harness correctly and not fall into the traps
below.

## What you're producing

For each sequence-separation range — `all` (sep ≥ 6), `short` (6–11),
`medium` (12–23), `long` (≥ 24) — the mean over the 554 proteins of:
**R-precision** (precision at the top-`#true-contacts` predicted pairs), plus
precision@{L, L/2, L/5} and ranking AUC. Report MarinFold next to the Protenix-v2
(single-seq + MSA) / ESMFold / ESMFold2 baselines, like the exp89 headline table.

The headline people usually want is **long-range R-precision** and **aggregate
(`all`) R-precision** — lead with those, but produce the full per-range table.

## The workflow

Run the three-step harness. Steps (1) and the baselines are checkpoint-independent
and already exist; only export + scoring are per-checkpoint.

1. **Get the checkpoint into HF safetensors.** If you were handed a Levanter
   checkpoint (a `gs://…/checkpoints/<run>/checkpoints/step-N/` path), export it
   first with the `export_contacts_v1_best_to_hf.py` pattern — levanter's
   `export_lm_to_hf` on CPU, with the exp75 Qwen3 config and the contacts-v1
   tokenizer co-located (vocab 2845). If you already have an HF export
   (`…/hf/step-N/`, a `Qwen3ForCausalLM` with the tokenizer alongside), skip this.
   *Co-locate the tokenizer with the weights — the scorer builds the sequence
   prefix from it, and a mismatched/absent tokenizer produces silently wrong
   prompts.*

2. **Pull the ground-truth universe — do not rebuild it.** It's checkpoint-
   independent (same 554 proteins every time) and lives on the bucket:
   ```
   hf buckets cp hf://buckets/open-athena/MarinFold/data/contacts-v1-model-eval-exp89/gt_universe.jsonl data/gt_universe.jsonl
   ```
   Rebuilding it from scratch needs the exp78 pyconfind venv + the staged GT
   structures and reproduces the same bytes — only do that if the bucket copy is
   somehow unavailable. The README keeps the rebuild command as a fallback.

3. **Score, then compute metrics.** Scoring is exp82 **pairwise** — the symmetrized
   geometric-mean log-prob of `<contact> <pi> <pj>` over every candidate pair,
   which exp82 found extracts the most LM-only signal. Local CUDA transformers
   (`score_eval_set.py`) is the canonical, fastest path; the vLLM/iris-TPU variant
   (`score_eval_set_vllm.py`) is the same scoring definition if you need it.
   Feed the score matrices to **exp89's `compute_metrics.py`** with its committed
   `--exp78-*`/`--exp74-*` baseline-splice args (they're in the README). Then
   `plot.py` if you want the boxplots.

## Guardrails — the ways this goes silently wrong

- **Use exp89's `compute_metrics.py`. Never hand-roll the metric, and don't reuse
  exp82's `metrics()`.** exp82's own implementation disagrees with exp89's by up to
  0.4/protein (float16 tie-breaking + small-protein edge cases). Cross-predictor
  comparison is only valid under one metric implementation, and exp89's is it.
- **The eval unit is (dataset, stem), not stem.** Two stems (`7ur7_A`, `8ah9_A`)
  appear in two datasets each, so it's **554 evaluated proteins over 552 unique
  stems**. If your counts say 552, you deduplicated on the wrong key.
- **Score the whole resolved universe over the shared candidate pairs** the GT
  universe defines — that's what makes MarinFold and the structure baselines
  comparable. Don't invent a different candidate set.
- **Ground truth is fixed**: pyconfind side-chain contacts on the experimental
  structure, `native_only`, degree ≥ 0.001, sequence-separation ≥ 6 — identical to
  the contacts-v1 training documents. Don't redefine it per run.

## Inputs

The one thing that varies per invocation is the checkpoint. Accept either:
- a GCS **Levanter** path: `gs://…/checkpoints/<wandb-run-name>/checkpoints/step-N/`
  (needs export, step 1), or
- an **HF export**: `…/hf/step-N/` (skip export).

If the user names a W&B run but not a path, the checkpoint layout convention is
`checkpoints/<wandb-run-name>/step-<N>/` on the bucket — find it there or ask.

## Going further (optional)

- **Stronger inference than pairwise:** exp82's **rollout+resample** (rank pairs by
  sampled-completion vote frequency, optionally with a pairwise tiebreak) edges out
  pairwise on this eval but costs many rollouts per protein. Use exp82's
  `score_rollout_resample_eval.py` + `build_comparison_table.py` if the ask is
  "best possible LM-only number" rather than "standard eval of this checkpoint".
- **Single-panel high-res plot:** exp82's `plot_all_only.py` renders one range of
  the comparison boxplot at publication resolution (PNG + vector PDF).
