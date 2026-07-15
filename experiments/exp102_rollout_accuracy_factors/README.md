# exp102 — what differentiates high-accuracy rollouts from average ones?

Data-generation + CPU-analysis scaffold for
[Open-Athena/MarinFold#102](https://github.com/Open-Athena/MarinFold/issues/102).

Contact prediction from the contacts-v1 LM has a wide per-target spread across
rollouts (see #98: F1 ranges a lot over n=1000 rollouts of one target). #102 asks
**what makes the good rollouts good** — are they longer? do they emit contacts in
a different order (long-range first? most-confident first? certain residues
first?)? within one target, is there a single informative contact the good
rollouts guess early?

This experiment **generates the data those questions need** so the analysis runs
on CPU (Allen has no GPU), and ships a starter that answers each question on the
generated data.

## Why new data (vs. reusing exp98's 1M rollouts)

exp98 already published 1M rollouts, but its per-rollout `pred` is **sorted by
(i,j)** and it kept only a whole-rollout `nll` — so **emission order and
per-contact confidence are gone**, and those are exactly what half of #102 needs.
Everything order-*independent* (length, composition, within-target
informative-contact) is answerable straight from exp98's data; this experiment
adds the order-*dependent* half.

The regeneration is a surgical variant of the exp98 worker (same model, same
`resample` recipe, `rollout_metrics.py` scoring copied verbatim), changing only:
1. keep contacts in **emission order** (no `sorted`), and
2. record each contact's **emission logprob** (the 3 sampled-token logprobs of its
   `<contact> <pI> <pJ>` statement), from the model's *raw* next-token
   distribution (`output_logits`, so top_p/top_k warping doesn't distort it).

Output rows keep `entry_id`+`r`, so they **join 1:1** to exp98's
`rollout_metrics_all.parquet`.

## What it runs on

- **Model:** the tuned contacts-v1 1.5B, eval loss 2.7566
  (`prot-exp75-cv1-1_5b-e8-lr1e-3-wd0p2-v1-bc3084`, `hf/step-35679`), staged from
  the open-athena bucket.
- **Targets:** a **length-stratified subset of ~200** of exp98's 1000 train
  targets (L 38–504), × **1000 rollouts** each (~200k rollouts) — enough spread
  across length for across-target trends, and 1000/target for within-target
  variance.
- **Hardware:** a single local GPU (RTX A5000, 24 GB) via **HF transformers**
  (not vLLM) — exact raw per-token logprobs, no vLLM/TPU rope/bf16 gotchas. SDPA
  attention; adaptive batch by length; ~1.5–2.6k tok/s; full run ~half a day,
  resumable (skips completed targets).

Model loading was validated by teacher-forcing five known exp98 rollouts (L up to
504) and matching their recorded `nll` to within 0.1–0.6% — rules out the
transformers-5/levanter rope mismatch (which would blow up at long L).

## Files / pipeline

| step | script | env | out |
|---|---|---|---|
| A select | `select_targets.py` | CPU | `data/targets.parquet` (~200, stratified from exp98) |
| B prompts | `gen_prompts.py` | CPU (marinfold) | `data/prompts/<entry>.parquet` (resampled prefixes) |
| C generate | `gen_rollouts_worker_hf.py` | **GPU** | `data/runs/<run>/rollout_metrics_ordered/<entry>.parquet` |
| D publish | `publish_to_hf.py` | CPU | bucket `data/contacts-v1-rollouts-ordered-exp102/` |
| E analyze | `analysis_starter.py` | **CPU** | printed worked examples (Allen's entry point) |

`rollout_metrics.py` (parsing/scoring, with the new ordered/token-level parser)
is unit-tested in `tests/`.

```bash
uv sync
uv run python select_targets.py --n 200 --seed 102
uv run python gen_prompts.py --targets data/targets.parquet -k 1000 --out data/prompts
uv run python gen_rollouts_worker_hf.py --out data/runs/full --n-rollouts 1000
HF_TOKEN=<open-athena-scoped> uv run python publish_to_hf.py --run data/runs/full
uv run python analysis_starter.py --source hf     # anyone, no GPU
```

## Output schema — `rollout_metrics_ordered.parquet`

One row per rollout, keyed `entry_id`+`r`:

- `pred` — flattened `[i0,j0,i1,j1,…]` in **emission order** (deduped first-wins,
  seq-index, sep≥6). **Not sorted** (unlike exp98) — index = prediction rank.
- `pred_logprob` — length `n_pred`; contact k's emission logprob.
- `nll`, `nll_per_tok`, `n_gen_tokens`, `finished`, `n_pred`, and per-band
  (`all`/`short`/`med`/`long`) `*_npred/_tp/_prec/_rec/_f1` — as exp98.

`targets.parquet` carries `sequence` + `gt_contacts`, so contact separation,
residue identities, and correctness are all derivable on CPU.

## Analysis

Two equivalent entry points, both CPU-only and auth-free:

- **`analysis_colab.ipynb`** — self-contained Colab notebook: pulls the published
  data anonymously, builds the tables, and renders a **plot per #102 question**.
  Defaults to a 40-target demo subset for snappy cells (`USE_ALL_TARGETS = True`
  for the full ~200). Open it in Colab:
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Open-Athena/MarinFold/blob/main/experiments/exp102_rollout_accuracy_factors/analysis_colab.ipynb)
- **`analysis_starter.py`** — the same analyses as a script (`--source hf` reads
  the bucket; `--source local` reads a run dir; `--max-targets N` to iterate).

Both build `contacts_frame` (one row per rollout×contact: rank, sep, band,
logprob, correct, rollout F1 quartile, residue identities) and cover: length
correlation, order-bias by band × F1-quartile, confidence/order signature,
amino-acid enrichment, and the within-target most-informative-contact search.
Scaffolds to extend, not conclusions.
