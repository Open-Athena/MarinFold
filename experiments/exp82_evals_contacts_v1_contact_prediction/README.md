---
marinfold_experiment:
  issue: 82
  title: "exp: contact-prediction inference algorithms for the contacts-v1 1.5B model"
  kind: evals
  branch: exp/82-contacts-v1-contact-prediction
---

# exp: contact-prediction inference algorithms for the contacts-v1 1.5B model

**Issue:** [#82](https://github.com/Open-Athena/MarinFold/issues/82) · **Kind:** `evals` · **Branch:** `exp/82-contacts-v1-contact-prediction`

## Question

How well can the contacts-v1 1.5B model (trained in #67) predict residue–residue contacts from sequence, and which inference algorithm extracts the most signal from it?

## Hypothesis

The model has learned a **weak-but-real** contact signal (early probe in #67: teacher-forced ranking AUC ≈ 0.59 vs 0.5 chance; free-generation ≈ random due to set-generation pathologies). We expect:
- Long-range contact precision well **above random** but **far below** a strong contact predictor (it was a quick/simple run; contacts-from-sequence is the folding problem).
- Structured inference — rollout-frequency voting and exp27-style **iterative growing-K** refinement — may extract more signal than naive pairwise scoring, the way exp27's iteration helped the prior model.

## Background

- **#67** — trained the contacts-v1 1.5B model (`protein-contacts-1_5b-3.5e-4-contacts-v1-unmasked-3b5cf2`, published to the open-athena bucket at `…/hf/step-11999/`). The early contact probe there motivated a proper eval harness.
- **#27** (`experiments/exp27_evals_improved_inference_algorithm/`) — developed sampled-rollout + iterative growing-K inference for the *previous* contacts-and-distances-v1 model (distogram readout). This experiment adapts that idea to contacts-only.
- contacts-v1 document format: `marinfold/.../document_structures/contacts_v1/` (ground truth contacts are in the document text; `min_seq_separation=6`).

## Approach

Score/rank candidate residue pairs (free-running generation is degenerate for an unordered set) and report standard CASP contact metrics. Inference variants:

- **pairwise** — autoregressive P(`<contact> <pi> <pj>` | sequence), symmetrized as the geometric mean of P(i)·P(j|i) and P(j)·P(i|j).
- **rollout** — N sampled completions of the contact section; rank pairs by occurrence frequency (ensemble voting).
- **iterative** — exp27-style growing-K: commit the top-ranked contacts as a prefix, re-score the rest conditioned on them, repeat over `[0.5L, 1L, 1.5L, 2.5L]`.
- **random** baseline.

Metrics: precision of the top-{L, L/2, L/5} predicted **long-range** (seq-sep ≥24) and medium+long (≥12) contacts, over held-out `test`-split proteins. Ground truth from the document text — no pyconfind needed.

- Eval harness: `eval_contact_prediction.py` (drafted in #67; this experiment is its basis).
- Runs on GPU (transformers); the model is the published `hf/step-11999` checkpoint.

### Running

The host runs CUDA 12.2, so `pyproject.toml` pins `torch==2.5.1+cu121` (the
exp67 training venv has CPU-only torch — don't reuse it). Fetch the model and
run:

```bash
cd experiments/exp82_evals_contacts_v1_contact_prediction
uv venv && uv sync                       # CUDA torch + transformers + gcsfs + pyarrow

# Get the published model locally (HF buckets aren't HfFileSystem-addressable):
hf buckets cp -r \
  hf://buckets/open-athena/MarinFold/checkpoints/protein-contacts-1_5b-3.5e-4-contacts-v1-unmasked-3b5cf2/hf/step-11999 \
  ./model

uv run python eval_contact_prediction.py --model ./model \
    --methods pairwise,rollout,iterative -n 25 --n-rollouts 100
```

Ground truth is read from the held-out `test`-split parquet on GCS
(`…/exp53_contacts_v1_5x/documents/test/`) — no pyconfind, no structures needed.

## Success criteria

- A working, reusable contact-prediction eval harness comparing the inference variants.
- Baseline long-range precision@top-{L,L/2,L/5} numbers for the contacts-v1 1.5B model, with the random baseline for context.
- A clear comparison of pairwise vs rollout vs iterative (does structured inference help?).
- **Stretch:** head-to-head vs the prior contacts-and-distances-v1 1.5B (the original #67 hypothesis).

## Results

**First run (2026-06-15):** `hf/step-11999`, 24 held-out `test` proteins
(seq-len 50–150, ≥5 contacts), 100 rollouts. Raw output:
[`data/results_step11999_n25.txt`](data/results_step11999_n25.txt).

Mean precision of the top-{L, L/2, L/5} ranked contacts (long = seq-sep≥24;
medlong = ≥12):

| method | long P@L | long P@L/2 | long P@L/5 | medlong P@L | P@(#gt) |
|---|---|---|---|---|---|
| pairwise  | 0.014 | 0.013 | 0.016 | 0.017 | 0.023 |
| rollout   | 0.013 | 0.013 | 0.008 | 0.013 | 0.020 |
| **iterative** | **0.017** | **0.014** | 0.014 | **0.020** | 0.022 |
| random    | 0.007 | 0.003 | 0.005 | 0.009 | 0.010 |

**Read:**
- All three methods beat random by **~2×** (e.g. iterative long P@L 0.017 vs
  random 0.007) — a *weak-but-real* contact signal, consistent with the #67
  probe (balanced teacher-forced AUC ≈ 0.59).
- **Structured inference barely helps**: `iterative` is marginally best (~+20%
  over `pairwise` on long/medlong P@L), but `rollout` does **not** beat
  `pairwise`. This is unlike exp27, where iteration gave +30% — there the base
  model already had a strong signal to refine; here it doesn't.
- **Absolute precision is very low (~1–2%).** Note the metric caveat: most of
  these small proteins have far fewer than L *long-range* contacts, so
  precision@top-L is capped low even for a perfect predictor — the
  **model-vs-random ratio (~2×)** is the meaningful quantity, not the absolute
  value.

### Head-to-head vs the prior contacts-and-distances-v1 1.5B (the #67 hypothesis)

`eval_prior_model_contacts.py` scores the **prior** 1.5B model
(`protein-contacts-1_5b-distance-masked-70f8f5`, step-49999) on the **same** 24
proteins / ground truth / metrics, via its native readouts: `statements`
(`<{range}-range-contact> <pi> <pj>`) and `distance` (P(CA–CA ≤ 8 Å)). Raw:
[`data/results_prior_1_5b_n24.txt`](data/results_prior_1_5b_n24.txt).

| model · method | long P@L | long P@L/2 | long P@L/5 | medlong P@L | P@(#gt) |
|---|---|---|---|---|---|
| contacts-v1 · pairwise | 0.014 | 0.013 | 0.016 | 0.017 | 0.023 |
| contacts-v1 · iterative | 0.017 | 0.014 | 0.014 | 0.020 | 0.022 |
| **prior c-and-d-v1 · statements** | **0.028** | **0.030** | **0.026** | **0.048** | 0.047 |
| prior c-and-d-v1 · distance | 0.018 | 0.013 | 0.009 | 0.033 | **0.050** |
| random | 0.008 | 0.004 | 0.005 | 0.009 | 0.011 |

**The prior 1.5B wins by ~2–4×** on every band (e.g. medlong P@L 0.048 vs the
contacts-v1 model's best 0.020; long P@L 0.028 vs 0.017). Its **contact-statement
readout beats its own distance readout** — even though it was distance-masked,
the `<…-range-contact>` statements carry the better contact signal (the CA–CA
distance proxy is also weaker against pyconfind side-chain ground truth).

Caveats: (1) the prior model trained ~50k steps on the larger
contacts-and-distances-v1 corpus vs exp67's 12k steps — this is "more training",
not necessarily a worse recipe; (2) the two models use different contact
*definitions* (contacts-v1 = pyconfind side-chain; prior = CB–CB ≤ 8 Å), and we
score both against the contacts-v1 ground truth — yet the prior still wins
*despite* that home-field disadvantage, which strengthens the result.

## Conclusion

The quick contacts-v1 1.5B model (#67) has only a **weak** contact signal
(~2× random at ranking long-range contacts), and **better inference doesn't
rescue it** — iterative refinement helps marginally, rollout voting not at all.
The bottleneck is the base model, not the readout algorithm.

**The #67 hypothesis was not met:** the prior contacts-and-distances-v1 1.5B
predicts contacts **~2–4× better** than the new quick contacts-v1 model on the
same proteins — so the simple/quick run did *not* beat the previous 1.5B at
contact recapitulation. That's expected given the prior model's ~50k-step / larger-corpus
training vs exp67's 12k steps; the lever is a stronger model (more training /
the carefully-tuned #61), not a cleverer decoder. The two harnesses
(`eval_contact_prediction.py`, `eval_prior_model_contacts.py`) are the reusable
deliverable — re-runnable against any future contacts-v1 / contacts-and-distances-v1
checkpoint.

**Open follow-ups:** (1) re-run on #61's tuned model when it lands (does careful
tuning close the gap to the prior 1.5B?); (2) larger protein set + true CASP
top-L normalization (here capped by small-protein contact counts); (3) common
contact-definition ground truth (compute CB–CB GT) for a fully apples-to-apples
prior comparison.
