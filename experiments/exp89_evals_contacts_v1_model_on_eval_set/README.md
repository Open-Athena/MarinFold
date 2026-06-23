---
marinfold_experiment:
  issue: 89
  title: "exp: evaluate best contacts-v1 model on current eval set"
  kind: evals
  branch: exp/89-contacts-v1-model-on-eval-set
---

# exp: evaluate best contacts-v1 model on current eval set

**Issue:** [#89](https://github.com/Open-Athena/MarinFold/issues/89) · **Kind:** `evals` · **Branch:** `exp/89-contacts-v1-model-on-eval-set`

## Question

How well does the best model @eric-czech trained in
[#61](https://github.com/Open-Athena/MarinFold/issues/61) (the one with eval
loss **2.7566**) predict residue–residue contacts on our current curated eval
set, alongside every other predictor (Protenix-v2 single-seq / MSA, ESMFold,
ESMFold2)?

## Hypothesis

N/A. (Prior from [#82](https://github.com/Open-Athena/MarinFold/issues/82): the
*quick* contacts-v1 1.5B from #67 was ~chance at de-novo contact prediction.
#89 is exactly exp82's named open follow-up — *does the carefully-tuned #61/#75
model close the gap?*)

## Background

- **The eval set** ([#74](https://github.com/Open-Athena/MarinFold/issues/74) /
  [#78](https://github.com/Open-Athena/MarinFold/issues/78)) — **554 proteins**:
  FoldBench-100 + 454 exp65 low-MSA / novel-fold candidates. Ground-truth
  contacts are pyconfind side-chain contacts on the experimental structure
  (`native_only=True`, degree ≥ 0.001, sequence-separation ≥ 6) — defined
  identically to the contacts-v1 training documents. exp74/exp78 already scored
  **Protenix-v2** (single-seq / MSA) and **ESMFold / ESMFold2** on it; we splice
  those in unchanged and add MarinFold.
- **The model** — eric-czech's #75 tuning sweep winner. Found via marin branch
  `eac/plm-exp75` → W&B `eric-czech/marin` (group `exp75-contacts-v1-tune`) → GCS:
  - W&B run **`prot-exp75-cv1-1_5b-e8-lr1e-3-wd0p2-v1`** (`…-bc3084`): Qwen3 1.47B,
    **epochs 8, lr 1e-3, wd 0.2**; final `eval/contacts-v1-val/loss` = **2.756602**
    at **step 35679** (the single permanent checkpoint).
  - Levanter checkpoint: `gs://marin-us-east5/checkpoints/prot-exp75-cv1-1_5b-e8-lr1e-3-wd0p2-v1-bc3084/checkpoints/step-35679/`.
  - Tokenizer `timodonnell/contacts-v1-tokenizer@5d68a24a899f` (2845 vocab).
- **Inference algorithm** — exp82 found **pairwise** scoring extracts the most
  signal (rollout/iterative did not beat it; structured inference can't bootstrap
  when the unconditional signal is weak). We use pairwise — "the best inference
  approach identified in #82".

## Method (reproduction spec)

A three-step harness; GT and model scoring run in separate venvs (pyconfind vs
torch), and the metric step is backend-agnostic.

0. **Export the checkpoint** — [`export_contacts_v1_best_to_hf.py`](export_contacts_v1_best_to_hf.py).
   The run was never HF-exported (its `hf_save_path` was empty), so we convert
   the levanter checkpoint to HF safetensors via levanter's `export_lm_to_hf`
   (CPU), with the exp75 Qwen3 config + contacts-v1 tokenizer co-located. Result:
   `Qwen3ForCausalLM`, vocab 2845, published to the open-athena bucket at
   `checkpoints/prot-exp75-cv1-1_5b-e8-lr1e-3-wd0p2-v1-bc3084/hf/step-35679/`.
1. **GT universe** — [`prepare_gt_universe.py`](prepare_gt_universe.py). For each
   protein, run pyconfind on the GT structure (exactly as exp74/exp78) and record,
   in input-sequence coordinates, `L`, the **resolved-residue set** (the
   candidate-pair universe, identical across all predictors), and every degree>0
   contact → `data/gt_universe.jsonl`. Verified bit-identical to the saved exp78
   GT contacts.
2. **MarinFold pairwise scores** — [`score_eval_set.py`](score_eval_set.py).
   Build the contacts-v1 **sequence-section prefix** from each input sequence (the
   official deterministic `build_document` with empty contacts), then score every
   candidate pair as the symmetrized geometric-mean log-prob of `<contact> <pi> <pj>`
   (exp82 pairwise). Output: one `[L,L]` score matrix per protein.
   - Canonical numbers were produced with the **exp82 transformers scorer**
     (CUDA, local) — identical math, faster to iterate, no shared-TPU contention.
   - [`score_eval_set_vllm.py`](score_eval_set_vllm.py) is the **vLLM / iris-TPU**
     reproduction (same scoring definition, same `npz` layout; generation-logprobs,
     the path eric's `eval_protein_contacts` already runs on TPU) — see *Running*.
3. **Metrics** — [`compute_metrics.py`](compute_metrics.py). Feed each MarinFold
   score matrix into the exp74 metric over the resolved universe: precision @
   {L, L/2, L/5}, **R-precision**, and **AUC**, per range (all / short [6,11] /
   medium [12,23] / long [≥24]). **AUC** is added here for *every* predictor
   (MarinFold + Protenix-structure SS/MSA + ESMFold + ESMFold2) over the same
   universe, since the existing tables carried only precision. Plots:
   [`plot.py`](plot.py); slides: `build_summary.py` → `plots/summary.pdf`.

### Running

```bash
# (0) export — in a marin checkout (CPU; ~6 GB HF model)
cd /home/bizon/git/marin && uv run --no-sync python \
    <exp89>/export_contacts_v1_best_to_hf.py --output-dir <local>/hf_step35679

# (1) GT universe — exp78 venv has pyconfind + the staged GT structures
<exp78-venv>/bin/python prepare_gt_universe.py --out data/gt_universe.jsonl

# (2) scores — local CUDA (canonical) …
PYTHONPATH=<repo>/marinfold uv run python score_eval_set.py \
    --model <local>/hf_step35679 --out-dir _scratch/scores --timings data/timings.csv
#     … or vLLM on iris TPU (canonical-platform reproduction):
HF_TOKEN=… uv run iris --config=lib/iris/examples/marin.yaml job run \
    --region us-east5 --tpu=v5p-8 --extra=vllm --extra=tpu -- \
    python -m score_eval_set_vllm \
      --model gs://marin-us-east5/checkpoints/prot-exp75-…-bc3084/hf/step-35679 \
      --out-dir gs://marin-us-east5/eval/exp89-contacts-v1/scores

# (3) metrics + plots
uv run python compute_metrics.py --gt data/gt_universe.jsonl --scores _scratch/scores \
    --exp78-precision <exp78>/data/contact_precision_all.csv \
    --exp78-raw <exp78-raw>.parquet --exp74-raw <exp74-raw>.parquet \
    --out data/contact_precision_all.csv
uv run python plot.py --precision-csv data/contact_precision_all.csv --out plots
```

## Success criteria

Eval succeeds for all 554 eval-set proteins; AUC + contacts @ {R, L, L/2, L/5}
for MarinFold reported next to every other predictor.

## Results

All **554** proteins scored, **0 failures** (`data/timings.csv`; mean 4.9 s,
max 63 s on one A5000). Full table: `data/contact_precision_all.csv`; plots:
`plots/`; slides: `plots/summary.pdf`.

**Headline — long-range (seq-sep ≥ 24), mean over 554 proteins:**

| predictor | AUC | R-precision | contacts@L |
|---|---|---|---|
| **MarinFold-cv1 1.5B (sequence only)** | **0.881** | 0.269 | 0.188 |
| Protenix-v2 · single-seq (structure) | 0.815 | 0.572 | 0.326 |
| Protenix-v2 · MSA (structure) | 0.935 | 0.795 | 0.465 |
| ESMFold (single-seq) | 0.892 | 0.732 | 0.418 |
| ESMFold2 (single-seq) | 0.916 | 0.769 | 0.443 |

(Aggregate-range AUC: MarinFold **0.904**, Protenix-SS 0.830, Protenix-MSA
0.941, ESMFold 0.901, ESMFold2 0.923.)

**1. Ranking AUC — the sequence-only LM is competitive.** MarinFold's
long-range contact-ranking AUC (**0.881**; 0.904 aggregate) **matches ESMFold**
and **beats single-sequence Protenix-v2** (0.815), trailing only Protenix-MSA
and ESMFold2. It is **robust to MSA depth** (long-range AUC 0.87 / 0.85 / 0.91 /
0.90 over orphan / marginal / low / deep tiers) and **fold novelty**, where it
**edges ESMFold2 on novel folds** (0.81 vs 0.79). This answers exp82's open
question: careful tuning moved the model from **~chance** (the quick #67 model)
to **ESMFold-class ranking AUC from sequence alone**.

**2. Top-K precision — structure methods win decisively.** At R-precision and
contacts@L, MarinFold (0.269 / 0.188 long-range) sits **well below every
structure predictor** (ESMFold2 0.769 / 0.443; Protenix-MSA 0.795 / 0.465). It
orders contacts broadly well but does not concentrate confidence on the true top
contacts. (AUC also somewhat favours a continuous ranker — MarinFold scores
every pair — over structure-derived contact sets that tie most pairs at degree
0; the top-K precision metrics are the fairer "did you find the contacts"
comparison.)

**Heatmaps (`plots/heatmap_{1QYS,7BNY,1UBQ}.png`,
[`benchmark_heatmaps.py`](benchmark_heatmaps.py)).** For the three canonical test
proteins — 1QYS (Top7), 7BNY, 1UBQ — each figure shows ground-truth pyconfind
contacts next to the model's **P(contact)** (the probability it emits each pair
as its next contact statement). The model puts **structured off-diagonal mass
that tracks the real contacts** (β-hairpin corners, long-range bands) — *not* the
diagonal-only sequence-separation gradient exp82 saw for the near-chance #67
model. Visual confirmation of finding 1.

The vLLM/iris-TPU scorer ([`score_eval_set_vllm.py`](score_eval_set_vllm.py))
is provided as the canonical-platform reproduction (same scoring definition and
`npz` layout) but was not the source of these numbers — see *Method*.

## Conclusion

The carefully-tuned contacts-v1 1.5B (eval loss 2.7566) has a **real,
MSA-depth-robust contact signal**: from sequence alone it ranks residue–residue
contacts about as well as ESMFold by AUC (and better than single-sequence
Protenix-v2), a large jump over the near-chance #67 model that exp82 measured.
It is **not yet a high-precision contact predictor** — its top-K precision
(contacts@L, R-precision) is far below the structure-based methods, so for
"high-confidence contact" use it trails ESMFold / ESMFold2 / Protenix-MSA. Net:
tuning closed the *ranking* gap but not the *top-K precision* gap; the next
lever is sharpening the model's top predictions (or a larger / longer-trained
model), not the decoder.

_(Suggested follow-ups, for a human to decide: (a) protenix-distogram AUC needs
the saved distograms — only its precision is included here; (b) a higher-quality
calibration / sharpening of the top predictions; (c) the vLLM/iris-TPU run as the
canonical platform reproduction.)_
