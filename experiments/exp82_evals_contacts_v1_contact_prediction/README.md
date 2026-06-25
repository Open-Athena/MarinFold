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

### Canonical benchmark proteins (1QYS / 7BNY / 1UBQ) — heatmaps + seeding

Built fresh contacts-v1 documents (sequence + pyconfind ground-truth side-chain
contacts) for the canonical test proteins from prior evals — Top7 (1QYS),
7BNY (chain A), ubiquitin (1UBQ) — via `contacts-v1 generate` (experimental PDB
structures, not AFDB). Docs: [`data/benchmark_docs.parquet`](data/benchmark_docs.parquet);
analysis: `benchmark_analysis.py`.

**Contact-probability heatmaps** ([`plots/legacy_step11999/heatmap_{1QYS,7BNY,1UBQ}.png`](plots/legacy_step11999/)) —
predicted contact score vs ground truth. They explain *why* the contacts-v1
model is weak:
- **contacts-v1 model** → a smooth **sequence-separation gradient** (bright near
  the diagonal, decaying out); it has learned "closer in sequence ⇒ more likely
  contact" but **not the specific off-diagonal contact structure**.
- **prior model** (P(CA–CA ≤ 8 Å)) → sharp **structured off-diagonal blobs** that
  track the real contact clusters — it genuinely predicts structure.

**AUC vs # ground-truth contacts seeded into the prompt**
([`plots/legacy_step11999/auc_vs_seeded.png`](plots/legacy_step11999/auc_vs_seeded.png)) — seed N true contacts,
then measure AUC of ranking the *remaining* (sep ≥ 12) contacts vs decoys:

| #seeded | 0 | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|---|
| 1QYS | 0.49 | 0.54 | 0.59 | 0.68 | 0.75 | 0.87 | 0.94 |
| 7BNY | 0.53 | 0.56 | 0.58 | 0.57 | 0.73 | 0.79 | 0.86 |
| 1UBQ | 0.47 | 0.50 | 0.62 | 0.54 | 0.83 | 0.81 | 0.95 |

Unconditionally (0 seeds) the model is **at chance** (~0.5), but AUC climbs to
**~0.86–0.95 by 32 seeds**. So the model learned the **joint / co-occurrence
structure** of contact maps (contact *completion* — given some contacts it ranks
correlated ones well) but **not de novo contact prediction from sequence**. That
is consistent with: a quick run learned contact-map statistics, not folding —
and it's exactly the regime where exp27-style iterative seeding *can't*
bootstrap, because the N=0 starting point is chance.

**Poly-Ala control** (dashed lines in the plot) — re-run with every residue in
the prompt replaced by `<ALA>`, so the model sees only the protein *length* plus
the seeded contacts, no real sequence:

| #seeded | 0 | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|---|
| 1QYS poly-Ala | 0.43 | 0.48 | 0.52 | 0.60 | 0.72 | 0.83 | 0.90 |
| 7BNY poly-Ala | 0.40 | 0.49 | 0.55 | 0.59 | 0.68 | 0.73 | 0.80 |
| 1UBQ poly-Ala | 0.46 | 0.42 | 0.53 | 0.53 | 0.76 | 0.74 | 0.86 |

The poly-Ala curves track the native ones with the **same shape**, sitting only
**~0.05–0.13 AUC lower** and still climbing to ~0.80–0.90 at 32 seeds. So the
"seeding helps" effect is **almost entirely contact-map topology**, not
sequence: given some contacts, the model ranks correlated ones from the contact
*geometry* alone. The native sequence contributes a **small, consistent** boost
(real but minor) — further evidence the model leans on contact-map statistics
rather than reasoning from the sequence.

## Re-run on the tuned #61/#75 model (eval loss 2.7566) — the inference search, redone

exp82's open follow-up #1 was *"re-run on #61's tuned model."*
[#89](https://github.com/Open-Athena/MarinFold/issues/89) exported that model
(`prot-exp75-cv1-1_5b-e8-lr1e-3-wd0p2-v1-bc3084` step-35679) and showed it is far
stronger than the #67 quick model (long-range ranking AUC 0.62 → 0.88) — but it
scored **`pairwise` only**, taking exp82's "smarter decoders don't help" as
settled. That verdict was measured on the *weak* model; exp27 got **+30% from
iteration on a strong base model**. So this PR re-runs the full inference-algorithm
search on the strong model.

**Methodology — select on dev, never on test.** Inference algorithms are
*selected* on a 16-protein **FoldBench dev** set ([`prepare_foldbench_dev.py`](prepare_foldbench_dev.py)
→ `data/foldbench_dev.jsonl`; 100 ≤ L ≤ 250, fixed seed-0 sample). Held out for the
final evaluation: the other 84 FoldBench proteins, the 454 denovo/CASP/CAMEO
proteins, and the entire contacts-v1 `test` split. Candidate universe = the
**resolved** residues; ground truth = pyconfind contacts (sep ≥ 6) — defined
exactly as #89. The shared harness gained a resolved-universe option + R-precision,
and rollout's generation budget changed from `3·|GT|+60` (a ground-truth leak) to
`4·L+64`.

### Dev: `rollout` > `pairwise` > `iterative` — a flip from the weak model

[`eval_search_foldbench_dev.py`](eval_search_foldbench_dev.py); raw:
[`data/results_foldbench_dev_step35679.txt`](data/results_foldbench_dev_step35679.txt).
Mean over 16 dev proteins:

| method | long P@L | long P@L/2 | long P@L/5 | long R | P@#gt |
|---|---|---|---|---|---|
| **rollout** | **0.158** | **0.211** | **0.305** | **0.192** | **0.233** |
| pairwise | 0.131 | 0.165 | 0.226 | 0.160 | 0.204 |
| iterative | 0.096 | 0.144 | 0.213 | 0.132 | 0.202 |
| random | 0.013 | 0.011 | 0.009 | 0.010 | 0.013 |

The order **inverts** vs #67: `rollout` beats `pairwise` on every metric (+20%
R-precision), and `iterative` now **hurts** (−18% R). Mechanism: `iterative`
commits the model's own top-0.5L–2.5L predictions as fixed context, and at
~13–33% precision that seeds mostly *false* contacts, which the model's learned
co-occurrence prior then propagates — exactly the regime exp82's seeding curve
predicted would fail (the model completes maps well from *true* contacts, but
cannot yet self-seed). `rollout` is variance reduction over 100 sampled
contact-sets; its gain is largest at the very top of the ranking (P@L/5 +35%).

### Sampling sweep — a wash ([`sweep_rollout_dev.py`](sweep_rollout_dev.py))

Raw: [`data/results_rollout_sweep_dev.txt`](data/results_rollout_sweep_dev.txt).
Sweeping temperature (1.0 / 0.7 / 0.5), top-p (0.90 / 0.95 / 1.0), and a
domain-aware **top-k = L/5** moves long-range R-precision by < ±0.006 — within the
noise band for 16 proteins. **T = 0.5 clearly hurts** (P@L/5 0.257 vs 0.283;
over-sharpening collapses the vote). Default T = 1.0 / p = 0.95 is near-optimal:
the lever is the *method*, not the sampling distribution's shape.

### Resampling the document per rollout — a small, free bonus ([`eval_rollout_resampled_dev.py`](eval_rollout_resampled_dev.py))

Raw: [`data/results_rollout_resampled_dev.txt`](data/results_rollout_resampled_dev.txt).
Each rollout draws a *fresh* contacts-v1 realization (resample the N-terminus start
+ the `<pX> <AA>` statement order — #89's test-time augmentation), so the frequency
vote averages over the document nuisance as well as the sampling noise. Because all
realizations of a protein share prefix length, this costs ~nothing extra on the GPU
(`Scorer.sample_completions`). Mean over 16 dev proteins:

| method | long P@L | long P@L/2 | long P@L/5 | long R | P@#gt |
|---|---|---|---|---|---|
| pairwise | 0.135 | 0.176 | 0.248 | 0.163 | 0.203 |
| rollout | 0.160 | 0.217 | 0.312 | 0.197 | 0.231 |
| **rollout + resample** | **0.162** | **0.228** | **0.331** | **0.204** | **0.242** |

Resampling improves every metric (+0.002–0.019; R +0.007) — small but consistent,
and free. It is smaller than #89's +0.05 *pairwise* TTA because `rollout` already
ensembles, so the document nuisance is largely averaged out. **Settled recipe:
`rollout + resample`** at default sampling.

### Full curated-set evaluation (554 proteins): vs every other predictor

The dev recipe is scored on the **whole #74/#78 curated set** (FoldBench-100 + 454
denovo/CASP/CAMEO) by **exp89's exact `compute_metrics`** — the *same* metric code
as the structure predictors, so every bar is comparable (our exp82 `metrics()`
disagrees with it by up to 0.4/protein from float16 tie-breaking, so it can't be
mixed in). Pipeline: [`score_rollout_resample_eval.py`](score_rollout_resample_eval.py)
saves per-pair vote-score matrices → [`build_comparison_table.py`](build_comparison_table.py)
computes metrics with exp89's copied-verbatim code → [`plot_comparison.py`](plot_comparison.py)
draws the bars ([`plots/cmp_*_by_config_and_range.png`](plots/), plus by MSA-depth
tier and fold novelty).

**Tie-break refinement (raised in review).** The vote score is an integer 0–N, so
~⅔ of candidate pairs tie at 0 votes — a tie mass that, ordered arbitrarily, drags
ranking **AUC** below pairwise's despite far stronger top-K precision. Breaking ties
with the pairwise log-prob — `combined = votes + min-max(pairwise) ∈ [0, 0.5)`, so
votes stay the primary key — reorders that tail informatively. It is **free** (the
pairwise score matrices already exist) and recovers AUC at **zero top-K cost**.

Long-range, mean over 554 (exp89 metric):

| predictor (long-range) | R-precision | contacts@L | AUC |
|---|---|---|---|
| **MarinFold rollout+resample +tiebreak** | **0.353** | **0.231** | **0.898** |
| MarinFold rollout+resample (plain) | 0.355 | 0.232 | 0.851 |
| MarinFold ×10 ens · pairwise (#89) | 0.315 | 0.209 | 0.899 |
| MarinFold pairwise (#89) | 0.269 | 0.188 | 0.881 |
| Protenix-v2 · single-seq (structure) | 0.466 | 0.302 | 0.815 |
| Protenix-v2 · MSA (structure) | 0.628 | 0.414 | 0.935 |
| ESMFold (single-seq) | 0.732 | 0.418 | 0.892 |
| ESMFold2 (single-seq) | 0.769 | 0.443 | 0.916 |

**rollout+resample is the best LM-only inference** — long-range R-precision **0.355**
and contacts@L **0.232**, well above pairwise (0.269 / 0.188, what #89 used) and the
K=10 pairwise ensemble (0.315 / 0.209). The **tie-break** lifts AUC **0.851 → 0.898**
(above pairwise's 0.881 and ESMFold's 0.892, level with the ensemble) at no top-K
cost — so the headline recipe is **`rollout+resample +tiebreak`**: rollout's top-K
with pairwise's global ranking. It still **trails every structure predictor on
top-K** (ESMFold2 0.769 / 0.443; Protenix-MSA 0.628 / 0.414) — better inference
narrows the LM-vs-structure gap from the sequence side but does not close it; that
needs a stronger model. Per-model summary:
[`data/eval_comparison_summary.csv`](data/eval_comparison_summary.csv).

### Cost ([`data/eval_full_timings.csv`](data/eval_full_timings.csv), [`plots/eval_full_resample_time_vs_length.png`](plots/eval_full_resample_time_vs_length.png))

`rollout+resample` is **~50 s/protein** mean on one A5000 at n=100 (median 44 s,
p95 98 s, max 225 s at L=738), scaling ~linearly with L (the full 554 took 8.4 h);
`pairwise` is ~0.3 s. Rollouts terminate cleanly — **0 % hit the `4·L+64` cap** —
each emitting ~2–2.5·L tokens / ~80–125 contacts before `<end>`. So the top-K gain
costs ~150× the per-protein compute of pairwise: cheap in absolute terms
(decoder-only, no retraining), but a real cost at eval-set scale. The score-matrix
re-run (adaptive batch up to 64) did all 554 in **3.9 h**; the pairwise tie-break is
**free** (arithmetic on the existing matrices).

## Conclusion

**Headline (strong model).** Re-running the inference search on the tuned #61/#75
model (eval loss 2.7566) **flips exp82's original verdict**: with a strong base
model, better inference helps a lot. The recipe — **`rollout` voting + per-rollout
document resampling, with pairwise tie-breaking** — is the **best LM-only inference**
on the 554-protein curated eval set: long-range R-precision **0.355** and contacts@L
**0.231** (vs pairwise 0.269 / 0.188, what #89 used; and the K=10 pairwise ensemble
0.315 / 0.209), with AUC **0.898** once the tie-break recovers it from 0.851. A free,
decoder-only gain (no retraining). It still **trails every structure predictor on
top-K** (ESMFold2 0.769 / 0.443; Protenix-MSA 0.628 / 0.414) — better inference
narrows the LM-vs-structure gap but does not close it; that needs a stronger model.
`iterative`, marginally *best* on the weak model, now *hurts* (self-seeding the
model's own ~13–33%-precision predictions backfires). Net: tuning made the
**decoder a real lever again**.

**Original finding (#67 quick model).** The quick contacts-v1 1.5B model (#67) has
only a **weak** contact signal (~2× random at ranking long-range contacts), and
**better inference doesn't rescue it** — iterative refinement helps marginally,
rollout voting not at all. The bottleneck is the base model, not the readout
algorithm.

**What the model actually learned** (benchmark heatmaps + seeding curve): a
**sequence-separation prior** plus the **co-occurrence structure** of contact
maps — given some true contacts it ranks correlated ones well (AUC →~0.9 at 32
seeds) — but **not de novo contact prediction from sequence** (unconditional AUC
≈ chance; its heatmap is a diagonal gradient, not the real off-diagonal
structure). It learned contact-map *statistics*, not folding.

**The #67 hypothesis was not met:** the prior contacts-and-distances-v1 1.5B
predicts contacts **~2–4× better** than the new quick contacts-v1 model on the
same proteins — so the simple/quick run did *not* beat the previous 1.5B at
contact recapitulation. That's expected given the prior model's ~50k-step / larger-corpus
training vs exp67's 12k steps; the lever is a stronger model (more training /
the carefully-tuned #61), not a cleverer decoder. The two harnesses
(`eval_contact_prediction.py`, `eval_prior_model_contacts.py`) are the reusable
deliverable — re-runnable against any future contacts-v1 / contacts-and-distances-v1
checkpoint.

**Open follow-ups:** (1) ✅ **done in this PR** — re-ran the search on #61's tuned
model (see the section above): the gap to the prior 1.5B is closed and the
inference-algorithm verdict flips (`rollout+resample` > `pairwise` > `iterative`);
(2) larger protein set + true CASP top-L normalization — also addressed by the
554-protein curated-set eval above; (3) common contact-definition ground truth
(compute CB–CB GT) for a fully apples-to-apples prior comparison; (4) confirm the
recipe on the contacts-v1 `test` split (kept untouched for a final check) and
push the rollout+resample run to the iris/vLLM-TPU path for scale.
