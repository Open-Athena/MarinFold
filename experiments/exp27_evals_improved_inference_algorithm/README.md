---
marinfold_experiment:
  issue: 27
  title: "exp: identify an improved inference algorithm"
  kind: evals
  branch: exp/27-improved-inference-algorithm
---

# exp: identify an improved inference algorithm

**Issue:** [#27](https://github.com/Open-Athena/MarinFold/issues/27) · **Kind:** `evals` · **Branch:** `exp/27-improved-inference-algorithm`

## Question

Can we find an inference-time algorithm that improves MarinFold 1B's
mean LDDT on a 10-protein subset of the FoldBench monomers — without
fine-tuning, without changing the test set or scoring code, without
external models, and without cheating on the proteins themselves?

## Hypothesis

Yes. The current readout is naive: prompt with the sequence and read
all 64-bin distance probabilities for every (i, j) in parallel. There
is no autoregressive generation, no use of contact statements, and no
inference-time search. Multi-rollout aggregation, seeded-contact
bootstrapping, contact-then-distance prompting, or simple sampling
ensembles should be able to clear a +10% mean LDDT bar within the
runtime budget.

## Background

- Issue #20 / `experiments/exp20_evals_marinfold_1b_foldbench/` ran
  the naive readout against all 100 FoldBench monomers; aggregate
  mean `lddt_distogram_cb` was **0.27** (median 0.23). That's the
  baseline shape we need to improve on.
- Issue #12 / `experiments/exp12_data_protenix_foldbench_monomers/`
  collected the Protenix GT mmCIFs we score against, plus the
  per-protein `manifest.csv` we draw the train set from.
- Document-structure spec: exp1
  (`contacts-and-distances-v1`). The vocabulary is the constraint —
  any new inference algorithm has to stay inside the prompt grammar
  the 1B model was trained on
  (`<begin_sequence> <AAs> <begin_statements> [<*-range-contact> <p_i> <p_j>]* <distance> <p_i> <p_j> <atom_i> <atom_j>`).

## Approach

Train set: 10 proteins, sampled uniformly at random from the
exp20 manifest **after filtering to `n_residues <= 400`**, with
`random.Random(27).sample(pool, 10)` and frozen to
[`data/train_proteins.csv`](data/train_proteins.csv). The cap exists
purely so we can iterate locally on V100 fp32 — without it the seed
draws a 738 aa protein whose naive readout costs ~9 hours wall-clock
by itself (V100 sm_70 has no native bf16 and no FlashAttention, so
fp32 is the only safe dtype; fp16 produces NaN logits on this bf16-
trained checkpoint, see naive_inference.py's smoke test). 86 of the
100 FoldBench monomers are ≤400 aa, so the cap discards ~14 of them
from the sampling pool. We do NOT look at the discarded or unsampled
proteins for the duration of this experiment.

Scoring is byte-identical to exp20: `score_marinfold.py` is copied
verbatim, distogram-only metrics, intersection range `[2.31, 21.69]`
Å, CB-CB / CA-for-GLY/UNK. Headline metric for the +10% bar is the
point `lddt_distogram_cb` (issue says "mean LDDT" — point LDDT is
exp20's headline column too).

Runtime metric is **total wall-clock time** to score all 10 train
proteins end-to-end, parallelised across the 8 local V100s. The
runtime constraint is `total_wall_seconds <= 5 * baseline_wall_seconds`.

Every algorithm we try gets one row in `data/experiments.tsv`:

```
experiment_id  description  mean_lddt  median_lddt  total_wall_seconds
runtime_ratio_vs_baseline  mean_lddt_delta_pct  notes
```

Baseline row (`baseline_naive`) is the current naive distogram readout
from exp20 — copy-forked here as `naive_inference.py` so the experiment
dir is self-contained.

### Files

- `select_train_proteins.py` — seed=27 sample of 10 from
  `protenix_data/.../manifest.csv`, writes `data/train_proteins.csv`.
- `fetch_protenix_data.py` — copy from exp20; pulls Protenix GT
  mmCIFs + manifest into `protenix_data/`.
- `canonical_sequence.py` — copy from exp20; reads the
  `entity_poly_seq` 1..N sequence + representative-atom convention.
- `naive_inference.py` — one-protein vLLM worker (baseline
  algorithm; same as exp20's `_predict_distogram`). Binds to one GPU
  via the `--gpu` arg, writes
  `outputs/<stem>/{distogram.npz, provenance.json}`.
- `run_baseline.py` — 8-GPU work-stealing orchestrator; launches up
  to 8 `naive_inference.py` subprocesses (one per V100), records
  total wall time.
- `score_marinfold.py` — copy from exp20; scores `outputs/<stem>/`
  against the GT. Used unchanged by every algorithm we try.
- `append_experiment_row.py` — append a row to
  `data/experiments.tsv`. Computes runtime ratio / LDDT delta vs
  baseline automatically.

### Compute

Local box: 8 × Tesla V100-SXM2-16GB (sm_70 / Volta). The 1B model
loads in ~30 s and fits comfortably (~4 GB bf16 weights + KV cache).
V100s don't have native bf16 — vLLM falls back to a slower software
path. If numerics drift vs the H100 / A5000 reference, fall back to
`dtype=float32`.

No Modal, no iris, no TRC — local only per the issue.

## Success criteria

1. Baseline established on the frozen 10-protein train set with
   per-protein LDDT, total wall-clock time, and a deterministic
   experiment_id in `data/experiments.tsv`.
2. At least one subsequent algorithm achieves
   `mean_lddt >= 1.10 * baseline_mean_lddt` while
   `total_wall_seconds <= 5 * baseline_wall_seconds`.
3. Final write-up identifies the best inference algorithm by name +
   the gain, and gives enough detail to reproduce it.

## Results

Full per-experiment narrative lives in
[`RESULTS_LOG.md`](RESULTS_LOG.md); aggregate rows are in
[`data/experiments.tsv`](data/experiments.tsv). Summary below.

### Baseline

`baseline_naive` (exp20's naive distogram readout, A100/bf16) on the
10-protein train set:

  mean LDDT 0.2496 · median 0.2500 · wall 1386.7 s

Per-protein range 0.151 — 0.449. Soft LDDT 0.268 (the model's
probability mass is closer to GT than its expected-value point
estimate; classic multimodal failure).

### The +10% bar (mean LDDT ≥ 0.2746): cleared

Two cheap wins each clear it on their own; stacking them gives the
first decisive result.

- **Distogram sharpening** (`p' = softmax(log(p+ε)/T)`, T=0.05). Pure
  post-process, ~30 s. Mean LDDT **0.2738, +9.68%** — just below
  the bar alone but stacks.
- **Self-bootstrapped contact seeding** (`seeded_contacts_kL1.0_min0.3`):
  one extra readout where K=L most-confident contacts from a full
  baseline are dropped into the `<begin_statements>` prefix as
  `<{range}-range-contact><pi><pj>`. Mean LDDT **0.2816, +12.83%**.
  Stack with sharpening: **0.2894, +15.94%**.

### The +50% bar (mean LDDT ≥ 0.3744): NOT cleared

Best honest algorithm: **`iter_R4_grow_05_10_15_25`** —
iterative contact-only with a growing-K schedule (kc per round =
0.5L, 1.0L, 1.5L, 2.5L; min_contact_prob=0.1; range-ordered
long → medium → short). Round 1 picks contacts from baseline. Each
subsequent round picks contacts from the previous round's distogram,
which is sharper because iteration enriches the model's high-
confidence contact pool (8baq_A goes from 3 contacts >0.5 in
baseline to 227 after iter R=3).

  mean LDDT **0.3511** · median 0.3169 · +40.66%
  (sharpening *doesn't* help here — T=1.0 wins, same pattern as
  the GT-oracle: better algos have less-spread distributions to
  rescue.)

Chain wall: 1386.7 (baseline prior — necessary because seeds must
come from an unfiltered distogram) + 2986.4 (iter R=4 grow) =
**4373 s, 3.16× baseline**. Within the 5× budget.

### GT-oracle ceiling (diagnostic only)

Seeding the model with TRUE contacts (every pair where `gt_d < 8 Å`)
gives mean LDDT **0.7167, +187%**. This tells us:
1. The model has the capacity to score >0.7 with right contacts.
2. The bottleneck for honest algorithms is contact-prediction quality
   on harder proteins, not the model's expressivity.
3. Sharpening doesn't help the oracle distogram (T=1.0 best),
   confirming the diminishing-returns pattern for sharpening as
   algorithms improve.

### What got tried and rejected

Full table in `RESULTS_LOG.md`. Headline misses:

- **Sampled contact prefixes** (idea 6, model-generated contact
  statements as prefix): +2.88% single rollout, +0.28% M=5 average.
  M=10 union: +26.92%. All worse than top-K marginal seeding because
  the model emits few contacts before transitioning to `<distance>`
  and the sampled set has lower precision than top-K.
- **Distance commits** (idea 2, one-hot rows for high-confidence
  modal distances): +12.4% — *worse* than plain seeded. One-hot rows
  zero LDDT when modes are wrong; per-pair variance kills the gain.
- **K = 2L seeded contacts**: −1.2% vs K=L. Adding lower-precision
  tail seeds poisons the readout. **Precision > count.**
- **Multi-rollout averaging** of distograms (M=5): blurs the
  distributions, 8/10 proteins regress.
- **Mixture-of-distograms** across algorithms: doesn't beat single
  best; the variants aren't differently wrong, just less wrong.
- **Per-pair max-confidence** ensembling: worse than any individual.
- **Sharpening + better algo**: the sharpening contribution shrinks
  as algorithms improve (matches GT-oracle finding).

## Conclusion

**Best honest algorithm: `iter_R4_grow_05_10_15_25` at mean LDDT
0.3511 (+40.66%).** Clears the issue's original +10% bar (raised
mid-experiment to +50%) by a wide margin; falls short of the +50%
bar by 0.023 absolute / 9 percentage points.

Per-protein status vs +50% bar (0.3744): 4/10 pass on this single
algorithm (7y5j, 7ykm, 7ur2 newly passes from iteration, 7zs2
nearly there at +0.012). The hard misses (8baq, 7uk8, 7xz3,
8cba, 8eb9, 7ylr) all gain +0.05 to +0.13 from baseline but their
oracle ceilings are 0.62 — 0.73; the model can score well on these
proteins given the right contact set but doesn't honestly identify
those contacts. The bottleneck is contact-prediction quality on
sparser-confidence (mostly larger) proteins.

Reproducing the best algorithm:

```
uv run python run_baseline.py --dtype bfloat16 --n-gpus 1
uv run python snapshot_distograms.py --to distogram_baseline_naive.npz
uv run python run_iterative.py --dtype bfloat16 --n-gpus 1 \
  --algorithm iter_R4_grow_05_10_15_25 \
  --n-rounds 4 \
  --k-contacts-per-L-per-round 0.5 1.0 1.5 2.5 \
  --k-distances-per-L-per-round 0.0 0.0 0.0 0.0 \
  --min-contact-prob 0.1 \
  --prior-name distogram_baseline_naive.npz
```

Total wall: ~4373 s on a single A100 40 GB in bf16.
