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

  mean LDDT 0.2496 · median 0.2273 · wall 1386.7 s

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

Best honest algorithm: **`iter_R4_grow_on_sampled_uniform_M5`** —
combines two ideas:

1. **Sampled-uniform prior.** Sample contact statements directly
   from the model (vLLM `logits_processors` masks everything but the
   3 contact-range and N position tokens). The model's natural prior
   over the 3 range tokens is heavily medium-biased (~99% at T=0.7),
   so we overwrite range-token logits to 0 → uniform 1/3 of each
   range. M=5 independent rollouts, take the UNION of unique
   (i, j) pairs as the seed prefix. Run one readout under that
   prefix. mean LDDT 0.3142 alone.

2. **Iterative contact-only with growing K**, starting from that
   sampled prior instead of the naive baseline distogram.
   kc per round = 0.5L, 1.0L, 1.5L, 2.5L. min_contact_prob=0.1.
   Range-ordered long → medium → short.

  mean LDDT **0.3564** · median 0.3560 · +42.81%

The reported 0.3564 uses the **standard expected-distance readout**
(`pred_d = sum(probs * midpoints)`) with **no sharpening** —
sharpening on top hurts (T=1.0 wins; T=0.3 gives 0.3537, T=0.1
gives 0.3520). Same diminishing-returns pattern the GT-oracle showed:
sharpening rescues high-entropy distributions from bad context;
once context is good the distributions are already sharp.

Chain wall: 2458 (sampled M=5 union prior) + 3155 (iter R=4 grow) =
**5613 s, 4.05× baseline**. Within the 5× budget.

**Why sampling helps where iteration didn't:** sampled rollouts emit
contact pairs autoregressively, so they capture *joint* structural
hints (which positions co-occur as plausible contacts) that the
independent marginal contact-probabilities (used by pure iteration)
miss. 8eb9_A — a protein where iteration alone got stuck at 0.31 —
jumps to 0.35 with the sampled prior + iteration; 7ylr_A and 7zs2_A
newly clear the +50% bar.

**Previous in-budget headline** (kept for the record):
  `iter_R4_grow_05_10_15_25` — pure iteration from a naive baseline
  prior. 0.3511, +40.66%, chain 4373 s.

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

**Best honest algorithm: `iter_R4_grow_on_sampled_uniform_M5` at mean
LDDT 0.3564 (+42.81%).** Clears the issue's original +10% bar by 4×;
falls 0.018 short of the mid-experiment-raised +50% bar (0.3744).

Per-protein status vs +50% bar: 5/10 pass on this single algorithm
(7y5j, 7ykm pass easily; 7ylr and 7zs2 newly cross the bar with the
sampled prior added; 7ur2 within 0.01). The 5 misses (8baq, 7uk8,
7xz3, 8cba, 8eb9) need +0.07 to +0.12; their oracle ceilings are
0.62 — 0.73. The bottleneck is contact-prediction quality on
sparser-confidence proteins — the model has the *capacity* but
neither sampling nor iteration can fully extract those contacts
from this checkpoint.

Reproducing the best algorithm:

```
# 1. Produce the sampled-uniform-M5 union distogram (no naive prior needed)
uv run python run_sampled_contacts.py --dtype bfloat16 --n-gpus 1 \
  --algorithm sampled_uniform_M5_union_T0.7 \
  --n-rollouts 5 --temperature 0.7 --top-p 0.9 --max-sample-tokens 900 \
  --range-strategy uniform --aggregation union
uv run python snapshot_distograms.py --to distogram_sampled_uniform_M5_union.npz

# 2. Iterate R=4 growing K on top of that prior
uv run python run_iterative.py --dtype bfloat16 --n-gpus 1 \
  --algorithm iter_R4_grow_on_sampled_uniform_M5 \
  --n-rounds 4 \
  --k-contacts-per-L-per-round 0.5 1.0 1.5 2.5 \
  --k-distances-per-L-per-round 0.0 0.0 0.0 0.0 \
  --min-contact-prob 0.1 \
  --prior-name distogram_sampled_uniform_M5_union.npz
```

Total wall: ~5613 s on a single A100 40 GB in bf16.

## Cross-protein check: held-out 10 proteins (1B)

Knobs were tuned on the train 10. To estimate overfit, picked
**10 NEW proteins** with `random.Random(42).sample(...)` from the
same FoldBench pool (`n_residues ≤ 400`, excluding the train 10) and
ran the headline algorithm with identical knobs.

| | train (10) | held-out (10) | delta |
|---|---:|---:|---:|
| baseline_naive | 0.2496 | 0.2797 | +0.030 |
| sampled\_uniform\_M5\_union (stage A) | 0.3142 | 0.3189 | +0.005 |
| combined (headline) | **0.3564** | **0.3685** | **+0.012** |
| **lift over baseline** | **+42.81%** | **+31.75%** | **−11 pp** |

**Every held-out protein gains.** Per-protein lifts range from +11.7%
(7qsj) to +54.6% (7v3o). 7y8i reaches 0.7179 (near GT-oracle range).
Mean lift drops by 11 percentage points vs train — real overfit cost
of tuning on a 10-protein set, but a modest one. **The algorithm
generalizes; we didn't just memorize knobs that flatter these
specific 10 proteins.**

Per-protein held-out results:

| stem | L | base | combined | lift |
|---|---:|---:|---:|---:|
| 7t9r |  38 | 0.3455 | 0.3980 | +15.2% |
| 7y8i |  97 | 0.4689 | **0.7179** | +53.1% |
| 7zoi | 151 | 0.2362 | 0.3083 | +30.5% |
| 7wz5 | 161 | 0.2222 | 0.3023 | +36.0% |
| 8bau | 189 | 0.2518 | 0.3270 | +29.9% |
| 8gmy | 236 | 0.2746 | 0.3139 | +14.3% |
| 7xg9 | 288 | 0.2980 | 0.4252 | +42.7% |
| 7x4p | 307 | 0.2319 | 0.3014 | +30.0% |
| 7v3o | 328 | 0.1594 | 0.2465 | +54.6% |
| 7qsj | 373 | 0.3080 | 0.3440 | +11.7% |

## Cross-model check: same algorithm on 1.5B

After rebasing onto main, `MODELS.yaml` gained a `1.5B` entry pointing
at `protein-contacts-1_5b-distance-masked-70f8f5/step-49999` in the
`open-athena/MarinFold` HF bucket. The 1.5B has 24 hidden layers
(1.5× the 1B's 16) and GQA (8 KV heads vs 32). The checkpoint name
suggests step 49999 — likely undertrained vs the 1B production
checkpoint.

Re-ran the exact same algorithm with `--model 1.5B`, output to
`outputs_1.5b/`:

| | mean LDDT | median | wall (s) | lift |
|---|---:|---:|---:|---:|
| 1B baseline (naive) | 0.2496 | 0.2273 | 1387 | — |
| 1B combined (headline) | **0.3564** | 0.3560 | 3155 | **+42.81%** |
| 1.5B baseline (naive) | 0.2627 | 0.2315 | 1866 | — |
| 1.5B sampled\_uniform\_M5\_union (stage A only) | 0.2038 | 0.2126 | 3472 | **−22.42%** |
| 1.5B combined (headline) | **0.2864** | 0.2545 | 4230 | **+9.04%** |

**The 1B algorithm does not transfer cleanly to 1.5B.** The same
pipeline gives +42.81% on 1B but only +9.04% on 1.5B. Stage A alone
is *worse* than baseline on 1.5B (−22%) — sampled-uniform contacts
appear to mislead the 1.5B more than they help.

Per-protein, 1.5B regresses on the largest proteins (7ylr: −30%,
7zs2: −26%, 7uk8: −15%) but still gains on small/mid ones (8eb9:
+78%, 8cba: +35%, 8baq: +31%). The large-protein regression is the
defining failure: those proteins also dominate the mean.

**Likely causes (not investigated further here):**
- Real architectural differences (24 layers + GQA vs 16 layers + MHA)
  produce different conditional-distance distributions.
- The `range_strategy=uniform` fix was tuned to undo 1B's
  ~99% medium-range prior. 1.5B may have a different range-token
  prior, in which case forcing uniform actively hurts. A
  re-run of `probe_range_entropy.py` on 1.5B would say.
- (The "1.5B is undertrained" hypothesis was ruled out — both 1B
  and 1.5B saw the same number of training steps per the model
  authors. So this transfer failure is about the model, not training
  budget.)

The algorithm's hyperparameters (`range_strategy`, K schedule,
`min_contact_prob`) were tuned on 1B and need re-tuning per model.

## Overfit decomposition

Combining the two generalization checks gives a clean decomposition of
where the tuning-on-train lift is rooted:

|  | drop in lift vs train | which dimension changed |
|---|---:|---|
| held-out 10 (same model, different proteins) | −11 pp | protein set |
| 1.5B on train (different model, same proteins) | −34 pp | model |

**The algorithm's tuning is ~3× more model-specific than
protein-specific.** Most of the "+42.81% headline" reflects the
algorithm exploiting features specific to *this* 1B checkpoint;
the protein-set portion is much smaller.

The +31.75% on a fresh protein set is the more honest "generalizing"
number for the algorithm at this model. The full +42.81% should be
read as "what's achievable when the algorithm's range/K knobs are
let to overfit the eval set."

## Addendum: re-tuning the algorithm for 1.5B

After establishing that the 1B-tuned pipeline only gives +9.04% on
1.5B, I re-tuned the algorithm from scratch on 1.5B using the **same
10 train proteins**, then evaluated on the 10 held-out proteins.

Two hypotheses were tested on train:

1. **Drop stage A (sampled-uniform contacts).** On 1.5B the
   sampled-uniform stage *hurts* (−22% on its own); the 99%
   medium-range bias that motivated it on 1B is a 1B-specific
   pathology.
2. **Use fewer iteration rounds.** Long proteins on 1.5B over-iterate
   under R=4. Test R=4 → R=3 → R=2.

1.5B sweep on the 10 train proteins:

| run | mean LDDT | median | wall (s) | lift vs 1.5B baseline |
|---|---:|---:|---:|---:|
| 1.5B baseline (naive) | 0.2627 | 0.2315 | 1866 | — |
| 1B headline algorithm transferred | 0.2864 | 0.2545 | 4230 | +9.0% |
| 1.5B iter R=4 grow `[0.5, 1, 1.5, 2.5]` from baseline | 0.3295 | 0.2743 | 4272 | +25.5% |
| 1.5B iter R=4 grow `[0.5, 1, 1.5, 2.0]` (smaller K_final) | 0.3320 | 0.2754 | 4187 | +26.4% |
| 1.5B iter R=3 grow `[0.5, 1.0, 1.5]` from baseline | 0.3398 | 0.2842 | 2732 | +29.4% |
| **1.5B iter R=2 grow `[0.5, 1.0]` from baseline (WINNER)** | **0.3403** | **0.3106** | **1531** | **+29.6%** |

The 1.5B winner is `iter_R2_grow_from_baseline`: two rounds, K=0.5L
then K=1.0L, min_contact_prob=0.1, prior = the naive baseline
distogram. No sampling stage. It is also faster than the 1B winner
(~1.7× over baseline vs ~4× for 1B).

### Cross-model held-out evaluation

The 1.5B-tuned algorithm applied to **both** models on the 10
held-out proteins (per-protein numbers in
`data/heldout_1B_iter_R2_scores.csv` and
`data/heldout_1.5B_iter_R2_scores.csv`):

| model | baseline | tuned (iter R=2 grow) | lift |
|---|---:|---:|---:|
| 1B | 0.2797 | **0.3314** | **+18.5%** |
| 1.5B | 0.3150 | **0.3605** | **+14.4%** |

(For reference, the original 1B-headline algorithm gives 0.3685 on
the same 10 held-out proteins, +31.75%. So when each model gets
its own tuned algorithm, 1.5B's absolute LDDT comes close to 1B's
but the relative lift is smaller — partly because 1.5B has a
stronger baseline on this set.)

**Reproducer (1.5B winner on either model):**

```sh
uv run python run_iterative.py --dtype bfloat16 --n-gpus 1 \
    --train-csv data/heldout_proteins.csv \
    --out outputs_heldout_1.5B \
    --model 1.5B \
    --algorithm heldout_1.5B_iter_R2_grow_05_10 \
    --n-rounds 2 \
    --k-contacts-per-L-per-round 0.5 1.0 \
    --k-distances-per-L-per-round 0.0 0.0 \
    --min-contact-prob 0.1 \
    --prior-name distogram_heldout_1.5B_baseline.npz
```

(`--prior-name` is the naive-baseline distogram for the same model;
generate it first with `run_baseline.py` + `snapshot_distograms.py`.)

## Library functions for downstream use

Both winning algorithms are exposed as library functions in
[`combined_algorithm.py`](combined_algorithm.py), so downstream
experiments can run them without going through the CLI / on-disk
snapshot dance:

- `predict_distogram_combined(rt, residue_names, pair_mask, ...)` —
  the 1B-tuned headline (`iter_R4_grow_on_sampled_uniform_M5`).
  Default knobs reproduce the +42.81%-on-train number byte-for-byte.
- `predict_distogram_iter_from_baseline(rt, residue_names, pair_mask, ...)` —
  the 1.5B-tuned addendum (`iter_R2_grow_from_baseline`). Default
  knobs reproduce the +29.6%-on-train number byte-for-byte.

Each returns `(probs, meta)`. `probs` is `[N, N, 64]` and is
score-equivalent to the `.npz` the experiment scripts write to
`outputs/<stem>/distogram.npz`. `meta` carries the algorithm name and
per-stage counts; pass `include_history=True` to also get every
contact/distance statement picked, or `include_intermediate_distograms=True`
to also get the per-round distograms (large).

`*_from_cif` convenience wrappers derive `residue_names` and the
LDDT-shell `pair_mask` from a Protenix GT CIF in one line. Use the
plain function with a full-upper-triangle mask when you don't have a
GT structure on hand.

Full per-protein numbers and discussion in [RESULTS\_LOG.md](RESULTS_LOG.md).
