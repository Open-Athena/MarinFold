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

_(Filled in as experiments land.)_

### Baseline

_(Filled in after the baseline run completes.)_

## Conclusion

_(Filled in once at least one algorithm clears the +10% bar.)_
