---
marinfold_experiment:
  issue: 9
  title: "exp: zero-shot distance heatmap eval on 10 random test proteins (1B model)"
  kind: evals
  branch: exp/9-test-distance-heatmaps
---

# exp: zero-shot distance heatmap eval on 10 random test proteins (1B model)

**Issue:** [#9](https://github.com/Open-Athena/MarinFold/issues/9) · **Kind:** `evals` · **Branch:** `exp/9-test-distance-heatmaps`

## Question

How well does the [`1B`](https://huggingface.co/timodonnell/LlamaFold-experiments/tree/main/marin-experiments.protein-contacts-1b-3.5e-4-distance-masked-7d355e)
model predict the CA-CA distance map of held-out (`contacts-and-distances-v1-5x`, `split=test`)
AFDB structures it has never seen? We want a small, interactive,
runnable record of this so anyone training a new model can drop
their checkpoint in and see how it stacks up.

**Kind:** `evals`

## Hypothesis

On the test split, zero-shot CA-CA MAE should be comparable to the
in-training distogram benchmark on the small hand-curated FoldBench
set (~3-4 Å). The heatmaps should look like noisy versions of the
ground-truth distance maps — diagonal band correct, off-diagonal
contacts roughly placed, with detail degrading at long range.

## Background

Existing eval surfaces are either (a) the training-time tagged
loss in `protein_distogram_eval.py` (CA-CA on 13 fixed PDB targets,
ran every `steps_per_eval`) or (b) the offline distogram script
`eval_protein_distogram.py` in marin (one PDB, vLLM, top-K logprob
renormalization over the 64 `<d_X.X>` bins).

This experiment is the smallest useful "did the model learn the
test distribution?" check: 10 random AFDB entries from
`contacts-and-distances-v1-5x` split=test, fetched fresh from the
EBI AFDB endpoint, scored CA-CA with vLLM-driven inference using
the prefix-shared batch pattern that exp1 already implements in
`inference._query_pairs`. Selection is RNG-seeded so re-running
gives the same 10 entries.

## Approach

- `select_test_proteins.py`: loads `timodonnell/protein-docs` (subset
  `contacts-and-distances-v1-5x`, split `test`), uses a fixed seed
  to sample 10 `entry_id`s, returns them plus the currently-live
  AFDB `.cif` URLs (`model_v6.cif` as of writing).
- `inference_helpers.py`: shared vLLM helpers (model load,
  distance-bin resolution, prompt build, per-pair / full-matrix
  queries, GT contact extraction, sampled-pair MAE). Both notebooks
  use it.
- Both notebooks resolve the model via the top-level
  [`MODELS.yaml`](../../MODELS.yaml) — bump `MODEL_NICK` (or add
  a new entry) and re-run to retarget a fresh checkpoint.
- Initial smoke run on this machine's RTX A5000 (24 GB) with the
  `1B` model.

### Variant 1 — `eval_notebook.ipynb` (zero-shot heatmaps)

For each protein, runs CA-CA inference at every (i, j), i < j with
a shared base prefix (`<task> <begin_sequence> <AAs>
<begin_statements>`) and per-pair `<distance> <p_i> <p_j> <CA>
<CA>` tails. Renormalizes top-128 logprobs over the 64 distance-bin
tokens, takes the expected distance from the bin midpoints. Plots
GT, predicted, and `|residual|` heatmaps per protein, a pooled
expected-vs-GT scatter, and a 10×3 grid.

### Variant 2 — `contact_seeding_search.ipynb` (minimal seeded contacts)

Same 10 proteins. **Greedy-searches for the smallest set of true
long-range contacts (≤5) whose presence in the prompt drops the
sample MAE below 1.0 Å.** At each round 1..5 we try every
remaining GT long-range contact (CB-CB ≤ 8 Å, sep ≥ 24) as the
next seeded contact, measure MAE on a deterministic 500-pair
CA-CA sample, and pick the one minimizing it. Stop early when the
target is hit. After the search, the chosen contacts are re-run on
the full N×N matrix to produce final heatmaps and a comparison
grid against the zero-shot variant.

## Success criteria

1. **Variant 1**: notebook runs end-to-end on a single GPU in
   under ~15 minutes for 10 small (\<150 res) proteins; macro
   CA-CA MAE is finite and within ~2x of the in-training
   FoldBench numbers (we expect ≤ the FoldBench number since the
   test split is in-distribution AFDB).
2. **Variant 2**: search notebook completes per-protein in a
   bounded time (~5 min each at the worst case) and records, for
   each of the 10 proteins, the smallest k ∈ {0..5} that brings
   sample MAE below 1.0 Å (or reports that the target is not met
   at k=5).
3. Sampling is deterministic: re-running the selection cell
   gives the same 10 entry_ids in both notebooks.
4. Per-protein artifacts are saved to `plots/`; per-protein
   metrics to `data/`.

## Results

Both runs on an RTX A5000 (24 GB), 2026-05-18, seed=0,
`max_seq_len=150`, model `1B` from `MODELS.yaml`. The executed
notebooks (`eval_notebook.ipynb`, `contact_seeding_search.ipynb`)
hold the cells + embedded plots; this section is the prose
summary.

### Variant 1 — zero-shot (`eval_notebook.ipynb`)

**Macro CA-CA MAE = 3.29 Å** across 10 AFDB proteins (40,442
evaluable pairs).

| entry_id | n_res | n_pairs | MAE (Å) |
|---|---:|---:|---:|
| AF-A0A1N7G8C0-F1 | 60 | 1,900 | 1.03 |
| AF-A0A6B0Z5B5-F1 | 112 | 10,186 | 2.20 |
| AF-A0A1C5BRX1-F1 | 72 | 3,886 | 2.85 |
| AF-R7G5V6-F1 | 132 | 8,774 | 3.21 |
| AF-A0A1H0PBF4-F1 | 94 | 6,658 | 3.23 |
| AF-A0A1G4A0Q3-F1 | 114 | 7,722 | 3.43 |
| AF-A0A2P2Q6H4-F1 | 55 | 2,286 | 3.74 |
| AF-A0A7W4UDR7-F1 | 131 | 15,228 | 4.05 |
| AF-E6UJZ8-F1 | 112 | 9,758 | 4.35 |
| AF-C6S3E2-F1 | 140 | 14,486 | 4.81 |

Per-protein heatmaps are in `plots/`. Source CSV:
`data/per_protein_mae.csv`. Sampling sometimes skips entries
whose UniProt accession has been retired from AFDB since the
training data was built — at seed=0 four such entries
(AF-A0A352P7D8-F1, AF-A0A1K1WRR2-F1, AF-A0A6M1XDR8-F1,
AF-A6D053-F1) were skipped; the next ten valid entries were used.

Wall time on a single A5000: ~5 minutes end-to-end including
model staging, with vLLM at ~85% GPU util during inference.

### Variant 2 — minimal seeded-contact search (`contact_seeding_search.ipynb`)

Greedy search over GT long-range contacts (CB-CB ≤ 8 Å, sep ≥ 24),
cap 12 random candidates per round, sample MAE on 500
deterministic CA-CA pairs, target MAE < 1.0 Å, MAX_CONTACTS = 5.
Wall time ~25 min on a single A5000.

**No protein reached the 1.0 Å target within 5 seeded contacts.**
All 8 proteins with at least one long-range GT contact terminated at
k=5; the other 2 (AF-A0A1C5BRX1-F1, AF-A0A1N7G8C0-F1) have no
long-range contacts in their AFDB structure (they're small,
extended folds), so the search reports k=0.

| entry_id | n_res | n_cand | k | full MAE @ k=0 | full MAE @ k=5 |
|---|---:|---:|---:|---:|---:|
| AF-A0A6B0Z5B5-F1 | 112 | 59 | 5 | 2.20 | **1.21** |
| AF-A0A1H0PBF4-F1 | 94 | 37 | 5 | 3.23 | **1.29** |
| AF-A0A2P2Q6H4-F1 | 55 | 17 | 5 | 3.74 | **1.50** |
| AF-A0A1G4A0Q3-F1 | 114 | 45 | 5 | 3.43 | 2.14 |
| AF-R7G5V6-F1 | 132 | 21 | 5 | 3.21 | 2.16 |
| AF-A0A7W4UDR7-F1 | 131 | 116 | 5 | 4.05 | 2.91 |
| AF-E6UJZ8-F1 | 112 | 113 | 5 | 4.35 | 2.91 |
| AF-C6S3E2-F1 | 140 | 156 | 5 | 4.81 | 3.34 |
| AF-A0A1C5BRX1-F1 | 72 | 0 | 0 | 2.85 | 2.85 |
| AF-A0A1N7G8C0-F1 | 60 | 0 | 0 | 1.03 | 1.03 |

Pattern: each seeded contact reliably drops MAE, but the curve
flattens — 5 contacts cuts MAE roughly in half (e.g. 4.81 → 3.34
for the worst case) but doesn't reach 1 Å for any of these
proteins. The cleanest reductions are on small / well-folded
proteins where the long-range contacts strongly constrain the
geometry (AF-A0A2P2Q6H4-F1 at 55 residues goes from 3.74 to
1.50 Å with 5 contacts). See `plots/contact_search_trace.png`
for the per-protein curves and `plots/contact_search_grid.png`
for the final heatmaps.

Source CSV: `data/contact_search_summary.csv` (selected
contacts as `i-j; i-j; …`).

## Conclusion

**Zero-shot.** The `1B` model predicts unseen AFDB CA-CA distance
maps with **macro MAE 3.29 Å zero-shot**, consistent with the ~3-4
Å hypothesis and very close to the in-training FoldBench-monomer
benchmark (~3.6 Å on real PDB targets). The heatmaps look right:
the diagonal neighborhood is tightly matched, off-diagonal contacts
are placed roughly correctly, residual is dominated by the
saturated-bin ceiling at >32 Å.

**Seeded.** **5 true long-range contacts is not enough** to reach
< 1.0 Å MAE on any of the 10 test proteins (with greedy
selection, cap 12 candidates/round). 5 contacts approximately
halves the zero-shot MAE; the best-case proteins reach
~1.2 Å. This is consistent with the in-training distogram
benchmark, where the FoldBench-monomer set needs ~50 seeded
contacts to reach ~1.4 Å. The headline number from this
experiment: **on this test split, the `1B` model needs more
than 5 long-range contacts to break 1 Å MAE.** Whether the
boundary is closer to 10 or 50 is a follow-up — extend
`MAX_CONTACTS` (and the candidate cap) and re-run.

The two notebooks are the deliverable — a future "did the new
model learn?" check is just: add a nickname to `MODELS.yaml` and
re-run them.
