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
  to sample 10 `entry_id`s, returns them plus their pinned
  AFDB v4 `.cif` URLs.
- `eval_notebook.ipynb`: orchestrates the eval interactively.
  Imports the existing `parse.iter_parsed_structures` and
  `inference.predict` / `_query_pairs`. For each protein, computes
  GT CA-CA matrix from coordinates and runs the model with the
  shared prefix + per-pair `<distance>` tails. Plots side-by-side
  heatmaps (GT vs. prediction) and a residual heatmap, plus a
  scatter of expected vs. GT distance.
- Initial smoke run on this machine's RTX A5000 (24 GB) with the
  `1B` model from [`MODELS.yaml`](../../../../MODELS.yaml).

## Success criteria

1. Notebook runs end-to-end on a single GPU in under ~15 minutes
   for 10 small (\<150 res) proteins.
2. Sampling is deterministic: re-running the selection cell gives
   the same 10 entry_ids.
3. Each protein produces three artifacts: GT heatmap, predicted
   expected-distance heatmap, residual heatmap. Saved to `plots/`.
4. Macro CA-CA MAE across the 10 proteins is finite and within ~2x
   of the in-training FoldBench numbers (the test split is
   easier than FoldBench monomers, so we expect lower if anything).

## Results

Run on an RTX A5000 (24 GB), 2026-05-18, seed=0, `max_seq_len=150`.
The notebook (`eval_notebook.ipynb`) holds the executed cells +
embedded plots; this section is the prose summary.

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

## Conclusion

The `1B` model predicts unseen AFDB CA-CA distance maps with
**macro MAE 3.29 Å zero-shot**, consistent with the ~3-4 Å
hypothesis and very close to the in-training FoldBench-monomer
benchmark (which sees ~3.6 Å on real PDB targets, slightly harder
than in-distribution AFDB). The heatmaps look right: the diagonal
neighborhood is tightly matched, off-diagonal contacts are
placed roughly correctly, and the residual is dominated by the
saturated-bin ceiling at >32 Å. The notebook is the deliverable
— a future "did the new model learn?" check is just: add a
nickname to `MODELS.yaml` and re-run.
