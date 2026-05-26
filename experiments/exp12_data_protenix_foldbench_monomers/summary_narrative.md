# Summary slides — Protenix on FoldBench monomers (single-seq vs MSA)

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide.
     Keep this current as the experiment progresses. -->

## What we're doing

Collect distogram + structure (mmCIF) outputs for Protenix v2 on a
subset of the FoldBench protein-monomer set, in single-sequence mode
and with MSAs. Compute per-protein MAE (vs expected distances from
the distogram) and dRMSD (CA-CA pairwise distances) and produce a
comparison plot of single-seq vs MSA. Modal-hosted fan-out across
`(protein × {single_seq, msa})` on H100. Distograms captured via a
forward hook on `runner.model.distogram_head`.

## Why

Eventual goal: compare MarinFold's distograms head-to-head against
Protenix in both modes. This experiment supplies the Protenix
reference numbers under conditions MarinFold will be scored on. No
hypothesis on Protenix's numbers beyond "single-seq performs worse
than MSA."

## Reproducibility pins

FoldBench at commit `4273f687…`. `protenix==2.0.0`,
`torch==2.7.1`, `gemmi==0.6.7`, Python 3.11.5. Protenix v2 weights
from the HF mirror (`TMF001/pxdesign-weights`). MSAs pre-computed
once via Protenix's `colabfold` backend (ColabFold MMseqs2 API),
persisted to the weights Volume so reruns are deterministic and
free. 5 seeds × 8 diffusion samples per (protein, mode); top-1 by
Protenix's `ranking_score`.

## Results — distance metrics on CB-CB (in-range pair set)

GT in `[2.31, 21.84] Å` (Protenix v2's distogram range).

| Mode       | MAE CB        | dRMSD CB       |
|------------|---------------|----------------|
| single_seq | 2.69 / 2.96   | 3.75 / 4.15    |
| msa        | 0.47 / 0.39   | 0.79 / 0.66    |

(mean / median, in Å, lower is better)

## Results — contact regime (GT ≤ 8 Å)

| Mode       | MAE CB (contacts) | dRMSD CB (contacts) |
|------------|-------------------|---------------------|
| single_seq | 3.56 / 3.97       | 5.49 / 6.18         |
| msa        | 0.40 / 0.32       | 0.73 / 0.54         |

## Results — structure & CASP contact precision

Median Kabsch CA-RMSD: single_seq **14.4 Å**, msa **1.18 Å** —
global-fold-correct in MSA mode, global-fold-wrong otherwise.
CASP contact precision (long range) @ top L: msa **0.99**,
single_seq 0.28. LDDT-CA: msa 0.95, single_seq 0.42.

## Conclusion

On 100 FoldBench monomers, Protenix v2 + MSA produces near-native
structures (median CA-RMSD 1.18 Å, dRMSD 0.84 Å) — a regime where
downstream MarinFold comparison is meaningful. Single-sequence mode
degrades sharply on natural proteins (CA-RMSD 14.4 Å); only a
viable MSA-free baseline for designed peptides. The distogram is a
strong distance signal when the model is otherwise performing well;
MSA-mode distogram MAE-CB is 0.47 / 0.39 Å (in-range filter).

For MarinFold cross-comparison, score against the same pair sets —
intersection of Protenix's [2.31, 21.84 Å] with MarinFold's
[0, 32 Å] is [2.31, 21.84 Å]. Raw outputs uploaded to HF at
`open-athena/MarinFold/data/protenix-foldbench-monomers/`.
