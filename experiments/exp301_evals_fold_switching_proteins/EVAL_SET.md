# How exp301 landed on these 68 fold-switching pairs

Short version: **93 pairs come from one published table; 68 survive a three-part
gate that asks whether a pair can discriminate two folds at all.** Nothing was
hand-picked, and every exclusion is recorded with its reason in
[`data/premise_gate.csv`](data/premise_gate.csv).

## Where the proteins come from

The set is **not** ours. It is the benchmark the Porter lab built and has used
across a series of papers on AlphaFold and fold switching, taken unmodified.

| | |
|---|---|
| **Table** | `supporting tables/TableS1.xlsx` in [`github.com/ncbi/AF2_benchmark`](https://github.com/ncbi/AF2_benchmark) |
| **Archived at** | Zenodo [10.5281/zenodo.13221957](https://doi.org/10.5281/zenodo.13221957) |
| **Primary citation** | Chakravarty D, Schafer JW, Chen EA, Thole JF, Ronish LA, et al. (2024). *AlphaFold predictions of fold-switched conformations are driven by structure memorization.* **Nature Communications** 15:7296. [doi:10.1038/s41467-024-51801-z](https://doi.org/10.1038/s41467-024-51801-z) · [PMC11344769](https://pmc.ncbi.nlm.nih.gov/articles/PMC11344769/) |
| **The set's origin** | Chakravarty D, Porter LL (2022). *AlphaFold2 fails to predict protein fold switching.* **Protein Science** 31:e4353. [doi:10.1002/pro.4353](https://doi.org/10.1002/pro.4353) |
| **How it was assembled** | Porter LL, Looger LL (2018). *Extant fold-switching proteins are widespread.* **PNAS** 115:5968–5973. [doi:10.1073/pnas.1800168115](https://doi.org/10.1073/pnas.1800168115) — an exhaustive PDB search for near-identical sequences with different secondary structure, yielding ~100 literature-validated fold switchers |
| **Framing we borrowed** | Lee M, Porter LL (2026). *Fold-switching proteins push the boundaries of conformational ensemble prediction.* [arXiv:2601.01740](https://arxiv.org/abs/2601.01740) — frames fold switching as **contact redistribution**, which is why a contact model is the right instrument |

Each row of TableS1 gives two PDB entries with chain — `Fold1` and `Fold2`,
e.g. `6c6sD` / `2ougC` — plus the amino-acid sequence of the region that
switches. The pairs have mean 99% / median 100% sequence identity by
construction: they are the *same protein* solved in two different folds.

Using this table rather than curating our own set is deliberate. It is the set
AlphaFold was measured on, so our numbers sit beside Porter's without a
selection argument in between.

## The funnel

```
 93   pairs in TableS1                     (97 spreadsheet rows, 4 of them blank)
 93   built with no error                  177 of 178 PDB entries came from the
                                           local RCSB mirror; only 1fzp was fetched
 68   pass the premise gate                <- the eval set
```

**All 93 built.** No pair was lost to a parsing, chain-extraction or
contact-calculation failure. That matters: a silent build failure would have
selected on structure quality.

## The gate, and why each criterion exists

A pair is only useful if it can distinguish the two folds. Three criteria, each
fatal on its own; 25 pairs fail at least one.

| criterion | threshold | pairs failing | why |
|---|---|---:|---|
| **`min(\|A\|, \|B\|)`** | ≥ 10 | 17 | `A` and `B` are the contacts unique to each fold. If one side has almost none, the two structures are the same fold observed twice and there is nothing to measure. |
| **chain coverage** | ≥ 50% of the reference | 14 | The two chains must actually overlap. A pair whose chains cover different domains has a common universe far smaller than the protein, and its "unique" contacts are mostly residues the other structure never resolved. |
| **switching region located** | found in the reference | 3 | Without it there is no way to compute the FS-restricted metrics, which are what separate real fold switching from ligand or domain motion. |

Attributed to a single first-listed reason, the 25 break down as 17 / 6 / 2.

Excluded pairs are **not deleted** — every one is a row in `premise_gate.csv`
with its measured values and its reason, so the exclusions can be audited or
re-litigated with a different threshold.

## What the 68 are

| tier | definition | identical-sequence | homolog |
|---|---|---:|---:|
| **A** | both folds are two chains of **one PDB entry** | 6 | 0 |
| **B** | X-ray / X-ray, two entries | 47 | 0 |
| **C** | NMR or cryo-EM involved | 12 | 3 |

Tier A is the cleanest arm in the whole experiment: one crystal, so identical
construct, conditions, resolution and refinement. TableS1 contains **8** such
pairs and 6 survive the gate. The `homolog` class is the 3 pairs whose two
chains are not the same sequence (point mutants or homolog pairs); they are
flagged, never pooled silently with the identical-sequence pairs.

Medians over the 68: L = 273, common universe = 239 residues, |A| = 70,
|B| = 57, Jaccard between the two folds = 0.515.

## Why this set can carry a conclusion

The gate answers "can this pair discriminate?", not "does the model do well on
it" — no criterion touches model output, so the set is fixed before any
inference runs.

And the premise is checked, not assumed. Two contact maps always differ
somewhat; the question is whether these differ *more than crystallography alone
would produce*. [`build_noise_floor.py`](build_noise_floor.py) measures that from
the same entries — two chains of the same sequence in the same fold inside one
crystal — giving a median Jaccard of **0.873** against the fold-switch pairs'
**0.515**. All 68 sit below the replicate median. That floor independently
reproduces [#224](https://github.com/Open-Athena/MarinFold/issues/224)'s DsbA
crystal-replicate figure of 0.85–0.88 on a completely different set of proteins.

## Reproducing it

```bash
uv run python prepare_inputs.py     # TableS1 -> data/foldswitch_universe.jsonl + premise_gate.csv
uv run python build_noise_floor.py  # -> data/noise_floor.csv
```

`prepare_inputs.py` downloads TableS1 itself and caches it, so the only local
dependency is the RCSB mirror at `/data/tim/af3-db/mmcif_files` (and it falls
back to fetching from RCSB for anything missing).
