# Summary slides — Low-MSA / structurally-novel eval proteins (issue #65)

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide.
     Keep this current as the experiment progresses. -->

## What we're doing

Sourcing candidate eval structures that are deliberately **far from
MarinFold's AFDB training set**, for issue #65 (continuation of #41). The #41
Foldseek train-similarity tool showed FoldBench-100 is almost entirely close to
training (48 redundant / 51 same-fold / 1 novel). To actually probe
generalization we need candidates from the novel / shallow-MSA regime, which a
depth-agnostic PDB sample barely contains.

## Why these three sources

A single-sequence model has the most to prove where MSA-based models lose
their coevolution signal (shallow MSA) and where the fold is novel. Three
sources hit that, each for a different reason:

- **PDB de novo proteins** — no evolutionary lineage (Neff ~= 1), often novel
  folds, mostly outside UniProt/AFDB.
- **CASP14/15 free-modeling domains** — the community "no usable template, few
  homologs" gold standard; blind and temporally honest.
- **CAMEO hard targets** — a rolling weekly stream of difficult monomers.

## What we fetched

Three idempotent fetch scripts sharing one manifest schema (so the sources
stack and join onto exp12/exp41 CSVs):

- de novo: RCSB search → **396** experimental monomers (40-400 aa, res <= 3.0)
  out of 2,007 in the DE NOVO PROTEIN class.
- CASP FM: **32** FM/FM-TBM domains (17 CASP14 + 15 CASP15); **26** resolved
  (19 from the public domain tarballs + 7 clipped out of the deposited PDB
  entry for oligomeric / late-release targets), 6 with no released structure
  recorded without coords (not silently dropped).
- CAMEO hard: **35** hard targets over the latest 1-year window, references
  pulled from RCSB by PDB id.

## Fold novelty vs the training set

Foldseek-TM each candidate vs the 1.33M AFDB-24M train reps (exp41 tool).
**Novel folds: CASP FM 50% / CAMEO hard 34% / de novo 6%** — versus
FoldBench-100's 1/100. De novo designs cluster near known folds structurally
but are sequence-novel (only 41% have a >=30%-id training-sequence match).

## MSA depth (the named axis)

ColabFold MMseqs2 MSA per candidate (no Modal; the search is on the free
ColabFold server either way) -> Neff. **221/454 (49%) have Neff < 10** — the
shallow-MSA regime where a single-sequence model has the most headroom. The
assessor-released CASP FM monomers are fold-novel but homolog-rich (deep MSA);
the oligomeric / late-release FM targets recovered from the PDB are mostly
orphan-MSA.

## The headline: fold novelty x MSA depth

The 2-D label populates the cell FoldBench could not: **33 candidates are
novel-fold AND shallow-MSA** (21 de novo, 6 CAMEO, 6 CASP FM). The
per-protein table (`data/candidate_2d_label.csv`) carries all three axes
(fold novelty, sequence leakage, MSA depth) so the eval can stratify
MarinFold vs MSA baselines. Next: accuracy on these strata.
