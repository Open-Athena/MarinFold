# Summary slides — exp: curate 30%-held-out experimental complexes and evaluate contacts plus Helico structures

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide.
     Keep this current as the experiment progresses. -->

## Question

Can FoldBench PPI support a useful experimental-complex benchmark under a 30% identity threshold?

We compare a strict component-held-out claim with a pair-held-out claim, then freeze a FoldBench-only set released after the inherited Protenix v1 training cutoff.

## Two holdout claims

A qualifying alignment has exact identity >=30% over >=50% of the shorter sequence.

Component-held-out rejects a dimer when either chain has any training homolog. Pair-held-out rejects it only when one complex-training document has two chain instances that can be assigned one-to-one to homologs of both candidate chains.

The pair rule tests a new relationship between chain families. It permits prior single-chain exposure.

## Survival

The strict rule leaves 0/184 scoped FoldBench dimers and 1/1,929 PINDER dimers after locally auditable MarinFold arms; the last PINDER target is present in Helico fine-tuning.

The pair rule leaves **35 FoldBench + 183 PINDER = 218** after both MarinFold complex arms. These include 55 homodimers and 163 heterodimers.

The conservative Helico same-PDB pair screen leaves **32 + 65 = 97** candidates.

## Confirmation and evidence

The broad complex search hit its 100,000-result median cap. We repeated all 220 initial survivors at a 1,000,000-result cap; its median was 86,868 and it found two additional PINDER exclusions.

Exact MMseqs backtraces provide integer identity counts. Commands, hashes, compact witnesses and final status tables are committed. Complete compressed alignments and logs are published in the public MarinFold HF bucket.

## Decision

Use FoldBench under the explicit **pair-held-out** claim. Its release-date design protects the Helico comparison from inherited Protenix v1 pretraining exposure. We still remove three targets with a paired homolog in Helico's later fine-tuning pool.

PINDER is not part of the frozen benchmark.

## Frozen FoldBench evaluation

The 35 MarinFold-clean pairs become **30 natural dimers** after the three Helico fine-tuning exclusions and manual removal of two de novo binder targets.

The set contains 13 homodimers and 17 heterodimers in 26 connected 30%-identity groups. One five-target ubiquitin-related group is kept intact. The deterministic split has **8 development and 22 test targets**; both cuts have the same mean length (448.25 vs 448.27 residues).

Every target ships with its exact FoldBench assembly, canonical two-chain input, label-to-author residue map, resolved cross-chain candidate universe, and contacts-v1 pyconfind ground truth.

## Evaluation contract

Contact evaluation ranks only resolved cross-chain residue pairs. R is the number of degree >=0.001 pyconfind interface contacts; no within-chain sequence-separation filter is applied across chains. Contact-budget choices use development only; the 22-target test cut is read once for the final result.

The structural bundle uses FoldBench's native layout for Helico/DockQ and retains the exact two scored chains even when a deposited assembly has extra copies. Model sample selection must use model confidence; ground-truth best-of-N remains an oracle diagnostic.

The complete 12.3 MB bundle is public at `hf://buckets/open-athena/MarinFold/data/evals/exp350_foldbench_pair_holdout/v1`.
