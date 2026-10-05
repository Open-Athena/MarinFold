# Summary slides — exp: curate 30%-held-out experimental complexes and evaluate contacts plus Helico structures

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide.
     Keep this current as the experiment progresses. -->

## Question

Can FoldBench PPI and PINDER test support a useful experimental-complex benchmark under a 30% identity threshold?

We compare a strict component-held-out claim with a pair-held-out claim, then screen the resulting pairs against Helico's documented fine-tuning PDB pool.

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

Proceed to benchmark freezing under the explicit **pair-held-out** claim. Start with 218 MarinFold-clean pairs and use the 97-candidate conservative subset for Helico reporting.

Before inference, cluster and deduplicate assemblies, remove prior development targets, and split development/test by connected homology group. Helico's inherited Protenix pretraining remains unknown, so structural results cannot claim a fully certified end-to-end holdout.
