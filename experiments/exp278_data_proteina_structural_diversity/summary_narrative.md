## Scale generation: independent batch workers

First production submission: September 9, 21:32 UTC. Review due September 10, 15:32 UTC / 11:32 EDT. Continue generation during review.

Frozen plan: 4.82 million raw backbones at lengths 60–500, four class arms, 768 independently admitted worker queues. East02 has 256 H100s. RNO2A transfer approval remains pending.

Preserve all backbones and sequences, including rejects. Quality-pass documents require decontamination and global diversity selection before training use.

## Exact scale recipe

Compiled Proteina 200M no-triangle checkpoints: short through 250 aa, long from 251 aa. Every integer length 60–500; unconditional, alpha, beta and mixed class requests; 400 steps; one CA ProteinMPNN sequence per backbone, temperature 0.1.

ESMFold refolds each sequence (four recycles). Require CA self-consistency RMSD ≤2 Å, pLDDT ≥70 and cis-proline-aware CA geometry checks. Contacts-v1 labels use refolded geometry. Each independent single-GPU worker alternates sampling and refolding by case.

## Initial scale results — September 10, 13:20 UTC

15 h 49 m after launch: 476,976 saved backbones; 428,992 saved sequences; 427,152 committed refolds; 263,735 quality-pass provisional documents (61.7% of refolds).

Quality retention by length: 60–100 aa 82.7%; 101–200 aa 72.2%; 201–300 aa 66.2%; 301–400 aa 62.7%; 401–500 aa 53.1%. These percentages precede decontamination and global structural selection.

88 H100 pods active and 680 waiting, all batch priority; no terminal job failures at the check. The 18-hour review starts at 15:32 UTC / 11:32 EDT; keep generation running during review.

Sources: RUN_SUMMARY.md and data/scale-20260909/reports/snapshot-20260910T132043Z.json, plus its per-case CSV.

## Proteina initial screen

We filed experiment #278 and completed a 1,536-backbone screen on Iris H100s. The end-to-end pipeline produces valid contacts-v1 documents, and class conditioning changes secondary-structure composition.

494 documents survive quality, evaluation decontamination and fine-cluster capping. The user authorized a larger run to measure retention and diversity. This is an initial-screen result, not completion of all proposed pilot arms.

## What survived

Proteina CA traces → CA ProteinMPNN → ESMFold full backbone → quality checks → evaluation decontamination → structural selection → contacts-v1.

1,536 raw; 1,044 quality-pass; 523 decontaminated; 494 selected in 453 fine clusters. Retention at 60/100/200/300/400/500 aa is 53% / 53% / 34% / 21% / 21% / 11%.

All selected sequences, contact counts and endpoints validate. No documents are truncated. No generated data entered training.

## Diversity: composition control, unresolved coverage benefit

Conditioning changes alpha/beta composition, but this is not independent CATH topology assignment. Fine-cluster effective count is 0.965× a matched unconditional sample (108 per arm; subsampling range 0.881–1.057).

The 1.5× target is poorly calibrated here: the control already has about 95 effective clusters out of 108 samples, leaving a maximum median gain of only 1.14×. This screen cannot rule out a benefit from larger samples or A/T conditioning.

Next design: diversity accumulation curves and architecture/topology coverage against a frozen natural-data baseline, plus long-chain rejection analysis.

## Measured cost replaces the initial estimate

The screen and auxiliary probes consumed 4.706 H100-hours including setup and failed starts. All GPU jobs finished.

At screening yield, one million retained documents uniformly spanning 60–500 aa project to 13,382 H100-hours (17.4 days on 32 GPUs), including 20% overhead. Balancing accepted contributions across all four requested arms projects to 15,823 hours.

Illustrative $2–4/H100-hour accounting gives $26,800–53,500 for the primary projection. Fleet is prepaid; these are not contract quotes. Million-scale duplicate growth, CPU selection and storage/transfer costs remain unmeasured.

## Auxiliary probes and scope

500-aa TF32 sampling: 5.00 vs 10.16 seconds/backbone; 30/48 vs 27/48 pass quality. Matched original seeds can diverge, and MPNN seeds differ between arms: promising speed, not established quality noninferiority.

A second sequence attempt rescued 5 of 32 identical 500-aa backbones: first 18/32; best-of-two 23/32; 231 extra inference seconds before decontamination.

A/T labels, higher noise, the 400M triangle model and a larger pilot remain untested; 250/251-aa crossover canaries subsequently passed integration checks. A scale run is now authorized; issue #278 and draft PR #282 retain the code, timing records and small result artifacts.
