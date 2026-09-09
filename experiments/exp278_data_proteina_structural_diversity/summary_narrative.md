## Proteina initial screen: production held

We filed experiment #278 and completed a 1,536-backbone screen on Iris H100s. The end-to-end pipeline produces valid contacts-v1 documents, and class conditioning changes secondary-structure composition.

494 documents survive quality, evaluation decontamination and fine-cluster capping. The million-document run is held: yield is below plan and the proposed diversity target is unproven. This is an initial-screen result, not completion of all proposed pilot arms.

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

A/T labels, higher noise, the 400M triangle model, checkpoint overlap and a larger pilot remain untested. Production is held for redesign; issue #278 and draft PR #282 retain the code, timing records and small result artifacts.
