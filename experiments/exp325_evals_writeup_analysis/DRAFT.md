---
title: Sampling protein contacts from a single sequence
slug: marinfold-single-sequence-contacts
author: Open Athena
date: 2026-09-23
published: false
math: false
toc: false
tags: [MarinFold, proteins]
summary: A figure-first look at contact prediction, structural accuracy, and useful sampling diversity.
---

Accurate folding remains harder for proteins with shallow benchmark MSAs. Improving here could also help protein design. These depth counts do not establish how many homologs appeared in a predictor's training data.

Depth panels now use effective depth, Neff at 80% identity, on the same alignments.
[Definition and comparison with raw depth](NEFF_ANALYSIS.md).

AlphaFold2, AlphaFold3 and Boltz-2 use the shared benchmark MSAs here, with templates disabled.

![Existing predictors across Neff tiers](plots/01_predictors_neff.png)

More AF3 sampling brings 8oxk_A from TM 0.472 to 0.947, close to ESMFold2’s
existing 0.940. Two low-depth proteins remain below 0.8 after 1,000 runs.
These AF3 runs use ten recycles and fixed benchmark MSAs without templates;
the baseline is five seeds × five samples, the expansion 1,000 seeds × one.
[Protocol and per-protein plots](AF3_SAMPLING.md).

![AlphaFold3 sampling at low MSA depth](plots/01b_af3_sampling.png)

ESMFold2 still has the higher mean TM on these five proteins: 0.854 versus
0.719 for pTM-selected AF3 at 1,000 runs (0.681 with official ranking).

![AF3 sampling versus the other predictors](plots/01c_af3_context.png)

The existing AF3 baseline is stronger than ESMFold2 in the three deeper Neff
bins. Extra sampling covers only the original five raw-depth<10 proteins.

![TM-score across Neff tiers](plots/01d_af3_depth_context_neff.png)

Helico does much better with just L/2 randomly sampled true contacts. All other pairs stay unknown; Helico receives no MSA.
Mean GDT-TS is 0.872 / 0.839 / 0.844 / 0.841 across the four Neff tiers.

![Helico with oracle contacts](plots/02_oracle_neff.png)

On the original five raw-depth<10 proteins, random L/5 true contacts give mean GDT-TS 0.622;
L/2 gives 0.906, versus 0.932 for the full oracle map. Ten contacts give 0.362.

![Sparse oracle contacts, per protein](plots/02e_oracle_budget.png)

![Sparse oracle contacts versus existing predictors](plots/02f_oracle_budget_context.png)

Can MarinFold extend a few known contacts? With 10 true seed contacts, filling
to L/2 raises mean GDT-TS from 0.362 to 0.469, but helps only three of five
proteins. With L/5 seeds, all five improve; the mean rises from 0.622 to 0.747.

![Seeded contact completion, per protein](plots/02g_seed_completion.png)

The added contacts also become more accurate as the seed set grows.
[Protocol and paired results](SEED_COMPLETION.md).

![Accuracy of newly predicted contacts](plots/02h_seed_contact_precision.png)

Aside: ranking by pTM puts the oracle first for four low-depth proteins and fifth for one.

![Oracle versus ESMFold2 contact maps](plots/02b_confidence.png)

Among plausible candidates, TM-score versus pTM shows a stronger relationship for one protein and weaker relationships for the other four.

![TM-score versus Helico pTM](plots/02c_accuracy_confidence.png)

This suggests generating candidate contact sets directly from sequence, trained on structures from existing predictors.

![MarinFold](plots/03_method.png)

Our current 1.47B-parameter model saw 248.584B raw tokens. Here is its contact accuracy, followed by the structures Helico produces from its top-L/2 contacts.

![Contact R-precision across Neff tiers](plots/04_contacts_neff.png)

Precision at L/2 uses a sequence-length-based contact budget.

![Precision at L/2 contacts across Neff tiers](plots/04_contacts_pl2_neff.png)

[Metric details](PL2_ANALYSIS.md) · [L/5 comparison](PL5_ANALYSIS.md).

Ranking structure-derived contacts by predicted lDDT product times contact
degree improves every predictor's mean P@L/5. Confidence alone usually hurts,
except for single-sequence Protenix. [Paired comparison and both rankings](CONTACT_CONFIDENCE.md).

![Confidence times degree, precision at L/5](plots/04d_pl5_plddt_degree_neff.png)

![Long-range precision at L/5 with confidence times degree](plots/04d_pl5_plddt_degree_neff_long.png)

MarinFold also clears the Sequence KNN baseline.

![MarinFold versus KNN across Neff tiers](plots/04b_knn_neff.png)

![From predicted contacts to structures across Neff tiers](plots/05_folding_neff.png)

The top-L/2 prediction arm matches the oracle's contact budget.
Mean GDT-TS is 0.478; the L/2 oracle reaches 0.843.
[Exact protocol](MARINFOLD_L2.md) · [L, L/2 and L/5 budget comparison](MARINFOLD_L5.md).

Low-depth proteins remain the challenge. Consensus R-precision is 0.561; oracle selection of an individual sample gives 0.528. The maps are distinct, but useful whole-map alternatives remain limited.

![Consensus versus oracle selection](plots/06_sampling.png)

The same comparison at P@L/5 again favors consensus.

![Consensus versus oracle selection at L/5](plots/06_sampling_pl5.png)

At L/2, consensus reaches 0.714 versus 0.570 for oracle best-of-100.

The same panel includes the standard predictors at the L/2 contact budget.

![Sampling choices and standard predictors at L/2](plots/06_sampling_pl2.png)

[L/2 analysis and long-range variant](SAMPLING_L2.md) · [L/5 details](PL5_ANALYSIS.md) · [Poster-ready files](POSTER_NEFF.md).

Next: better candidate diversity, inference-time search, and post-training.

<!-- Placeholder prose. Copy figure captions from theme.py when integrating into
the site; each states the population, uncertainty, conditioning and oracle limits.
Use data/summary.csv and data/paired_deltas.csv for exact values.
This draft embeds PNGs for GitHub. See RUNBOOK.md for the corresponding
interactive website embeds. -->
