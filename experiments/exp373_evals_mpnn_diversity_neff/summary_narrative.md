# ProteinMPNN-only diversity and Neff dependence

## What we tested

Does redesign-only training broaden contact predictions and reduce their dependence on available evolutionary information?

We compared the exp364 MPNN-only 1.5B checkpoint at step 186884 with the exp277 full-data default at step 266344. The primary cohort is 97 natural eval-val proteins; 19 designs are reported separately. Eval-test was not scored.

The original design has 300 fixed-prompt samples plus 100 matched randomized prompts per model/protein, at temperatures 0.7, 1.0 and 1.3. A requested follow-up extends temperature-1 fixed prompts to 1,000 samples, preserving every original sample and adding 162,400. All 440,800 samples are saved. Intervals bootstrap proteins, preserving model pairing, with 10,000 draws.

## Diversity increases after the controls

On natural proteins, mean fixed-prompt Jaccard distance rises from 0.708 to 0.813. The paired difference is +0.1046, with 95% CI [0.0866, 0.1229].

Matching every paired sample's contact count gives +0.1064 [0.0891, 0.1244]. Randomized prompts give +0.1008 [0.0836, 0.1189]. Long-range contacts show the same effect. Independent seed-block differences are +0.1050, +0.1042 and +0.1046.

MPNN-only generates modestly more contacts, but the long-range fraction is similar. Count, prompt and seed-block controls support a real difference in the contact-sampling distribution.

## Extending discovery and ensembles to 1,000

The extension reveals a small MPNN-only coverage advantage: 99.0% versus 98.5% for default. The paired difference is +0.50 percentage points [0.11, 0.96]. The original 300-sample difference was inconclusive.

At 1,000, MPNN-only discovers 255.6 true and 12,290.8 false contacts per protein, versus default's 254.2 true and 8,011.0 false contacts. This is 1.4 more true contacts alongside 4,280 more false contacts: 53% more false discovery.

Ensembles largely plateau after 300: default R-precision changes from 0.5394 to 0.5414, and MPNN-only from 0.4541 to 0.4549. Both gains have intervals spanning zero. Default remains higher at every plotted k. Original temperature, diversity and Neff analyses retain their original sample counts.

## Oracle best@k and mean individual F1

For each protein, oracle best@k selects the highest individual-rollout contact F1 among the first k samples. We then average equally across proteins. Selection requires ground truth; this is an upper bound, not an inference-time selector. All- and long-range oracles select independently.

On natural proteins, mean individual F1 over all 1,000 samples is 0.4115 for default and 0.3020 for MPNN-only. These are the dashed comparison lines. Oracle best@1,000 is 0.5607 versus 0.4628: paired MPNN-only deficit 0.0980 [0.0799, 0.1166]. Bands are pointwise protein-bootstrap intervals.

Sampling beyond 300 finds better individual predictions in both models. Oracle F1 gains are +0.0161 [0.0120, 0.0205] and +0.0218 [0.0168, 0.0275], respectively. Long-range best@1,000 is 0.5625 versus 0.4679. Default remains stronger in both ordinary and oracle F1.

## Neff gives contrast-specific support

Neff is redundancy-weighted depth at 80% identity, not raw MSA sequence count. Fixed tiers contain 4, 10, 28 and 55 natural proteins.

The all-range high-minus-low accuracy drop is 0.399 for default and 0.329 for MPNN-only: difference -0.0705 [-0.1026, -0.0401]. However, absolute low-Neff accuracy is 0.280 versus 0.263; MPNN-only does not improve it.

The continuous interaction is inconclusive: -0.0080 [-0.0276, 0.0165] per Neff decade. Long range, length/viral adjustment and nonviral-only slopes are also inconclusive. The largest accuracy gap occurs in the middle tiers. Four lowest-tier proteins limit generalization. No natural protein has raw MSA depth below 10.

## Patterns and mechanisms

Both models have contact dependencies beyond sample-size and contact-frequency margins. Excess Jaccard SD over exact-margin shuffled controls is 0.0197 for default and 0.0169 for MPNN-only. Half/full-chain differences are small relative to that excess. This does not identify physical states or uniquely stronger MPNN-only coordination.

Eight contact-map examples were chosen by Neff rank before inference. Some retain similar recurring patterns; others show diffuse MPNN-only frequencies and lower accuracy.

Existing sequence searches cover native AFDB/ESM-Atlas. The actual 162.6-million-document redesign corpus has no complete available sequence-search or KNN audit. Removing native documents retains structural ancestry and cannot establish an absence of retrieval or memorization.

## Secondary designs and execution checks

For 19 designs, the original Jaccard difference is inconclusive: +0.0457 [-0.0017, 0.0976]. At 1,000, oracle F1 is 0.7428 versus 0.7158, paired difference -0.0270 [-0.0472, -0.0077]; mean individual F1 is 0.5205 versus 0.4670. These secondary intervals are unadjusted. Designs are never pooled with natural proteins.

All 12 extension shards succeeded after one initialization failure was restarted unchanged. The 162,400 new samples have no continuations or caps and add 2.57 GPU-hours of pure inference. Original sampling used 5.20 GPU-hours and retained its 17 final caps at other temperatures. Setup and retries are excluded from these timings.

Every original 300-sample prefix passes its frozen hash. All 3,248 original discovery/ensemble rows reproduce within 1e-12. Validation also checks paired prompt tokens, seeds, checkpoint identity, parser consistency and bounded, monotonic oracle curves. Thirteen unit tests pass.

## Interpretation and reproducibility

Greater diversity is supported on natural proteins. At 1,000, it yields a small MPNN-only true-contact coverage benefit alongside substantially more false discovery. Extra sampling improves oracle F1 in both models; default still has better mean F1, best@1,000 F1 and ensemble accuracy. Neff support remains specific to the extreme-tier contrast.

One selected checkpoint per recipe, different training exposure/schedules, observational Neff associations, sparse low-Neff data and incomplete redesign-sequence searches limit causal interpretation.

The README is the complete report. Figures regenerate from committed tables. Original raw files (921 MB) remain immutable under v1; new files (442 MB) and the updated report live under v2-1000. Prefix and publication manifests preserve provenance. Individual F1 scores are committed; each appendix figure prints its generating command.

Report and source: https://github.com/Open-Athena/MarinFold/pull/374
Public bucket prefix: open-athena/MarinFold/data/exp373-mpnn-diversity-neff/v2-1000
