# ProteinMPNN-only diversity and Neff dependence

## What we tested

Does redesign-only training broaden contact predictions and reduce their dependence on available evolutionary information?

We compared the exp364 MPNN-only 1.5B checkpoint at step 186884 with the exp277 full-data default at step 266344. The primary cohort is 97 natural eval-val proteins; 19 designs are reported separately. Eval-test was not scored.

The frozen design has 300 fixed-prompt samples plus 100 matched randomized prompts per model/protein, at temperatures 0.7, 1.0 and 1.3. All 278,400 samples are saved. Temperature 1 is primary. Intervals bootstrap proteins, preserving model pairing, with 10,000 draws.

## Diversity increases after the controls

On natural proteins, mean fixed-prompt Jaccard distance rises from 0.708 to 0.813. The paired difference is +0.1046, with 95% CI [0.0866, 0.1229].

Matching every paired sample's contact count gives +0.1064 [0.0891, 0.1244]. Randomized prompts give +0.1008 [0.0836, 0.1189]. Long-range contacts show the same effect. Independent seed-block differences are +0.1050, +0.1042 and +0.1046.

MPNN-only generates modestly more contacts, but the long-range fraction is similar. Count, prompt and seed-block controls support a real difference in the contact-sampling distribution.

## Extra diversity mainly adds benchmark errors

Single-rollout precision falls from 0.411 to 0.298. After 300 fixed-prompt samples, true-contact coverage is essentially tied: 0.964 versus 0.967, paired difference +0.0029 [−0.0033, 0.0095].

Distinct false contacts rise from 5,642 to 8,902 per protein: 58% more, with a paired increase of 3,260 [2,367, 4,298]. Fixed-prompt ensemble R-precision falls from 0.539 to 0.454.

The standard 100-randomized-prompt R-precision is 0.554 versus 0.463, difference −0.0911 [−0.1113, −0.0719]. Default ensemble accuracy remains higher throughout the temperature sweep. Increased diversity does not establish useful alternatives or physical conformations.

## Neff gives contrast-specific support

Neff is redundancy-weighted depth at 80% identity, not raw MSA sequence count. Fixed tiers contain 4, 10, 28 and 55 natural proteins.

The all-range high-minus-low accuracy drop is 0.399 for default and 0.329 for MPNN-only: difference −0.0705 [−0.1026, −0.0401]. However, absolute low-Neff accuracy is 0.280 versus 0.263; MPNN-only does not improve it.

The continuous interaction is inconclusive: −0.0080 [−0.0276, 0.0165] per Neff decade. Long range, length/viral adjustment and nonviral-only slopes are also inconclusive. The largest accuracy gap occurs in the middle tiers. Four lowest-tier proteins limit generalization. No natural protein has raw MSA depth below 10.

## Patterns and mechanisms

Both models have contact dependencies beyond sample-size and contact-frequency margins. Excess Jaccard SD over exact-margin shuffled controls is 0.0197 for default and 0.0169 for MPNN-only. Half/full-chain differences are small relative to that excess. This does not identify physical states or uniquely stronger MPNN-only coordination.

Eight contact-map examples were chosen by Neff rank before inference. Some retain similar recurring patterns; others show diffuse MPNN-only frequencies and lower accuracy.

Existing sequence searches cover native AFDB/ESM-Atlas. The actual 162.6-million-document redesign corpus has no complete available sequence-search or KNN audit. Removing native documents retains structural ancestry and cannot establish an absence of retrieval or memorization.

## Secondary designs and execution checks

For 19 designs, the Jaccard difference is +0.0457 [−0.0017, 0.0976], inconclusive and smaller after matching counts. Randomized-prompt all-range R-precision is 0.708 versus 0.661, difference −0.0469 [−0.0857, −0.0159]. Designs are never pooled with natural proteins.

All 12 H100 production shards succeeded. Retained conditions total 5.20 GPU-hours of inference, excluding setup and retries. All malformed and capped outputs remain included: 1,501 continuations and 17 final caps, with no continuation at temperature 1.

Validation covers 1,392 conditions, 27,840 metric rows, exact prompt/seed pairing, continuation prefixes, context limits and checkpoint hashes. Ten unit tests pass. Upload recovery changed publication only; the scientific worker/parser programs were identical.

## Interpretation and reproducibility

Greater diversity is supported on natural proteins, but it mostly adds false-contact discovery and lowers ensemble accuracy. The Neff hypothesis has partial support from the extreme-tier contrast, not a clear continuous interaction or better absolute low-Neff prediction. The full-data default remains stronger here.

One selected checkpoint per recipe, different training exposure/schedules, observational Neff associations, sparse low-Neff data and incomplete redesign-sequence searches limit causal interpretation.

The README is the complete report. Figures regenerate from committed small tables; 921 MB of compressed raw samples and all inputs are public, with checksums and anonymous-read verification. Each appendix figure prints its generating command.

Report and source: https://github.com/Open-Athena/MarinFold/pull/374
Public bucket prefix: open-athena/MarinFold/data/exp373-mpnn-diversity-neff/v1
