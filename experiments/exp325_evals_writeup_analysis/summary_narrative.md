## Sampling contacts from sequence

A figure-first writeup on existing predictors, oracle contacts, MarinFold and useful sampling diversity.

Fixed model: exp277 step 266344, 1.47B parameters, 248.584B raw training tokens.
Natural proteins lead: 97 validation + 217 publication test. Designs remain separate.
AF2/AF3/Boltz-2 comparators use shared archived MSAs and no templates, with confidence selection.

Predictor generation → cached analysis tables → fast static and interactive rendering.
Every numerical figure element maps back to a source CSV row.

## What the contact and confidence analyses show

AF2 / AF3 / Boltz-2 GDT-TS: 0.854 / 0.860 / 0.861 on 305 natural proteins.
At MSA depth <10: 0.196 / 0.332 / 0.239, with only five proteins.

MarinFold contact R-precision: 0.561 overall; 0.345 at MSA depth <10 versus 0.617 at ≥1000.
Only five natural proteins occupy the shallowest tier.

Oracle pTM ranks 1, 5, 1, 1, 1 out of 101 maps for the five low-depth proteins.
Highest pTM selects each map’s Helico sample; no ipTM or clash penalty.
100 ESMFold2 maps/protein, a shared eligible-pair mask, three Helico samples/map.
Original ESMFold2 median TM-score: 0.848, 0.860, 0.881, 0.649, 0.839.
pTM–source TM-score Spearman: 0.127, 0.083, 0.254, 0.288, 0.613.
Protein order: 8ii8_A, 8oxk_A, 8qoh_A, 8ux2_A, 8wrx_A. Only five biological examples.

Top-L contacts raise Helico GDT-TS from 0.150 to 0.504 on 305 matched natural proteins.
The paired gain is 0.354 [0.323, 0.387]. All cuts were fixed before test inference.

## Can more AlphaFold3 sampling recover accurate folds?

Five natural proteins with MSA depth <10, all in the authorized test split.
100 fresh full AF3 runs per protein are complete; the 1,000-run extension is running.
One diffusion sample per independent seed, ten recycles, fixed archived MSA, no templates.

8oxk_A: best TM rises from 0.516 in the original 25 to 0.818 in 100 fresh runs.
The first TM >=0.8 appears on draw 85; it ranks third by pTM in the pilot.
pTM selects a different structure with TM 0.643.
8qoh_A and 8ux2_A have no TM >=0.8 in the first 100; best TM 0.473 / 0.505.
8ii8_A and 8wrx_A already fold well; best TM 0.876 / 0.919.

Per-protein plots separate oracle best-of-N from pTM-selected TM.
Five proteins remain five biological examples; no inference recipe tuning.

## Useful diversity is the open question

314 proteins × 100 samples: consensus R-precision 0.561; oracle best individual map 0.528.
Paired difference −0.032, 95% bootstrap interval [−0.038, −0.025].

All 100 maps are distinct for every protein; mean Jaccard 0.277, true-contact union recall 0.943.
This is not an absence of contact-set diversity, and it does not directly measure distinct folds.

Next: better whole-map candidates, inference-time search and post-training.

## Low benchmark MSA depth versus ESMC training sources

ESMC documents UniRef 2023_02, MGnify 2023_02 and JGI (July 2023), not Logan.
Low retrieved alignment depth does not establish novelty to the language model.

New historical UniRef search, with MSA filtering disabled:
Significant hits covering at least half the query: 1, 4, 1, 15, 1.
Protein order: 8ii8_A, 8oxk_A, 8qoh_A, 8ux2_A, 8wrx_A.
14 of the 15 well-covered 8ux2_A hits are 97.9–100% identical to the query.
Counts exclude the added query; they are not Neff or final training-cluster counts.

Online MGnify30 hits count heavily clustered representatives, not training-source depth.
ESMC uses 70%-identity clustering; historical MGnify90 is the closer public search target.
Full historical MGnify and JGI are not yet searched. Complete training depth is unknown.
See TRAINING_SOURCE_SEARCH.md and data/training_source_depths.csv for details.
Offline reduction: uv run python generation/search_training_sources.py prepare
