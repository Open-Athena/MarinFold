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

5,000 predictions: 1,000 independent full runs for each of five natural test proteins.
One diffusion sample per seed, ten recycles, fixed archived MSA, no templates.

8oxk_A: best TM 0.516 in the original 25 → 0.818 in 100 fresh runs → 0.947 in 1,000.
pTM selects the 0.947 structure (seed 10470, draw 471).
20 / 1,000 predictions reach TM >=0.8; the first is draw 85.
At budget 100, pTM selected a different structure with TM 0.643.

8qoh_A / 8ux2_A: no TM >=0.8 in 1,000; best TM 0.502 / 0.635.
8ii8_A / 8wrx_A: already accurate; best TM 0.881 / 0.936.

Per-protein plots separate oracle best-of-N from pTM-selected TM.
The first 100 runs are unchanged; original baseline scores reproduce.
Five biological examples, fixed generation settings, no model tuning.

## AF3 sampling in predictor and MSA-depth context

8oxk_A: ESMFold2 already reaches TM 0.940; AF3 at 1,000 reaches 0.947.
Five low-depth proteins, mean TM: ESMFold2 0.854; AF3 baseline 0.603;
AF3 at 1,000: 0.681 with official ranking, 0.719 with pTM, 0.780 oracle best.
TM >=0.8: ESMFold2 4/5; original AF3 2/5; expanded AF3 3/5.

At depths 10–99 / 100–999 / >=1000, baseline AF3 means: 0.891 / 0.932 / 0.955.
ESMFold2: 0.691 / 0.874 / 0.944. Same 305 proteins; bin n=5 / 20 / 60 / 220.
Different proteins occupy each bin; extended sampling only exists at depth <10.

Official defaults: ten recycles, five diffusion samples per user-specified seed.
Our baseline: 5 full seeds x 5 samples; expanded: 1,000 full seeds x 1 sample.
Both use fixed archived MSAs and no templates, not the native search pipeline.
40x structures, 200x trunk evaluations; neither is a measured runtime ratio.
Official ranking includes disorder/clash terms; pTM ranking is shown separately.

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


## Precision at L/5 and the KNN comparison

k=max(1,floor(L/5)); L is the input sequence length, not the resolved-residue count.
Same 314 natural proteins, nine predictors, candidate pairs and archived predictions.
All-range / long-range P@L/5: MarinFold 0.830 / 0.761; sequence-KNN 0.657 / 0.595.
At depth <10 (five proteins): MarinFold 0.595; KNN 0.102; ESMFold2 0.727.
KNN transfers from ten native decontaminated-corpus neighbors; no redesign index.

Same 100-rollout pools: mean individual 0.400; consensus 0.829; oracle best 0.609.
Oracle reselected by P@L/5. Emission order ranks individual maps; short maps retain k.
Invalid individual maps receive zero. Consensus follows the original vote rules.
All 5,994 predictor and 1,884 sampling R-precision cells reproduce before rescoring.

Every static plot has native vector PDF/SVG and a 7,200-pixel-wide PNG.
At 300 dpi the raster supports 24-inch-wide panels; vector PDFs scale further.
POSTER.md links the public collection. Preprocessing remains separate from rendering.
