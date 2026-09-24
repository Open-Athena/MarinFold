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

## Useful diversity is the open question

314 proteins × 100 samples: consensus R-precision 0.561; oracle best individual map 0.528.
Paired difference −0.032, 95% bootstrap interval [−0.038, −0.025].

All 100 maps are distinct for every protein; mean Jaccard 0.277, true-contact union recall 0.943.
This is not an absence of contact-set diversity, and it does not directly measure distinct folds.

Next: better whole-map candidates, inference-time search and post-training.
