## Sampling contacts from sequence

A figure-first writeup on existing predictors, oracle contacts, MarinFold and useful sampling diversity.

Fixed model: exp277 step 266344, 1.47B parameters, 248.584B raw training tokens.
Natural proteins lead: 97 validation + 217 publication test. Designs remain separate.

Predictor generation → cached analysis tables → fast static and interactive rendering.
Every numerical figure element maps back to a source CSV row.

## What the contact and confidence analyses show

Natural contact R-precision: 0.561 overall; 0.345 at MSA depth <10 versus 0.617 at ≥1000.
Only five natural proteins occupy the shallowest tier.

Oracle confidence exceeds every matched random-map control on all 20 preselected proteins.
Three diffusion samples per map, identical positive/negative counts and known mask.
Random maps are weak negatives, not plausible alternatives.

Top-L contacts raise Helico GDT-TS from 0.150 to 0.504 on 305 matched natural proteins.
The paired gain is 0.354 [0.323, 0.387]. All cuts were fixed before test inference.

## Useful diversity is the open question

314 proteins × 100 samples: consensus R-precision 0.561; oracle best individual map 0.528.
Paired difference −0.032, 95% bootstrap interval [−0.038, −0.025].

All 100 maps are distinct for every protein; mean Jaccard 0.277, true-contact union recall 0.943.
This is not an absence of contact-set diversity, and it does not directly measure distinct folds.

Next: better whole-map candidates, inference-time search and post-training.
