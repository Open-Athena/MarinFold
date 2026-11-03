## Individual contact rollouts through Helico: 200 predictions complete

Ten lowest-Neff natural eval-val proteins, ten rollouts each, two MarinFold models.
Current default exp277 step266344 vs ProteinMPNN-only two-epoch lr5e-4 step373768.
Each contact set produced one independent Helico structure on Modal H100.
All 200 predictions passed scoring and have full reference CA coverage.

## No clear checkpoint winner in this pilot

Default / two-epoch mean TM-score:
Oracle: 0.483 / 0.481. Median: 0.340 / 0.350.
Confidence pick: 0.433 / 0.440. Individual-rollout mean: 0.347 / 0.348.

For confidence picks, two-epoch minus default:
TM-score +0.007, 95% paired interval [-0.086, +0.090].
lDDT +0.008 [-0.059, +0.073]; GDT-TS -0.005 [-0.098, +0.080].
All model-difference intervals include zero; this is not an equivalence claim.

## Sampling helps; confidence captures part of the oracle gain

Confidence selection improves TM-score over the mean individual rollout:
Default +0.086 [0.017, 0.159]; two-epoch +0.092 [0.037, 0.154].
Oracle gains over that mean are +0.136 and +0.134.
The remaining oracle-minus-confidence gaps are 0.050 and 0.041.
Confidence selects the TM-score oracle on 4/10 proteins for each model.
This pilot does not compare against folding an aggregated contact-vote map.

## Individual proteins tell different stories

8eb9_A: confidence-selected TM-score improves from 0.531 to 0.728.
7xp9_A: it falls from 0.738 to 0.417.
8cba_A: the default median is 0.198, but confidence finds a 0.510 rollout.
8bux_A: both models struggle, with oracle TM-scores only 0.209 and 0.246.
Two proteins are viral; separate subgroup tables are descriptive only.
Ten proteins and ten rollouts cannot establish full eval-val performance.

## Frozen protocol and standard structural scoring

Reuse exp373 rollouts 0-9: randomized prompts, temperature 1, top-p .95.
Prompts and sampling seeds are paired across models; no accuracy filtering.
Map all contacts into the verified resolved sequence; no voting or top-L cut.
Helico step6000, six recycles, one diffusion sample, no MSA, paired seeds.
Fixed-residue TMscore supplies reference-normalized TM-score and optimized GDT-TS.
Biotite supplies all-heavy-atom lDDT, excluding within-residue pairs.
Missing SE atoms remain penalized in lDDT; heavy-atom coverage is >=99.58%.

## Reproduction and compute

Small CSVs, plotting scripts, environment lock and PDF are in experiment #381.
Public HF artifacts contain all ten inputs and all 200 coordinate arrays/PDBs.
Run analyze.py and build_summary.py to regenerate figures without GPU access.
Use restore_public.py, setup_metrics.py and score.py to rescore saved structures.
Measured inference: 0.380 H100-hours; median 6.66 seconds per structure.
Inference-only GPU cost component: $1.50; startup and other charges are additional.
See the README for the exact protocol, run links, artifact hashes and limitations.
