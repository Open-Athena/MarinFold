# Summary slides — ESMFold / ESMFold2 contact prediction (exp78)

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide.
     Keep this current as the experiment progresses. -->

## What we're doing

How well do ESMFold (`facebook/esmfold_v1`) and ESMFold2 (`biohub/ESMFold2`)
predict pyconfind-defined contacts on our eval set, compared to Protenix v2
(single-sequence and MSA)? We extend exp74 with two more single-sequence
structure predictors, scored the same way: run pyconfind on the predicted
structure and rank candidate pairs by contact degree, against one shared
pyconfind ground truth.

## Method

Same 554-protein eval set as exp74 (FoldBench-100 + 454 exp65 low-MSA /
novel-fold candidates), same pyconfind ground truth (degree ≥ 0.001, sep ≥ 6),
same metric (contacts @ L / L/2 / L/5 and R-precision; aggregate + short /
medium / long). ESMFold runs once per protein (deterministic, num_recycles=4).
ESMFold2 runs single-sequence, best-of-N diffusion samples (num_loops=20,
num_sampling_steps=100), keeping the top-1 by confidence. Both on Modal (H100);
all predicted structures saved to the HF bucket for re-scoring. Protenix v2
numbers are reused from exp74.

## Results so far

_(Fill in as predictions land and scoring completes.)_
