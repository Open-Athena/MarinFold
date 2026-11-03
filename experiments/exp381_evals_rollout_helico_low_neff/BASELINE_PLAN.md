# Matched ESMFold2 and AlphaFold3 extension

Frozen before baseline inference in `data/baseline_plan.json`. The original
Helico protocol and its 200 results remain unchanged.

- Same ten lowest-Neff eval-val targets and exact resolved sequences used by
  Helico (76–289 residues). No new target selection or eval-test prediction/scoring.
- Three arms: Biohub ESMFold2 single sequence; official AlphaFold3 single
  sequence; official AlphaFold3 with archived ColabFold MSA. Ten independently
  seeded full-model runs per arm per protein, one diffusion sample per seed:
  **300 new predictions**, no hidden best-of-five filtering.
- Fixed seeds `381000 + 100*cohort_index + rollout_index`, paired across
  baseline arms. Smoke uses rollout zero of the longest target, retained in
  the final ten. Failed infrastructure attempts are recorded; no successful
  prediction is discarded or rerun based on accuracy.
- AF3 source `3c89cc7b89aa7042b72885af9016a45b262da008`, ten recycles, one
  diffusion sample, default sampler, no templates in either arm. Single
  sequence uses explicitly empty paired and unpaired MSAs. MSA uses the same
  archived non-pairing A3Ms underlying exp260's Neff ranking. Match columns
  are projected onto resolved residues; sequence membership/order is kept.
  Insertions between consecutive retained positions are retained; discarded
  unresolved positions and their insertions are removed. AF3 applies its
  standard MSA processing afterward. This is an archived ColabFold-MSA
  baseline, not a newly searched full AF3 database baseline.
- ESMFold2 source `8f0834cbc350aeaf507f755475873f7c61fa44a1`, released
  `biohub/ESMFold2` snapshot `1ebf0e3481a5184eb6171d40615c79e384b48796`,
  20 loops, 200 diffusion steps, LM dropout 0.3, ESMC bf16; no MSA.
  Uses current pinned public inference API and its paper-evaluation dropout.
- Select by unrounded native confidence: ESMFold2 pTM, AF3 ranking_score.
  Helico remains selected by its native ranking_score. Select one structure
  for all three accuracy metrics; ties use earliest rollout. Oracle is
  separately best for each metric. Retain median and individual-sample mean.
- Reuse `score.py` unchanged: reference-normalized TM-score and optimized
  GDT-TS with fixed residue correspondence; all-heavy-atom lDDT at 15 Å,
  thresholds 0.5/1/2/4 Å, excluding same-residue pairs and penalizing missing
  atoms. Use identical reference atom arrays and exact atom-name mapping,
  including the existing MSE selenium-versus-MET sulfur convention.
- Plot k=1..10 curves and k=10 summaries for all five arms, with equal protein
  weighting and 10,000 paired protein bootstrap draws, seed381. The cohort
  has ten independent units; 100 predictions per arm are not 100 proteins.
- Record per-seed inference time, feature time where separable, model setup,
  worker hardware, source and weight hashes. AF3 timings include cold-shape
  JIT; ESM's fold API includes preparation. Do not infer a speed ranking.
- Modal H100 execution uses existing model caches; export only predictions,
  inputs and provenance. Public v2 artifacts exclude private AF3 parameters.
  Preserve official AF3 output notices alongside its structures.

This matches sample budgets, not training exposure, architecture, or compute.
It evaluates the resolved-sequence pilot and does not establish performance
on all eval-val or the full prompt sequences.
