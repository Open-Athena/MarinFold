# Frozen plan: individual rollout contacts through Helico

Issue: https://github.com/Open-Athena/MarinFold/issues/381

Select the ten smallest exp260 `n_eff_0.8` values among the 97 natural
FoldBench eval-val targets, breaking ties by stem. This is a Neff-ranked pilot,
not the separate raw-MSA-depth <10 benchmark. Never consult eval-test.

Compare the current default exp277 step 266344 with the ProteinMPNN-only
two-epoch exp364 lr=5e-4 step 373768. Reuse exp373's first ten randomized-prompt,
temperature-1, top-p=.95, top-k-disabled rollouts per target and model. The
selection depends only on saved rollout index, not contact or structure quality.
Verify raw checksums, completion status, parsed contacts, matched prompts and
sampling seeds. Preserve the source raw records and measured CoreWeave batch
timings; do not present divided batch times as measured per-rollout timings.

Map all live contacts into Helico's resolved sequence by the verified exp14
residue mapping. Remove unresolved endpoints and pairs with mapped separation
<6; record both counts. No contact voting, top-L cut, or contact resampling.
Unlisted pairs remain UNKNOWN. Use the established contacts-msafree-01 step6000
Helico checkpoint (SHA256 pinned in plan.json), source b10385d, six trunk
cycles, one diffusion sample per contact set, bfloat16, no MSA. Use distinct
SHA256-derived diffusion seeds per (target, rollout), paired across models.
The 200 predictions run on Modal H100, at most four concurrent workers.

Save each full-precision structure, viewable PDB, confidence, input identity,
per-input timing and worker metadata durably before analysis. A failed call
must be resumed, not omitted or replaced by another target/rollout. Gate the
fan-out on two predictions (rollout zero, both models) of the lowest-Neff
target; these belong to the 200, not additional cherry-picked samples.

Compute reference-normalized CA TM-score and optimized GDT-TS with the official
TMscore program (pylelab/USalign commit 23f14ff), matching fixed residue numbers.
Compute all-heavy-atom lDDT with Biotite 1.5.0, 15-Angstrom inclusion radius,
0.5/1/2/4-Angstrom thresholds, excluding within-residue pairs. Match atoms by
residue position/name; retain missing reference atoms in the denominator and
report coverage. No structural realignment or single-Kabsch-fit GDT substitute.
Report all scores on [0,1]. No side-chain symmetry relabeling is performed.

For each protein/model and each metric, report the maximum (metric-specific
oracle), median, and the score of the prediction with highest Helico
`ranking_score` (ties: lowest rollout index). The confidence pick is shared
across metrics and uses no reference labels. Also show mean single-rollout
accuracy and prefix curves for k=1..10. Average summaries equally over proteins;
bootstrap paired proteins (10,000 draws, seed381) for model differences. Show
individual-protein results and confidence-selection regret. This pilot cannot
establish performance on the full eval-val set or disentangle all training
differences between the two checkpoints.

Publish small tables, figures, a prose README and summary PDF in this experiment;
publish the raw inputs and 200 structures in the public MarinFold HF bucket.
