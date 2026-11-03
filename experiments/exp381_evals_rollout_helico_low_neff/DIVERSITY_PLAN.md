# Ten representative rollouts selected from 1,000

Question: with the structural prediction budget fixed at ten, does clustering
1,000 MarinFold contact rollouts improve Helico accuracy?

Freeze before generating the additional contacts or inspecting selected structures:

- Same ten lowest-Neff eval-val targets and two checkpoints as the original pilot.
  Retain the same resolved sequences, contacts conditioning and structural scorer.
- Extend the verified exp373 randomized-prompt T=1 pools from 100 to 1,000 each:
  20 pools, 20,000 total contact rollouts, 18,000 additional rollouts. The previous
  1,000-sample fixed-prompt pools are not interchangeable and are not used.
  Keep exp373's prompt identifiers, seed formula, top-p .95, disabled top-k,
  6L+128 initial generation budget and recorded continuation to the context limit.
  Completed stop status and parser/input identity are required for every sample.
- Cluster the final contact sets after the exact Helico mapping: retain resolved
  residues and contacts at resolved-sequence separation >=6. Native contact or
  structure accuracy, model confidence and structural metrics are not selectors.
- Primary arm: k=10 medoids under binary Jaccard distance, d=1-|A∩B|/|A∪B|;
  d(empty,empty)=0. Duplicate contact sets retain their sampling multiplicity.
  Ten starts, deterministic seeds; k-medoids++ initialization (D² weighting),
  alternate assignments and within-cluster medoid updates until convergence or
  100 iterations. The first start begins at the global medoid; remaining starts
  begin uniformly at random. Keep the solution with minimum mean nearest-medoid
  distance, breaking ties lexicographically by sorted source rollout indices.
  Every representative is an actual rollout. Order representatives by source
  rollout index, not by native quality or cluster size.
- Random control: ten indices drawn uniformly without replacement from the same
  1,000-sample pool, with one frozen draw per protein and the same indices for both
  checkpoints. Order by source rollout index. This includes natural duplicates.
- First-ten reference: reuse the original 200 Helico structures. For each new
  arm, sample slot j uses the original protein/slot-j Helico seed. Six recycles,
  one diffusion sample, unchanged checkpoint/runtime. Thus 400 additional Helico
  predictions, ten per (protein, checkpoint, selection arm).
- Primary endpoint: paired per-protein change in confidence-selected TM-score,
  medoids minus random control, separately for the two checkpoints. Report oracle,
  median and individual mean alongside it, plus lDDT and GDT-TS, and comparison
  with the first-ten reference. Native ranking_score chooses a single structure
  for all metrics; oracle is metric-specific. Equal protein weighting and 10,000
  paired protein-bootstrap draws with seed381. No tuning after structural scoring.
- Diagnose the intervention using mean pairwise Jaccard distance, mean nearest
  representative distance over the full pool, cluster sizes and contact counts.
  After selections are frozen, contact F1 may be reported as an explanatory
  diagnostic; it never changes selection or hyperparameters.
- Generate contacts on CoreWeave beside existing regional checkpoints, fold on
  Modal H100s, record timings and provenance, and publish v3-diversity artifacts.
  Preserve original v1/v2 results. This is a single random-control draw and a
  ten-protein pilot; bootstrap intervals do not quantify all subset/seed variance.
