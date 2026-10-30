# Extension to 1,000 fixed-prompt rollouts

Frozen before extension inference, 2026-10-30, after inspecting the original
300-rollout results and following the user's request. This is a follow-up,
not a new prespecified primary analysis.

- Retain all original 300 temperature-1 fixed-prompt samples byte for byte.
  Pin their compressed-file SHA256 hashes in `data/extension_prefix_inventory.csv`.
  Add rollout indices 300–999 (seed blocks 3–9) for both existing checkpoints
  and all 116 proteins: 162,400 new samples. All other conditions stay at
  their original sample counts. Natural proteins (97) and designs (19) remain
  separate; no eval-test access.
- Reuse checkpoint files, tokenizer, prompt `{stem}:r0`, sampling parameters,
  canonicalization, continuation policy and same-region CoreWeave H100 execution.
  Publish only the new 700 samples per condition under a new `v2-1000` root;
  the immutable original `v1` files supply the first 300. No sample replacement,
  cherry-picking or termination-based exclusion. Capture timings and worker metadata.
- Extend ensemble R-precision, true-contact coverage and distinct true/false
  discovery through k=1,000, preserving the original plotted points. Use k=1,
  10, 25, 50, 100, 200, 300, 400, 500, 600, 750, 900, 1,000 for these curves.
- For every rollout, compute contact-set F1 = 2 TP / (predicted contacts +
  true contacts) in exactly the resolved-pair scoring universe, separately
  for all range (separation ≥6) and long range (≥24). An empty prediction with
  nonempty truth scores zero; empty truth is undefined and must be reported
  if encountered. Do not calculate F1 from cohort-averaged precision and recall.
- Oracle best@k is the maximum individual-rollout F1 among the first k samples,
  first computed within each protein, then averaged equally across proteins.
  Compute all k=1…1,000. It uses ground truth for selection and is an upper
  bound, not an available inference-time selector. The comparison baseline is
  each protein's mean individual F1 over all 1,000 samples, macro-averaged and
  displayed as a horizontal dashed line for each model. Also save running means
  and seed-block means to diagnose sampling stability.
- Use the original paired protein bootstrap (10,000 draws, seed 373) for
  pointwise 95% intervals, paired model differences, and k=300→1,000 gains.
  Intervals describe protein uncertainty, not independent rollout replication
  or simultaneous coverage of the full curves. No temperature selection or
  best-checkpoint selection is added.
- Validate exact original-prefix hashes, complete index ranges and seed
  identities, paired prompt tokens, checkpoint hashes, parser consistency,
  retained caps, F1 bounds, monotonic oracle curves, and exact agreement with
  the original discovery/ensemble tables at k≤300. Publish raw outputs and
  reproducible small tables, add the plots and results to the canonical README
  and summary PDF, and update PR #374.
