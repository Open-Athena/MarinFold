# Secondary test: explicitly spread-out contacts

The frozen k-medoids selector improved coverage but reduced mean pairwise
Jaccard distance relative to random ten (default .904 vs .937; MPNN .918 vs
.941). Before inspecting any new structural accuracy results, add this
secondary arm to directly test the user's request for diverse samples.
The original protocol, selections, and primary endpoint remain unchanged.

- Reuse exactly the twenty validated 1,000-rollout pools and their resolved
  contact mapping. No new contact generation, ground-truth selection, filtering,
  confidence input, or tuning.
- Pick ten by greedy farthest-point (max-min Jaccard) selection. Start at the
  global medoid, then repeatedly take the sample with greatest distance to its
  nearest selected sample. Break all ties by lowest source rollout index.
  Sort the final ten indices for assigning the same per-slot Helico seeds.
- Produce 200 additional Helico structures (ten proteins, two checkpoints,
  ten representatives), with the existing checkpoint, one sample, six recycles,
  identical scoring and confidence selection. Four experiment arms now include
  first ten, random ten, k-medoids, and farthest-point ten.
- These are secondary contrasts, reported alongside the original primary
  medoids-minus-random contrast. Show all four structural summaries and all
  three metrics with paired protein-bootstrap intervals; no multiplicity
  correction or claim of a new pre-registered primary endpoint.
- Freeze and publicly publish these selected inputs before their Helico calls.
  This adds 200 predictions to the original 400 new predictions, not extra
  structures within any ten-sample arm. The control subset remains fixed.
