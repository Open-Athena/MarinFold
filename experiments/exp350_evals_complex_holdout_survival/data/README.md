# Stage A artifacts

`survival.csv` and `per_complex.csv` record the strict component-held-out audit.
`pair_survival.csv` and `pair_per_complex.csv` record the pair-held-out audit;
`pair_exclusion_witnesses.csv` retains one two-chain witness per candidate and
training arm. Candidate chain membership and both full/resolved sequence
representations are in `query_membership.csv` and `queries.fasta`.

`pair_queries.fasta` is the 706-sequence confirmation query set selected from
the 220 survivors of the initial pair screen. `pair_queries.json` pins that
selection. The 1,000,000-result confirmation removed two PINDER candidates;
`pair_confirmation_search.json` pins its commands and diagnostics. The final
pair-held-out counts are 35 FoldBench and 183 PINDER before the Helico screen,
and 32 plus 65 afterward.

The JSON files pin input hashes, decoded-complex shard counts, MMseqs commands,
query hashes, result-list diagnostics, and the conservative Helico fine-tuning
reference. `evidence_manifest.json` hashes the complete compressed alignment
and log bundle published at
<https://huggingface.co/buckets/open-athena/MarinFold/tree/main/data/evals/exp350_complex_holdout_survival/evidence>.

`native_queries.fasta` contains all representations for the 46 candidates that
survived the two complex-training arms under the strict component rule. It is
retained to reproduce that negative result.

# Final contact and Helico artifacts

`contact_r_precision.csv` contains one strict 100-rollout inter-chain result per
context-eligible target. `contact_r_precision_summary.csv` aggregates by frozen
split with 10,000-replicate homology-group bootstrap intervals. `timings.csv`
records per-target timing and worker metadata for 23 targets scored by six
CoreWeave jobs. `contact_rollout_manifest.json` pins the checkpoint, targets,
six CoreWeave jobs, raw score-part hashes and public artifact prefix.

`helico_arms/` contains the six length-budget contact inputs evaluated on
development. `helico_per_target.csv` records exact pair-specific, symmetry-aware
DockQ, iRMSD, lRMSD and Fnat for all ten Helico arms. `helico_summary.csv`
aggregates each arm; `helico_comparisons.csv` gives prespecified paired test
deltas; `helico_timings.csv` contains all 104 predictor timing rows.
`helico_eval_manifest.json` pins the selected budget, Helico checkpoint and git
revision, inference/scoring settings, Modal volume paths, hashes and the public
HF prefix.
