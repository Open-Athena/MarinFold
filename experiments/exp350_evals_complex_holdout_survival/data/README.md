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
