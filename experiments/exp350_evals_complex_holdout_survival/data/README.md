# Stage A artifacts

`survival.csv` is the review table. `per_complex.csv` retains every source
candidate and its terminal status, while `exclusion_witnesses.csv` retains the
best alignment witness per query sequence and training arm. Candidate chain
membership and both full/resolved sequence representations are in
`query_membership.csv` and `queries.fasta`.

The JSON files pin input hashes, decoded-complex shard counts, MMseqs commands,
query hashes, result-list diagnostics, and the conservative Helico fine-tuning
reference. Large decoded FASTAs, MMseqs databases, alignments, and logs remain
under `/data/exp350`; their small manifests are committed here.

`native_queries.fasta` contains all representations for the 46 candidates that
survived the two complex-training arms. Searching only this subset is valid for
the cumulative survival result because every omitted candidate already has a
preserved complex-corpus exclusion witness.
