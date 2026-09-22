# Training set explorer

Interactive local explorer for the exact corpora used by exp277 (232,090,905
documents) and the original exp199 AFDB + ESM-Atlas training set (70,889,604
documents). The eval view contains every exp245 `eval-val` and `eval-denovo`
protein, including the entry excluded from contact scoring only if it belongs to
one of those two sets.

From the repository root, run
`python -m http.server 8766 --bind 127.0.0.1 --directory apps/training_explorer`
and open <http://127.0.0.1:8766/>. The page reads
`data/{latest,original,eval}.json` and backbone previews for all 316 displayed
proteins from the public `open-athena/MarinFold` artifact bucket.
The viewer also loads Mol* from jsDelivr; network access is needed for 3D
previews. A rotatable canvas backbone is used when Mol* cannot initialize.
Do not call a query "uniform" or show neighbors until its materialization
manifest records the full source counts and completed search.

The explorer is a visual inspection tool, not a homology or contamination
verdict. Neighbors are ranked by MMseqs2 local alignment bit score, with
identity, coverage, and E-value shown separately. A search has a finite
reporting depth and may miss remote homologs. Identical training rows remain
distinct because the models see documents rather than unique sequences.

`build_eval.py` materializes the complete evaluation query catalog. The
sampling/search pipeline is documented in `PIPELINE.md`.
