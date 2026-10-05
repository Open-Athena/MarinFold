# Data pipeline and interpretation

## Populations

The latest view is the **exp277 first full epoch** source mixture, sampled by
document. Its four source counts are pinned in
`experiments/exp277_models_single_mpnn_pilot/config.py`:

| Source | Documents |
| --- | ---: |
| Native AFDB, decontaminated in exp225 | 3,963,003 |
| Native ESM-Atlas, decontaminated in exp225 | 65,553,178 |
| ProteinMPNN redesigns of AFDB source backbones | 31,702,680 |
| ProteinMPNN redesigns of ESM source backbones | 130,872,044 |
| **Total** | **232,090,905** |

The original view is exp199's unfiltered AFDB + ESM-Atlas set:
4,129,682 + 66,759,922 = **70,889,604 documents**. The same sequence or
backbone can appear more than once, especially in the redesign arms. Sampling
is uniform over *training documents*, not distinct sequences or folds.

The evaluation view is exactly the **97 eval-val + 19 eval-denovo** rows of
exp245's `data/eval_sets.csv`. It does not read eval-test. All 116 are shown,
including any protein excluded only from contact scoring.

## Rebuild

Run Python entry points with `uv run` from this directory. The corpus source
locations and counts are checked during sampling; failures stop the pipeline.
The workstation already has exp213's original MMseqs database at
`/data/exp213_overlap/targetDB`, exp225's exact drop list at
`/data/exp225_decontam/droplist_final.parquet`, and a configured `cw` AWS
profile for CoreWeave S3.

1. `uv run python build_eval.py` creates the complete evaluation catalog.
2. `uv run python sample_original.py` draws 100 distinct document ordinals
   uniformly from the entire 70,889,604-entry original MMseqs database.
3. `uv run python launch_sample_latest.py` runs beside CoreWeave S3. It reads
   every Parquet footer of the four exp277 source corpora, asserts their full
   counts, and draws 100 distinct global row ordinals with seed `20260922`.
   Copy its small output back with:

   ```bash
   aws --profile cw --endpoint-url https://cwobject.com s3 cp \
     s3://marin-us-east-02a/MarinFold/training_explorer/2026-09-22/latest.json \
     data/latest.json
   ```

4. `uv run python search_original.py` searches the original 100 against the
   original 70.9M database. It excludes only the query's own document, and
   sorts results by MMseqs2 local alignment bit score.
5. `uv run python search_native.py` searches the latest 100 + eval 116 against
   the original target index, then removes exp225's dropped documents by exact
   `(arm, shard, row)` key. This yields the exp277 native-arm candidates.
6. `uv run python launch_search_mpnn.py` submits 16 disjoint batch-priority
   CoreWeave CPU jobs. Together they decode all **162,574,724** MPNN documents,
   index their sequences, and search all 216 queries. Each job writes a manifest
   with its actual row counts and a compressed alignment table. Jobs read S3
   through the in-region LOTA endpoint; they do not copy the source corpus to
   the workstation.
7. `uv run python assemble_neighbors.py` requires all 16 manifests and checks
   that their AFDB and ESM totals equal the pinned exp277 counts. It merges
   native and redesigned hits by bit score and writes up to 10 per protein.
8. `uv run python fill_weak_neighbors.py` searches the original native index
   permissively for any query with fewer than ten hits, excludes dropped rows
   for the latest and eval views, and completes those lists. Supplemental hits
   with E-values above 10 are marked **WEAK** in the page; they are ranked
   sequences, not evidence of homology. An extreme low-complexity synthetic
   sequence requires a separate unmasked, composition-bias-disabled pass;
   all of its unmasked hits are also marked **WEAK**.
9. For the exceptional sample `latest:148268521`, run
   `uv run python launch_search_mpnn.py --unmasked-query-id latest:148268521`.
   Once all 16 manifests are present, run
   `uv run python merge_low_complexity_mpnn.py` to include the complete
   162.6M synthetic corpus in its unmasked candidate ranking.
10. `uv run python extract_eval_structures.py` selects the specified polymer
   chain from each RCSB experimental mmCIF, then saves a compact local
   backbone. This prevents multi-chain complexes from appearing as the eval
   protein itself.
11. `uv run python extract_afdb.py` reconstructs the exact staged backbone for
   MPNN AFDB samples. For native AFDB samples, it fetches the *current* public
   AlphaFold DB model for the accession, which may be a newer model version
   than the training source. The viewer labels this distinction.
12. `uv run python extract_structures.py` restores the 179 ESM backbones for
    this pinned sample from the published compact artifact bucket. It checks
    each PDB's committed SHA-256 digest, residue count, and sequence labels.
    The original source-CIF extraction located entries by Parquet footer and
    row-group entry-ID statistics and required 2.81 GB of compressed source
    row groups. Rebuilds now transfer only the selected compact PDBs, rather
    than repeating that multi-GB public-source read. A different sample needs
    a new, compute-local mirror of selected source row groups before raw CIF
    extraction. For MPNN designs, the coordinates are the reused parent
    backbone and the displayed residue identities are the designed sequence.
13. `uv run python publish_structures.py` uploads the 316 compact PDB previews
    to the public `open-athena/MarinFold` bucket under
    `data/training-explorer/2026-09-22/structures/`, verifies every filename
    and byte size, writes a SHA-256 manifest, and puts public structure URLs
    into the three snapshots. The local `structures/` directory is a generated
    staging area and is ignored by git.
14. On a GCP-authenticated workstation, run `uv run python
    prepare_neighbor_afdb.py --billing-project <project>` and include the
    generated, gitignored `legacy_afdb/` directory in the CoreWeave workspace
    bundle. This reads the requester-pays AFDB v4 archive used to build the
    corpus, including entries retired from the current API.
15. `uv run python extract_neighbor_structures.py` runs next to the mirrored
    ESMFold2 source on CoreWeave, range-reads only selected row groups, and
    publishes the 2,690 distinct source backbones shared by the 3,155 unique
    displayed sequence hits. MPNN hits use the parent backbone on which the
    sequence was redesigned. The script checkpoints each output to a durable,
    co-located working copy and verifies the public Hugging Face listing.
16. `uv run python attach_neighbor_structures.py` validates complete hit
    coverage and adds the public structure URL, format, and provenance note to
    every neighbor row.

The source document decoder was checked against
`marinfold.document_structures.contacts_v1.read.sequence_from_document` on
500 actual exp266 documents; their concatenated sequence SHA-256 digests
matched byte for byte. All 200 emitted PDBs were parsed with gemmi and have
exactly the sampled document's residue count.

## Similarity meaning

The neighbors are the highest **reported MMseqs2 local-alignment bit scores**
at sensitivity 7.5 and finite reporting depth (1,000 for the original set;
3,000 native candidates and 500 per MPNN search shard for the latest set).
The standard search uses E-value limit 10. For any list shorter than ten, a
permissive native-arm search with E-value limit 1,000,000 supplies additional
ranked sequences. These weak hits are labeled and should not be read as
homology. The native index is complete, but MMseqs is heuristic and report
limits mean the displayed ten are the best **reported** hits, not a guarantee
of exact global nearest neighbors. Percent identity is over the aligned
region; query coverage is shown separately. E-values from MPNN shards and
native hits are approximately scaled to the 232.1M-document combined search
space for display; bit score determines rank.

The source structure preview shows only backbone atoms, so 3D rotation stays
responsive. Exact ESM source coordinates and exact staged MPNN AFDB backbones
are preserved. The experimental eval chain is saved from RCSB; unresolved
residues may be absent from its coordinate preview even when present in the
query sequence. The page uses Mol* when WebGL is available and a rotatable Cα
canvas view otherwise.

## Launch the page

```bash
python -m http.server 8766 --bind 127.0.0.1 --directory apps/training_explorer
```

Then open <http://127.0.0.1:8766/>. The latest/original control is a switch;
evaluation is a separate view. Search, source filters, copyable sequence,
structure rotation, and neighbor metrics work in the browser after the JSON
snapshot and selected remote preview load.
