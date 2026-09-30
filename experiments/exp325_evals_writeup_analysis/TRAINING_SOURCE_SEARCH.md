# Homolog searches against ESMC source databases

Searched 2026-09-30. **The historical UniRef search returns few homologs for
these five proteins. We still cannot establish their depth across ESMC's
complete training corpus: historical MGnify and JGI remain unsearched.**

ESMC's [paper](https://www.biorxiv.org/content/10.64898/2026.06.03.729735)
lists UniRef 2023_02, MGnify 2023_02 and JGI downloaded in July 2023. Logan is
not listed. The [model card](https://huggingface.co/biohub/ESMC-6B#training-data)
describes clustering at 70% sequence identity. Source-database hits are not
counts of the final clusters sampled during training or proof of membership.

## Historical UniRef

ColabFold's [database history](https://github.com/sokrypton/ColabFold/wiki/MSA-Server-Database-History)
documents UniRef30 2302 with UniRef100 expansion. This matches the documented
UniRef source release; the job does not provide an independent database hash.
We searched all five fixed benchmark query sequences with `env-nofilter`,
disabling MSA filtering, and retained hits with E-value ≤0.001.

| Protein | Archived plot depth | Significant UniRef hits | Cover ≥50% of query | Cover ≥80% of query |
|---|---:|---:|---:|---:|
| 8ii8_A | 3 | 1 | 1 | 1 |
| 8oxk_A | 6 | 4 | 4 | 2 |
| 8qoh_A | 3 | 1 | 1 | 0 |
| 8ux2_A | 6 | 16 | 15 | 15 |
| 8wrx_A | 4 | 1 | 1 | 1 |

Hit counts exclude the artificially added query and retain separate database
records, even exact query matches. Add one to a hit count to construct an
alignment depth with one query row. The archived plot depth includes two query
copies plus hits from two sources; it is not directly comparable and is not
Neff. These new counts do not replace the benchmark alignment-depth labels.

For 8ux2_A, 14 of the 15 well-covered hits have reported identity 97.9–100%;
the other is approximately 70% identical. The sixteenth hit is a fragment.
Its larger count mainly reflects similar sequences retained when MSA filtering
is disabled, rather than 15 diverse homologs. Four proteins already had nearly
identical public matches before ESMC's source snapshots, as shown in the
[accession-history audit](LOW_DEPTH_TRAINING_AUDIT.md). Shallow coverage therefore
does not imply that the sequences were unseen during training.

The search remains heuristic: three MMseqs profile iterations, cluster expansion,
search E-value 0.1, and a later alignment cutoff of 10. The archived server script
records these settings. Disabling MSA filtering does not remove every search
filter or turn retrieval into an exhaustive census.

## Newer hosted searches, reported separately

EBI HMMER `phmmer` searched UniProt **2025_01** and MGnify30-C2 **2026_07** with
sequence and domain E-value cutoffs of 0.001. These are newer than ESMC's sources.
Each cell gives **all significant sequence hits / hits covering ≥50% / ≥80%**.

| Protein | UniProt 2025_01 | MGnify30-C2 2026_07 |
|---|---:|---:|
| 8ii8_A | 2 / 1 / 1 | 1 / 0 / 0 |
| 8oxk_A | 9 / 6 / 4 | 0 / 0 / 0 |
| 8qoh_A | 1 / 1 / 0 | 2 / 0 / 0 |
| 8ux2_A | 4 / 2 / 2 | 3 / 1 / 0 |
| 8wrx_A | 5 / 2 / 2 | 0 / 0 / 0 |

MGnify30-C2 contains representatives of 30%-identity clusters with at least
two members, according to the [HMMER database documentation](https://hmmer-web-docs.readthedocs.io/en/development/databases.html).
It excludes singleton clusters and does not report their member multiplicities.
Most returned hits here match short, repeated regions. Coverage counts the union
of query positions occupied by residues in reported domains with independent
E-value ≤0.001, excluding insertions and deletions. A significant whole-sequence
score from several short domains does not automatically yield a well-covered hit.
Zero results in this restricted subset do not establish absence from MGnify.

The older ColabFold environmental mix (ColabFoldDB 202108) returned 0, 0, 0, 1
and 1 significant hits, respectively. It combines several sources and is not a
search of MGnify 2023_02. Differences across columns also reflect different
search algorithms and clustering; counts should not be summed.

## Remaining source searches

The [MGnify 2023_02 archive](https://ftp.ebi.ac.uk/pub/databases/metagenomics/peptide_database/2023_02/)
is available as 729,215,663 representatives clustered at 90% identity and
90% coverage. Its compressed sequence file is **83,473,342,442 bytes**. We have
downloaded only its small release metadata, not that sequence archive.
The URL, checksum and proposed bounded-memory search are pinned in
[the search plan](data/training_source_search_plan.json). The large download
awaits the explicit approval required by the repository's transfer policy.
Even this search would measure source representatives, not the unpublished
processed ESMC training subset.

The July 2023 JGI snapshot is not available in this workspace; the broad
IMG/M live sequence-search interface requires authenticated access. JGI is
**not searched**, rather than a zero-hit result.

## Reproduction and evidence

Run from this experiment directory:

```bash
uv run python generation/search_training_sources.py prepare
```

This fast, offline step rebuilds both tables from committed inputs. New network
searches are separate: run the script with `submit`, then `collect` once jobs
have finished. New jobs may use updated server databases; keep versions separate.

`uv run python publish_to_hf.py --source-search-only --upload` publishes this
small evidence package and the updated summary PDF, without rebuilding predictor
archives, to `hf://buckets/open-athena/MarinFold/data/exp325-writeup-analysis/esmc-training-source-search-2026-09-30`.

- [Depth table](data/training_source_depths.csv): one row per protein and source,
  including zeros and unique aligned-sequence counts.
- [Every returned hit](data/training_source_hits.csv): accession, E-value,
  coverage, aligned sequence and exact raw file/record index.
- [Manifest](data/training_source_search_manifest.json): input and script hashes,
  thresholds, source versions and counting definitions.
- [Raw requests, job IDs, results, A3Ms and server script](data/inputs/training_source_search/).
  All ten HMMER jobs and the five-query ColabFold job completed; no result pages
  were truncated. Queries are the same resolved-residue sequences used for the
  archived benchmark MSAs, with lengths 225, 94, 274, 239 and 86.

For the writeup, the supported description remains **low depth in the retrieved
benchmark MSAs**, with the new UniRef search providing additional evidence of
shallow coverage in that historical source. Low depth in ESMC's entire training
corpus remains unverified.
