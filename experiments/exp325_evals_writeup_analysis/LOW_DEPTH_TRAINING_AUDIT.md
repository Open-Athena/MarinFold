# Low alignment depth versus ESMC training exposure

Follow-up: [new source-database searches on 2026-09-30](TRAINING_SOURCE_SEARCH.md)
measure historical UniRef hits and separate newer hosted-search results.

Checked 2026-09-29. **We have not established that these five proteins have few
homologs in ESMC's training corpus.** The depth strata describe the archived
alignments supplied to the benchmark predictors.

ESMC's [paper](https://www.biorxiv.org/content/10.64898/2026.06.03.729735)
identifies UniRef 2023_02, MGnify 2023_02 and JGI downloaded in July 2023.
Logan is not a listed source. The [model card](https://huggingface.co/biohub/ESMC-6B#training-data)
describes 70%-identity clustering. No exact processed training corpus or
membership manifest was available for this audit, so training-set homolog counts
and actual training membership remain unverified.

The archived alignments themselves already contain these hits. Each includes
two copies of the query; the plotted count retains both. All five still qualify
as shallow after removing repeated query records.

| Protein | Plotted rows | Non-query hit rows | Best existing hit | Reported identity | Query coverage | Accession public since |
|---|---:|---:|---|---:|---:|---|
| 8ii8_A | 3 | 1 | UniRef100_A0A2Z5U5S2 | 100% | 100% | 2018 |
| 8oxk_A | 6 | 4 | UniRef100_N1J9C5 | 98.9% | 97.9% | 2013 |
| 8qoh_A | 3 | 1 | UniRef100_A0A0L1KJ08 | 27.8% | 58.8% | 2015 |
| 8ux2_A | 6 | 4 | Q7NY09 | 100% | 100% | 2003 |
| 8wrx_A | 4 | 2 | UniRef100_A0A4D6AQM2 | 100% | 100% | 2019 |

Best means lowest archived E-value. Identity is copied from the search header;
coverage is the fraction of query positions occupied by a hit residue in the
A3M. These denominators differ for gapped alignments. Dates come from UniSave
histories, including deleted/demerged accessions; all five histories contain
only sequence version 1. These dates establish prior public availability, not
membership in the final ESMC training set. Four proteins have a nearly identical,
almost full-query match. A close match also does not establish a large family.

The supported description is **low depth in the benchmark's retrieved MSAs**.
Neither sequence novelty to ESMC nor low homolog abundance in its complete
training corpus follows from this measurement. Answering the latter would
require searching the actual historical training sequences, including JGI,
with a declared search, coverage and redundancy policy. A current MGnify or
Logan search would answer a different question.

## Reproduction and evidence

From this experiment directory, run `uv run python generation/audit_low_depth_msas.py`.
This reads the committed snapshots without network access. `--snapshot` refreshes
the five local cached MSAs and UniSave metadata; it performs no new sequence search.

- [Per-protein audit](data/low_depth_training_audit.csv)
- [Every archived non-query hit](data/low_depth_msa_hits.csv), including source record indices
- [Inputs and hashes](data/low_depth_training_audit_manifest.json)
- [Saved A3Ms, query sequences and UniSave responses](data/inputs/low_depth_training_audit/)
