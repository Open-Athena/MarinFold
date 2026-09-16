---
marinfold_experiment:
  issue: 294
  title: 'exp: curate three million predicted complex documents'
  kind: data
  branch: exp/294-predicted-complexes
---

# exp: curate three million predicted complex documents

**Issue:** [#294](https://github.com/Open-Athena/MarinFold/issues/294) · **Kind:** `data` · **Branch:** `exp/294-predicted-complexes`

## Question

Can we curate at least **3,000,000 usable predicted protein-complex contacts-v1 training documents** from the NVIDIA/AlphaFold Database complex release, with relaxed but calibrated confidence tiers, without allowing a handful of repeated interfaces to dominate the training mixture?

## Hypothesis

The NVIDIA/AFDB complex release already contains enough usable predictions to reach three million training documents without running AlphaFold-Multimer ourselves. The earlier [#145](https://github.com/Open-Athena/MarinFold/issues/145) attempt became small because it was designed as a validation set: it restricted candidates to homologues of validation monomers and then selected one representative per coarse pair of independently clustered chains. For training, retaining multiple documents per **interface-aware** cluster while controlling their sampling weight should preserve the desired scale and prevent extreme redundancy.

We expect all usable high-confidence predictions to provide the core corpus. A calibrated medium-confidence tier should fill the remainder to three million. Homodimers and heterodimers, confidence tiers, and source datasets must remain separately identifiable so later training experiments can tune or ablate the mixture.

## Background

- The public [NVIDIA/AFDB complex release](https://ftp.ebi.ac.uk/pub/databases/alphafold/collaborations/nvda/) contains more than 30 million AlphaFold2 complex predictions. The associated [AFCDB paper](https://research.nvidia.com/labs/dbr/assets/data/manuscripts/afdb.pdf) reports about 1.8 million high-confidence predictions grouped into about 225,000 interface-aware clusters. The live release and website use evolving quality summaries, so the first deliverable must reproduce a fresh metadata census rather than assume an old count.
- The [#145 artifact](https://claude.ai/code/artifact/7e190150-da28-487a-af44-243576cdd511) started from 29,025,020 dimers, retained 2,010,800 under its `ipSAE >= 0.6` and `pDockQ2 >= 0.23` gate, restricted that set to 881,010 validation-homologous candidates, and collapsed it to 17,083 chain-cluster-pair representatives. Its interface-size filter was nearly inert. This issue reuses its useful metadata/tar-indexing work if the unpushed branch can be recovered, but not its validation-only selection policy.
- [#222](https://github.com/Open-Athena/MarinFold/issues/222) implemented multi-chain contacts-v1 documents and produced 85,660 PDB multimer documents. This experiment should reuse that generator and validator.
- [#225](https://github.com/Open-Athena/MarinFold/issues/225) defines the current evaluation-decontamination policy: remove sequence matches at >=30% identity and >=50% shorter-chain coverage (plus the remote-homology significance rule), and remove near-identical structural leakage without purging every shared fold.
- Additional sources worth auditing, but not required to make the first AFCDB corpus, are [Protein Complex Atlas](https://pmc.ncbi.nlm.nih.gov/articles/PMC13181095/) (1.1 million predicted binary complexes, including 181,671 classified high-confidence), [PINDER](https://github.com/pinder-org/pinder) (1.56 million bound experimental dimer examples but only 42,220 interface clusters), [Predictomes](https://predictomes.org/downloads), and [PPIRef](https://ppiref.readthedocs.io/en/stable/ppiref_dataset.html). Their overlap, licenses, and incremental interface diversity must be measured before treating their headline counts as additive.

## Approach

### Stage A — metadata census and deterministic selection

Implement a local, metadata-only pipeline before downloading coordinates:

1. Stream the current AFCDB homodimer and heterodimer metadata CSVs into normalized Parquet.
2. Preserve the complete upstream provenance and quality fields, including model/accession IDs, homo/heterodimer status, taxon, chain lengths, ipTM, directional ipSAE, pDockQ/pDockQ2, pLDDT, interaction counts, clashes, and source tar member.
3. Reproduce and compare the paper's confidence gate, the #145 gate, and score-binned relaxed alternatives. Report yield by homo/heterodimer, taxon, length, and contacts-v1 context eligibility.
4. Remove exact duplicate sequence pairs and apply the frozen evaluation sequence-decontamination reference to **every subunit**.
5. Select every eligible high-confidence model as Tier A. Select Tier B deterministically from successively relaxed, calibrated score strata until the post-filter target is at least 3,000,000 documents. Prefer evidence-backed heterodimers over the weak homodimer tail.
6. Emit a complete selection manifest and a rejection ledger. Every source row must have a terminal status and named reason; unexpected parse or I/O failures raise rather than becoming silent drops.

The selection output must remain cheap to regenerate when thresholds change. It is separate from coordinate extraction.

### Stage B — stratified calibration pilot

Before freezing Tier B, extract a deterministic approximately 50,000-model pilot stratified by confidence bin, homo/heterodimer status, taxon, and length. Generate contacts-v1 documents and measure:

- parse/generation success and context-window fit;
- backbone clashes, chain breaks, finite-coordinate validity, and interface size;
- inter-chain contact count and density;
- agreement with matching PDB/PINDER structures where available;
- per-document fetch, parse, generation, and total wall time; and
- the estimated acceptance yield, storage, and full-run wall clock for every proposed tier.

Use a second predictor such as Boltz or Protenix only for a small consensus audit of ambiguous bins, not to regenerate millions of complexes.

### Stage C — interface-aware redundancy control

Use the published AFCDB Multimercluster assignments if they are available; otherwise build interface-aware Foldseek Multimercluster assignments for accepted structures. Do **not** choose one example per sorted pair of single-chain clusters.

- Keep all accepted documents in the durable corpus.
- Remove exact sequence-pair duplicates.
- Publish natural-mixture and cluster-balanced training manifests.
- Carry `interface_cluster_id`, `cluster_size`, `confidence_tier`, and a proposed sampling weight per document.
- Default the balanced view to an inverse-square-root cluster weight and a configurable per-cluster sampling cap; choose the final cap from the census so no small collection of clusters owns a large fraction of sampling probability.
- Keep homodimer/heterodimer and confidence-tier shards separately addressable.

### Stage D — production generation

Build an I/O-bound Zephyr pipeline around the #222 contacts-v1 generator:

- select IDs before reading structure archives;
- make input work units source-tar-aware so each archive is streamed once rather than fetched once per model;
- project only required manifest columns;
- use in-shard concurrent fetch/parse where the storage layout benefits from it;
- run one-CPU preemptible workers pinned to the data's region;
- write one Parquet per input shard under `gs://marin-<region>/protein-structure/MarinFold/exp<N>/`, with compute and output co-located;
- record a zero-silent-drop status ledger and per-input timing/worker metadata; and
- publish consolidated corpus shards, tokenizer, manifests, rejection tables, and public QC artifacts under `buckets/open-athena/MarinFold/data/` without overwriting prior corpora.

Run local tests and then a small Iris smoke test. Report measured throughput, projected wall time, storage, region, and output prefix before requesting approval for the full multi-million-document run. No bulk cross-region transfer is authorized by this issue.

### Stage E — optional source audit

Perform a metadata/license/overlap audit of Protein Complex Atlas, PINDER, Predictomes, and PPIRef. For each, report coordinate availability, redistribution terms, exact sequence-pair overlap with AFCDB/PDB, added heterodimers, and added interface clusters. Add an external source only as a separately labeled arm if it contributes defensible incremental data.

Training on the resulting corpus is a follow-up experiment. That experiment should compare Tier A alone, Tier A+B, natural versus cluster-balanced sampling, and predicted versus predicted-plus-experimental mixtures under a fixed token budget.

## Success criteria

- A deterministic accepted manifest containing at least **3,000,000 predicted complex documents after hard validity and evaluation-leakage filters**. If the calibration shows that reaching three million would require an indefensible quality collapse, stop and report the measured frontier rather than silently weakening the policy.
- Raw model count, generated-document count, unique sequence-pair count, and interface-aware cluster count are reported separately.
- Homodimer/heterodimer, confidence tier, taxon, length, interface size, inter-chain-contact, and cluster-size distributions are committed as small tables/plots; target at least 500,000 heterodimer documents if the calibrated source yield supports it.
- The default balanced manifest prevents severe concentration: report the top 1, 10, 100, and 1% of clusters' shares of both documents and sampling probability, and choose the cap/weighting so no single interface cluster contributes more than 0.1% of balanced sampling probability.
- Every selected source model is accounted for by an output document or a named designed-in rejection. Unexpected fetch, parse, and generation failures are fatal.
- All generated documents round-trip, and a deterministic sample's inter-chain contacts matches a direct coordinate-based calculation.
- The Iris smoke test accounts for every input and records measured per-document latency, worker count, projected full-run wall time, storage, and region before the full run is launched.
- Corpus and manifests are published at new versioned paths with source licenses/provenance and the tokenizer co-located; existing datasets remain untouched.

## Results

### Implementation status (2026-09-15)

The metadata-first path is implemented:

- `metadata.py` converts both live AFCDB metadata schemas to lossless typed
  Parquet, emits a common selection schema, and produces confidence/hard-filter
  yield curves. It deliberately requires a local one-time mirror of the CSVs so
  iterative development cannot re-stream several gigabytes from EBI.
- `annotations.py` streams one or more UniProt/AFDB FASTA files into exact
  sequence hashes and lengths, joins the versioned eval-homology drop list, and
  fails if any selected AFCDB accession is missing or maps to conflicting
  sequences.
- `selection.py` removes contaminated and invalid candidates, deduplicates by
  exact unordered sequence pair, keeps every eligible source-quality-pass model
  as Tier A, fills the heterodimer and total-document quotas from ranked Tier B,
  and writes both the selected manifest and a terminal-status ledger for every
  source row. The current `pilot-v0` relaxed floor is a placeholder for the 50k
  calibration pilot, not a frozen production threshold.
- Six local tests cover both source schemas, score derivation, tar provenance,
  sequence hashing, missing-accession failure, two-subunit decontamination,
  exact pair deduplication, tier/quota behavior, hard rejection reasons, and the
  fail-loud unmet-quota gate. `uv run pytest -q` and `pyrefly check` pass.

The live metadata census and structure pilot have not run yet. The current EBI
metadata mirror is being staged once under `/data`; no structure archive or
full production job has been launched.

Stage A commands:

```bash
uv run python metadata.py normalize \
  --homodimer-csv /data/afcdb/homodimer_metadata.csv \
  --heterodimer-csv /data/afcdb/heterodimer_metadata.csv \
  --out /data/exp294/metadata

uv run python metadata.py census \
  --input '/data/exp294/metadata/normalized_*.parquet' \
  --out /data/exp294/census

uv run python annotations.py \
  --normalized '/data/exp294/metadata/normalized_*.parquet' \
  --fasta /data/exp294/afcdb_sequences.fasta \
  --decontam-drop-list /data/exp294/eval_accession_droplist.parquet \
  --out /data/exp294/accession_annotations.parquet

uv run python selection.py \
  --input '/data/exp294/metadata/normalized_*.parquet' \
  --annotations /data/exp294/accession_annotations.parquet \
  --out /data/exp294/selection
```

## Conclusion

_(Fill in after results are in.)_
