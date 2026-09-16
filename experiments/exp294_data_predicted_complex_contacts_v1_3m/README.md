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

### Live metadata census (2026-09-16)

The full 8.45 GB metadata mirror is normalized and censused. No structure
archive has been downloaded and no Iris job has been launched.

**Source scale.** 29,025,020 models: 21,430,663 homodimers and 7,594,357
heterodimers. This reproduces #145's 29,025,020 dimer count exactly, so the
release has not changed size since that attempt.

**Tier A.** 1,923,625 models pass both the hard validity filter and the
upstream/proxy quality gate: 1,848,043 homodimers and **75,582 heterodimers**.

**Yield frontier** (hard filter + `quality_ratio` floor, before sequence-pair
dedup and eval decontamination; `quality_ratio = min(ipSAE/0.6, pDockQ2/0.23)`,
so 1.0 is the nominal Tier-A gate):

| floor | heterodimer | homodimer | total |
| --- | --- | --- | --- |
| 1.00 | 75,582 | 1,848,043 | 1,923,625 |
| 0.70 | 121,744 | 2,431,095 | 2,552,839 |
| 0.50 | 158,384 | 2,788,229 | 2,946,613 |
| 0.40 | 181,648 | 2,992,987 | 3,174,635 |
| 0.30 | 210,967 | 3,244,146 | 3,455,113 |
| 0.25 | 230,230 | 3,398,949 | 3,629,179 |

Two conclusions follow directly.

1. **The `pilot-v0` floor of 0.5 cannot reach 3M.** It yields 2,946,613
   candidates *before* dedup and decontamination, so it fails the target on
   arithmetic alone. The 3M target needs a floor at or below ~0.4, and only
   ~0.3 leaves margin for the decontamination loss (#225 removed 1.8–4.0% on
   comparable corpora). Whether a floor that low is defensible is exactly what
   the Stage-B pilot must decide; it is not being set here.
2. **The 500,000-heterodimer floor is not reachable from AFCDB at any
   defensible confidence.** Only 922,359 hard-eligible heterodimers have a
   *nonzero* ipSAE at all, and just 353,774 reach ipSAE >= 0.1. Clearing 500k
   would mean accepting models whose predicted interface is essentially absent.
   The issue's success criteria already condition this floor on "if the
   calibrated source yield supports it"; the measured answer is that it does
   not. The realistic heterodimer ceiling is ~210k at floor 0.30 and ~354k if
   gated on ipSAE alone at 0.1.

**Heterodimer diversity is also narrow.** At floor 0.30 the heterodimers span
only 79–81 distinct taxa, against 8,198 for homodimers. AFCDB heterodimers come
from a small set of model organisms, so they add interface variety but little
taxonomic breadth.

**Schema verification.** Three upstream semantics were checked rather than
assumed:

- `n0chn` is the complex residue count, not just a scaling term: every value is
  a whole number, every homodimer value is even, range 60–3,000. Using it as
  `source_total_residues` is sound.
- `max_ipSAE` and `max_pDockQ2_AB` are already maxima over both directions
  (7,594,357 / 7,594,357 rows equal `greatest(_AB, _BA)`), despite the `_AB`
  suffix. The normalized projection is therefore consistent across sources.
- The homodimer Tier-A proxy's use of `min` rather than `max` directional ipSAE
  is immaterial: mean |ipSAE_AB - ipSAE_BA| is 0.00137 and the two gates differ
  by 0.9% of models (2,593,499 vs 2,616,575).

**The clash filter is quality-correlated, not mis-calibrated.** It looks
brutal in aggregate (it rejects ~47% of heterodimers), but 94.7% of Tier-A
heterodimers and 97.4% of Tier-A homodimers pass it, against 41.8% / 54.4% in
the `quality_ratio < 0.25` band. It is removing the low-confidence tail, which
is the intended behaviour.

**Exact-pair dedup will not come from accessions.** Within the hard-eligible
set there are 12,379,757 distinct homodimer accession pairs for 12,380,049
models (mean 1.000). AFCDB stores essentially one model per accession pair, so
*all* of the deduplication value rests on distinct accessions that share an
identical sequence — which is unreachable without the sequence join. This
confirms the decision to require `annotations.py` rather than dedup on
accession pairs.

### Archive reconnaissance (2026-09-16)

Stage D was sized before being designed, because the numbers change the design:

- **The coordinate archive is 48.8 TB**: 35.20 TB of homodimers (4,000
  `chunk_*.tar` averaging 7.48 GB, plus 5,892 `shard_*_batch_*.tar` averaging
  0.90 GB) and 13.58 TB of heterodimers (8,209 tars averaging 1.65 GB). The
  metadata references 17,820 of these tars.
- **Members are individually addressable.** Each model is one member named
  `AF-<modelEntityId>-model_v1.cif.zst`, so the member name is a pure function
  of `modelEntityId` with no index required. Homodimer `chunk_*` tars
  additionally carry a redundant `.pdb.zst` per model plus PaxHeader entries,
  roughly doubling their bytes for data we do not need.
- **EBI serves HTTP range requests** (`Accept-Ranges: bytes`, verified with 206
  responses on both a 12 GB and an 8 GB tar). Tar headers can therefore be
  walked with 512-byte reads and selected members fetched by byte range, so a
  selective extraction moves roughly the selected payload (~1.5–2.5 TB) instead
  of the full 48.8 TB.
- **This cannot run on the workstation.** Measured single-stream EBI throughput
  here is ~113 KB/s (10.1 MB in 90 s); 48.8 TB would take years and even a
  selective ~2 TB pass would take months. Stage D has to be a cloud fan-out, and
  the same applies to the sequence dependency below.

### Blocking dependency: the sequence source

The AFCDB release ships no FASTA, and `modelEntityId` is an opaque numeric id
(`AF-0000000065760001`) that does not encode the accession. The sequences must
come from AlphaFold DB's bulk `sequences.fasta`, which is **110 GB** — far above
the 10 GB local-mirror threshold, and 11+ days at the measured local rate. It
needs a cloud-side streaming filter that keeps only the 21,437,370 accessions
AFCDB references (3,326,577 of them at the 0.30 floor) and stores only the
filtered result.

`annotations.py` was written against UniProt-style headers and would have
mis-parsed this file: AFDB writes `>AFDB:AF-A0A919MGV6-F1 ... UA=A0A919MGV6`,
whose leading `AFDB:` defeated the `AF-...-F<n>` pattern and would have silently
produced `AFDB:AF-A0A919MGV6-F1` as the "accession" — which would then have
failed the missing-accession check for every row. Fixed, with a regression test
on the real header, plus gzip-safe reads so the filtered mirror can stay
compressed.

The eval-decontamination drop list is still to be produced.

### Sequence fetch: measured, sharded, and hardened (2026-09-16)

An Iris smoke run in `europe-west4-a` (4.29 GB, 16 ranges, one 2-vCPU pod)
measured **14.52 MB/s** and confirmed the source is **118.0 GB**, not the 110 GB
the directory listing rounds to. A single pod would therefore take ~2.3 hours,
because the scan is Python-bound rather than network-bound. `fetch_sequences.py`
takes `--shard-index`/`--shard-count`: shard *i* owns the global byte window
`[total*i/N, total*(i+1)/N)` and subdivides it into ranges, applying the same
start-offset ownership rule at both levels so pod seams are exact for the same
reason range seams are. A sharded run cannot judge its own completeness — each
pod sees only a fraction of the accessions — so `--verify` checks the union:
that the windows tile the file contiguously from 0 to its length, and that every
requested accession was matched exactly once.

Two failure modes showed up that are worth recording, because both are silent:

- **EBI refuses concurrent connections.** Seven of the first sixteen pods died
  immediately on `URLError: [Errno 111] Connection refused` from the opening
  HEAD request. Sixteen pods opening streams inside thirty seconds is past what
  the public FTP accepts. The reader now retries with exponential backoff and
  jitter and resumes a dropped stream from the first byte it has not yet
  returned, rather than losing a multi-GB range.
- **A truncated transfer reads as a clean EOF.** Writing the reconnect test
  surfaced a worse bug than the one it was written for: when a server closes
  early, `read()` returns `b""`, so an interrupted transfer was
  indistinguishable from the end of the file. The range would stop short, report
  success, and the run would ship a short sequence set — which
  under-decontaminates the corpus, silently, weeks downstream. Ranges now carry
  the source length and treat running dry before the range end as fatal unless
  they own the file's tail.

The tests cover eight worker counts, four shard/worker combinations, a missing
shard, an injected mid-stream drop, and a truncated source.

**EBI throttles aggregate bandwidth, not per connection.** One pod reached
14.5 MB/s with 4 concurrent ranges and the same with 16, so at low concurrency
the limit is the client's Python parsing. But with 7–9 pods running at once,
per-pod throughput fell to ~5 MB/s while the total stayed roughly flat. More
pods does not buy proportional throughput from this source, which is a direct
constraint on how Stage D should be sized.

**The accession → sequence join checks out.** On the first completed shard,
341,636 of 342,375 joined models (**99.78%**) have joined UniProt subunit
lengths exactly equal to the modelled `n0chn`, and the disagreements are small
(±2, ±4, ±10, ±18 residues) — consistent with UniProt entries revised since
AFDB folded them. That is strong evidence the accession mapping is right, and
it prices the exact-match default in `selection.py` at roughly 0.2% of
candidates.

### Stage B: the pilot draw

`pilot.py` deliberately does **not** sample proportionally to the selected
mixture. The pilot exists to set the production Tier-B floor, and the census
cannot decide that on its own: it can say how many candidates sit above a floor,
but only generated documents say whether they are usable. A proportional draw
would spend most of its 50,000 on Tier A, which nobody is arguing about, so the
draw is equalised across strata — complex type x confidence bin x length band —
which over-samples the low-confidence tail the decision actually turns on. Rows
are ordered within a stratum by `hash(model_id)`, so the draw is reproducible
from the manifest alone with no stored seed and no dependence on row order, and
a stratum thinner than its share redistributes its shortfall rather than quietly
shrinking the pilot.

### Stage D: what the archive reconnaissance implies

Not yet implemented, and not to be launched before the pilot reports. The
reconnaissance already rules one design in and one out:

- **Streaming whole tars is out.** That is the full 48.8 TB for ~10% of the
  members, and the homodimer `chunk_*` tars carry a redundant `.pdb.zst` per
  model on top.
- **Range-addressed extraction is in.** Walking a tar's headers costs one
  512-byte read per member (next header = `off + 512 + roundup(size, 512)`), and
  selected members are then fetched by byte range. Walking and extracting can be
  fused into a single pass, so the transfer is roughly the selected payload
  (~1.5–2.5 TB) rather than 48.8 TB.

The counter-intuitive part is that the header walk is *not* the expensive
half. Walking is latency-bound (512-byte reads), streaming is bandwidth-bound,
and EBI's binding constraint is aggregate bandwidth:

| | cost model | at EBI's measured ~50 MB/s aggregate |
| --- | --- | --- |
| stream whole tars | 48.8 TB / bandwidth | **~11 days** |
| walk + fetch selected | members x RTT, then ~1.5-2.5 TB / bandwidth | ~75 min of walking + **8-14 h** of payload |

Latency parallelises across tars without touching the bandwidth cap, so the
walk's ~45M small reads (about 23 GB in total) cost roughly an hour spread over
a couple of hundred workers, while the unselected payloads that streaming would
pull are what actually costs days. Per tar the walk looks slower than streaming
— 48k members x 20 ms RTT beats 7.5 GB at a *per-connection* rate — which is
exactly the trap: the per-connection comparison is the wrong one once the source
throttles in aggregate.

Two things the pilot has to measure rather than assume:

- **Whether EBI's ~50 MB/s is really an aggregate cap.** It was inferred from
  per-pod throughput falling from 14.5 MB/s to ~5 MB/s as 7-9 pods ran, with the
  total roughly flat. If it is per-IP rather than global, wider fan-out helps and
  the payload transfer shrinks proportionally.
- **Whether ~45M small requests is acceptable to a public FTP.** This experiment
  has already had EBI refuse connections at 16 concurrent pods. Request rate,
  backoff and total volume need a measured, polite setting, and it may be worth
  asking EBI rather than discovering the limit by hitting it.

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
