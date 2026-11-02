# Plan: a PINDER-style split for the AFDB dimer release

Status: phases 0 and 2-pilot complete, 2026-09-05. Phases 1 and 3 onward not run.

Context: [`README.md`](README.md) is the survey that issue #145 asked for. This
plan covers the curation work that followed from it, and the decision recorded in
conversation to follow PINDER's split protocol rather than the sequence-only split
already built.

## What this dataset is for, and why that decides its size

**Primary deliverable: a held-out multimer validation set**, a few thousand
non-redundant complexes usable to validate models trained on contacts-v1
monomers. Not a training corpus. This is the decision that sizes every download
below, so it is stated first rather than left implicit.

The consequence is that the clustering population is the **val-eligible 881,010
complexes**, not all 2,010,800. The other 1,129,790 would be clustered only to
define communities nobody draws from.

One caveat, stated because it is the thing that would change if the goal changed:
PINDER's deleaking step exists to keep test systems away from *train* systems. If
we never train on AFDB dimers there is no train side to deleak against, and the
leakage that does matter (against the contacts-v1 monomers our models have seen)
is handled upstream by the sequence split. Reintroducing a training goal
reintroduces the train-side download and the deleaking pass together.

## What already exists

Five steps have run to completion and their outputs are on disk under
`data/curation/`:

| Artifact | Contents |
|---|---|
| `confident_{homo,hetero}dimers.csv.gz` | 1,930,552 + 80,248 complexes passing `ipSAE >= 0.6 and pDockQ2 >= 0.23`, exact counts from a full stream of both metadata CSVs |
| `val_monomers.fasta` | 41,954 contacts-v1 validation monomer sequences, zero unresolved |
| `subunit_shards/*.fasta` | 1,979,542 dimer subunit sequences, zero unresolved, 767 MB |
| `dimer_split_assignment_id30_cov50.csv.gz` | per-complex train/val on subunit homology to the validation monomers, 881,010 val-eligible |
| `fetch_pilot_summary.json`, `fetch_timings.csv` | the phase 2 rate pilot, 8,012 structures |

The confident set is already a filtered slice: the release is 29,025,020 models
(21,430,663 homodimers + 7,594,357 heterodimers) and the gate keeps 6.9% of it.

`dimer_split_assignment_id30_cov50.csv.gz` is **not** superseded by this plan. It
answers a question PINDER's protocol does not: whether a dimer's chains are
homologous to the monomers our contacts-v1 models were validated on. PINDER
controls leakage *within* a dimer set; this controls leakage *between* the dimer
set and our existing monomer validation split. The two compose, and phase 5
combines them.

## Why the sequence-only split is not enough

Two complexes can share no chain above 30% identity and still present the same
binding geometry, and the interface is what a multimer model learns. The 45-model
smoke test in `README.md` already produced a case: a *Merluccius polli* SNX4
homodimer and a *Glycine max* heterodimer clustered at multimer TM 0.62 with no
sequence relationship between them.

The sequence rule also treats heterodimers unfairly, and that is an artifact
rather than a real difference in risk. It routes a complex to val if *either*
subunit hits, so a heterodimer gets two chances and a homodimer one. At 30%
identity that produces 69.9% of heterodimers in val against 42.7% of homodimers.
Assigning whole interface clusters to splits removes the asymmetry, because the
decision is made once per cluster rather than once per chain.

## What PINDER does

Read on 2026-08-24 from `github.com/pinder-org/pinder` at `--depth 1`, files
`src/pinder-data/pinder/data/{config.py,foldseek_utils.py,get_clusters.py,graph_utils.py}`.
The preprint's full text is not open in Europe PMC (record PPR884594, `inEPMC=N`,
`OA=N`), so the code is the source. Where the code and the abstract disagree, the
code wins.

**PINDER does not use `easy-multimercluster`.** An earlier draft of this plan was
written around it and was wrong. What PINDER actually does:

1. Runs Foldseek **`easy-search`, all against all, on individual chains**, not on
   complexes (`foldseek_utils.py:206`).
2. Builds a graph whose nodes are chains and whose edges are alignments passing
   the filters in the parameter table (`graph_utils.py:25`).
3. Runs **asynchronous label-propagation community detection** on that graph to
   get monomer communities (`get_clusters.py:298`, `cluster_from_graph`).
4. Defines a dimer's interface cluster as the **sorted pair of its two chains'
   community IDs**: `cluster_{min(cR, cL)}_{max(cR, cL)}`
   (`get_clusters.py:111`, `sysid_to_cluster_string`).

So "interface cluster" means "this pair of monomer structural communities", and
the interface enters through a filter (both sides must have at least 7 interface
residues) rather than through a multimer alignment. This is cheaper than
`easy-multimercluster` and it scatters into independent jobs.

Their published outcome: 2,319,564 dimeric systems, split into 1,560,682 training
dimers from 42,220 clusters, **1,958 validation representatives** and 1,955 test
systems with interface leakage removed.

Three properties to copy:

1. Cluster on the **interface**, not on chain sequence.
2. Assign **whole clusters** to splits, never individual complexes.
3. Draw validation from **cluster representatives** (`top_n = 1`), so the set is
   small and non-redundant by construction. PINDER's validation set is 0.08% of
   its corpus; our sequence rule produced 43.8%, which is a quarantine pool, not
   a validation set.

## Parameters

The Foldseek search, graph-edge and clustering rows below are **recorded, not
used**: phase 3 clusters directly with `easy-cluster` at TM >= 0.5 and coverage
0.5, and the reasons are there. They stay in the table because the rejection is
only meaningful next to what was rejected, and because a training-side dataset
would need the deleaking rows after all.


| Stage | Parameter | PINDER value | Source |
|---|---|---|---|
| Foldseek search | `-s` sensitivity | **11.0** (Foldseek's own default is 9.5) | `FoldseekConfig` |
| | `-e` evalue | **0.05** (default 0.001, so more distant hits) | `FoldseekConfig` |
| | `--max-seqs` | 1000 | `FoldseekConfig` |
| | `--alignment-type` | **2** (3Di+AA Gotoh-Smith-Waterman, local and fast) | `FoldseekConfig` |
| | score type | **lddt** (not `alntmscore`) | `FoldseekConfig` |
| Graph edges | score band | **> 0.5 and < 1.1** | `GraphConfig.score_threshold`, `upper_threshold` |
| | min alignment length | 10 residues | `GraphConfig` |
| | min interface length | **7 residues on both sides** | `GraphConfig.min_interface_length` |
| | coverage threshold | 0.5 | `GraphConfig` |
| Clustering | algorithm | AsynLPA community detection, seed 40 | `ClusterConfig` |
| | edge threshold for clustering | **0.70** | `foldseek_cluster_edge_threshold` |
| | canonical method | `foldseek_community` | `ClusterConfig` |
| Deleaking | edge threshold for neighbour search | **0.55** (looser than clustering) | `foldseek_edge_threshold` |
| | depth limit | 2 hops | `ClusterConfig.depth_limit` |
| | max node degree assumed leaky | 1000 | `ClusterConfig.max_node_degree` |
| Splitting | representatives per cluster | **`top_n = 1`** | `ClusterConfig.top_n` |
| Batching | chains per Foldseek sub-database | **50,000** | `ScatterConfig.foldseek_db_size` |
| Secondary (MMseqs) | `-s`, `-e`, `--min-seq-id` | 11.0, 0.05, 0.2 | `MMSeqsConfig` |
| | pident score band | 30 to 110 | `GraphConfig.mmseqs_*_threshold` |

`top_n = 1` is how the validation set gets to 1,958 systems: one representative
per cluster, and the rest of each cluster is simply not drawn. Our phase 5 target
of a few thousand complexes follows the same mechanism, not a separate quota.

Two threshold pairs are worth noticing because they are deliberately asymmetric.
Clustering cuts edges below 0.70, but leakage detection searches neighbours down
to 0.55, so a complex can be excluded from the test set for a similarity that was
too weak to have put it in the same cluster. And the search runs at `-e 0.05`
against Foldseek's default `0.001`, which deliberately admits more distant hits
than a normal search would.

## What does not transfer

PINDER's test-set criteria are properties of PDB depositions and have no analogue
in a predicted set: `resolution_thr 3.5`, `method "X-RAY DIFFRACTION"`,
`interface_atom_gaps_4A 0`, `prodigy_label "BIO"`, `oligomeric_count 2`,
`alphafold_cutoff_date "2021-10-01"`. Our equivalent quality filter is the
confidence gate already applied upstream (`ipSAE >= 0.6 and pDockQ2 >= 0.23`),
and our equivalent of the AF2-cutoff holdout does not exist, since every model in
the release was produced by AlphaFold-Multimer v2.3.0 weights.

`min_chain_length 40` and `min_atom_types 3` do transfer and are cheap to apply.

## Phases

### Phase 0. Read the PINDER methods and fix parameters. DONE 2026-08-24

Deliverable is the parameter table above, with divergences marked.

### Phase 1. Sequence-collapse the val-eligible set before downloading anything

New, and it comes first because it is free and it sizes everything after it.

The val-eligible 881,010 complexes contain **858,360 distinct subunit
accessions**, whose sequences are already on disk. Cluster them with
`mmseqs easy-cluster` and count the clusters. Complexes whose chains fall in the
same sequence clusters will almost always land in the same interface community,
so one complex per distinct sequence-cluster pair is enough to define the
structural clustering.

The risk is real and stateable rather than unknown: near-identical chains can
still present different interface geometry (domain swaps, alternative binding
modes), so this merges some things PINDER would split. For a validation set that
errs toward fewer and more diverse representatives, which is the safe direction.
Record how many complexes the collapse discards so the cost stays visible.

**Result, 2026-09-05, at 80% bidirectional coverage:**

| min identity | sequence clusters | complexes to fetch | share of val side | wire |
|---|---|---|---|---|
| 0.3 | 58,410 | 63,907 | 7.3% | 7.5 GB |
| 0.5 | 138,230 | 146,309 | 16.6% | 17.1 GB |
| 0.7 | 281,002 | 291,522 | 33.1% | 34.1 GB |

**Decision: do not collapse.** PINDER does not have this step. Its MMseqs2
clustering (`--min-seq-id 0.2`) is an *optional parallel* clustering of the
complete system set producing an alternative cluster label, skipped outright when
no graph is supplied (`get_clusters.py:322`), and `canonical_method` is
`foldseek_community` (`config.py:324`). It never decides which structures to look
at, because PINDER already holds every PDB structure. The collapse is our
invention, forced by having to fetch over the wire, and it discards between
589,488 and 817,103 complexes unseen on the assumption that a shared
sequence-cluster pair implies a shared interface community.

Note the direction of risk, which is the opposite of the split's. In the split, a
lower identity threshold was *safer*: it caught more remote homology and pushed
more complexes into val. Here a lower threshold merges more aggressively and
discards more, so 0.3 is the riskiest row in the table, not the safest.

Phase 2 therefore fetches all 881,010. The numbers above stay in
`val_cluster_summary.json` as a measured alternative if the transfer ever needs to
shrink.

### Phase 2. Fetch the structures

**Rate pilot: DONE 2026-09-05**, 8,012 structures via
`curate_fetch_structures.py --sweep`. Results:

| workers | files/s | speedup | 429s | retries |
|---|---|---|---|---|
| 1 | 1.44 | 1.00x | 0 | 0 |
| 2 | 2.85 | 1.98x | 0 | 0 |
| 4 | 5.96 | 4.14x | 0 | 0 |
| 8 | 11.54 | 8.01x | 0 | 0 |

Throughput is linear in workers with no throttling of any kind. The mechanism:
`alphafold.ebi.ac.uk/files/` is served out of Google Cloud Storage, not EBI's own
web tier (`server: UploadServer`, `x-guploader-uploadid`, `x-goog-storage-class`
headers, a Google load balancer at 34.149.152.8), with
`cache-control: public,max-age=86400`.

**We cap at 8 workers anyway.** No rate limit is published: EMBL-EBI's terms of
use state only that "any attempt to use EMBL-EBI Data Resources and Tools to a
level that prevents, or is likely to prevent, EMBL-EBI providing services to
others, will result in the user being blocked", and their `robots.txt` disallows
`/api` and `/search/sequence` while allowing `/files/`. The only quantitative EBI
figure anywhere is 10 req/s per IP, forum-sourced from Europe PMC staff for a
different service. 11.5 files/s sits at that order and is demonstrated safe. If
we ever need more, the honest route is to email `afdbhelp@ebi.ac.uk` and ask for
the GCS bucket name, which would make this an in-network copy.

**Measured sizes** over 8,012 files: 117.0 kB on the wire, 508.1 kB decoded,
4.34x. Per population:

| population | complexes | wire | decoded | at 11.5 files/s |
|---|---|---|---|---|
| val-eligible | 881,010 | 103 GB | 448 GB | 21 h |
| all confident | 2,010,800 | 235 GB | 1,022 GB | 49 h |

**The release has gaps.** 28 of 2,000 files in one batch returned 404, all
homodimers in the ID band `AF-...74043945` to `...74276307`. Re-requesting all 28
serially returned 404 again, so they are missing rather than load-shed, and a
random 400-model sample across the whole confident set was 400/400 present. The
fetcher records a 404 and continues; **the list of missing models ships as a
published artifact**, because the dataset is defined by what actually downloaded.

Shard the ID list by `index % n_shards` over a sorted list, not in blocks, for the
same reason as the sequence fetch. Record per-file timings to CSV as `AGENTS.md`
requires.

**Runbook: [`AWS_FETCH.md`](AWS_FETCH.md).** One `c7i.large`, eight workers,
881,010 complexes into 441 tar shards on S3, ~21 h, ~$2 of compute. We do not fan
out across hosts: 11.5 files/s is the rate demonstrated safe, and multiplying our
footprint against a public resource to save a day is not a trade worth making.

**Run: DONE 2026-09-06.** One `c7i.large` in us-west-2a, eight workers, headless
under the instance role. Launched 17:20 UTC 2026-09-05, published `_DONE_rc0` and
self-terminated at 12:13 UTC 2026-09-06: **18.9 hours** wall, against the 21 h
estimate above.

| | |
|---|---|
| requested | 881,010 (824,126 homodimers, 56,884 heterodimers) |
| fetched | 880,248 |
| 404 | 762 (0.09%) |
| non-404 failures | 0 |
| retries | 117 |
| written | 441 tars + 441 sidecars, 103.7 GiB |
| median shard rate | 10.34 files/s (min 8.93, max 18.68) |

Nothing throttled. Zero non-404 failures across 880k requests over 19 hours at 8
workers, and 117 retries total (0.013% of requests), which is the strongest
evidence we have that the 8-worker cap is well inside what the endpoint tolerates.

Shard rate on EC2 (median 10.34 files/s) came in slightly *below* the pilot's 11.5
measured from a workstation, and individual shards ranged over 2x. The pilot ran
four fixed-size batches back to back; the full run competed with nothing but
itself, so the spread is the endpoint's, not ours.

**The transfer estimate above was wrong, and by 4x.** It assumed
`Accept-Encoding: gzip` would be honoured. It was honoured for 15.8% of requests:
**740,852 of 880,248 responses (84.2%) came back uncompressed**, and `fetch_one`
gzipped them locally so the archive stays uniform. Actual bytes pulled from EBI
were **396 GiB, not the 103 GB projected**; decoded mmCIF totals 448 GiB, so
essentially all the compression is ours. Stored size is unaffected (103.7 GiB) and
so is cost, since ingress to EC2 is free. Two consequences:

- The `wire_bytes` column understates the real wire for those 84% of rows. They
  are identifiable: `fetch_one` negates the status, so `http_status = -200` marks
  a reply that arrived uncompressed. Any future bandwidth estimate has to branch
  on that column rather than sum `wire_bytes`.
- Scaling this to the full confident set means roughly **900 GiB** off EBI, not
  235 GB. That does not change the AWS bill, but it changes what we are asking of
  a public resource, and it is the number to quote if we ever email
  `afdbhelp@ebi.ac.uk` about the GCS bucket.

**Artifacts.** `data/curation/fetch_shard_summary.csv` (441 rows, one per shard)
and `data/curation/fetch_missing_404.csv` (762 rows, the models that do not
exist) are in git. The per-input timings CSV is 880,248 rows and 13.5 MB gzipped,
too large for the repo, and lives beside the shards at
`s3://marinfold-exp91-usw2/MarinFold/exp145-afdb-dimers/fetch_timings.csv.gz`.
Phase 3 must exclude the 762 missing models; they are counted in the split
assignment but have no structure.

### Phase 2a. Storage and region

Local disk has 58 GB free, so this needs cloud storage regardless of population.
Decision recorded in conversation: **AWS**, which exp91 already used (EC2 workers
reading and writing S3).

Costs at current us-east-1 pricing, first month:

| line item | val-eligible (881,010 / 103 GB) | all confident (2,010,800 / 235 GB) |
|---|---|---|
| EC2, `c7i.large`, 21 h vs 49 h | $2 on demand | $4 |
| ingress to AWS | $0 | $0 |
| S3 PUT at $0.005/1,000 | $4.41 | $10.05 |
| S3 Standard at $0.023/GB-mo | $2.37/mo | $5.41/mo |
| later egress to the HF bucket, $0.09/GB after 100 GB free | $0.27 | $12.15 |
| **total** | **~$9** | **~$32** |

Two things follow. The PUT charge is a third of the total at 2 M objects, so
**pack structures into tar or parquet shards** rather than writing one key per
structure. And the eventual S3-to-HF move is over the `AGENTS.md` 10 GB
cross-region threshold and needs explicit sign-off; the val-eligible population
keeps that leg under the 100 GB free-egress allowance, which the full set does
not.

Per [[ec2-worker-no-boto3]]: on ephemeral EC2 workers, read and write S3 through
`pyarrow.fs.S3FileSystem` with the instance role. Do not `pip install boto3`; it
wedges the box.

### Phase 3. Cluster the chains with Foldseek. DONE 2026-09-11

The unit is the distinct chain, not the complex: 858,360 of them on the val side,
one per UniProt accession, extracted by `curate_extract_chains.py`. A homodimer's
two chains are one accession and contribute once.

**One `foldseek easy-cluster` over all of them, TM-score >= 0.5, coverage 0.5.**
That is the whole of this phase. Earlier drafts specified PINDER's route instead
(chain-against-chain `easy-search`, a graph over the hits, AsynLPA communities,
sharded into 50,000-chain sub-databases and run as 171 pairs). That route was
measured and rejected; see "Why not PINDER's graph" below.

Measured by `curate_foldseek_scaling.py --op cluster` on 3,376 pilot chains, 8
threads on 4 physical cores:

| chains | seconds | peak RSS | clusters | representatives | largest | singletons |
|---|---|---|---|---|---|---|
| 400 | 8.6 | 0.85 GB | 222 | 55.5% | 16 | 154 |
| 800 | 11.0 | 0.85 GB | 354 | 44.3% | 32 | 207 |
| 1,600 | 15.8 | 0.86 GB | 481 | 30.1% | 44 | 220 |
| 3,376 | 28.3 | 0.86 GB | 712 | 21.1% | 128 | 312 |

Cost fits **5.5 s + 7.2 ms per chain**, which is linear. A bare power law fits
these points as `n^0.645`, but that is 5.5 s of fixed startup masquerading as
sublinear scaling and would extrapolate to nonsense; the affine form is the one to
project with. Memory is flat at 0.86 GB across the sweep because clustering never
materializes an all-pairs hit table.

**The full val side projects to about 1.7 hours on four cores, at single-digit
GB.** One instance, one command. Nothing here needs a fan-out.

**Why TM >= 0.5.** `easy-cluster` has no lddt threshold, so PINDER's lddt 0.70 cut
does not carry over and we are choosing a line rather than inheriting one. TM 0.5
is the conventional boundary above which two chains are called the same fold, and
it is the value in `curate_foldseek_scaling.py`. At TM 0.7 the same 3,376 chains
give 999 clusters instead of 712, so the threshold moves the count by about 40%
in the range worth arguing about.

**Why not PINDER's graph.** Three reasons, in descending order of how much they
matter.

1. **We do not deleak.** PINDER's graph exists to serve leakage detection between
   train and test, which is a transitive property: a test system similar to
   something similar to a train system is a leak path, so you need connected
   components and the two asymmetric thresholds (cut at 0.70, search neighbours
   down to 0.55). Nothing trains on AFDB dimers, so there is no train side to
   deleak against, and the leakage that does matter is handled upstream by the
   sequence split. The graph would be machinery serving a goal we do not have.
2. **Transitivity is wrong for this deliverable.** Community detection merges A
   with C whenever some B sits between them, so a single representative can stand
   for structures quite unlike each other. `easy-cluster` is set-cover: every
   member resembles its representative directly. For "give me N mutually
   dissimilar complexes", that is the property we want.
3. **It costs 14x more and the gap widens.** Measured at the same chain counts:

   | chains | easy-search | easy-cluster | ratio |
   |---|---|---|---|
   | 400 | 28.1 s | 7.8 s | 3.6x |
   | 800 | 61.0 s | 10.3 s | 5.9x |
   | 1,600 | 174.0 s | 19.4 s | 9.0x |
   | 3,376 | 404.1 s | 29.0 s | 13.9x |

   The ratio grows because search is `n^1.21` while clustering is linear. Search
   RSS fits 0.87 GB + 77 kB/chain, so one all-against-all over 858,360 chains
   wants ~67 GB, which is why PINDER shards at 50,000 in the first place. Sharding
   is not free either: 171 pairs rebuild the k-mer index 171 times and total 492
   hours against 90 for a single pass, 5.5x more work to buy the memory back. The
   full search route came to roughly $210 across 171 fanned-out jobs. Clustering
   is a few dollars on one.

The measurements behind all of this are in `data/curation/foldseek_scaling.json`
(search, size), `foldseek_scaling_threads.json` (search, threads) and
`foldseek_scaling_cluster.json` (cluster, size).

**Two things the pilot could not tell us**, both now answered by the full run:

- **How many clusters 858,360 chains give.** 857,599 chains gave **13,478**, a
  representative fraction of 1.57% against 21.1% at n=3,376. The guess here was
  "anything from 15,000 to 150,000", so the real answer lands just under the low
  end of a range that spanned an order of magnitude. Wrong in the safe direction:
  13,478 is still far more than the few thousand phase 5 draws.
- **Whether the pilot's redundancy is typical.** It is not, and it understated
  the redundancy badly. The full set is much flatter than one ID band suggested:
  the largest cluster holds 15,861 chains, more than four times the entire pilot.

The affine cost fit held. 5.5 s + 7.2 ms per chain projects 1.72 h for 857,599
chains; the run took 0.97 h on 8 vCPUs against the pilot's 4 physical cores, so
the projection was conservative by the thread count and little else. Peak RSS was
4.12 GB, above the pilot's flat 0.86 GB but nowhere near a constraint.

Deliverable: `chain_cluster.tsv` mapping every accession to its representative,
plus the cluster-size distribution. Both are in `data/curation/`, and
[`CLUSTERING_REPORT.md`](CLUSTERING_REPORT.md) reads them: the distribution, the
chain-length check on the head, and the 17,083 distinct cluster pairs that cap
the val set.

**Run record.** The first attempt, `i-018e48483b1e9a989` (c7i.2xlarge, us-west-2a,
launched 2026-09-10T02:10:04Z), never left the extraction stage. It was terminated
after 8.3 hours having read about 12.5 GB of the 111.3 GB of tar shards, roughly
11%.

The cause was in `curate_extract_chains.py`, not in Foldseek. `iter_members`
handed a `pyarrow.fs` `open_input_stream` to `tarfile.open(mode="r|")`, and
tarfile reads in 10240-byte records, so every record became its own round trip to
S3. CloudWatch showed the shape of it plainly: NetworkIn flat at 750 MB per 30
minutes (1.5 GB/h) and CPUUtilization flat at 1.02% of 8 vCPUs, from minute 30 to
termination. I/O bound at 1/25th of the needed rate, which puts the extraction
pass at 74 hours and the run at roughly $27.

Measured on `shard_000_00000` from the workstation (cross-region, so read the
ratio and not the absolute numbers):

| read path | throughput |
|---|---|
| `open_input_stream`, 10240-byte reads (what ran) | 0.08 MB/s |
| `open_input_file().read()`, one whole-object read | 12.06 MB/s |

`io.BufferedReader` over the stream did not reliably fix it (3.77 MB/s on one
trial, 0.09 on another) because pyarrow returns short `readinto`s that leave the
buffer partly filled. Shards are ~250 MB, so the fix is to read each one as a
single object into a `BytesIO`. The script also gained a per-shard progress line
and a per-50,000-chain counter: the first attempt left nothing in the log but
5-minute heartbeats, so there was no way to see it was stuck without CloudWatch.

Verified after the fix, locally against the real bucket: 1,200 chains out of
`shard_000_00000` in 269.5 s, most of that the download.

Relaunched as `i-079d80d205e37cd42` (same instance type, subnet, and 200 GB gp3
volume) at 2026-09-10T10:43:12Z. That one extracted cleanly, 857,597 chains and
45.3 GB across all 441 shards in 14,471.9 s (4.02 h), and then died on the
completeness guard in `ec2_cluster_userdata.sh`, which compared the result against
a hard-coded `EXPECTED_CHAINS=858360`. That constant never subtracted the 762
complexes the fetch could not retrieve, so a complete extraction published
`_FAILED_extract_short_857597` and clustering never ran.

**The chain arithmetic, reconciled against `fetch_timings.csv.gz`.** The
assignment holds **881,009** val complexes, not the 881,010 quoted elsewhere in
this plan. 762 of them 404'd, 880,247 distinct models were fetched, and one,
`AF-0000000208803286`, was fetched twice, which is why the timings carry 880,248
rows. Of the 858,360 distinct accessions, 760 are reachable only through a 404'd
complex and 2 more fall back to a second complex that also contains them, so
**857,600 chains are expected**. The 4.02-hour run wrote 857,597, leaving exactly
one accession unexplained; `curate_extract_chains.py` now writes every absent
accession and its source model to `chains_absent.csv`, so the next run names it.

The guard also moved out of the shell and into the script, which knows the 404
list and can say which accessions are missing rather than reporting a count the
caller has to re-derive. There is deliberately no chain count left in the
user-data. Extraction is now a spawn-based `multiprocessing.Pool` over shards, one
worker per vCPU, which should take the 4.02 hours to roughly 35 minutes; verified
on two real shards, with 400 outputs byte-identical to the serial path's.

Relaunched again as `i-0ef9e10000084db14` at 2026-09-11T00:12:47Z. **That run
succeeded**, publishing `_DONE_rc0` and all five artifacts at 01:51:18Z: 1.64
hours end to end, about $0.58 at the c7i.2xlarge on-demand rate.

| stage | result |
|---|---|
| extract | 857,599 of 857,600 chains, 45,294,441,724 bytes, 51.6 kB mean, 441 shards, **2,227.6 s (37.1 min)** on 8 workers |
| cluster | **13,478 clusters** over 857,599 chains, **3,478.0 s (58.0 min)**, 4.12 GB peak RSS |

Parallelising extraction took it from 14,471.9 s to 2,227.6 s, a **6.5x speedup**
on 8 vCPUs, so clustering is now the longer of the two stages rather than the
shorter.

**The one absent chain, and its cause.** `chains_absent.csv` held exactly one row,
`A0A3Q1MTM2,AF-0000000208803286,A`. `AF-0000000208803286` appears **twice** in the
assignment as two different complexes, the homodimer `A0A3Q1MTM2/A0A3Q1MTM2` and
the heterodimer `A0A044RNL9/A0A044S3F6`, and it is the only model id in the val
assignment that does. `init_worker` inverted the accession-to-source map with
`by_model[model][chain] = accession`, so the heterodimer's chain A overwrote the
homodimer's and that accession was never requested. The same duplicate explains
both the 881,010-versus-881,009 complex count and the duplicate row in
`fetch_timings.csv.gz`.

`by_model` now holds a list of `(chain, accession)` pairs, which reproduces all
857,600 expected chains from the same assignment. The fix is not reflected in the
published `chain_cluster.tsv.gz`: one chain in 857,599 cannot move 13,478
clusters, and re-running extraction and clustering to add it would cost another
1.6 hours and $0.58. It is recorded here instead, and the missing accession is
excluded downstream like the 762 unfetchable models.

### Phase 4. Label each complex by its pair of chain clusters

Cheap, and the only step that was never the expensive part. For each complex, look
up the cluster of `uniprot_1` and of `uniprot_2` and label it
`cluster_{min}_{max}` from the sorted pair. Homodimers get `cluster_c_c`, which
falls out of the construction rather than needing a special case.

**Interface residues are dropped from the plan.** PINDER's 7-residues-both-sides
filter removes systems whose two chains barely touch, and it needs per-complex
contact residues, which is a full pass over the downloaded structures. It belonged
in the graph route, where interface length was one of the edge filters. Here it
would be a quality filter on the drawn set rather than an input to clustering, and
it is far cheaper to apply it to the few thousand complexes phase 5 draws than to
all 881,010. Moved to phase 5 as a filter on candidates.

Deliverable: `cluster_id` per `model_entity_id`, plus the distribution of distinct
cluster pairs.

**Run record. DONE 2026-09-11**, by `curate_label_pairs.py`, locally in about a
minute. All 881,010 val rows are in the output, each carrying a `status`, so the
count reconciles against the assignment and nothing is dropped quietly:

| status | rows |
|---|---|
| `labeled` | 880,247 |
| `unfetchable` (404'd during the fetch) | 762 |
| `unmapped_chain` | 1 |

Of the labeled complexes, 824,125 are homodimers and 56,122 heterodimers, giving
**17,083 distinct cluster pairs**, 10,151 of them same-cluster. The median pair
holds one complex, the largest holds 15,846, and **8,621 pairs (50.5%) hold
exactly one**, which is the number phase 5's interface filter can delete outright.

`model_entity_id` is not a unique key. `AF-0000000208803286` is two different
complexes and appears twice, once as `unmapped_chain` and once as a labeled
heterodimer; phase 5 must key on the row, not the model id.

`complex_cluster_pairs.csv.gz` is 881,010 rows and 14 MB, so it stays local and
gitignored. Committed alongside the script: `cluster_pair_index.csv.gz` (17,083
rows, one per distinct pair, with its complex and homodimer counts) and
`cluster_pair_summary.json`.

### Phase 5. Draw the validation set from cluster-pair representatives. DONE 2026-09-12

For each distinct `cluster_{min}_{max}`:

1. Confirm every member is flagged val-eligible by
   `dimer_split_assignment_id30_cov50.csv.gz`. With a val-only download it should
   not arise otherwise, and if it does, that is a finding worth reporting rather
   than silently resolving.
2. Drop candidates with fewer than **7 interface residues on both sides**,
   computed on the candidates only. This is PINDER's `min_interface_length`,
   applied here rather than in phase 4.
3. Exclude the 762 models in `data/curation/fetch_missing_404.csv`, which are in
   the split assignment but have no structure.
4. Draw **one representative per cluster pair** (`top_n = 1`), targeting a few
   thousand.

Report, so the cost of the criterion stays visible: how many cluster pairs exist,
how many complexes are drawn, how many are discarded for a thin interface, and how
many are discarded merely for sharing a cluster pair with the drawn one.

**Interface measurement, measured locally 2026-09-11.**
`curate_interface_residues.py` counts, per side, the residues with a heavy atom
within a contact radius of the partner chain. PINDER publishes
`min_interface_length = 7` without the radius it was computed at, so the radius is
a flag defaulting to 5.0 A (the CAPRI/DockQ heavy-atom convention) rather than an
inherited constant.

Cost on the 45 local dimers in `sample/structures` (median 5,470 heavy atoms, max
15,926), single core: **74 ms per structure**, 18 ms parsing the mmCIF and 56 ms
on contacts at one radius. An exact residue-level prefilter does most of the work:
a residue pair can only touch if their centroids are within the sum of their
measured enclosing radii plus the contact radius, which cuts the atom-against-atom
pass by 4.6x (11.59 s to 2.53 s over the 45) and reproduces the unfiltered counts
exactly on every one of them.

That puts all 880,247 labeled complexes at 18 h single core, **2.3 h on 8 vCPUs**,
against 2.6 min for the 17,083 drawn candidates alone. Reading the 111.3 GB of
shards costs 37 min either way, so computing the interface for every complex adds
about $0.83 and produces a table that can be re-thresholded without touching S3
again. Run it over everything.

**The threshold looks close to a no-op on this data**, on the evidence available.
All 45 sample dimers pass at every radius from 4 to 10 A, and the thinnest
interface among them is 10 residues at 4 A against a threshold of 7. These 45 came
from the confidence-gated smoke fetch rather than a random draw, so this is a hint
and not a yield estimate; the real numbers come from the run. Per-structure
counts and timings: `data/curation/interface_sample_45.csv`.


**Run record. DONE 2026-09-12.** `i-02756293aca0ea4e1` (c7i.2xlarge, us-west-2a)
measured all 31,956 candidates in **486 s**, launched 00:26:35Z and `_DONE_rc0` at
00:35:33Z, about $0.06. It read the 441 shards at 229 MB/s, 4.6x faster than the
37 minutes projected from the extraction pass, because that pass was also writing
45 GB of gzip to EBS. No candidate was missing from the shards.

The interface filter barely bites, as the 45-structure sample suggested.
**31,929 of 31,956 candidates clear 7 residues on both sides at 5 A**, a 0.08%
failure rate, and the drawn set's median thinnest side is 37 residues against a
threshold of 7.

| the four counts this phase owes | |
|---|---|
| cluster pairs | 17,083 |
| complexes drawn | **17,067** |
| discarded for a thin interface | 24 measured, costing 16 pairs |
| discarded only for sharing a pair with the drawn one | 863,156 |

17,064 pairs are represented by their top-ranked candidate, so carrying three
candidates per pair instead of one rescued **3 pairs**. Cheap insurance at 9
minutes of runtime, and not worth repeating at a larger multiple.

The draw is **40.8% heterodimers** (6,963 of 17,067) against 6.4% in the corpus,
which is the chain clustering doing its job: a homodimer-heavy release collapses
hard, and the rarer heterodimers survive deduplication in much greater proportion.
It covers 13,473 of the 13,478 chain clusters. The 16 lost pairs are named in
`data/curation/val_set_pairs_lost.csv`.

**Confidence is not gated, and the tail is soft.** ipTM across the drawn set runs
from 0.099 to 0.970 with a median of 0.768; **22.3% sit below 0.5**, and 3,017 of
those 3,811 are pairs holding a single complex, where there was no more confident
alternative to pick. Ranking by ipTM cannot fix a pair whose only member is a weak
prediction. Adding a floor would trade cluster-pair coverage for per-structure
quality, and that trade is a decision for whoever uses the set rather than one to
bury in the draw.

Deliverable: `data/curation/val_set_dimers.csv`, 17,067 rows carrying the model id,
the pair label, both clusters, both accessions, ipTM, pDockQ2 and the measured
interface counts.

### Phase 6. Publish

Structures and per-complex tables to the HF bucket under
`data/afdb-nvda-dimers/`, following the `AGENTS.md` prefix rule. Small artifacts
(cluster-size distribution, split counts, scaling table, the 404 manifest) stay in
`data/` in git. A dataset README stating the gate, the split rule, the thresholds
and the exact counts ships with it.

## Costs and unknowns

| Item | Status |
|---|---|
| EBI aggregate rate limit | **measured**: linear to 8 workers, 11.5 files/s, no throttling; GCS-backed |
| Structure transfer | **measured** over 880,248 models: 122.7 kB stored gzip, 533.7 kB decoded, 472.2 kB actually on the wire (84% of replies arrive uncompressed) |
| Missing models | **measured**: 762 of 881,010 (0.09%); manifest in `data/curation/fetch_missing_404.csv` |
| Storage and AWS cost | **measured** for val-eligible: $1.69 compute (18.9 h x $0.08925), $0.004 PUT (882 objects, not 881,010, which is what the tar packing bought), $2.37/mo storage; full set still estimated at ~$32 |
| PINDER's exact parameters | **read**, table above |
| Sequence collapse factor | **measured**: 7.3% / 16.6% / 33.1% at id 0.3 / 0.5 / 0.7; not used, see phase 1 |
| Foldseek clustering cost | **measured**: easy-cluster is linear, 5.5 s + 7.2 ms/chain at 0.86 GB, so the full 858,360 chains project to ~1.7 h on 4 cores. The rejected search route was n^1.21 and ~$210 |
| Cluster count at 858k chains | **unknown**. Representative fraction falls 55% to 21% over the measured range; extrapolating 254x is not honest. Any plausible value exceeds what phase 5 draws |
| Interface-residue computation | **descoped**: applied in phase 5 to the few thousand drawn candidates, not to all 881,010 |

The sequence-only split needed no structures. This one does, and the download is
now the largest thing here at 103.7 GiB; everything downstream of it turned out to
be small. Both of the expensive-looking phases shrank once measured. Phase 1's
sequence collapse was dropped because PINDER does not collapse before clustering,
and phase 3's fan-out was dropped because Foldseek clustering is linear and fits
in one job. What is left of the plan is one clustering run, a join, and a draw.
