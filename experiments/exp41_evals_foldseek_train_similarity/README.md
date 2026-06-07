---
marinfold_experiment:
  issue: 41
  title: "curate an eval set of designed proteins and low-MSA proteins"
  kind: evals
  branch: exp/41-foldseek-train-similarity
---

# Foldseek train-set similarity tool (issue #41)

**Issue:** [#41](https://github.com/Open-Athena/MarinFold/issues/41) · **Kind:** `evals` · **Branch:** `exp/41-foldseek-train-similarity`

## Question

For any candidate protein structure (a FoldBench monomer, a de novo
design, a low-MSA natural protein), **how structurally close is it to
anything in MarinFold's training set?** This is the reusable capability
[#41](https://github.com/Open-Athena/MarinFold/issues/41) asks for:
"code for taking additional test structures and map those back onto
structural clusters to see if they are very similar to things in our
training set."

## Hypothesis

MarinFold trains on [`afdb-24M`](https://huggingface.co/datasets/timodonnell/afdb-24M),
which is *already* Foldseek-clustered: every row carries a
`struct_cluster_id` (its structural cluster representative) and a `split`
(train/val/test) hashed from that id, so a whole fold lands in one split.
So we don't re-cluster — we `foldseek easy-search` a candidate against a
target DB of the training-set cluster **representatives**
(`--alignment-type 1` → TM-score) and join each hit back to its split.
A candidate that matches a *train* representative at high TM is close to a
trained fold; one whose best train match is below the same-fold boundary
is genuinely novel.

## Approach

Two pieces, deliberately separated:

- **Query tool (reusable, the deliverable):** `query_similarity.py` takes a
  dir of candidate `.cif`/`.pdb` files and a prebuilt representative DB +
  manifest, runs `foldseek easy-search`, and writes a per-candidate CSV
  with the nearest training representative, its TM-score, its split, the
  free sequence-identity signal (`fident`), and a verdict. It never touches
  `afdb-24M` directly, so the repeatedly-run path is light and offline.
- **Representative DB (built once on Modal):** `build_db_modal.py` extracts
  all 1.33M cluster representatives (`uniprot_accession == struct_cluster_id`)
  across the 12,005 `afdb-24M` shards into one `foldseek createdb` target DB
  + a `representative_id -> split` manifest, on a Modal Volume; `fetch_db.py`
  pulls it local.

### Files

| File | What it does |
|---|---|
| `foldseek_env.py` | Locate/install the Foldseek binary; `run_foldseek()`. |
| `query_similarity.py` | **The deliverable.** candidate cifs → easy-search → similarity CSV. |
| `build_db_modal.py` | Full DB build on Modal: all 12,005 shards → 1.33M-rep `targetDB` + manifest on a Volume. |
| `fetch_db.py` | Download the built `targetDB*` + manifest from the Modal Volume to a local dir. |
| `fetch_foldbench_candidates.py` | Pull the FoldBench-100 GT cifs as the first candidate set. |
| `collect_timings.py` | Batched (+ per-candidate) search wall-time + worker metadata → `data/timings.csv`. |
| `plot_similarity.py` | Verdict histogram, structure-vs-sequence scatter, search-timing curve. |
| `tests/test_query.py` | Unit tests for the parse/join/verdict path. |

### Running it: smoke test vs full run

The same scripts do both; only the scale changes (a `--limit-shards` smoke
build vs the whole scan). There are no separate smoke-test scripts. Always
start with:

```bash
uv sync
uv run python foldseek_env.py install        # one-time: foldseek binary -> local cache
uv run python fetch_foldbench_candidates.py   # the 100 GT candidate cifs
```

**Smoke test** (minutes, ~$0.05) — proves the pipeline end to end without the
full 1.3 TB scan:

```bash
# (a) pure logic, no foldseek / network / Modal (<1 s):
uv run --extra test pytest tests/
# (b) real end-to-end on a THROWAWAY Volume (so it can't collide with the full
#     build's committed batches): build ~20 shards -> a few-hundred-rep DB in
#     ~2 min (no --detach; it finishes before the client exits), then query.
MARINFOLD_FOLDSEEK_VOLUME=afdb-foldseek-smoke \
    uv run --extra modal modal run build_db_modal.py --wait \
    --limit-shards 20 --shards-per-batch 10 --snapshot-tag smoke-20shards
uv run --extra modal python fetch_db.py --volume afdb-foldseek-smoke --out db_smoke
uv run python query_similarity.py \
    --candidate-dir candidates/foldbench/data/protenix-foldbench-monomers/gt \
    --db db_smoke/db/targetDB --reps-manifest db_smoke/reps_manifest.csv \
    --out /tmp/foldbench_vs_smoke_similarity.csv --db-tag afdb-24M-smoke-20shards
```

Against such a sparse DB most candidates come back `same_fold`/`novel_fold` —
that is expected. The smoke test checks that extract → merge → fetch → query
runs and emits a well-formed CSV, *not* the verdicts (those need the full DB).

**Full run** (the real measurement; ~5 h build, ~$20–30):

```bash
# 1. build the full 1.33M-rep DB on Modal (server-side detached; survives client exit):
uv run --extra modal modal run --detach build_db_modal.py
#    ...or skip the build and pull the already-built DB:
uv run --extra modal python fetch_db.py --volume afdb-foldseek-reps-full --out db_full
# 2. query the candidates against it:
uv run python query_similarity.py \
    --candidate-dir candidates/foldbench/data/protenix-foldbench-monomers/gt \
    --db db_full/db/targetDB \
    --reps-manifest db_full/reps_manifest.csv \
    --out data/foldbench_vs_full_reps_similarity.csv \
    --db-tag afdb-24M-full-reps-1331330
```

`fetch_db.py` and the `modal` CLI need the `modal` extra; the query / plot /
test path stays Modal-free.

### Output schema (`*_similarity.csv`, one row per candidate)

`stem, n_residues, best_target_rep, best_target_split, best_alntmscore,
best_qtmscore, best_train_target_rep, best_train_alntmscore,
best_train_qtmscore, best_train_fident, tm_field, n_hits_tm_ge_fold,
n_train_hits_tm_ge_fold, verdict, fold_tm, redundant_tm, foldseek_version,
db_snapshot_tag`.

The `best_train_*` columns are the leakage-relevant signal (nearest
*training* representative). Verdict (on the nearest-train TM in `tm_field`):
`redundant` ≥ 0.9, `same_fold` ≥ 0.5, else `novel_fold`. The `(stem,
n_residues)` keys join onto exp20's `data/timings.csv`.

### Things verified against the live tools (not assumed)

- `struct_cluster_id` is the representative's **UniProt accession**
  (e.g. `K7TTU0`), not an AFDB `entry_id`. A representative's own row is
  `uniprot_accession == struct_cluster_id` (≈5.5% of rows).
- Foldseek names DB entries by **filename stem** (it ignores the internal
  `data_` block), so we name representative cifs `<struct_cluster_id>.cif`
  and the join is exact.
- Foldseek's `alntmscore` can slightly **exceed 1.0** on self-matches
  (e.g. 1.004); `qtmscore`/`ttmscore` are bounded in [0, 1]. The verdict
  uses `qtmscore` by default for this reason (and because it's
  query-normalized, fair to short candidates). Tunable via `--tm-field`.

## Success criteria

A reusable query tool that, given candidate cifs, emits a per-candidate
similarity/verdict CSV. Validated by unit tests on the parse → join → verdict
path and by the real FoldBench-100 run against the full 1.33M-rep DB (sane
TM-scores, splits join, verdicts as expected).

## Results

The full representative DB is **built** (see "Full representative DB"
below): all 12,005 `afdb-24M` shards scanned on Modal, **1,331,330**
cluster representatives (`uniprot_accession == struct_cluster_id`)
extracted, `foldseek createdb` per shard-batch, `concatdbs`-merged into
one 2.6 GB target DB. Manifest split fractions **0.980 / 0.0099 / 0.0099**
(train/val/test) match the expected ~0.98 / 0.01 / 0.01, confirming the
split-hash assumption; DB integrity checks out (`targetDB.index` =
1,331,330 entries = manifest rows, no duplicate ids).

Querying the **FoldBench-100** monomers against the full 1.33M-rep
training set:

| verdict | rule (nearest-train `qtmscore`) | candidates |
|---|---|---|
| `redundant` | >= 0.9 | **48** |
| `same_fold` | 0.5 to 0.9 | **51** |
| `novel_fold` | < 0.5 | **1** |

So **99 / 100** FoldBench monomers have a same-fold-or-closer match in
MarinFold's training set, and **48** are structurally near-identical to a
training representative (qtm >= 0.9). The lone novel candidate is `7xcd_A`
(nearest-train qtm 0.485). Nearest-train `qtmscore` across the 100: min
0.485, median **0.895**, mean 0.865, max 0.985. For 97/100 the single
nearest representative is a *train*-split rep; the 3 whose global nearest
is val/test carry a near-equal train match anyway (e.g. `7wzm_A`: best
test 0.908 vs best train 0.907), so the leakage verdict is unchanged.
The unit tests pass (`uv run --extra test pytest tests/`).

**The overlap is structural, not sequence** (`plots/struct_vs_seq.png`).
Of the 99 candidates with a same-fold-or-closer match, **65 sit below 0.30
sequence identity** (`fident`) to that representative and 31 below 0.20;
42 of the 48 `redundant` matches are under 0.50 fident (median 0.33). The
scatter is a band at high structural TM spanning the full range of
sequence identity. A sequence-only filter (e.g. MMseqs at 30% id) would
clear most of these as "novel" while structurally they are near-duplicates
of trained folds (exactly the leakage a single-sequence model faces, and
the reason structural rather than sequence clustering is the right axis).

### Performance / timing

`collect_timings.py` records per-input search wall-time + worker metadata
to `data/timings.csv` (schema mirrors exp20/exp12: `stem, n_residues,
elapsed_seconds, model_nickname, runner_tag, hostname, platform,
timestamp_utc`, plus foldseek-specific `n_db_reps, alignment_type, n_hits,
foldseek_version`; no GPU/torch columns (CPU TM-align). On an Intel
i7-10610U (8 cores), FoldBench-100:

| DB | mode | wall-time |
|---|---|---|
| 229-rep prototype | `per_candidate` | min 0.93 s · median 2.03 s · max 4.68 s (761 res) |
| 229-rep prototype | `batched` | 74.7 s → **0.75 s/candidate** amortized |
| **1.33M-rep full** | `batched` | 338 s → **3.38 s/candidate** amortized |

Search time scales with candidate length on the prototype DB
(`plots/search_timing.png`); the per-candidate mode pays Foldseek's
process + query-DB setup each call, so batched is ~3x faster per
candidate. The full-DB per-candidate mode is **deliberately skipped**:
each call reloads the 2.6 GB DB and rebuilds the ~37M-residue index, so
100 separate calls would be ~100x-redundant setup. The meaningful full-DB
number is the batched throughput: even against 1.33M reps the amortized
cost is **3.4 s/candidate** on 8 CPU cores (Foldseek's 3Di prefilter keeps
TM-align to a small candidate set; an `--gpu` search path is available).
Re-run with:

```bash
uv run python collect_timings.py \
    --candidate-dir candidates/foldbench/data/protenix-foldbench-monomers/gt \
    --db db_full/db/targetDB --db-tag afdb-24M-full-reps-1331330 \
    --no-per-candidate
```

## Full representative DB (built)

The full DB is built and lives on a Modal Volume (`afdb-foldseek-reps-full`
by default; override with `MARINFOLD_FOLDSEEK_VOLUME`); `fetch_db.py` pulls
it to `db_full/`. A server-side, detached Modal app (`build_db_modal.py`, `max_containers=10`)
scanned all 12,005 `afdb-24M` shards in ~101 batches: each batch filters
rows where `uniprot_accession == struct_cluster_id` (its own cluster
representative), writes their `cif_content` to a worker tmpdir, and runs
`foldseek createdb` into a per-batch sub-DB; a reducer `concatdbs`-merges
the 101 sub-DBs into one `targetDB`. Result: **1,331,330** representatives
(~5.5% of the 24M rows), a **2.6 GB** four-component structure DB
(3Di + AA + Cα coords + headers; ~8 B/residue), splits 0.980 / 0.0099 /
0.0099. The build is idempotent/resumable (committed batches skip).

**The DB is published** as a public HF bucket,
[`silterra/afdb-24M-foldseek-train-reps`](https://huggingface.co/buckets/silterra/afdb-24M-foldseek-train-reps),
so it can be fetched without rebuilding or Modal access. Fetch + query:

```bash
# easiest: pull the published DB from the HF bucket (needs the `hf` CLI)
hf buckets sync hf://buckets/silterra/afdb-24M-foldseek-train-reps db_full
# ...or, if you rebuilt it yourself, pull from the Modal Volume instead:
#   uv run --extra modal python fetch_db.py --volume afdb-foldseek-reps-full --out db_full
uv run python query_similarity.py \
    --candidate-dir candidates/foldbench/data/protenix-foldbench-monomers/gt \
    --db db_full/db/targetDB \
    --reps-manifest db_full/reps_manifest.csv \
    --out data/foldbench_vs_full_reps_similarity.csv \
    --db-tag afdb-24M-full-reps-1331330
```

The bucket carries `db/targetDB*`, `reps_manifest.csv`, and a provenance
`README.md` (afdb-24M revision, foldseek version, build date, rep count).
See `.dev/eval-strategy-summary.md` for how the nearest-train TM feeds the
eval split.

## Conclusion

The reusable query tool works against the real training set. Against the
full 1.33M-rep `afdb-24M` representative DB, **99 / 100 FoldBench-100
monomers fall in a fold MarinFold trained on** (qtm >= 0.5), 48 of them
near-identically (qtm >= 0.9), with just one genuinely novel candidate
(`7xcd_A`). The prototype's 229-rep slice had reported 32 same-fold / 68
novel; that "novelty" was almost entirely a sparse-DB artifact, which is
exactly why the full build mattered. Critically the overlap is
**structural, not sequence** (65 / 99 matches below 0.30 sequence
identity), so a sequence-only leakage filter would badly undercount it.
For building a held-out eval set, FoldBench monomers are a poor source of
*novel* folds relative to this training set; the nearest-train `qtmscore`
(median 0.895) is the per-candidate knob for sub-stratifying any candidate
pool by how far it sits from trained structure.
