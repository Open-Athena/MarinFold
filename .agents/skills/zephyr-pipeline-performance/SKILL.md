---
name: zephyr-pipeline-performance
description: Write a Zephyr/Iris data-generation pipeline that finishes in minutes, not hours. Use when authoring or reviewing an `exp<N>_data_*/cli.py` (or any `map_shard`-based job that fetches per-row inputs and emits parquet). Covers the per-row / per-shard / per-worker / per-job decisions that separate an 8-minute run from an overnight one.
---

# Skill: write performant Zephyr pipelines

A Zephyr pipeline that *works* and one that *finishes in budget* differ on a
small set of decisions you make once at the top of `cli.py`. Most of them
are unobvious from the marin-zephyr docs because the costs aren't local —
they show up as straggler tails, silently-truncated reads, or cluster bills,
not as wrong output.

This skill captures the lessons from two real MarinFold pipelines that
chose differently:

* [`experiments/exp5_data_contacts_and_distances_v2_zephyr/`](../../../experiments/exp5_data_contacts_and_distances_v2_zephyr/) —
  1.6 M structures, ~8 min wall-clock end-to-end on Iris.
* [`experiments/exp53_data_contacts_v1_zephyr/`](../../../experiments/exp53_data_contacts_v1_zephyr/) —
  4.2 M structures (pyconfind-heavy), ~31 min projected after two hard-won fixes.
  The pre-fix version of exp53 (commits prior to
  [`43a0535`](https://github.com/Open-Athena/MarinFold/commit/43a0535)
  and [`daa18e1`](https://github.com/Open-Athena/MarinFold/commit/daa18e1))
  silently dropped 98 % of rows and re-parsed a multi-second per-worker
  setup every shard — translating into hours of cluster wall-clock for the
  same output.

When you're done, your pipeline should clear every box in the
[Pre-launch checklist](#pre-launch-checklist) at the end of this skill.

---

## TL;DR — the five decisions that dominate

1. **Inside `map_shard`, run a `ThreadPoolExecutor` around the per-row
   fetch+parse.** GCS GETs are ~30–80 ms; gemmi parse releases the GIL.
   Threading them is the single biggest per-shard speedup — exp5's prior
   sequential version landed at ~128 s/shard; threaded landed at ~few s.
2. **`--worker-cpu 1`, scale via `--max-workers`.** Per-shard CPU is
   single-threaded Python once the in-shard thread pool overlaps the I/O.
   Asking for more cores per worker wastes cluster capacity.
3. **Pin `--region` and use `--preemptible`.** Without region pinning, an
   over-requested on-demand pool spills cross-continent (exp53 spilled to
   `europe-west4` and got a multi-hour straggler tail). With preemptible
   workers you get more in-region capacity at lower cost; Zephyr retries
   preemptions.
4. **Fetch per-row objects via
   `marinfold.document_structures.io.read_object_bytes`** (a full
   `cat_file` GET), not `fsspec.open().read()`. GCS objects with
   `Content-Encoding: gzip` report compressed size in `Content-Length`;
   a size-based read truncates large objects mid-content. The bug is
   silent — your parser fails downstream with a confusing
   "unexpected end" error, not an obvious I/O fault.
5. **Source per-row data via the manifest's `gcs_uri` pointer column,
   not the inline `cif_content` column.** Datasets like afdb-1.6M ship
   every cif twice: a one-line `gcs_uri` pointer to AFDB's public GCS
   bucket, *and* the full inline mmCIF text as `cif_content`. Reading
   `gcs_uri` means a small fan-out of in-cloud (often in-region) GCS
   GETs; reading `cif_content` means every worker streams the bulk
   mmCIF back from HuggingFace cross-cloud per shard. Same downstream
   output, ~2,000× less manifest I/O plus a fundamentally cheaper
   per-row fetch. Default to `--cif-uri-column=gcs_uri`; reserve
   `--cif-text-column` for local testing or inputs without a URI column.

The rest of this skill walks through each layer of the pipeline (per-row →
per-shard → per-job → per-output) with the code patterns these decisions
imply.

---

## Per-shard: thread the I/O, memoize the heavy init

A Zephyr `map_shard` body runs once per input shard. Two things matter:

### 1. ThreadPoolExecutor inside `map_shard` for per-row fetches

Without it, your per-shard wall-clock is the **sum** of per-row GCS GETs
(sequential I/O). With it, fetches overlap each other *and* overlap the
CPU work of rows already fetched. gemmi (and pyconfind) release the GIL
during the C++ parse, so the threads make real progress in parallel.

The pattern is captured as
[`marinfold.document_structures.io.thread_per_row_in_shard`](../../../marinfold/marinfold/document_structures/io.py)
so each new pipeline gets the correctness details (pool size cap,
`pool.map` for input-order output, `None`-skip for failed rows) by
construction rather than re-deriving them:

```python
from functools import partial
from marinfold.document_structures.io import thread_per_row_in_shard

def generate_shard(items, shard_info, *, cfg, fetch_concurrency=32, ...):
    worker = partial(_generate_doc_for_row, cfg=cfg, ...)
    yield from thread_per_row_in_shard(
        items, worker=worker,
        fetch_concurrency=fetch_concurrency,
        thread_name_prefix="exp<N>-fetch",
    )
```

**Default `--fetch-concurrency=32`** is well-calibrated for 30–80 ms GCS
GETs and ~6 ms of CPU per row. Bump it if your CPU step is heavier (e.g.
pyconfind-based contact computation: exp53 still uses 32, but its CPU
dominates so the threads matter less).

If you have *no* per-row I/O (inline cif text in the manifest), skip the
helper — there's nothing to overlap and a thread pool is pure overhead.
exp53's `generate_shard` shows the branch:

```python
if cif_text_column is not None:
    # Inline path: no I/O to overlap, run sequentially.
    for row in items:
        out = worker(row)
        if out is not None:
            yield out
    return
# URI path: thread the fetches via the shared helper.
yield from thread_per_row_in_shard(items, worker=worker, ...)
```

### 2. Memoize per-worker init in a module global

A Zephyr worker process serves many shards over its lifetime. If your
algorithm requires a heavy per-process resource (parsed reference library,
loaded model, JIT-compiled kernel), **do not initialize it per shard or per
row** — cache it in a module-level global so it's parsed once per worker.

exp53's
[commit `daa18e1`](https://github.com/Open-Athena/MarinFold/commit/daa18e1)
is the canonical example: pyconfind's rotamer library takes tens of seconds
to parse. Before the fix, every shard paid that cost. After:

```python
_ROTAMER_UNSET: Any = object()
_ROTAMER_LIBRARY: Any = _ROTAMER_UNSET

def _load_rotamer_library() -> Any | None:
    global _ROTAMER_LIBRARY
    if _ROTAMER_LIBRARY is not _ROTAMER_UNSET:
        return _ROTAMER_LIBRARY  # second-and-onward shards: instant
    try:
        from pyconfind import load_library, cached_rotamer_library
        _ROTAMER_LIBRARY = load_library(cached_rotamer_library())
    except Exception as exc:
        warnings.warn(f"preload failed ({exc}); per-call load", stacklevel=2)
        _ROTAMER_LIBRARY = None
    return _ROTAMER_LIBRARY
```

Use a sentinel (`_ROTAMER_UNSET`) rather than `None`, so the failure path
(genuinely returned `None`) caches and doesn't retry on every call.

This pattern compounds: a 30 s per-worker init across 500 workers × 100
shards-per-worker is a **41-hour cluster bill** if you do it per shard;
30 s × 500 once-per-worker is a 15-second blip.

---

## Per-job: cluster resources

The `ZephyrContext` / Iris `job run` call is where you decide how the
cluster spends its time. Two rules dominate.

### `--worker-cpu 1`, scale via `--max-workers`

Per-shard CPU is single-threaded Python once the in-shard thread pool
overlaps the I/O. A worker with 4 CPUs just leaves 3 idle. Use the smallest
worker that fits one Python process + your peak memory, then scale out:

```bash
uv run iris --cluster=marin job run --cpu 1 --memory 2GB -- \
    python cli.py generate \
        --worker-cpu 1 --worker-memory 4g --worker-disk 32g \
        --max-workers 512 --fetch-concurrency 32 \
        ...
```

Note the *two* cpu/memory pairs: the **outer** `--cpu`/`--memory` on
`iris job run` size the **launcher container** (tiny); the **inner**
`--worker-cpu`/`--worker-memory` (consumed by `ZephyrContext`) size each
**worker pod**. They're not the same thing and overspending on the launcher
is wasteful but harmless; overspending on the workers multiplies by 500+
and is not.

### Pin `--region`, request `--preemptible`

The shared marin Iris cluster straddles regions and continents (see the
project root [`AGENTS.md`](../../../AGENTS.md) "Cross-region data transfers
& the shared Iris cluster"). Without pinning, large worker pools spill to
fallback regions and your job's tail latency becomes "the slowest worker
in `europe-west4` dragged across the Atlantic":

```python
ctx = ZephyrContext(
    max_workers=args.max_workers,
    resources=ResourceConfig(
        cpu=args.worker_cpu,
        ram=args.worker_memory,
        disk=args.worker_disk,
        regions=[args.region] if args.region else None,   # pin
        preemptible=args.preemptible,                      # cheaper + more capacity
    ),
)
```

```python
# argparse:
p.add_argument("--region", type=str, default="us-central1",
               help="Pin workers to this GCP region (match the iris cluster) "
                    "so a large pool can't spill cross-region/continent.")
p.add_argument("--preemptible", action=argparse.BooleanOptionalAction,
               default=True,
               help="Request preemptible/spot workers — far more in-region "
                    "capacity + cheaper than on-demand; zephyr retries "
                    "preemptions.")
```

exp53's
[commit `7c19d5d`](https://github.com/Open-Athena/MarinFold/commit/7c19d5d)
added these knobs after the first full run got a multi-hour straggler tail
from cross-continent on-demand spills.

### Match output bucket to the cluster's region

`gs://marin-<region>/protein-structure/MarinFold/<exp>/` — the bucket
**must** be in the same region as the workers, or every per-row PUT is a
cross-region transfer (cost + latency). For the Iris cluster's
`us-central1` default, that's `gs://marin-us-central1/...`. See the root
[`AGENTS.md`](../../../AGENTS.md) "GCS bucket" section.

---

## Per-input: source via `gcs_uri`, project columns aggressively

Two levers, in order of impact.

### 1. Pick the right data-source column

afdb-1.6M (and other AFDB-derived manifests on HuggingFace) ship every
cif *twice*: as a one-line `gcs_uri` pointer to the public AFDB GCS
bucket, **and** as the full inline mmCIF text in a `cif_content` column.

The choice between them dominates everything else in this section:

- **`gcs_uri`** → workers fetch from AFDB's GCS bucket directly. For a
  GCP-based cluster (the marin Iris pool), that's in-cloud and usually
  in-region — fast network, no cross-cloud egress charges. Reading the
  manifest costs ~160 KB/shard (just the URI strings + provenance).
- **`cif_content`** → every worker reads the bulky inline mmCIF back
  from HuggingFace, cross-cloud, per row. Reading the manifest costs
  ~70 MB/shard (~2,000× more I/O), and that bulk crosses the open
  internet rather than staying inside GCP.

Default to `--cif-uri-column=gcs_uri`. Reserve `--cif-text-column` for
local tests (where inline cif keeps the path off the network entirely)
or for input manifests that don't ship a URI column.

A subtle related trap: if you have to enumerate the HF manifest's
*shards* up front, do it through an authenticated `HfFileSystem` with a
pre-warmed dircache + the `/resolve/` CDN URLs (one API call to list, N
CDN reads for the data). Per-shard `hf://` reads make one `paths-info`
API call each, which blows HF's 3,000 req / 5 min quota on a
multi-thousand-shard manifest. See the "HuggingFace API quota" trap
below.

### 2. Read only the columns you need

Parquet is columnar.
`Dataset.from_files(input).load_parquet(columns=…)` reads only the
requested columns. With `gcs_uri` picked, the minimal set is
`[entry_id, gcs_uri, <optional passthrough>]` — ~160 KB/shard. Reading
the full row (including `cif_content`) is ~70 MB/shard regardless of
which fetch path you take downstream — column projection is what
*activates* the data-source win.

The exp5 pattern: peek the manifest schema once at submission time,
decide which optional passthrough columns are present, then pass the
closed-over list to every shard's load:

```python
_OPTIONAL_PASSTHROUGH = ("split", "seq_cluster_id", "struct_cluster_id",
                         "uniprot_accession", "tax_id", "organism_name", "gcs_uri")

def _resolve_input_columns(input_path, cif_col):
    """Peek the first parquet file, intersect with desired columns."""
    import pyarrow.parquet as pq
    fs, _ = fsspec.core.url_to_fs(input_path)
    matches = sorted(fs.glob(input_path))
    with fsspec.open(fs.unstrip_protocol(matches[0]), "rb") as f:
        present = set(pq.ParquetFile(f).schema_arrow.names)
    # Validate required + collect optional passthrough.
    if "entry_id" not in present or cif_col not in present:
        raise ValueError(...)
    passthrough = [c for c in _OPTIONAL_PASSTHROUGH if c in present]
    columns = ["entry_id", cif_col] + [c for c in passthrough if c not in ("entry_id", cif_col)]
    return columns, passthrough
```

The schema peek happens once on the controller, not 1,000 × on workers.
The same column list is used for every shard so the output schema is
stable across shards (downstream readers can concatenate without
schema-merging).

---

## Per-output: one parquet per input shard

Use a `{shard}` placeholder in `--out` so Zephyr writes one output file
per input shard:

```bash
--out "gs://marin-us-central1/.../corpus-{shard:05d}-of-{total:05d}.parquet"
```

This is load-bearing for two reasons:

1. **Streaming write throughput** — many small writers in parallel beat one
   centralized writer's serialized output stream.
2. **Provenance + restartability** — if shard 042 fails, you re-run with a
   filter on input shard 042 and the rest of the output stays intact. See
   `experiments/exp53_data_contacts_v1_zephyr/rerun_missing.py` for the
   resume pattern.

For smoke tests, omit `{shard}` (or pass `--num-docs N`) — that collapses
to a single output file:

```python
if "{shard" not in args.out:
    out_rows = out_rows.reshard(1)
```

The output ordering policy is up to you; exp5 + exp53 both use
`executor.map` (preserves input order within a shard) for determinism.
exp53 additionally orders Stage A's input shards in descending pLDDT-round
order so the highest-quality data is read *last* by streaming consumers.

---

## I/O traps that silently destroy throughput

### AFDB GCS objects are gzip-transcoded; use the shared reader

`Content-Encoding: gzip` objects report their **compressed** size in the
content-length header. A size-based read (`fsspec.open(uri, "rb").read()`
or `compression="infer"` + slicing) reads only that many bytes of the
decompressed stream — silently truncating large cifs mid-`_atom_site`.
gemmi then raises `"Wrong number of values in loop _atom_site"`.

The fix is to use a single full GET (the filesystem's `cat_file`) that
reads to EOF. This is what
[`marinfold.document_structures.io.read_object_bytes`](../../../marinfold/marinfold/document_structures/io.py)
encapsulates — same `None`-on-failure contract as the threading helper
above, so they compose directly:

```python
from marinfold.document_structures.io import (
    read_object_bytes, thread_per_row_in_shard,
)

def _generate_doc_for_row(row, *, cif_uri_column, ...):
    cif = read_object_bytes(row[cif_uri_column])  # full GET, gzip-safe
    if cif is None:
        return None
    ...
```

Symptom check: if your smoke test produces a suspiciously low success rate
(e.g. < 5 % of rows yielding output), **inspect a sample of failures**
before scaling up. A near-total silent drop with an otherwise-working
pipeline shape is a near-certain I/O truncation bug.

### Requester-pays buckets (AFDB)

The AFDB GCS bucket is requester-pays. Local user credentials get
truncated reads; Iris workers' service account reads fine. Means:
**the `gcs_uri` per-row fetch path is only validatable on-cluster**, not
on your laptop. Plan a small `--num-docs 100` smoke on Iris before the full
run (exp53 did this).

### HuggingFace API quota when listing many shards

afdb-24M has ~12,000 parquet shards. Per-shard `hf://` reads make one
`paths-info` API call each, blowing through HF's 3,000 req / 5 min quota
on a single dataset listing. The mitigation lives in exp53's
[`fetch_manifest_columns.py`](../../../experiments/exp53_data_contacts_v1_zephyr/fetch_manifest_columns.py):
use an authenticated `HfFileSystem` with a **pre-warmed dircache** + the
`/resolve/` CDN URLs for the actual reads. One API call to list, N CDN
reads for the data.

### `httpx<1` for marin-iris

The marin-iris GCP provider calls `httpx.Client(timeout=...)`, removed in
the 1.0 prerelease. Pin `httpx<1` in your `pyproject.toml` base deps. (exp53
discovered this in its HANDOFF; pin it preemptively.)

---

## Stage the work: cheap selection, then heavy generation

Large data pipelines benefit from a **two-stage** layout:

* **Stage A — selection** is metadata-only. Read a few small columns from
  the input manifest (entry_id, pLDDT, cluster ids, ...), compute the
  selection / shuffling / round assignment in Python or DuckDB, and emit a
  *new* parquet manifest. This stage runs in seconds-to-minutes on one
  process and is trivially re-runnable.
* **Stage B — generation** consumes Stage A's manifest and does the
  per-row fetch + heavy compute. This is the Zephyr/Iris stage with all
  the lessons above.

exp53 split this way: `selection.py` reduces 24 M structures → 4.2 M
selected (entry, round) records in ~13 s, then `cli.py generate` runs the
heavy per-row work on the 4.2 M-row manifest.

**Why**: the selection logic changes more often than the generation logic
(filter thresholds, split policies, round counts) and the iteration cost
should match. Don't recompute structure parsing every time you want to
test a new selection.

---

## Validation flow: smoke before full run

```
1. Local unit tests (no Zephyr): test_cli.py against file:// URIs.
2. Local Zephyr smoke (in-process): ZephyrContext with 2-3 inline rows.
3. Iris smoke: --num-docs 100, one input shard, single output file.
   Verify success rate, per-doc latency, output schema.
4. GATE: report measured rate + worker count + projected wall-clock to
   the user. Don't auto-trigger the full run.
5. Full run after explicit go-ahead.
```

Step 3 is the one that catches almost everything: the gzip truncation, the
requester-pays issue, the cross-region spill, the per-worker init bug.
Run it on every change; don't skip it because "the local tests passed."

Per-doc latency from the smoke is what you use to project full-run
wall-clock: `n_docs / (workers × docs_per_sec_per_worker)`. exp5 measured
~6 ms/doc → 1.6 M docs / (512 workers × 167 docs/s) ≈ 19 s of compute
(actual wall-clock ~8 min with overhead). exp53 measured ~225 ms/doc
(pyconfind-heavy) → 4.2 M docs / (512 × 4.45) ≈ 31 min.

---

## Finding the next perf opportunity

If your smoke run's projected wall-clock is out of budget — or you just
want to know whether there's headroom — these are the steps that
actually surface the right thing to fix, in order. The order matters:
each step is cheap and rules out a class of bottlenecks the next step
would otherwise misdiagnose.

### 1. Project before optimizing

Don't reach for cProfile yet. Compute the **per-row latency** from the
smoke and break it into the parts you can change vs the parts you can't:

```
per_row = network_GET  +  parse  +  generate  +  write
        ~30–80 ms      +  ms     +  ms        +  µs   (typical)
```

If the network GET dominates (it usually does for cif fetches), no amount
of Python-side optimization moves the needle — `thread_per_row_in_shard`
is your win, and you're done. If parse or generate dominates, then
profiling is worthwhile. **Decide which regime you're in before
investing in tools.** Most of the time the answer is "I/O dominates,
thread it, ship."

### 2. cProfile a representative single-row run, sorted by `tottime`

Once you've decided the CPU side is worth investigating, profile one row
on a *real* input (not a synthetic fixture — pathological-fast structures
hide the very loops you want to find):

```python
import cProfile, pstats
prof = cProfile.Profile()
prof.enable()
for _ in range(200):                       # amortize fixed overhead
    generate_doc_for_row(real_row, ...)
prof.disable()
pstats.Stats(prof).strip_dirs().sort_stats("tottime").print_stats(15)
```

Read **`tottime`**, not `cumtime` — `tottime` is time spent *in* the
function excluding callees, which is what points at the hot loop.
`cumtime` is dominated by your top-level entry point and tells you
nothing actionable. Run the loop ≥ 200 times so the per-call overhead of
the profiler itself doesn't dominate cheap calls.

### 3. Isolate the components

End-to-end timing tells you the total; **component timing tells you
where the total comes from**. Time each layer independently against the
same input:

```python
# Component A: parse only
t = time.perf_counter()
for _ in range(N): parse_structure(path)
parse_ms = 1000 * (time.perf_counter() - t) / N

# Component B: generate only (over already-parsed structures)
parsed = [parse_structure(path) for _ in range(N)]
t = time.perf_counter()
for s in parsed: generate_one(s, ...)
gen_ms = 1000 * (time.perf_counter() - t) / N

# Component C: end-to-end
# ... and check that parse_ms + gen_ms ≈ end_to_end_ms (often it doesn't,
# which is itself the finding — something in the glue is non-trivial).
```

This is how exp5's perf-engineering thread (which Tim ultimately rejected
in favor of one source of truth — but the *measurement* was sound)
narrowed the 9.5 → 5.7 ms/doc journey: each iteration measured parse and
generate separately, so we knew which half a given change moved.

### 4. Multi-trial measurement; report min + median

Single-shot wall-clock numbers are noisy: cold caches, JIT-ish warmups,
GC, system load. **Run ≥ 5 trials and report min + median.** First-run
inflation can be 20-50 %; pretending the first run is representative
will send you optimizing the wrong thing.

```python
trials = [time_one_run() for _ in range(5)]
print(f"min={min(trials):.2f}  median={sorted(trials)[2]:.2f}  trials={trials}")
```

Use **min** as the "true" cost (least noise added) and **median** as
"typical." When they diverge significantly, something in your environment
is variable and worth fixing before continuing (background processes,
disk cache state, etc.).

### 5. Hypothesis-driven changes — predict, then measure, then report honestly

For each optimization, **predict the win before implementing it** and
write the prediction down. Then measure. Three possible outcomes:

- **Predicted win materialized** → ship, update the skill if the lesson
  generalizes.
- **Predicted win was wrong (smaller, or zero)** → roll back. This is
  the most important branch; the temptation to keep the change because
  it's "cleaner" or "more correct in principle" leaks complexity into
  the codebase for no measurable benefit.
- **Predicted win was wrong in an *interesting* way** (different
  bottleneck surfaced) → keep the prediction in your notes; the gap
  between expected and measured is itself a finding about where the
  cost actually lives.

The exp5 perf thread had all three: the two-phase distance loop was
predicted to save 0.3 ms/doc; measurement showed it broke even (gains
from vectorization were eaten by Python-side plan-list construction).
That's a "rollback" outcome — the code went into a different branch, and
the *finding* (per-element numpy access has fixed overhead comparable to
~12 scalar Python ops) made it into the surrounding decisions.

### When to stop

You're done optimizing when the remaining cost is one of:

- **Real network latency** — `cat_file` of a 50 KB object across the
  internet has a floor. Add concurrency, not cleverness.
- **C-extension parse cost** (gemmi, pyconfind) — Python can't do
  anything here; the GIL is already released, you're competing with C.
- **Algorithmic floor** — the doc generator inherently has to do O(N²)
  contact work over N residues; you can vectorize but not avoid it.
- **The cluster's wall-clock is now bounded by something else** — at
  some point shaving 0.5 ms/doc off a 4 ms/doc pipeline saves
  ~minutes on a 30-minute run; cluster startup overhead and
  Zephyr scheduling start dominating. Below that threshold, the
  return on more profiling is essentially zero.

Each of these is a perfectly acceptable stopping point — the goal isn't
maximum theoretical throughput, it's "fits the budget the experiment
needs." If your projected wall-clock is already under the user's
tolerance, **don't optimize**; the time you'd spend has better uses.

---

## Pre-launch checklist

Before triggering the full run, verify each of these:

**Per-row**
- [ ] Object bytes are fetched via
      `marinfold.document_structures.io.read_object_bytes` (or
      equivalent: a full `cat_file` GET that handles gzip-transcoded
      objects), not `fsspec.open().read()` with size inference.
- [ ] The worker returns `None` (not raises) on transient I/O / parse
      failures so a single bad row can't kill a Zephyr worker.

**Per-shard**
- [ ] URI-path `map_shard` body delegates to
      `marinfold.document_structures.io.thread_per_row_in_shard` (or
      equivalent: a thread pool sized at `min(concurrency, len(rows))`
      that yields in input order and skips `None`); inline-cif path
      runs sequentially (no I/O to overlap, pool is pure overhead).
- [ ] Any heavy per-process init (parsed libraries, JIT kernels) is
      memoized at module scope with a sentinel, not re-run per shard.

**Per-job**
- [ ] `--worker-cpu 1`, `--worker-memory` set to peak working set + a
      small margin (4 GB is fine for most gemmi/numpy workers).
- [ ] `--max-workers` set explicitly. Default `None` means cluster
      default, which may be too small for a full run.
- [ ] `--region` pinned to the cluster's region (default `us-central1`
      for the marin Iris cluster), `--preemptible` on by default.
- [ ] Output bucket is `gs://marin-<same-region>/protein-structure/MarinFold/<exp>/`.

**Per-input / per-output**
- [ ] Input parquet is read with `columns=…` projected to only what the
      worker needs (one schema peek at submission time).
- [ ] `--out` contains `{shard:05d}-of-{total:05d}` for the full run;
      smoke runs may omit it.

**Validation**
- [ ] Local unit tests pass.
- [ ] Iris smoke run (`--num-docs 100`, single input shard) produced
      ≥ 95 % valid docs and measured per-doc latency matches the
      projection.
- [ ] Smoke output sample manually inspected for shape + content.
- [ ] User has gated the full run.

If any of these is missing, fix it before scaling out. The cost of an
unnecessary full run on 512 preemptible workers is real (cluster capacity
+ $) and the cost of *finding out at hour 3* that you needed the
gzip-safe reader is realer.

---

## When you're done

After the full run completes:

1. Record the measured per-doc rate + wall-clock + worker count + region in
   the experiment's README "Results" section. Include the output GCS URI.
2. Commit a `data/timings.csv` (per the root `AGENTS.md` "Capture timings
   for every predictor run" rule).
3. If you discovered a new trap not in this skill, *open a PR adding it
   here* — the cost of writing it down is much less than the cost of the
   next agent rediscovering it.
