---
marinfold_experiment:
  issue: 266
  title: 'exp: ProteinMPNN-redesign the decontaminated contacts-v1 training set — 8 sequences per backbone'
  kind: data
  branch: exp/266-mpnn-redesign
---

# exp: ProteinMPNN-redesign the decontaminated contacts-v1 training set — 8 sequences per backbone

**Issue:** [#266](https://github.com/Open-Athena/MarinFold/issues/266) · **Kind:** `data` · **Branch:** `exp/266-mpnn-redesign`

## Question

Does adding ProteinMPNN-redesigned sequences for backbones we already have
improve contacts-v1 — and is inverse-folding augmentation a cheaper substitute
for generating novel structures?

Concretely: take all **3,963,003** documents of the decontaminated AFDB corpus
([`contacts_v1_decontam`](https://huggingface.co/buckets/open-athena/MarinFold/tree/main/data/document_structures/contacts_v1_decontam),
[#225](https://github.com/Open-Athena/MarinFold/issues/225)), redesign each
backbone into **8 new sequences**, re-run pyconfind on each, and publish the
resulting **~31.7 M-document** corpus.

This issue ships the corpus. The accuracy question is a follow-up
`kind/models` issue against the [#232](https://github.com/Open-Athena/MarinFold/issues/232)
decontaminated recipe.

## Hypothesis

Redesign augmentation supplies a contrast the current corpus cannot: today
every fold appears with exactly one sequence, so the model has no way to
separate "this contact is geometry" from "this contact is this rotamer". Eight
sequences per backbone provides exactly that contrast, at essentially the cost
of the pyconfind pass alone.

## Background: why this and not a structure generator

This is the cheap null version of a larger idea — an unconditional backbone
generator (BoltzGen / Proteina / RFdiffusion3) plus ProteinMPNN to manufacture
millions of *novel* training structures. That version costs GPU-weeks and
changes two things at once (novel folds **and** MPNN-flavoured sequences).
This one changes only the second, so a flat or negative result here is strong
evidence against the expensive version — and a positive one validates the whole
pyconfind-on-designed-sequences path before any generator is bought.

Cost of the generator version for reference (3 M monomers, L≈256, H100-hours,
generation only): distilled Proteina 75, Proteina-400-step 1,750,
RFdiffusion3 ~2,500, **BoltzGen ~15,000**. BoltzGen is the wrong tool — a
conditional *binder-design* model at ~24 s/design on an H100, most of whose
machinery we would not use.

**The prior in this repo is not encouraging.**
[#120](https://github.com/Open-Athena/MarinFold/issues/120) is the closest
thing we have run and synthetic documents lost: regenerated (even
only-correct-filtered) docs vs re-epoching the original at matched budget
scored R-precision **0.330 vs 0.350** overall and **0.262 vs 0.279** on long
proteins, two orders of magnitude above
[#204](https://github.com/Open-Athena/MarinFold/issues/204)'s 0.0023 noise
floor. The difference here is that #120 regenerated the *documents* by
self-distillation over the same structures; this regenerates the *sequences*
with an independent, structure-aware model MarinFold has never seen.

**The headroom argument for doing it at all.**
[#139](https://github.com/Open-Athena/MarinFold/issues/139)'s 66.8 M ESM-Atlas
structures are already the 40 %-identity linclust representatives, so we are
near the end of readily available *non-redundant* real predicted structures.
Inverse-folding augmentation is the only axis that multiplies the corpus
without either new structures or new redundancy.

## What was verified before writing the pipeline

### 1. pyconfind ignores input side chains — bit-identically

confind's contact degree is a rotamer-ensemble quantity: it rebuilds side
chains from the Dunbrack library rather than reading the ones in the file.
Stripping to backbone atoms (`N/CA/C/O`, no `CB`) and re-running
`contacts_v1.parse.analyze_structure` (`probe_pyconfind.py`):

| structure | residues | contacts (all-atom) | backbone-only | identical pair set | max Δdegree |
|---|---|---|---|---|---|
| 1crn | 46 | 57 | 57 | ✅ | 0.000 |
| 1ubq | 76 | 162 | 162 | ✅ | 0.000 |
| 101m | 154 | 352 | 352 | ✅ | 0.000 |
| 1mbn | 153 | 327 | 327 | ✅ | 0.000 |

**A backbone plus a residue-name assignment is a complete pyconfind input.**
So the redesigned corpus is computed under exactly the same contact operator as
`contacts_v1` — no silent train-distribution mismatch — and all-atom generators
would buy us nothing for this document format.

Pinned by `tests/test_backbone.py`, which asserts the stronger property the
pipeline actually depends on: a document built from *stripped backbone + native
sequence written back on* is **byte-identical** to the document built from the
untouched structure.

### 2. The contact label is strongly sequence-dependent at fixed geometry

Holding the backbone fixed, changing only residue names, applying the
contacts-v1 selection rule (`degree ≥ 0.001`, `|i−j| ≥ 6`)
(`probe_seq_sensitivity.py` → `data/sequence_sensitivity_probe.csv`):

| backbone | native contacts | shuffled native | random uniform | poly-LEU | poly-ALA | poly-GLY |
|---|---|---|---|---|---|---|
| 1crn (46 res) | 28 | J=0.364 | J=0.341 | J=0.463 | 0 contacts | 0 contacts |
| 1ubq (76 res) | 67 | J=0.409 | J=0.426 | J=0.586 | 0 contacts | 0 contacts |
| 101m (154 res) | 130 | J=0.353 | J=0.313 | J=0.492 | 1 contact | 0 contacts |

Only ~43–54 % of native contacts survive a sequence shuffle on an identical
backbone. **The 8 redesigns are not 8 near-duplicates** — each is a genuinely
different document, and pyconfind genuinely has to be re-run per
(backbone, sequence) pair rather than copied from the parent.

It also flags the risk: contact count collapses for small side chains, so an
Ala-rich design distribution would systematically shorten the documents. That
is measured below.

## Approach

Three stages. The split is forced by one fact: **CoreWeave task pods carry only
CoreWeave S3 credentials** (`iris-task-env` holds `AWS_*` / `CW_*` / `FSSPEC_S3`
and no GCP), so a CoreWeave worker cannot read AFDB's requester-pays GCS bucket.
The structures have to cross clouds as data.

**Stage A — keep-list** (`select_backbones.py`, local, minutes). Read `entry_id`
+ provenance from the published `contacts_v1_decontam/train` corpus (2,067
shards; schema verified) and emit a length-sorted Stage-B manifest. `gcs_uri` is
rebuilt from `entry_id` rather than joined back to the 12,005-shard afdb-24M
manifest, since AFDB's layout makes it a pure function of the accession.

Decontamination is **inherited, not re-derived**: the keep-list is read out of
the published decontaminated corpus, so we only ever redesign a backbone that
survived #225, and a redesigned sequence cannot reintroduce an eval *sequence*
because it is a new sequence.

**Stage A2 — stage backbones** (`cli.py stage` → `stage_rows.py`, GCP Iris).
Fetch each cif once (gzip-safe `read_object_bytes`), strip to backbone, and
write a compact row. This is the only stage that *must* run on GCP. It is also
genuinely I/O-bound — a ~30–80 ms GCS GET against a few ms of parse-and-encode —
so it is the textbook case for `thread_per_row_in_shard` at the default fetch
concurrency of 32, unlike the document stage where seconds of CPU dominate.

What crosses is small because we only need backbones: N/CA/C/O plus sequence and
per-residue pLDDT is **53 bytes/residue** (12 int32 coords + float32 pLDDT + 1
sequence char) against ~180 KB of all-atom mmCIF. At the corpus's mean length of
277.7 residues (measured over a 12-shard Stage-A sample) that is 14.4 KB/protein
= **~58 GB staged once** (less after ZSTD) instead of ~700 GB fetched
repeatedly. The artifact is reusable by
any future backbone-based experiment, the same argument #139 makes for having
saved its raw pyconfind contacts.

**Stage B — redesign + documents** (`dispatch_redesign_cw.py` →
`redesign_worker_cw.py`, CoreWeave rno-2a). N independent 1×H100 tasks, batch
priority, no gang. Per chunk: design on the GPU in exact-length batches, then
fan document generation out over a process pool while the next chunk designs.

8 designs per backbone on a **temperature ladder**
`{0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.5, 0.5}`, recorded per row alongside
`mpnn_score` and `identity_to_native`, so a training experiment can subset the
near-native ↔ diverse axis without regenerating anything.

`cli.py generate` is the all-CPU fallback for when the GPU fleet is busy — same
worker code, `--device cpu`.

### Why CoreWeave, and why this shape

Measured live (2026-08-31) across both CoreWeave clusters:

| cluster | pool | nodes | vCPU | requested | idle | H100 idle |
|---|---|---|---|---|---|---|
| cw-us-east-02a | `cpu-genoa` | 4 | 768 | 33 | **735** | — |
| cw-us-east-02a | `h100-8x` | 32 | 4,095 | 844 | 3,250 | 24 / 256 |
| cw-rno2a | `cpu-turin` | 1 | 192 | 23 | 169 | — |
| cw-rno2a | `h100-8x` | 64 | 8,189 | 1,456 | **6,734** | **224 / 512** |

rno2a has no CPU fan-out pool at all (`cpu-turin` is `max_slices: 1`, it hosts
the controller), but its H100 fleet is prepaid and `buffer_slices: 64` pins it
fully warm — "the fleet is prepaid, so there is nothing to save by autoscaling
it down". Idle GPUs there are free at the margin, and **224 were idle**.

That is why Stage B is GPU-shaped even though ProteinMPNN barely needs a GPU.
The two halves of the per-backbone work land on different hardware and roughly
balance — ~0.1 s of H100 for 8 designs against ~4.2 s of CPU for 8 pyconfind
runs, i.e. ~0.3 s of wall-clock across a task's 15 cores — so one task per GPU
with a slice of the node's 128 vCPUs keeps neither device idle.

### Four decisions the measurements forced

**Stage the backbones as int32 milli-ångströms, not floats.** AFDB mmCIF writes
`Cartn_x` with 3 decimals, so `round(x * 1000)` is exact and `v / 1000.0`
reproduces the very same double gemmi parsed — lossless, 4 bytes per number, and
integers with spatial locality compress far better than float32. float32 would
perturb coordinates at the ~1e-3 Å level, which is exactly the scale at which a
marginal pyconfind contact flips.
`tests/test_backbone.py::test_staged_backbone_round_trip_is_byte_identical`
asserts the whole hop is byte-identical at the **document, `sha1` and
`global_plddt`** level, so the staged corpus is provably computed from the same
coordinates as the parent.

**Fold the 8 design slots into the batch dimension.** ProteinMPNN uses
`temperature` only in broadcast-compatible divisions, so a `[B, 1]` tensor works
where a scalar is expected — one `sample()` call produces all 8 designs at 8
different temperatures. Measured on an A5000 at L=154: 10.38 s → 1.30 s.
`tests/test_redesign.py::test_per_item_temperature_matches_scalar` asserts
bit-equality with the scalar path, because a silent failure here would draw
every sequence at the wrong temperature.

**Exact-length batches** (`redesign.batch_by_exact_length`). Not an optimization
but a correctness requirement: `tied_featurize` pads `omit_AA_mask` (an
`[L, 21]` array) with the 1-D pad spec `[[0, L_max - l]]`, widening the
*alphabet* axis, and raises on any batch of mixed lengths (`could not broadcast
input array from shape (476,426) into shape (476,21)`). Upstream never hits it
because the stock CLI batches N copies of one protein. Rather than monkey-patch
a dependency we feed it only the shape it handles, which also means zero padding
waste. Stage A's length sort keeps the equal-length groups full.

**Override `proteinmpnn`'s `numpy<2` pin.** It is unsatisfiable against
`marin-zephyr`'s `numpy>=2`. The pin is the packager's caution, not a real
constraint — the module contains none of the APIs numpy 2 removed — and the
CPU-parametrized tests run the model under numpy 2.5.2 rather than taking it
on faith.

## Measurements so far (local, RTX A5000 + 1 CPU core)

Two runs of `smoke_local.py --limit 48`: **direct** (before the staging hop
existed) and **staged** (through `encode_backbone`/`decode_backbone`, the path
that actually runs). They draw different 48-protein samples, because
`encode_backbone` rejects the non-contiguous author numbering that experimental
PDB entries routinely have — a filter that should never fire on AFDB's complete
1..L models. `data/local_smoke_*.csv`.

### Throughput

| path | ms/sequence | note |
|---|---|---|
| GPU, one `sample()` per design, B=1 | 1298 | what the stock CLI does |
| GPU, designs folded in, B=1 | 162 | 8× from the batch-dimension fold |
| GPU, designs folded in, B=8 (L=154) | 33 | +another 5× from batching |
| GPU, designs folded in, B=32 (L=46) | 6.3 | |
| CPU, 1 thread, designs folded in (L=154) | 602 | 4.82 s/backbone for 8 designs |
| pyconfind → document, 1 thread | 378–463 ms/**document** | at L≈220–250 |
| mmCIF → staged backbone row | 177 ms | I/O-bound in practice, not this |

Device memory scales cleanly at **0.52 MB per padded residue** in the effective
batch, independent of L and B — what `--max-batch-residues` bounds.

Staged size: **53 bytes/residue** — 11.2 KB/protein on the L=217 PDB smoke
sample, but the corpus mean is 277.7 residues (12-shard Stage-A sample), so
**14.4 KB/protein and ~58 GB** for the full corpus before ZSTD. (An earlier
draft said ~46 GB; that applied the PDB sample's mean length to the corpus.)

### Composition drift — the risk flagged in the issue

ProteinMPNN's known biases are present and in the expected direction, and
consistent across both runs. Largest shifts (staged run):

| over-used | Δpp | under-used | Δpp |
|---|---|---|---|
| A | +2.23 | Q | −2.06 |
| P | +1.98 | M | −1.33 |
| L | +1.46 | S | −1.27 |
| E | +0.92 | H | −1.12 |
| K | +0.83 | F | −0.92 |

And the number that matters, given that contact count collapses for small side
chains:

| run | native contacts/residue | designed | ratio |
|---|---|---|---|
| direct (48 proteins) | 0.930 | 0.900 | **0.968** |
| staged (48 proteins) | 0.902 | 0.850 | **0.942** |

So the feared artifact — Ala-rich designs producing systematically shorter
documents — is real and small, somewhere around 3–6 %. **The two runs disagree
by 0.026, which is the honest precision of a 48-protein sample**; this is a
pilot-scale measurement, not a settled number, and it is on PDB rather than
AFDB structures. The staging hop is not the cause of the difference — that is
asserted byte-identical by
`tests/test_backbone.py::test_staged_backbone_round_trip_is_byte_identical`.

Temperature behaves as intended in both runs, though the ladder spreads gently:

| T | identity to native (direct / staged) | MPNN score |
|---|---|---|
| 0.1 | 0.467 / 0.443 | 0.845 / 0.883 |
| 0.2 | 0.457 / 0.432 | 0.876 / 0.916 |
| 0.3 | 0.451 / 0.430 | 0.919 / 0.965 |
| 0.5 | 0.434 / 0.413 | 1.088 / 1.140 |

0.44–0.47 recovery at T=0.1 is in line with ProteinMPNN's published ~50 %.

## Cluster results (2026-09-02)

`data/cluster_smoke_stage_a2.csv`, `data/density_representative.csv`,
`data/composition_representative.csv`, `data/temperature_representative.csv`.

### Stage A + A2 — and the check that matters

Stage A read 12 of the corpus's 2,067 shards → 22,918 rows in **9.1 s** (full
keep-list projects to ~26 min; mean `seq_len` 277.7). Stage A2 on marin Iris
(`/bizon/iris-run-cli-20260902-141215`) **SUCCEEDED**: `records_in 22,918`,
`records_out 200`.

The staged rows carry `native_sha1` — each entry's document hash in the
*published* `contacts_v1_decontam` corpus. Rebuilding from the staged int32
coordinates, writing the native sequence back on, and generating:

> **200 / 200 sha1 match. 0 mismatch.**

So AFDB mmCIF → strip → int32 milli-ångström encode → parquet → object store →
decode → `generate_document` reproduces real published corpus documents
byte-for-byte, on cluster-produced data.

Staging rate (workstation, threaded): **41–65 structures/s at
fetch-concurrency 32**, 0 filtered and 0 raised over 700 real AFDB objects — so
the contiguity and exact-coordinate asserts hold on real input. Staged size
**14.3 KB/protein** at mean L 277, matching the 53 bytes/residue model behind
the ~58 GB estimate.

### Stage B on CoreWeave — it runs

`/bizon/exp266-smoke4-s0of1` **SUCCEEDED**: 200 backbones → **1,600 distinct
documents** written to CoreWeave object storage in 11 s (gpu 3 s, cpu 7 s over
14 processes), exactly 8 per backbone, provenance and `global_plddt` intact.

### The composition risk — resolved, on the right sample

`/bizon/exp266-rep-s0of1`: 400 length-representative backbones (mean L 276.1,
range 32–1209) → 3,200 documents, all distinct.

| stratum | n | native contacts/res | designed | ratio |
|---|---|---|---|---|
| **all** | 400 | 0.675 | 0.686 | **1.016** |
| 0–100 | 53 | 0.481 | 0.504 | 1.046 |
| 100–200 | 131 | 0.608 | 0.620 | 1.019 |
| 200–400 | 136 | 0.720 | 0.729 | 1.012 |
| 400–800 | 67 | 0.839 | 0.846 | 1.008 |
| 800+ | 13 | 0.821 | 0.809 | 0.985 |

**The feared artifact does not happen.** Redesigned documents carry
essentially the *same* contact density as native (ratio 1.016), monotonically
approaching 1 as length grows. Success criterion 4 is met.

This overturns the workstation estimates. Earlier samples gave 0.968 and 0.942
— both on **PDB** structures, which are better-packed crystal structures with a
different residue composition — and 1.040 on the 200 *shortest* AFDB entries.
On the corpus's own distribution the answer is ~1.0.

Composition drift is real and larger than the PDB estimate suggested
(P +4.33, A +4.16, S −3.25, E +2.98, L +2.59, Q −2.17 pp, vs a 2.2 pp maximum
on PDB) — but it does not translate into a density shift.

| T | identity to native | MPNN score | contacts/res |
|---|---|---|---|
| 0.1 | 0.356 | 0.990 | 0.684 |
| 0.2 | 0.352 | 1.032 | 0.684 |
| 0.3 | 0.345 | 1.106 | 0.687 |
| 0.5 | 0.326 | 1.325 | 0.688 |

Two observations. Sequence recovery on AFDB (0.356 at T=0.1) is well below
ProteinMPNN's published ~50 % on crystal structures — predicted models are
harder to recover. And **the temperature ladder is narrower than intended**:
0.1→0.5 moves identity only 0.356→0.326 and density not at all. The
near-native ↔ diverse axis is mostly not being exercised; a future
regeneration might want a wider ladder, and a training experiment should not
expect much from subsetting on `mpnn_temperature`.

## Traps the cluster smoke found

None of these are reachable from a local test — the experiment env resolves
the same packages the workers do only at job time, and two of them depend on
worker-side filesystem behaviour.

**1. exp53's launch imports no longer exist.** On 2026-09 marin main:

```
from fray import ResourceConfig             -> fray.types.ResourceConfig
from zephyr import Dataset, ZephyrContext   -> zephyr.dataset / zephyr.context
```

Same classes, same signatures, new homes. Two failed submissions, ~1 minute
each. `fray.types.ResourceConfig` also has a `device` union rather than a flat
`device_count`.

**2. `gcsfs >= 2026.5` 416s on any read past EOF.** This one killed a shard,
reading the pipeline's *own* Stage-A manifest:

```
gcsfs.retry.HttpError: Request range not satisfiable, 416
While loading from InputFileSpec(path='gs://.../manifest-00001-of-00005.parquet')
```

2026.5.0 introduced an experimental concurrent `cat_file` that splits a ranged
read into parallel sub-requests **without clamping `end` to the object size**.
fsspec's block cache reads past EOF as a matter of course (the parquet footer,
any tail block), so on a 299 KB object the split produces sub-requests that
*start* beyond EOF and GCS answers 416.

gcsfs has a guard for exactly this — `_fetch_range` catches `"not satisfiable"`
and returns `b""` — but it catches `RuntimeError`, and `gcsfs.retry.HttpError`
derives from `Exception`. The guard never fires.

It does not reproduce from a workstation: locally the adaptive-prefetch reader
is used, while the worker took the extended-gcsfs `cat` path. The fix is a
declarative pin, `gcsfs<2026.5` — no monkey-patching a dependency — found by
version archaeology (`_cat_file_concurrent` first appears in 2026.5.0).

**Any marin data pipeline reading parquet from GCS is exposed to #2.**

## Projected full-run cost — **measured**

Throughput has to be measured on a **contiguous length band**, not the strided
quality sample. Exact-length ProteinMPNN batching means a strided sample (~400
distinct lengths) runs batches of 1 — the worst case, and nothing like a real
shard, which is a contiguous slice of a 3.96 M-row length-sorted manifest.

| sample | shape | rate | gpu | cpu |
|---|---|---|---|---|
| strided, mean L 276 | ~400 distinct lengths (batch fill 1.0) | 1.6 backbones/s | 165 s | 76 s |
| **band, L 265–290** | 26 distinct lengths (batch fill 15.4) | **5.3 backbones/s** | 24 s | 50 s |
| shortest, mean L 47 | contiguous | 17.6 backbones/s | 3 s | 7 s |

The band run is the one to project from, and it is **CPU-bound** (50 s vs 24 s)
— which is the balance the design aimed for. That also means per-node
throughput is fixed by the node's 128 vCPUs, not by how the tasks are split:
8 tasks × 15 cores and 4 tasks × 30 cores both land at **~42 backbones/s/node**.
GPUs are the free resource here; cores are the constraint.

`project_full_run.py --backbones-per-second 5.3`:

| tasks | nodes | nominal | with 3× slack |
|---|---|---|---|
| 14 | 2 | 14.8 h | 44.5 h |
| **28** | **4** | **7.4 h** | **22.3 h** |
| 56 | 7 | 3.7 h | 11.1 h |
| 112 | 14 | 1.9 h | 5.6 h |

At 28 tasks that is **208–623 H100-hours** — on a prepaid, pinned-warm fleet
that had 224 GPUs idle, so free at the margin. The pipeline skill's warning
that a smoke rate is a lower bound is why the slack column is there; startup,
preemption retries and straggler tails all land on top.

Full pipeline, end to end:

| stage | where | cost |
|---|---|---|
| A — keep-list | local | ~26 min |
| A2 — stage backbones | GCP Iris, 512 workers | ~1 h, ~58 GB out |
| B — redesign + documents | cw-rno2a, 28–56 × 1 H100 | **3.7–7.4 h** (to 22 h with slack) |
| output | | **31,704,024 documents, ~40 B tokens** |

All-CPU fallback if the GPU fleet is busy: `cli.py generate --device cpu`,
~19.5 h on cw-us-east-02a's 735 idle `cpu-genoa` vCPUs.

## Launch

### Getting the backbones to CoreWeave

Neither cluster has both credentials: GCP Iris workers can read AFDB's
requester-pays bucket but have no CoreWeave keys; CoreWeave pods have CoreWeave
keys and no GCP. The workstation has both, but its ~2.5 MB/s uplink would take
~6.5 h for 58 GB.

The way out is that Stage A2's *output* path is just fsspec: give the **GCP**
staging job the CoreWeave credentials and have it write straight to
`s3://marin-us-east-02a/...`. One pass, no intermediate GCS copy, no 58 GB
double-handling — the worker reads GCS with its service account and writes to
CoreWeave over the internet (~$7 of GCP egress at 58 GB).

The keys live in the cluster's own `iris-task-env` secret:

```bash
export CW_KEY_ID=$(KUBECONFIG=~/.kube/coreweave-iris-rno2a kubectl -n iris \
    get secret iris-task-env -o jsonpath='{.data.CW_KEY_ID}' | base64 -d)
export CW_KEY_SECRET=$(KUBECONFIG=~/.kube/coreweave-iris-rno2a kubectl -n iris \
    get secret iris-task-env -o jsonpath='{.data.CW_KEY_SECRET}' | base64 -d)
```

### The four steps

```bash
# 1. Stage A — keep-list (local; huggingface_hub>=1.5 conflicts with this
#    experiment's pyproject, so run it out-of-project). ~26 min for all 2,067
#    corpus shards; add --max-shards N for a smoke slice.
uv run --no-project --with 'huggingface_hub>=1.5' --with pyarrow --with fsspec \
    python select_backbones.py --out /data/exp266/manifest

# 2. Stage A2 — stage backbones on GCP, writing straight to CoreWeave.
uv run iris --cluster=marin job run --cpu 1 --memory 2GB \
  -e AWS_ACCESS_KEY_ID "$CW_KEY_ID" -e AWS_SECRET_ACCESS_KEY "$CW_KEY_SECRET" \
  -e AWS_ENDPOINT_URL https://cwobject.com \
  -e FSSPEC_S3_ADDRESSING_STYLE virtual -- \
    python cli.py stage \
        --input 'gs://marin-us-central1/protein-structure/MarinFold/exp266_mpnn_redesign/manifest/manifest-*.parquet' \
        --out 's3://marin-us-east-02a/MarinFold/exp266/backbones/backbones-{shard:05d}-of-{total:05d}.parquet' \
        --worker-cpu 1 --worker-memory 4g --max-workers 512 \
        --region us-central1 --fetch-concurrency 32

# 3. Stage B — redesign on idle prepaid H100s. Smoke first (one task, one file).
uv run python dispatch_redesign_cw.py --shards 1 --max-files 1 --dry-run
uv run python dispatch_redesign_cw.py --shards 1 --max-files 1 --priority batch
# then the full fan-out, sized to the LIVE idle GPU count (see below)
uv run python dispatch_redesign_cw.py --shards 28 --priority batch

# 4. Publish to the public HF bucket.
```

Check live idle capacity before choosing `--shards` — the 224 idle GPUs above
were a point-in-time reading, not a reservation:

```bash
KUBECONFIG=~/.kube/coreweave-iris-rno2a kubectl get nodes -o json
```

## Success criteria

This issue ships a corpus; the accuracy question is a follow-up `kind/models`
issue against the #232 decontaminated recipe. For *this* issue:

1. **Fidelity.** Native-sequence relabelling reproduces the parent document
   byte-for-byte (`tests/test_backbone.py`) — ✅ on 5 structures.
2. **Staging is lossless.** The encode/decode hop reproduces the document,
   `sha1` and `global_plddt` exactly — ✅ on 5 structures, and coordinates are
   asserted exact at 0.001 Å.
3. **Completeness.** 3,963,003 × 8 = 31,704,024 documents, drops counted and
   reported by reason, fail-loud per the pipeline skill.
4. **Composition check.** AA frequency and contacts-per-residue vs native, per
   temperature, reported whatever it shows — measured locally at ratio
   0.968 / 0.942 over two 48-protein samples, which is too noisy to settle it;
   the pilot must repeat it at scale on AFDB structures.
5. Published to the public HF bucket with a `DATASET_README.md` and a
   reproducible `publish_to_hf.py`.

## Risks

- **The labels assume the design folds.** MPNN sequences are not refolded.
  Unlike the generator version the backbone is a real AFDB fold and the sequence
  is a high-likelihood design for it, so this is the mildest form of that
  assumption — but a refold-and-check on a 20 k subsample (~5 GPU-h with
  ESMFold) would measure the per-sequence self-consistency rate and is worth
  folding into the pilot.
- **`global_plddt` is inherited from the parent AFDB structure.** `decode_backbone`
  restores the staged CA B-factors, so contacts-v1 recomputes the same value. It
  correctly describes confidence in the *backbone* we reused, and says nothing
  about whether the designed sequence folds there.
- **The contiguity assert should be a no-op on AFDB.** `encode_backbone` raises
  on non-contiguous author residue numbering. AFDB models are complete 1..L so
  this should never fire; it fires readily on experimental PDB entries with
  missing loops, which is why the local smoke's sample changes when it is
  enabled. Any hit during Stage A2 is a real signal about the input, not noise
  to suppress.
- **Idle CoreWeave capacity is not a reservation.** The 224 idle H100s were a
  point-in-time reading; re-check before sizing `--shards`, and fall back to
  `cli.py generate --device cpu` if the fleet is busy.
- **T=0.5 may be past the designability cliff.** That is what the recorded
  `mpnn_temperature` column is for.
- **#120 says synthetic documents start at a deficit here.**

## Files

| file | role |
|---|---|
| `backbone.py` | strip / relabel / **lossless staged encode-decode** — the verified core |
| `redesign.py` | ProteinMPNN wrapper: exact-length batches, folded design dimension |
| `select_backbones.py` | Stage A — keep-list from the decontaminated corpus |
| `stage_rows.py` | Stage A2 per-row worker — mmCIF → staged backbone row |
| `cli.py` | Zephyr driver: `stage` (GCP) and `generate` (CPU fallback) |
| `redesign_worker_cw.py` | Stage B worker — one CoreWeave 1×H100 shard |
| `dispatch_redesign_cw.py` | Stage B fan-out, batch priority |
| `generate_rows.py` | staged row + designs → contacts-v1 documents |
| `smoke_local.py` | local end-to-end over the real staged path |
| `probe_pyconfind.py` | does confind need side chains? (no) |
| `probe_seq_sensitivity.py` | how much does the label move with sequence? (a lot) |
| `tests/` | fidelity + round-trip + Stage-B + pipeline tests (32 passing) |

## Results

Pending — nothing has run on the cluster yet.

## Conclusion

Pending.
