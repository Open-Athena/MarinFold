# Exp232 TRC LR-recovery operations

> This document governs a training sweep managed with the
> `run-training-sweep-trc` skill. Read it in full at the start of every
> heartbeat before inspecting code, SQLite, W&B, or Iris.

## Invariants

- Execute only on Google TRC TPUs at `interactive` priority. Never dispatch this
  sweep to CoreWeave and never reuse this priority choice for CoreWeave work.
- Every worker reads caches and the initialization checkpoint from its own region
  and writes only to its own regional prefix. Never copy state between TRC regions.
- Strictly restore the full Levanter `step-333960` state. Never initialize from or
  create an HF export.
- Allow one live writer per regional run and one winning regional run per trial.
- Production W&B routing is `open-athena/MarinFold`. Pass secrets only through the
  environment; never persist them in this document or SQLite.
- Include `--user eczech` on every Iris submission command that supports it.

## Sweep Definition

The training entry point and trial catalog are `exp232_train_trc.py`; its
`VARIANTS` mapping defines the three opaque logical trials `lr050`, `lr010`, and
`lr005`. The entry point owns all training, schedule, data, validation,
checkpoint, and hardware-parallelism semantics.

Each regional run must use the already-verified regional caches and regional copy
of the full Levanter source checkpoint. A same-region reslice resumes from that
run's regional checkpoint. A different-region replica starts independently from
the regional `step-333960` seed.

## Operator Choices

- Time limit: 14 days from the first production dispatch.
- Regional replicas: two simultaneous regions per logical trial.
- Compute: up to 512 actual TPU chips per regional replica and 3,072 submitted
  chips across the six replicas.
- Scope: every currently configured TPU family and slice with 32--512 actual
  chips in `europe-west4`, `us-east1`, `us-east5`, and `us-west4`.
- Exclusions: every other region, slices outside the actual-chip range, non-TPU
  backends, and CoreWeave.
- Priority: `interactive`, preemptible.
- Operations document: this tracked file.

## Operating Policy

- `heartbeat_every=1h`, scheduled only after the prior heartbeat completes;
  `reslice_after=1h`, `restart_after=3h`, `relocate_after=3d`, and
  `pending_target_limit=1`.
- Retry an isolated failure on the same regional target. Pause replacements and
  investigate when failures recur or cluster across independent runs.
- Maintain two distinct regional runs per unfinished trial. After one reaches
  `run_progress >= 1` and its expected checkpoint is reachable, mark it the winner
  and stop its nonterminal sibling.
- A trial completes only after its winning checkpoint is independently reachable.
  End the sweep when every trial completes or the 14-day limit is reached.

## Target Grid

Current Marin bucket mappings, Iris pool definitions, regional inputs, the entry
point placement guard, and every batch/mesh fit below have been validated.

| Region | Bucket | Slice | Chips | State | Reason |
| --- | --- | --- | ---: | --- | --- |
| `europe-west4` | `marin-eu-west4` | `v5litepod-32` | 32 | eligible | — |
| `europe-west4` | `marin-eu-west4` | `v5litepod-64` | 64 | eligible | — |
| `europe-west4` | `marin-eu-west4` | `v5litepod-128` | 128 | eligible | — |
| `europe-west4` | `marin-eu-west4` | `v5litepod-256` | 256 | eligible | — |
| `europe-west4` | `marin-eu-west4` | `v6e-32` | 32 | eligible | — |
| `europe-west4` | `marin-eu-west4` | `v6e-64` | 64 | eligible | — |
| `europe-west4` | `marin-eu-west4` | `v6e-128` | 128 | eligible | — |
| `europe-west4` | `marin-eu-west4` | `v6e-256` | 256 | eligible | — |
| `us-east1` | `marin-us-east1` | `v6e-32` | 32 | eligible | — |
| `us-east1` | `marin-us-east1` | `v6e-64` | 64 | eligible | — |
| `us-east1` | `marin-us-east1` | `v6e-128` | 128 | eligible | — |
| `us-east1` | `marin-us-east1` | `v6e-256` | 256 | eligible | — |
| `us-east5` | `marin-us-east5` | `v6e-32` | 32 | eligible | — |
| `us-east5` | `marin-us-east5` | `v6e-64` | 64 | eligible | — |
| `us-east5` | `marin-us-east5` | `v6e-128` | 128 | eligible | — |
| `us-east5` | `marin-us-east5` | `v6e-256` | 256 | eligible | — |
| `us-east5` | `marin-us-east5` | `v5p-64` | 32 | eligible | — |
| `us-east5` | `marin-us-east5` | `v5p-128` | 64 | eligible | — |
| `us-east5` | `marin-us-east5` | `v5p-256` | 128 | eligible | — |
| `us-east5` | `marin-us-east5` | `v5p-512` | 256 | eligible | — |
| `us-east5` | `marin-us-east5` | `v5p-1024` | 512 | eligible | — |
| `us-west4` | `marin-us-west4` | `v5litepod-32` | 32 | eligible | — |
| `us-west4` | `marin-us-west4` | `v5litepod-64` | 64 | eligible | — |
| `us-west4` | `marin-us-west4` | `v5litepod-128` | 128 | eligible | — |
| `us-west4` | `marin-us-west4` | `v5litepod-256` | 256 | eligible | — |

## Regional storage

Every region uses `gs://<bucket>/protein-structure/MarinFold/exp232_train_trc`:

| Region | Bucket |
|---|---|
| `europe-west4` | `marin-eu-west4` |
| `us-east1` | `marin-us-east1` |
| `us-east5` | `marin-us-east5` |
| `us-west4` | `marin-us-west4` |

Relative cache paths are:

- `tokenized/contacts_v1/afdb/2026.08.14`
- `tokenized/contacts_v1/esm/2026.08.14`
- `tokenized/contacts-v1-val/2026.07.25`

The regional Levanter seed is:

```text
checkpoints/protein/exp232-trc-init/
  prot-exp232-cw-cv1-decontam-recover-a03-skipstep-m2-p06-srcpeak-augcont/
  2026.08.21.1/checkpoints/step-333960
```

`exp232_cw_to_trc.py` enforces direct S3-to-each-region GCS ingress, validates
the pinned source object counts and bytes, and compares complete relative-path
and size inventories at the destination. It never accepts GCS as a source.

## Ingress jobs

| Region | Iris job | Result |
|---|---|---|
| `europe-west4` | `/eczech/exp232-cw-ingress-europe-west4-r02` | succeeded |
| `us-east1` | `/eczech/exp232-cw-ingress-us-east1-r02` | succeeded |
| `us-east5` | `/eczech/exp232-cw-ingress-us-east5-r02` | succeeded |
| `us-west4` | `/eczech/exp232-cw-ingress-us-west4-r02` | succeeded |

The first four submissions were killed while still pending and before they
executed. The `r02` jobs each read the original CW S3 objects directly and
wrote only to their own regional GCS bucket.

## Verified regional inputs

On 2026-08-21, all four regional destinations were independently compared to
their CW sources by complete relative-path and byte-size inventory:

| Artifact | Objects | Bytes | Inventory SHA-256 |
|---|---:|---:|---|
| AFDB train cache | 755 | 6,164,768,697 | `0bd54976c9d1a20cfca576e9aaab19470086bf72bad25494c70bf95500fea32c` |
| ESM train cache | 10,019 | 95,596,299,057 | `bb6cbb8a85581b2289af1fc4fd67d9226e46ac95cc2689cb6d98deaa77a8a291` |
| contacts-v1 validation | 17 | 66,618,514 | `7dcc705c378c2cd5128b105d47fd36da8397c8338227f620a2f04645f529be63` |
| Levanter checkpoint | 28 | 17,656,643,205 | `3ed8d9e17ced964036d82bfb78539c96a880c10f6c28317095109a1d37c69cd8` |

Marin cache stats also open successfully and match in every region:

| Cache split | Examples | Tokens |
|---|---:|---:|
| AFDB train | 3,963,003 | 4,432,940,838 |
| ESM train | 65,553,178 | 70,042,923,165 |
| contacts-v1 validation | 41,954 | 47,821,958 |

Every copied checkpoint reports permanent `step-333960` metadata and contains
`manifest.ocdbt`. No HF checkpoint was generated or copied because full-state
Levanter restore does not use one.

## Smoke validation

The approved two-step `lr010` smoke completed on 2026-08-21:

- Iris: `/eczech/exp232-trc-smoke-s92-lr010-use1-v6e4-a01` (`succeeded`).
- W&B: `eric-czech/marin`, run
  `prot-exp232-trc-cv1-decontam-train-smoke-s92-m2-p06-srcpeak-augcont-lr010-us-east1-v6e-4`
  (`finished`). Production remains pinned to `open-athena/MarinFold`.
- The us-east1 worker read only the regional caches and strict-restored the
  full Levanter `step-333960` state. TensorStore checking succeeded and training
  resumed at absolute step `333961`.
- W&B logged LR `0.0009999999310821295` at step `333961` and
  `0.0009999172762036324` at step `333962`, exactly matching the recovery
  schedule. Neither optimizer step was skipped.
- The run retained augmentation seed `166` and the original 145,200-step
  augmentation ramp; its resumed probability is clamped at `1.0`. Data seed
  `232` was independently visible in the worker log.
- The full-state output is under
  `tmp/checkpoints/<run-id>/checkpoints/step-333962`; its metadata is permanent
  and `manifest.ocdbt` is present.

The smoke also exposed that an HF interval beyond the run end still installs a
hook which Levanter forces at shutdown. The smoke therefore wrote an incidental
HF export. Production explicitly sets `hf_save_steps=None`; it writes only the
required full-state Levanter checkpoints.

The first v6e-4 compile used the original v6e memory correction and exceeded
HBM by 420 MiB before its first optimizer step. Commit `8dec032` calibrates the
v6e correction from `0.3` to `0.4`, selecting microbatch 8 with four-way
accumulation on v6e-4. The v6e-32 production shape remains microbatch 4 without
accumulation. Local Ruff and all six schedule, ingress, routing, augmentation,
and batch-fit tests pass.

Iris rejected the first post-review submission at its client build floor. The
local `eac-plm` Iris `BUILD_DATE` was set to the required `2026-08-07`; canary
`/eczech/exp232-trc-iris-floor-canary-20260821-a01` was accepted before the
successful smoke was submitted.

## Change Record

None.
