# Exp232 TRC LR-recovery operations

Status: regional ingress complete and independently verified; stopped at the
pre-smoke review gate. No training smoke or production run has been submitted.

## Contract

- Source: full Levanter `step-333960` from
  `prot-exp232-cw-cv1-decontam-recover-a03-skipstep-m2-p06-srcpeak-augcont`.
  Training never initializes from an HF export.
- Variants: `lr050`, `lr010`, and `lr005`, targeting `5e-4`, `1e-4`, and
  `5e-5` respectively.
- Restore step `333961` starts at the source peak LR `1e-3`. The first 10,890
  updates lower LR linearly, reaching the target at step `344850`. Hold through
  step `464640`, then linearly cool to exactly zero at final checkpoint
  `step-551760`.
- Keep skip-step enabled with its existing state and defaults. Restore is strict,
  not partial. Change the data seed to `232`; retain 100% augmentation, block
  shuffle, full validation, W&B watch, permanent checkpoints every 14,520 steps,
  and 30-minute temporary checkpoints.
- W&B must be `open-athena/MarinFold`. Every regional replica has a distinct run
  and checkpoint identity. Multiple regions may race the same LR variant.

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

## Pre-smoke gate

Before the first accelerator smoke:

1. All four regional ingress markers and full inventories must validate.
2. The lowered plan must use the region-local caches and exact Levanter
   `step-333960` through `TrainerConfig.initialize_from`.
3. Local tests must pass for every LR boundary and target.
4. Stop for user review. Do not submit a smoke test until approved.

Items 1-3 are complete. Item 4 is the current state.
