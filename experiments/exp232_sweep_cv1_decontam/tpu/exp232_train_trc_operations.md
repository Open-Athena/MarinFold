# Exp232 TRC LR-recovery operations

Status: regional ingress pending; no training smoke or production run submitted.

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
| `europe-west4` | pending | pending |
| `us-east1` | pending | pending |
| `us-east5` | pending | pending |
| `us-west4` | pending | pending |

## Pre-smoke gate

Before the first accelerator smoke:

1. All four regional ingress markers and full inventories must validate.
2. The lowered plan must use the region-local caches and exact Levanter
   `step-333960` through `TrainerConfig.initialize_from`.
3. Local tests must pass for every LR boundary and target.
4. Stop for user review. Do not submit a smoke test until approved.
