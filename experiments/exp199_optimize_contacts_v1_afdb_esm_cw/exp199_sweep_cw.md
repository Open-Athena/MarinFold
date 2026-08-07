# exp199 CoreWeave Sweep Operations

This is the lightweight operations record for `exp199_sweep_cw.py`. Submit
from this directory so Iris bundles this GPU workspace rather than the sibling
TRC workspace.

## Recipe

- Twenty trials: five unique points (`p01`, `p02`, `p03`, `p04`, `p06`) × two
  mixtures (`m1`, `m2`) × two augmentation variants (`base`, `aug`).
- Scratch initialization with model seed 0 and data seed 0.
- Linear WSD: 10% warmup, 70% stable, 20% decay to 10% of peak LR.
- Global batch 128, sequence length 8192, 145,200 steps, 152,253,235,200
  tokens (two step-rounded combined-cache epochs).
- Full validation every 2,230 steps.
- Permanent checkpoint every 14,520 steps (each 10% boundary through 90%) plus
  the final save: ten per trial. Step 116,160 is the 80% checkpoint immediately
  before WSD decay begins.
- Production W&B/checkpoint identity is independent of cluster and gang size.
- Priority is always `batch`; the child GPU gang inherits it from the driver.

Production W&B IDs have this form:

```text
prot-exp199-cw-cv1-s02-m1-p01-base
```

## Shared storage

All production CoreWeave clusters use the same S3 bucket. There are no
regional replicas or cross-region checkpoint moves.

```text
Experiment: s3://marin-us-east-02a/marin/protein-structure/MarinFold/exp199_optimize_contacts_v1_afdb_esm
AFDB:      s3://marin-us-east-02a/MarinFold/exp154_qwen_contacts_v1/tokenized/contacts-v1/2026.07.25
ESM:       s3://marin-us-east-02a/marin/protein-structure/MarinFold/exp137_contacts_and_crops_v1_1_5b/tokenized/contacts-v1-esm-atlas-train-568225
Validation:s3://marin-us-east-02a/MarinFold/exp154_qwen_contacts_v1/tokenized/contacts-v1-val/2026.07.25
```

AFDB and validation were compared with the exp199 GCS caches: all payload
objects match by size and MD5. The differing ledgers contain only backend-local
absolute paths.

### ESM restore receipt

The one-time direct Hugging Face-to-S3 restore completed successfully as Iris
job `/exedev/exp199-cw-restore-esm-cache` on 2026-08-07. Five concurrent
workers verified all 25 archive checksums and every uploaded member. The final
inventory is exactly 10,557 objects and 103,374,761,574 bytes.

The top-level training ledger reports 66,759,922 rows, 71,450,105,324 tokens,
1,669 shards, and only relative shard names. Old GCS paths remain in per-shard
construction ledgers, but Marin reads the relative top-level ledger and joins
those names against the S3 cache root.

Completion receipt:

```text
s3://marin-us-east-02a/marin/protein-structure/MarinFold/exp199_optimize_contacts_v1_afdb_esm/setup/esm-cache-restore/04ff0f2cc0d7530c062b027b3c14699f65e277dc/complete.json
```

The source archive revision is pinned to
`04ff0f2cc0d7530c062b027b3c14699f65e277dc`.

## Gang policy

Treat H100 and GB200 as equal production targets:

| Cluster | Node | Normal gang | Fallback gang |
| --- | --- | ---: | ---: |
| `cw-us-east-08a` | 4 × GB200 | 4 nodes / 16 GPUs | 2 nodes / 8 GPUs |
| `cw-us-east-02a` | 8 × H100 | 4 nodes / 32 GPUs | 2 nodes / 16 GPUs |
| `cw-rno2a` | 8 × H100 | 4 nodes / 32 GPUs | 2 nodes / 16 GPUs |

Four nodes on either system are roughly comparable aggregate BF16 compute and
match the historical exp166-cw GB200 shape. Fill all three clusters with as many
four-node trials as will schedule. Use two-node gangs when fragmentation would
otherwise leave a trial waiting. One-node gangs are calibration only; do not
use eight-node gangs unless measured scaling justifies the harder placement.

Before dispatching a wave, inspect live peers:

```bash
uv run --frozen iris --cluster marin rpc controller list-peers
```

Never run this experiment on `cw-us-west-04a`. `cw-rno2a` has only one CPU
pool, so avoid flooding it with idle drivers; its H100 nodes remain valid and
should be used for training when available.

## Smoke and calibration

The end-to-end augmented smoke completed on both GPU types. These runs exercised
scratch initialization, both training caches, scheduled augmentation, the full
validation cache, a final native checkpoint, and a final HF export.

Validated capacities:

```text
GB200: CLUSTER=cw-us-east-08a NODES=1 PER_DEVICE=32
H100:  CLUSTER=cw-rno2a       NODES=1 PER_DEVICE=8
```

| GPU | Iris child job | Result | Full eval | XLA-reserved GPU memory |
| --- | --- | --- | ---: | ---: |
| GB200 | `/exedev/exp199-cw-smoke-s02-m1-p01-aug-gb200-n1-pd32/run_levanter_train_lm-0ebf8ee8` | succeeded in 10m28s | 52 batches, loss 8.836 | 143 / 189 GiB |
| H100 | `/exedev/exp199-cw-smoke-s02-m1-p01-aug-h100-rno-n1-pd8/run_levanter_train_lm-db20a1a9` | succeeded in 17m32s | 104 batches, loss 9.290 | 61 / 81 GiB |

The H100 smoke used gradient accumulation 2 because one node has only eight
devices; every two- or four-node production shape uses accumulation 1. RNO
startup and checkpoint uploads were slower against the east S3 bucket, but the
run trained, evaluated, checkpointed, and exported successfully. Production
gangs split the same global batch across at least twice as many workers, so the
one-node input stalls and 104-batch eval are conservative rather than the
production shape.

Submit a smoke with the same driver template as production below, adding:

```text
-e SMOKE yes -e SMOKE_STEPS 10 -e PER_DEVICE <capacity>
```

The tested capacities are recorded in `MAX_SEQS_PER_DEVICE`. Compare
base/augmentation throughput during the first production wave before filling
each cluster.

## Submit one production trial

Set the placement and trial, then submit the small driver into the target
CoreWeave cluster. The driver creates the GPU gang locally.

```bash
set -a
source ~/marin.env
set +a

CW_PREFIX=s3://marin-us-east-02a/marin/protein-structure/MarinFold/exp199_optimize_contacts_v1_afdb_esm
CW_CLUSTER=cw-us-east-08a
CW_NODES=4
TRIAL=m1-p01-base
VERSION=2026.08.07.2

uv run --frozen iris --cluster marin job run \
  --target-cluster "$CW_CLUSTER" \
  --priority batch \
  --job-name "exp199-cw-s02-${TRIAL}-${CW_CLUSTER}-n${CW_NODES}" \
  --enable-extra-resources \
  --no-wait --cpu 1 --memory 4GB --disk 16GB \
  -e MARIN_PREFIX "$CW_PREFIX" \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  -e HF_TOKEN "$HF_TOKEN" \
  -e WANDB_ENTITY "$WANDB_ENTITY" \
  -e WANDB_PROJECT "$WANDB_PROJECT" \
  -e TRIAL "$TRIAL" \
  -e CLUSTER "$CW_CLUSTER" \
  -e NODES "$CW_NODES" \
  -- uv run --frozen python exp199_sweep_cw.py \
       --version "$VERSION" --run
```

At most one live dispatch may own a production run ID. Moving a trial to the
other cluster or changing from four to two nodes uses the same run/checkpoint
identity and therefore resumes it.

## Monitoring

- Iris establishes job state and logs; W&B establishes training progress.
- A queued four-node gang is normal on a crowded cluster. Prefer placing other
  trials or using a two-node fallback over repeatedly replacing the same job.
- Preemption recovery belongs to Iris. Do not launch another owner of the same
  run ID while its current driver or gang is alive.
- A trial is complete when W&B reaches the final step and the final checkpoint
  is present below `checkpoints/protein/<run-id>`.

| Trial | Cluster | Nodes | Iris job | W&B step | State |
| --- | --- | ---: | --- | ---: | --- |
| | | | | | |
