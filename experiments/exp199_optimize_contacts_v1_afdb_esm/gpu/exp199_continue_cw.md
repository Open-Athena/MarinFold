# exp199 CoreWeave continuation

`exp199_continue_cw.py` continues selected exp199 CoreWeave models from an exact
permanent Levanter checkpoint. The initial supported source is
`prot-exp199-cw-cv1-s02-m1-p06-aug` at `step-116160`, the 80% boundary immediately
before its original cooldown.

The checkpoint name is zero-indexed: `step-116160` restores
`TrainerState.step=116161`. The continuation adds exactly 435,600 optimizer updates
at global batch 128 and sequence length 8,192: 456,759,705,600 new tokens. Including
the source checkpoint's 121,803,636,736 tokens, training ends at trainer boundary
551,761 after 578,563,342,336 cumulative tokens. Its final completed update and
permanent checkpoint are `step-551760`.

The new LR cycle has no rewarmup. It holds the restored `1e-3` peak for 348,480
updates (80% of the continuation), then linearly decays toward `1e-4` over the final
87,120 updates. Cooldown begins at absolute step 464,641; the standard Optax
schedule reaches exactly `1e-4` at the terminal step-551761 boundary, immediately
after the final optimizer update.

Training retains the source's 50/50 AFDB–ESM mixture, AdamW state, RNG, data
position, global batch, model architecture, tokenizer, packing, and Block-Feistel
shuffle. Amino-acid statement permutation is applied to every new training example;
validation is never augmented.

## Identity and storage

Set `SOURCE=m1-p06-aug`. The production W&B ID and checkpoint root are:

```text
prot-exp199-cw-cv1-cont-sNN-m1-p06-srcaug-aug100
s3://marin-us-east-02a/marin/protein-structure/MarinFold/
  exp199_continue_contacts_v1_cw/checkpoints/protein/
  prot-exp199-cw-cv1-cont-sNN-m1-p06-srcaug-aug100
```

CoreWeave's production clusters share this S3 namespace, so placement is excluded
from production identity. The source is read directly from the existing sweep root;
the continuation does not copy checkpoints or token caches.

For now, production has exactly two eligible gang profiles:

| Profile | Cluster choices | Gang | GPUs | Measured step time |
| --- | --- | ---: | ---: | ---: |
| H100 | `cw-rno2a`, `cw-us-east-02a` | 8 nodes | 64 H100 | about 1.51--1.53 s |
| GB200 | `cw-us-east-08a` | 8 nodes | 32 GB200 | about 1.42 s |

Choose one profile according to current availability. Never run both against the
same production identity at once because W&B and checkpoint storage are shared.
Submit every Iris root driver with `--priority batch`; the GPU child gang inherits
that priority. Interactive and production priority are not permitted for this run.

Full validation retains the sweep's 2,230-step cadence. Permanent checkpoints retain
the sweep's 14,520-step cadence, producing 30 new permanent checkpoints through the
terminal boundary; temporary checkpoints retain the standard ten-minute cadence.

## Preview and smoke

Run from this directory after sourcing `~/marin.env`:

```bash
export MARIN_PREFIX=s3://marin-us-east-02a/marin/protein-structure/MarinFold/exp199_continue_contacts_v1_cw
export SOURCE=m1-p06-aug
export CLUSTER=cw-us-east-02a
export NODES=1
export SMOKE=1
export SMOKE_STEPS=20

uv run --extra gpu --frozen python exp199_continue_cw.py \
  --version 2026.08.10.1
```

Add `--run` only for an approved smoke. Production must omit `SMOKE` and
`SMOKE_STEPS`, use exactly 8 H100 or 8 GB200 nodes, and run only after the
lowered configuration and a real full-state smoke have been reviewed.
