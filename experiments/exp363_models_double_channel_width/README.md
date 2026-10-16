---
marinfold_experiment:
  issue: 363
  title: 'exp: double the channel width of the default exp277 model'
  kind: models
  branch: exp363/double-channel-width
---

# exp: double the channel width of the default exp277 model

**Issue:** [#363](https://github.com/Open-Athena/MarinFold/issues/363) · **Kind:** `models` · **Branch:** `exp363/double-channel-width`

## Question

Does doubling the channel width of the current default exp277 model improve contact accuracy at the same one-epoch training exposure?

## Hypothesis

The wider model will reduce held-out language-model loss and improve natural eval-val contact R-precision by at least 0.005 absolute over exp277. This is a single-seed model-capacity comparison, not a compute-matched comparison.

## Background

The current default is `contacts-v1-exp277-m2-p06-full-epoch-1.5B`, trained from scratch in #277 for one epoch over 232,090,905 native plus ProteinMPNN documents (248.584B raw tokens). Its published step-266344 config has 24 layers, hidden width 2048, MLP width 8192, 32 query heads, 8 KV heads, head dimension 64, vocabulary 2845, and untied embeddings. The exact parameter count including Q/K norms is 1,471,374,336.

The requested variant doubles both hidden and MLP widths, as explicitly confirmed by the requester. Keep head counts fixed so the head dimension doubles too:

| Setting | Current default | Wider variant |
| --- | ---: | ---: |
| Hidden width | 2048 | 4096 |
| MLP width | 8192 | 16384 |
| Layers | 24 | 24 |
| Query / KV heads | 32 / 8 | 32 / 8 |
| Head dimension | 64 | 128 |
| Parameters | 1,471,374,336 | 5,861,984,256 |

This is approximately 5.862B parameters, 3.984x the default. The parameter calculation includes both vocabulary matrices, attention projections, three gated-MLP matrices per layer, residual/final RMS norms, and Q/K norms. Validate it against the actual model parameter tree before launch.

## Approach

- Train from scratch on the exact four completed exp277 caches, with the same finite globally shuffled epoch: 34,092,146 packed examples and 266,345 optimizer updates at global batch 128 and sequence length 8192.
- Retain exp277's optimizer/schedule, seeds, RoPE, full validation set, contact document format, and scheduled amino-acid augmentation. Pin the existing tokenizer revision explicitly. This isolates width; optimizer tuning would be follow-up work.
- Use a feature branch and PR with an experiment-local launcher, committed dependency lock, architecture/count audit, and tests of the resolved training configuration.
- Inspect live CoreWeave capacity, keep data and training in US-EAST-02A, and run a short GPU smoke before production. Adjust only execution settings such as per-device microbatching to fit the larger model while preserving global batch.
- Log training to `open-athena/MarinFold`, create run history immediately when W&B initializes, and retain resumable checkpoints with run name and step.
- Export/publish checkpoints with tokenizer files. Evaluate contact rollout-v2 on eval-val and eval-denovo against exp277, recording per-input timings. Do not use eval-test for routine iteration.
- Maintain the experiment README and summary slides with observed results and runtime.

## Success criteria

- Exact architecture and parameter-tree count validated before GPU production launch.
- Smoke completes optimizer updates, validation, and checkpoint writing with finite metrics.
- Production completes the same full corpus epoch, with durable checkpoints and tokenizer.
- Report paired contact R-precision differences with protein-bootstrap intervals; a >=0.005 natural eval-val gain supports the stated hypothesis. Report eval-denovo separately and preserve the single-seed and increased-compute caveats.

## Results

The abstract Levanter model-tree audit confirms **5,861,984,256 parameters**, versus **1,471,374,336** for exp277 (3.984x). The audit allocates no model weights; its output is committed in `data/parameter_counts.csv`.

Twelve local tests pass, including the resolved Marin training contract and the inherited finite-epoch dataset tests. Only `hidden_dim` and `intermediate_dim` differ in the model configuration. Per-device microbatch is one; global batch remains 128. The tokenizer is pinned to revision `80f4e411b957641e8c350b07bbe1e2c832697518`.

The live federation inventory at 2026-10-13 14:43 UTC reports 248 available H100s in `cw-us-east-02a`, with no pending tasks. An initial 16-node / 128-H100 smoke was attempted at batch priority; subsequent placement was reduced to 64 H100s as recorded below. The data is already in the regional CoreWeave bucket; no corpus transfer is required. See `data/launch_capacity.json`.

The 64-H100 smoke completed successfully: ten optimizer updates, final train loss **8.56948**, two-batch validation loss **8.03575**, and **2.821 seconds/update**. Native and HF step-9 checkpoints are present; the HF export includes tokenizer files. `data/smoke_metrics.json` records the final W&B summary and exported file sizes. The compressed ten-step LR schedule produced finite but spiky training losses (including 31.25); the smoke establishes execution/memory/checkpoint correctness, not convergence. Production retains exp277's full 10% warmup (~26,634 updates), with no LR retuning.

Production uses **four nodes / 32 H100s**, microbatch one per GPU and accumulation four, preserving global batch 128 and all 266,345 updates. Driver [`/bizon/exp363-train-a02`](https://iris.oa.dev/#/job/%2Fbizon%2Fexp363-train-a02), source `56f10733`, submitted at 2026-10-13 15:14:24 UTC. Child: `/bizon/exp363-train-a02/exp363-train-4fae9eab`. The [production W&B run](https://wandb.ai/open-athena/MarinFold/runs/contacts-v1-exp363-m2-p06-full-epoch-5.86B) initialized at 15:15:36 UTC and reports 32 H100 devices. Subsequent progress and TPU exploration are recorded below. See `data/production_startup.json` and the committed run history. The first production submission requested 64 H100s but was cancelled before GPU/W&B initialization because other full training runs left only 56 H100s available. The smaller allocation starts within that capacity. The original smoke remains the 64-H100 execution check; the resolved 32-H100 production configuration also passed the batch/update-count check.

Production measured **5.608 seconds/update** on 32 H100s, approximately **17.3 days of optimizer time** for the full epoch, excluding validation, checkpointing, and preemptions. Repeated preemptions make a calendar completion estimate unreliable. The CPU parent has a 30-day timeout and waits for the GPU child. Production retains the baseline's full validation schedule.

The first GPU smoke `/bizon/exp363-train-smoke-a01` failed before model/W&B initialization because this Levanter release passes `repo@revision` to Hugging Face without parsing it. The tokenizer is now bundled as an unmodified, explicitly revisioned local snapshot using the loader's supported directory interface. A new integration test loads it through the actual training backend and checks the exported HF tokenizer vocabulary. The second smoke was cancelled before GPU startup because other jobs consumed capacity, leaving 88 free H100s. The third smoke uses eight nodes / 64 H100s and gradient accumulation to retain global batch 128. Its run is [contacts-v1-exp363-m2-p06-full-epoch-5.86B-smoke-n8-a03](https://wandb.ai/open-athena/MarinFold/runs/contacts-v1-exp363-m2-p06-full-epoch-5.86B-smoke-n8-a03), Iris `/bizon/exp363-train-smoke-a03/exp363-train-ecf49211`, source `65e00f6332d7ca39580d69c4eac8f900aebc2b26`. All eight workers completed distributed initialization and W&B reports 64 H100 devices.

## Conclusion

The requested architecture is implemented and its exact parameter count is verified. Contact-accuracy and training-quality conclusions await completed training.

## TPU trial (2026-10-14)

The GPU job has encountered eleven preemptions and is again queued. A completed
step-3848 optimizer checkpoint exists on CoreWeave. Its observed steady compute
speed is approximately 5.6 seconds/update; interrupted wall time is much longer.

The requester asked to try TPUs. `tpu_smoke.py` runs twenty updates on **v6e-32
(32 actual chips), pinned to us-east5-b**, at global batch 128, sequence length
8192, per-device microbatch one, and tensor parallelism one. It changes only the
attention implementation to the TPU Splash kernel; architecture and parameter
count are unchanged. Native AFDB/ESM and validation caches already exist under
`gs://marin-us-east5/protein-structure/MarinFold/exp232_train_trc/tokenized/`.
The trial uses those native caches, a fresh initialization and compressed smoke
schedule. It measures hardware compatibility/throughput, not production
convergence or a continuation of the native/MPNN epoch.

The initial v5p-64 attempt was cancelled before execution because the pool was
in degraded provisioning backoff. The replacement uses the same data region.
Driver: `/bizon/exp363-tpu-smoke-v6e-32-a01`; child:
`/bizon/exp363-tpu-smoke-v6e-32-a01/exp363-train-952f2c08`; source:
`8ddd24eba02e501d6314ba022d3af20782bf7f1f`. Submitted at
2026-10-14 20:43:52 UTC; awaiting TPU slice startup. The CPU driver and TPU child
use batch priority. All trial inputs, compilation caches, checkpoints and
exports remain in us-east5. Seven architecture, tokenizer and resolved GPU/TPU
configuration tests pass. The dependency lock adds libtpu and CPU Torch wheels
without removing or changing an existing package version.

Migrating production requires **250,073,316,791 bytes** of MPNN caches plus
**70,326,053,055 bytes** for the observed full step-3848 checkpoint: **320.4 GB**
total from CoreWeave to GCS. Native and validation caches need no copy.
`data/tpu_migration_assessment.json` records source prefixes, object counts and
byte totals. No bulk transfer has been performed; root AGENTS.md requires
explicit human sign-off for cross-region copies larger than 10 GB.

### 2026-10-15 recovery

The first v6e trial never initialized W&B or reached training. Its CPU driver
was preempted five times, killing and requeuing the TPU children, then exceeded
the two-hour execution timeout. The launcher now explicitly pins the CPU driver
to non-preemptible workers and allows 24 hours for queueing and the short test;
the TPU allocation remains preemptible. The replacement driver is
`/bizon/exp363-tpu-smoke-v6e-32-a02`, source
`288a63cdbf7a0aacde5b7ba389b72ae6c37eab83`, submitted 2026-10-15 14:39 UTC.
TPU optimizer throughput remains unmeasured.

GPU production has twelve preemptions and is queued. Last reported optimizer
step is 3918 (1.47% of the epoch), train loss 3.45197; the last validation loss
is 3.68509. Those metrics were logged on 2026-10-14 at 20:03:50 UTC, so they do
not establish overnight progress. The complete step-3848 checkpoint remains
reachable. The 320.4 GB production migration is still awaiting explicit transfer
approval; no bulk copy has been started.

### 2026-10-16 checkpoint failure and recovery

GPU training reached step **8438** (8,439 updates, **3.17%**), with train loss
**3.15777**, last validation loss **3.46485**, and **5.638 seconds/update**.
The job then failed in the asynchronous checkpoint barrier
`tensorstore_checkpoint_28::0` after a 30-minute timeout. Iris and W&B marked
the run finished despite the incomplete epoch. The last complete checkpoint is
**step 8120**, committed at 02:02:09 UTC; step 8279 has no completion metadata.
See `data/recovery_20261016.json` for the observed metrics and storage evidence.

The runtime now requires a committed final step-266344 checkpoint before
recording production success. Recovery bypasses the stale executor success
marker while retaining the fixed input versions, run identity, checkpoint path,
and full training schedule. Ten architecture, GPU/TPU configuration, tokenizer,
and recovery tests pass. Restart `/bizon/exp363-train-a03`, source
`241b26c99ea261a57a2695fe3b937471968e4da7`, was submitted at 15:42:01 UTC
on 32 H100s. Child `/bizon/exp363-train-a03/exp363-train-0cdc9fb5` is running
startup. Checkpoint restoration and new updates are not yet confirmed; this restart does not establish a fix for the underlying
checkpoint barrier failure.

The TPU a02 driver remained pending because it requested 8 GB while 2.1 GB was
available. Building and fingerprinting its actual plan used 451,380 KB peak RSS
locally, so the replacement requests 0.5 CPU and 2 GB on non-preemptible workers.
Trial `/bizon/exp363-tpu-smoke-v6e-32-a03` was submitted at 15:41:57 UTC from
the same source revision. Its CPU driver has started dependency setup; no TPU
child or optimizer throughput is confirmed yet.

The migration assessment now references the complete step-8120 checkpoint
(70,326,130,929 bytes). Together with MPNN caches it remains approximately
**320.4 GB**. Transfer approval is still pending and no bulk copy has occurred.

### TPU selection revision (2026-10-16)

The requester questioned the v6e choice. v6e was an availability fallback after
v5p provisioning failures; it had not demonstrated better training performance.
The v6e a03 TPU child was still pending and was cancelled before training.
The preferred trial is now **v5p-64: 32 chips across eight VMs in us-east5-a**,
submitted as `/bizon/exp363-tpu-smoke-v5p-64-a02` from source
`3ee08a5cf209e7d7e4b9d3af75ab13f03da757a9`. The existing tested configuration,
20-step budget, regional native data and stable CPU-driver settings apply.

Google documents v6e as supporting transformer training and serving; the
single-VM v6e-8 configuration is specifically optimized for inference. v5p
provides 95 GiB HBM/chip and 2,765 GB/s HBM bandwidth, versus v6e's reported
32 GB and 1,638 GB/s. v5p also has higher per-chip bidirectional ICI bandwidth
(1,200 versus 800 GB/s), while v6e has higher peak BF16 compute (918 versus
459 TFLOP/s/chip). We prioritize v5p's memory and communication headroom for
5.862B training at 8K context; superior measured throughput is not established.
Sources: [v5p](https://docs.cloud.google.com/tpu/docs/v5p),
[v6e](https://docs.cloud.google.com/tpu/docs/v6e).

At 15:49 UTC the regional v5p-64 pool reported one ready slice and one booting,
but provisioning health remained degraded with recent failures. Ready does not
establish that a slice is free. Placement and TPU optimizer results are pending.

## Reproduction

Run from the repository root:

```bash
JAX_PLATFORMS=cpu uv run --project experiments/exp363_models_double_channel_width --locked python -m experiments.exp363_models_double_channel_width.count_parameters
JAX_PLATFORMS=cpu uv run --project experiments/exp363_models_double_channel_width --locked --with pytest python -m pytest -q experiments/exp363_models_double_channel_width/test_train.py experiments/exp277_models_single_mpnn_pilot/test_epoch_data.py
uv run --no-project python experiments/exp363_models_double_channel_width/launch.py train-smoke --nodes 8 --attempt 3
# After the smoke succeeds:
uv run --no-project python experiments/exp363_models_double_channel_width/launch.py train --nodes 4 --attempt 2
uv run --no-project python experiments/exp363_models_double_channel_width/launch.py train --nodes 4 --attempt 3 --resume
uv run --no-project python experiments/exp363_models_double_channel_width/launch.py tpu-smoke --tpu v5p-64 --attempt 2
uv run --no-project --with matplotlib python experiments/exp363_models_double_channel_width/build_summary.py
```

The launcher submits a CPU parent through the current Iris CLI and bundles the pinned exp277 training runtime. Its GPU child is also batch priority; the parent waits for the child. Production has bounded task-failure recovery and resumes from its own checkpoint. Smoke attempts use separate run identities. No exp277 weights are restored.

Production run name: `contacts-v1-exp363-m2-p06-full-epoch-5.86B`.
Checkpoints: `s3://marin-us-east-02a/MarinFold/exp363_models_double_channel_width/runs/contacts-v1-exp363-m2-p06-full-epoch-5.86B/checkpoints/step-<N>/`.

The implementation reuses exp277's finite-epoch dataset, cache verification, and pinned CUDA setup. It retains the baseline's 1e-3 learning rate; whether that remains optimal at 5.86B is an experimental uncertainty, not a result already established by the smoke.
