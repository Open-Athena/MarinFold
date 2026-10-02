---
marinfold_run:
  user: zack
  launched_at: '2026-10-02T16:59:15Z'
  experiment: exp299_models_delta_stream_contacts
  kind: models
  short_description: 50-update TPU topology benchmark for exp277-scale delta-stream
    V2 on v6e-16
  wandb:
    url: https://wandb.ai/open-athena/MarinFold/runs/delta-v2-exp277-1_5b-v6e-16-benchmark50-a01
    entity: open-athena
    project: MarinFold
    run_id: delta-v2-exp277-1_5b-v6e-16-benchmark50-a01
    run_name: delta-v2-exp277-1_5b-v6e-16-benchmark50-a01
  git_sha: fa695ba5d37c935122c8dafe0dd1f2879ccfedbe
  iris_job_ids:
  - /zack/delta-v2-exp277-1_5b-v6e-16-benchmark50-a01
---

# 2026-10-02 · exp299_models_delta_stream_contacts · delta-v2-exp277-1_5b-v6e-16-benchmark50-a01

**Launched:** 2026-10-02T16:59:15Z by zack  
**Kind:** models  
**Experiment:** exp299_models_delta_stream_contacts  
**W&B:** [delta-v2-exp277-1_5b-v6e-16-benchmark50-a01](https://wandb.ai/open-athena/MarinFold/runs/delta-v2-exp277-1_5b-v6e-16-benchmark50-a01)  
**Git:** `fa695ba5`  

## Description

50-update TPU topology benchmark for exp277-scale delta-stream V2 on v6e-16

## Detailed plan

Measure steady-state throughput, MFU, compile time, and checkpoint time for the
fixed scientific configuration before choosing a TPU topology for cooldown
continuations. This run starts from scratch, uses global batch 128 and sequence
length 8,192, and trains for 50 updates on one `v6e-16` slice in `us-east5-b`.

## Changes from previous runs

- Doubles the prior compatibility-smoke topology from `v6e-8` to `v6e-16`.
- Uses per-chip microbatch 8 and the same Splash/blocked-attention model config.
- Disables periodic checkpoints, but Levanter still writes the forced final
  checkpoint.

## Notes

Succeeded at step 49. Final steady-state metrics were 136,025 tokens/s, 7.709
seconds/update, 12.68% instantaneous MFU, and 12.67% mean MFU. First-step JIT
was 32.5 seconds. The forced 16.50 GiB final checkpoint took about 83 seconds
from save start to the completed distributed barrier. This topology would need
roughly 7.5 days for the 84,198-update continuation from step 126,294.
