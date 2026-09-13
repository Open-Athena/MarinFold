---
marinfold_run:
  user: zack
  launched_at: '2026-09-12T19:03:13Z'
  experiment: exp177_models_compare_soft_target_contact_loss_against
  kind: models
  short_description: Corrected pos-token soft-target contacts-v1 run on v3 precomputed
    data, LR 1e-4, 2x4 GB200
  wandb:
    url: https://wandb.ai/open-athena/MarinFold/runs/exp177-soft-target-cw-r140-posv3-gb200-2x4-bs32-tp1-vecloss-full-lr1e-4
    entity: open-athena
    project: MarinFold
    run_id: exp177-soft-target-cw-r140-posv3-gb200-2x4-bs32-tp1-vecloss-full-lr1e-4
    run_name: exp177-soft-target-cw-r140-posv3-gb200-2x4-bs32-tp1-vecloss-full-lr1e-4
  git_sha: ffa820429b3d0ed8335ad46037f7a4a86763f32e
  iris_job_ids:
  - /zack/exp177-soft-target-cw-r140-posv3-gb200-2x4-bs32-tp1-vecloss-full-lr1e-4-driver
  - /zack/exp177-soft-target-cw-r140-posv3-gb200-2x4-bs32-tp1-vecloss-full-lr1e-4-driver/exp177-soft-target-cw-r140-posv3-gb200-2x4-bs32-tp1-vecloss-full-lr1e-4
---

# 2026-09-12 · exp177_models_compare_soft_target_contact_loss_against · exp177-soft-target-cw-r140-posv3-gb200-2x4-bs32-tp1-vecloss-full-lr1e-4

**Launched:** 2026-09-12T19:03:13Z by zack  
**Kind:** models  
**Experiment:** exp177_models_compare_soft_target_contact_loss_against  
**W&B:** [exp177-soft-target-cw-r140-posv3-gb200-2x4-bs32-tp1-vecloss-full-lr1e-4](https://wandb.ai/open-athena/MarinFold/runs/exp177-soft-target-cw-r140-posv3-gb200-2x4-bs32-tp1-vecloss-full-lr1e-4)  
**Git:** `ffa82042`  

## Description

Corrected pos-token soft-target contacts-v1 run on v3 precomputed data, LR 1e-4, 2x4 GB200

## Detailed plan

- Corrected-format exp177 soft-target run using the pos-token v3 precomputed dataset.
- Learning rate: `1e-4`.
- Shape: `2x4` GB200, global batch size 32, tensor parallelism 1, sparse precomputed batches.
- Monitor standard next-token `eval/loss`/`eval/bpb`, soft diagnostics (`teacher_ce`, `argmax_valid`), and optimizer stability metrics.

## Changes from previous runs

- Soft-target preprocessing now emits ordinary contacts-v1 position tokens (`contact.pos_i` / `contact.pos_j`), not raw sequence-index tokens.
- Soft-target document prefix now matches contacts-v1 generated prefix through `<begin_structure>`.
- Precomputed data prefix: `s3://marin-us-east-02a/protein-structure/MarinFold/exp177_soft_target_loss_h2h/preprocessed/soft_target_compact_v3/2026.09.12.pos-token.1/`.
- S3 validation job `/zack/exp177-soft-target-preprocess-v3-pos-token-validate16-driver` checked 512 rows across 16 shards successfully.
- Built from `exp177/soft-target-diagnostics` commit `ffa82042`.

## Notes

- False-start driver jobs r137/r138/r139 did not receive intended env vars from the launcher and defaulted to old H100/v1 settings; they failed quickly and should not be compared. r140/r141/r142 were relaunched with Iris `-e` env forwarding.
- First evaluation was expected around step 8920 for this full batch-32 run.

- 2026-09-13 first eval reached at step 8920. Snapshot:
  - `eval/loss=6.5744686`, `eval/bpb=0.8196763`, `train/loss=3.5986583`
  - soft diagnostics: `total_loss=51.0394111`, `first_endpoint_teacher_ce=5.6888954`, `second_endpoint_teacher_ce=5.3029857`, `second_endpoint_argmax_valid=0.0278637`, `actual_second_in_teacher_set_fraction≈1.0`
  - latest stability near step 10200: `grad_norm=1.1373082`, `update_norm=0.3142197`, `param_norm=790.1610`, `update_to_param_norm=3.97665e-4`
