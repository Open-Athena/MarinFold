---
marinfold_run:
  user: bizon
  launched_at: '2026-10-13T15:15:10Z'
  experiment: exp362_models_smaller_model_scaling
  kind: models
  short_description: 498M depth-only default model; full exp277 corpus and 266345
    updates
  wandb:
    url: https://wandb.ai/open-athena/MarinFold/runs/contacts-v1-exp362-depth8-full-epoch
    entity: open-athena
    project: MarinFold
    run_id: contacts-v1-exp362-depth8-full-epoch
    run_name: contacts-v1-exp362-depth8-full-epoch
  git_sha: 6aad8f15e5c092dc54022d900ef56fd1c245315a
  iris_job_ids:
  - /bizon/exp362-depth8-train-a01
  - /bizon/exp362-depth8-train-a01/exp362-train-3d12c3ab
  - /bizon/exp362-depth8-train-a01/exp362-train-0590ceee
---
# 2026-10-13 · exp362_models_smaller_model_scaling · contacts-v1-exp362-depth8-full-epoch

**Launched:** 2026-10-13T15:15:10Z by bizon  
**Kind:** models  
**Experiment:** exp362_models_smaller_model_scaling  
**W&B:** [contacts-v1-exp362-depth8-full-epoch](https://wandb.ai/open-athena/MarinFold/runs/contacts-v1-exp362-depth8-full-epoch)  
**Git:** `6aad8f15`  

## Description

498M depth-only default model; full exp277 corpus and 266345 updates

## Detailed plan

Train the eight-layer, 498,228,224-parameter model on the exact exp277 first-epoch corpus for all 266,345 updates. Compare final validation loss and matched contact evaluations with the four-layer, sixteen-layer, and existing twenty-four-layer models.

## Changes from previous runs

Depth is the only model change from the default recipe. Production placement uses four H100 nodes (32 GPUs) at batch priority in cw-us-east-02a; the ten-update smoke used two nodes and a separate checkpoint identity.

## Notes

W&B registered the run at 2026-10-13T18:47:04Z after initial admission delay. Live observation on 2026-10-14 confirmed advancing updates, all four child workers running, nine preemptions, and zero reported child failures. Keep the original dispatch while it progresses. This history was backfilled when monitoring access returned after repeated permission failures.

Verified completed at 2026-10-19 13:18 UTC: W&B `run_progress=1`, `_step=266344`, final validation loss 3.0833940505981445; exact Iris root and replacement child succeeded. Final native checkpoint timestamp: 2026-10-17T18:22:32.939237 UTC. Native metadata/manifests and HF architecture, tokenizer, tensor headers and object sizes passed verification; payload bytes were not downloaded or hashed. Source paths and object identities are recorded in `experiments/exp362_models_smaller_model_scaling/data/final_depth8_checkpoint.json`. Contact evaluation and public HF publication remain pending.
