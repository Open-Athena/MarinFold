---
marinfold_run:
  user: bizon
  launched_at: '2026-09-09T19:33:23Z'
  experiment: exp277_models_single_mpnn_pilot
  kind: models
  short_description: Single 1.5B native/MPNN mixture pilot, 152B tokens
  wandb:
    url: https://wandb.ai/open-athena/MarinFold/runs/contacts-v1-exp277-m2-p06-native-mpnn-1.5B
    entity: open-athena
    project: MarinFold
    run_id: contacts-v1-exp277-m2-p06-native-mpnn-1.5B
    run_name: contacts-v1-exp277-m2-p06-native-mpnn-1.5B
  git_sha: a8582d2d8394d66f6d762a95868153a5484ccaf7
  iris_job_ids:
  - /bizon/exp277-train-a01
  - /bizon/exp277-train-a01/exp277-train-1fca4ea5
---

# 2026-09-09 · exp277_models_single_mpnn_pilot · contacts-v1-exp277-m2-p06-native-mpnn-1.5B

**Launched:** 2026-09-09T19:33:23Z by bizon  
**Kind:** models  
**Experiment:** exp277_models_single_mpnn_pilot  
**W&B:** [contacts-v1-exp277-m2-p06-native-mpnn-1.5B](https://wandb.ai/open-athena/MarinFold/runs/contacts-v1-exp277-m2-p06-native-mpnn-1.5B)  
**Git:** `a8582d2d`  

## Description

Single 1.5B native/MPNN mixture pilot, 152B tokens

## Detailed plan

Train one scratch-initialized Qwen3 1.5B with exp232 m2-p06 settings: LR 0.001, weight decay 0.2, sequence length 8192, global batch 128, WSD warmup 10% / decay 20%, 145,200 steps (152.253B tokens). Preserve packed blocked attention and scheduled amino-acid order augmentation. Sample AFDB/ESM at 5.9522%/94.0478%, splitting each equally between native and ProteinMPNN-redesigned sequences. Compare later to the matched-budget native-only baseline on routine eval-val.

## Changes from previous runs

Adds the redesigned AFDB and ESM corpora from exp266. Uses Marin 0.2.86.dev32222909699 because the current Iris controller rejects exp232's original client; scientific settings remain fixed. The shorter original sweep budget is deliberate, distinct from the longer continued-training baseline.

## Notes

Launched at batch priority on cw-us-east-02a, 16 nodes / 128 H100s. All four caches were verified before submission; all 16 workers initialized and W&B opened. An isolated ten-step smoke passed with finite train/eval losses and verified native/HF checkpoints including tokenizer files. Production startup verification passed: W&B running at step 23, train loss 6.70270, step duration 0.86358 seconds, throughput 1.214M tokens/s. Building the packed-document indexes took about six minutes before compilation; no restart was needed. Approximately 35 hours of training compute at the observed early rate, plus validation and checkpoint overhead. Training remains in progress.

Checkpoint base: `s3://marin-us-east-02a/MarinFold/exp277_models_single_mpnn_pilot/runs/contacts-v1-exp277-m2-p06-native-mpnn-1.5B/checkpoints/step-<N>/`; HF exports use the sibling `hf/step-<N>/`. Temporary checkpoints every 15 minutes; permanent checkpoints every 14,520 steps and at completion.

Monitoring began at the user’s request on 2026-09-09 at 19:47 UTC. Both driver and training gang are running at batch priority. The first temporary recovery checkpoint completed at step 461 at 19:48:38 UTC, under the standard Marin `tmp/ttl=14d/checkpoints-temp/` mirror of this run’s checkpoint path. No production restart has been required.

First full LM validation completed on 2026-09-09 at 20:13:38 UTC: step 2,114, validation loss 3.76525235. Training resumed normally and reached step 2,180 by 20:14:42 UTC. Temporary recovery checkpoint rotation also passed at step 1,457.

2026-09-09 23:36 UTC: step 15,239 / 145,200, train loss 3.03161, latest full validation loss 3.24531 at step 14,798. All seven validation passes improved. The first permanent checkpoint completed at 23:24:45 UTC: `s3://marin-us-east-02a/MarinFold/exp277_models_single_mpnn_pilot/runs/contacts-v1-exp277-m2-p06-native-mpnn-1.5B/checkpoints/step-14520`. Warmup has ended; LR remains 0.001 through step 116,160, then decays linearly to 0.0001. No production restart.

2026-09-10 03:12 UTC: step 29,186 / 145,200 (20.1%), train loss 2.93443, latest full validation loss 3.16498. Thirteen validation passes completed. The permanent step-29,040 checkpoint saved at 03:10:26 UTC under the run’s native checkpoint base. No production restart.

2026-09-10 07:01 UTC: step 43,870 / 145,200 (30.2%), train loss 2.89204, latest full validation loss 3.12627. Twenty validation passes completed. The permanent step-43,560 checkpoint saved at 06:56:46 UTC under the run’s native checkpoint base. No production restart.

2026-09-10 10:46 UTC: step 58,402 / 145,200 (40.2%), train loss 2.90834, latest full validation loss 3.10894. Twenty-seven validation passes completed. The permanent step-58,080 checkpoint saved at 10:41:28 UTC under the run’s native checkpoint base. No production restart.

2026-09-10 13:38 UTC: intentionally stopped after the user changed the objective to one epoch over the complete combined corpus. Last observed step 69,556, train loss 2.84423, validation loss 3.10302. Latest recovery checkpoint: step 68,804 (temporary 14-day prefix); latest permanent checkpoint: step 58,080. No production failures or recovery restarts. The replacement uses a separate W&B identity, `contacts-v1-exp277-m2-p06-full-epoch-1.5B`, starts from scratch, and visits every packed example once. This partial weighted-mixture run is retained as a separate experiment record.
