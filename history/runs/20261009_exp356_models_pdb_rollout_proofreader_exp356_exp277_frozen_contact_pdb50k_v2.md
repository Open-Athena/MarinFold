---
marinfold_run:
  user: bizon
  launched_at: '2026-10-09T22:49:56Z'
  experiment: exp356_models_pdb_rollout_proofreader
  kind: models
  short_description: Frozen exp277 causal features with a bidirectional contact encoder
    on the 50000-structure PDB corpus
  wandb:
    url: https://wandb.ai/open-athena/MarinFold/runs/exp356-exp277-frozen-contact-pdb50k-v2
    entity: open-athena
    project: MarinFold
    run_id: exp356-exp277-frozen-contact-pdb50k-v2
    run_name: exp356-exp277-frozen-contact-pdb50k-v2
  git_sha: 8bc77612ff769edf54254b95efa459062bf7a1fb
  iris_job_ids:
  - /bizon/exp356-train-frozen-prod-a01
  - /bizon/exp356-train-frozen-prod-a02
  - /bizon/exp356-train-frozen-prod-a03
  - /bizon/exp356-train-frozen-prod-a04
  - /bizon/exp356-train-frozen-prod-a05
---
# 2026-10-09 · exp356_models_pdb_rollout_proofreader · exp356-exp277-frozen-contact-pdb50k-v2

**Launched:** 2026-10-09T22:49:56Z by bizon  
**Kind:** models  
**Experiment:** exp356_models_pdb_rollout_proofreader  
**W&B:** [exp356-exp277-frozen-contact-pdb50k-v2](https://wandb.ai/open-athena/MarinFold/runs/exp356-exp277-frozen-contact-pdb50k-v2)  
**Git:** `8bc77612`  

## Description

Frozen exp277 causal features with a bidirectional contact encoder on the 50000-structure PDB corpus

## Detailed plan

Compare against the fully bidirectional backbone on the identical frozen corpus and prefix sampler. Keep exp277 causal features frozen and train a four-layer 512-wide bidirectional encoder over contact triples (22,181,378 trainable parameters; 1,487,731,202 total). Global batch 64 on eight H100s. The three-epoch learning-rate schedule has 18,714 steps, with an intentional pause at step 2,000 for full validation before continuation.

## Changes from previous runs

Freezes the generator's causal backbone and trains a separate bidirectional contact encoder, rather than changing the backbone attention pattern and fine-tuning all weights. The corpus, prefix sampler, global batch and three-epoch schedule match the fully bidirectional comparison.

## Notes

Started 2026-10-09T22:48:13Z. The prerequisite eight-H100 smoke paused at step 2 and resumed in a separate job, reproducing all validation metrics exactly. Early training is about 0.27 seconds per optimizer step. Quality selection uses validation only; the new PDB test split remains reserved for the selected model.

Full validation at step 2,000 selected this architecture: AUROC 0.763, Brier 0.167, precision MAE 0.096, recall MAE 0.103; later context improves first-contact Brier by 0.0176. Job a02 resumes the same optimizer and 18,714-step schedule without the pilot pause. Submitted 2026-10-09T23:06:48Z; source base `daf06408`, with exact bundle hashes in the experiment dispatch ledger.

Job a02 stalled after logging step 16,070 at 2026-10-10T00:16:25Z, while the durable progress object remained at step 16,060. Rank 0 was idle and the other ranks waited; a fresh storage PUT from the same worker succeeded in 0.105 seconds. After three minutes without progress, a02 was cancelled and a03 was submitted at 00:19:55Z to resume the committed step-16,000 model and optimizer. At 00:22:03Z, a03 reproduced every saved validation metric exactly (loss 0.42063987723668106, Brier 0.1370822996157699) and continued beyond the interrupted steps. The same 18,714-step schedule remains in force. Source base `2cd6fc44`; exact submitted file hashes are in the dispatch ledger.

Job a03 reached step 17,000 and improved fixed-sample validation loss to 0.4157551147181948 (Brier 0.13474577970060864), then stalled during a multipart upload. A read-only process trace showed rank 0 waiting in fsspec. The complete local checkpoint, including optimizer, was salvaged through the same-region object-store origin and independently size/hash recorded at `recovered/checkpoints/exp356-exp277-frozen-contact-pdb50k-v2/step-17000`; only then were recovery pointers advanced. Job a04 reproduced all step-17,000 validation metrics exactly but stalled on a progress PUT through the origin as well. This rules out blaming only the LOTA proxy.

A same-worker probe with connection reuse disabled completed 200 consecutive writes in 6.60 seconds. Job a05 resumes the durable step-17,000 cursor with fresh HTTP connections and whole-operation upload deadlines (source `2455fb81`). These transport changes leave numerical training unchanged. All 21 tests passed before launch. Detailed recovery evidence is in `data/storage_recovery_step17000.json` and the dispatch ledger.

Job a05 reproduced step-17,000 validation exactly at 00:43:21Z and completed all three epochs at 2026-10-10T00:51:56.179463+00:00. Both subsequent large checkpoint uploads and progress writes succeeded after disabling connection reuse. The final step-18,714 model and optimizer are durable; best fixed-sample validation is step 18,000 (loss 0.4132416447291689). Final full-validation comparison and release selection follow the recorded protocol.
