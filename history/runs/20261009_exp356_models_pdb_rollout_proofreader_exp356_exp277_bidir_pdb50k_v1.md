---
marinfold_run:
  user: bizon
  launched_at: '2026-10-09T22:05:48Z'
  experiment: exp356_models_pdb_rollout_proofreader
  kind: models
  short_description: Production bidirectional proofreader on 50000 experimental PDB
    chain structures and 399213 generator rollouts
  wandb:
    url: https://wandb.ai/open-athena/MarinFold/runs/exp356-exp277-bidir-pdb50k-v1
    entity: open-athena
    project: MarinFold
    run_id: exp356-exp277-bidir-pdb50k-v1
    run_name: exp356-exp277-bidir-pdb50k-v1
  git_sha: f14a16959e554c90d42a1246e0e5f63b303d402c
  iris_job_ids:
  - /bizon/exp356-train-prod-a01
---

# 2026-10-09 · exp356_models_pdb_rollout_proofreader · exp356-exp277-bidir-pdb50k-v1

**Launched:** 2026-10-09T22:05:48Z by bizon  
**Kind:** models  
**Experiment:** exp356_models_pdb_rollout_proofreader  
**W&B:** [exp356-exp277-bidir-pdb50k-v1](https://wandb.ai/open-athena/MarinFold/runs/exp356-exp277-bidir-pdb50k-v1)  
**Git:** `f14a1695`  

## Description

Production bidirectional proofreader on 50000 experimental PDB chain structures and 399213 generator rollouts

## Detailed plan

Fine-tune the separate exp277 step-266344 backbone with bidirectional attention, per-contact correctness logits and a global recall head. Three epochs over 399,213 retained training rollouts from 50,000 experimental structures: 18,714 optimizer steps, global batch 64 on eight H100s. Prefixes are cropped before encoding and include single-contact inputs. Validation selects the final checkpoint; the new PDB test split is reserved for release evaluation.

## Changes from previous runs

First full-corpus proofreader run, following the one- and eight-H100 execution and recovery checks.

## Notes

Source-manifest SHA256: `d58b92ecf61ab5caaa127885a606f7e814f534e353b650e19d611d1fcd82592c`. Exact launch arguments and hashes are recorded in the experiment's `data/dispatches.jsonl`. Durable progress, recovery pointers and validation timings are under `s3://marin-us-east-02a/MarinFold/exp356/runs/exp356-exp277-bidir-pdb50k-v1/`. Training started at 2026-10-09T22:05:06Z.

Intentionally stopped at 2026-10-09T23:08:07Z after saving step 3,000. The frozen-causal-backbone comparison won the complete step-2,000 validation and continues as `exp356-exp277-frozen-contact-pdb50k-v2`. This run did not complete its originally planned three epochs; its saved checkpoints and comparison reports remain available.
