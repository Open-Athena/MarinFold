---
marinfold_run:
  user: bizon
  launched_at: '2026-10-05T14:12:20Z'
  experiment: exp347_models_qwen_base_contacts
  kind: evals
  short_description: Canonical 100-rollout R-precision; E8 reference
  wandb:
    url: https://wandb.ai/open-athena/MarinFold/runs/exp347-e8-reference-rollout
    entity: open-athena
    project: MarinFold
    run_id: exp347-e8-reference-rollout
    run_name: exp347-e8-reference-rollout
  git_sha: 62de43343497b2aae99f27273c6284d73dd80a19
  iris_job_ids:
  - /timodonnell/exp347-e8-reference-rollout-a01
  - /timodonnell/exp347-e8-reference-rollout-a02
---
# 2026-10-05 · exp347_models_qwen_base_contacts · exp347-e8-reference-rollout

**Launched:** 2026-10-05T14:12:20Z by bizon  
**Kind:** evals  
**Experiment:** exp347_models_qwen_base_contacts  
**W&B:** [exp347-e8-reference-rollout](https://wandb.ai/open-athena/MarinFold/runs/exp347-e8-reference-rollout)  
**Git:** `62de4334`  

## Description

Canonical 100-rollout R-precision; E8 reference

## Detailed plan

Validate the new pinned vLLM 0.30 evaluator against the exp75 E8 checkpoint at step35679 on the canonical legacy554 universe, using100 rollouts and unchanged exp89 metrics. This is a reference check, not eval-val.

## Changes from previous runs

Uses the new vLLM runtime needed for the text-only Qwen3.5 architecture. Launch included working-tree changes; the recorded git_sha is the base revision.

## Notes

Passed: all554 units and55400 rollouts, zero truncations. All-range R-precision0.424376548979845; long-range0.365988205633634, both within0.0004 of reference. Metrics and per-protein timings are committed under experiments/exp347_models_qwen_base_contacts/data/e8_reference/. Attempt1 failed before dispatch because a local child repeated the federation peer pin; attempt2 corrected locality and completed. The W&B metric key prefix is eval_val for this engineering reference run; its universe is legacy554, as recorded in config and artifacts.
