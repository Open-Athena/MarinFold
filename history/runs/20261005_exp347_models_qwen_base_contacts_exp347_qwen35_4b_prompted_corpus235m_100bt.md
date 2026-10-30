---
marinfold_run:
  user: bizon
  launched_at: '2026-10-05T16:27:55Z'
  experiment: exp347_models_qwen_base_contacts
  kind: models
  short_description: Qwen3.5 base full-weight contact fine-tuning
  wandb:
    url: https://wandb.ai/open-athena/MarinFold/runs/exp347-qwen35-4b-prompted-corpus235m-100bt
    entity: open-athena
    project: MarinFold
    run_id: exp347-qwen35-4b-prompted-corpus235m-100bt
    run_name: exp347-qwen35-4b-prompted-corpus235m-100bt
  git_sha: 5a17fc0c12509ff7ec5f6836f065d90c60d374f0
  iris_job_ids:
  - /timodonnell/exp347-qwen35-4b-prompted-corpus235m-100bt-a01
  - /timodonnell/exp347-qwen35-4b-prompted-corpus235m-100bt-a02
  - /timodonnell/exp347-qwen35-4b-prompted-corpus235m-100bt-a03
  - /timodonnell/exp347-qwen35-4b-prompted-corpus235m-100bt-a04
  - /timodonnell/exp347-qwen35-4b-prompted-corpus235m-100bt-a05
  - /timodonnell/exp347-qwen35-4b-prompted-corpus235m-100bt-a06
  - /timodonnell/exp347-qwen35-4b-prompted-corpus235m-100bt-a07
  - /timodonnell/exp347-qwen35-4b-prompted-corpus235m-100bt-a08
---
# 2026-10-05 · exp347_models_qwen_base_contacts · exp347-qwen35-4b-prompted-corpus235m-100bt

**Launched:** 2026-10-05T16:27:55Z by bizon  
**Kind:** models  
**Experiment:** exp347_models_qwen_base_contacts  
**W&B:** [exp347-qwen35-4b-prompted-corpus235m-100bt](https://wandb.ai/open-athena/MarinFold/runs/exp347-qwen35-4b-prompted-corpus235m-100bt)  
**Git:** `5a17fc0c`  

## Description

Qwen3.5 base full-weight contact fine-tuning

## Detailed plan

_(Why we ran this, what we expect to see, unusual parameters.)_

## Changes from previous runs

_(Bullet list of differences from the last run of this kind.)_

## Notes

Attempt a03 deployed the first-update DDP memory repair from d2836a26 on
October 21. It passed step 12157, saved all eight optimizer shards plus model
and tokenizer, and continued through step 12251. It was deliberately paused
for evaluation capacity, with complete step-12218 recovery state verified.
Attempt a04 was submitted at 2026-10-21 14:42:58 UTC after a03 was terminal;
it retains the same optimizer/data/RNG state, run identity, world size, and
100B-additional-token budget. The immutable step-12051 BF16 export scored
all R=0.15239 / long R=0.10738 on all 97 eval-val proteins.
At 14:47 UTC, a04 had advanced to step 12219 / 460,005,111 tokens and saved
a new complete 20-object checkpoint with all eight optimizer shards and tokenizer.

October 22: the preceding attempt failed after preemption restart, with seven
ranks waiting for the first token-count reduction while rank zero had printed
WANDB_RUN but not completed the run-registration write. Commit 7d9849d3 bounds
startup metadata writes and starts diagnostics earlier. Attempt a05 was
submitted with explicit valid CoreWeave credentials; at 19:53 UTC it remains
capacity-pending in east. No GPU progress with this repair is yet verified.

October 22 regional relocation: the east pending root was canceled before handoff. Production resumed on eight RNO2A H100s with commit 990e3641. Independently verified local checkpoint step 12539 / 476,967,387 additional tokens, all 20 objects (50,489,547,802 bytes), tokenizer and eight optimizer shards at `s3://marin-rno2a/MarinFold/exp347_qwen_base_contacts/checkpoints/exp347-qwen35-4b-prompted-corpus235m-100bt/step-12539`. All new checkpoints and periodic exports use marin-rno2a (verified physical region RNO2A). Source-corpus URIs and full optimizer/data/RNG state are preserved; subsequent recovery prefers local successors. No checkpoint objects were written to the east root after relocation.

October 30: a07 failed October 27 at 12:35 UTC because CoreWeave suspended RNO writes. Last observed step67826/3154711265 tokens; complete local checkpoint67797/3152923970 tokens independently verified (20 objects). Current regional PUT/GET and multipart upload probes pass. Replacement a08 submitted on the unchanged eight-H100 profile; preserve local resume state and100B target.
