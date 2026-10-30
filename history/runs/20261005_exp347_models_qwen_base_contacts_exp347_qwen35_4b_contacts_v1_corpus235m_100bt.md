---
marinfold_run:
  user: bizon
  launched_at: '2026-10-05T16:25:56Z'
  experiment: exp347_models_qwen_base_contacts
  kind: models
  short_description: Qwen3.5 base full-weight contact fine-tuning
  wandb:
    url: https://wandb.ai/open-athena/MarinFold/runs/exp347-qwen35-4b-contacts_v1-corpus235m-100bt
    entity: open-athena
    project: MarinFold
    run_id: exp347-qwen35-4b-contacts_v1-corpus235m-100bt
    run_name: exp347-qwen35-4b-contacts_v1-corpus235m-100bt
  git_sha: 5a17fc0c12509ff7ec5f6836f065d90c60d374f0
  iris_job_ids:
  - /timodonnell/exp347-qwen35-4b-contacts_v1-corpus235m-100bt-a01
  - /timodonnell/exp347-qwen35-4b-contacts_v1-corpus235m-100bt-a02
  - /timodonnell/exp347-qwen35-4b-contacts_v1-corpus235m-100bt-a03
  - /timodonnell/exp347-qwen35-4b-contacts_v1-corpus235m-100bt-a04
  - /timodonnell/exp347-qwen35-4b-contacts_v1-corpus235m-100bt-a05
  - /timodonnell/exp347-qwen35-4b-contacts_v1-corpus235m-100bt-a06
  - /timodonnell/exp347-qwen35-4b-contacts_v1-corpus235m-100bt-a07
---
# 2026-10-05 · exp347_models_qwen_base_contacts · exp347-qwen35-4b-contacts_v1-corpus235m-100bt

**Launched:** 2026-10-05T16:25:56Z by bizon  
**Kind:** models  
**Experiment:** exp347_models_qwen_base_contacts  
**W&B:** [exp347-qwen35-4b-contacts_v1-corpus235m-100bt](https://wandb.ai/open-athena/MarinFold/runs/exp347-qwen35-4b-contacts_v1-corpus235m-100bt)  
**Git:** `5a17fc0c`  

## Description

Qwen3.5 base full-weight contact fine-tuning

## Detailed plan

_(Why we ran this, what we expect to see, unusual parameters.)_

## Changes from previous runs

_(Bullet list of differences from the last run of this kind.)_

## Notes

October 19 bounded checkpoint I/O repair passed production saves. A subsequent
October 20 restart failed at the first backward after restoring step 4620, with
NCCL reporting CUDA out of memory. Commit d2836a26 synchronizes the first
microbatch after DDP construction to establish shared gradient bucket storage.
Attempt a03 passed step 4621 and saved the model, tokenizer, and all eight
optimizer shards; continued progress was observed beyond step 4800.
The immutable step-4620 BF16 export scored eval-val all R=0.14481 / long R=0.10787.
The full 100B-additional-token target and 235.49M-document source pool are unchanged.

October 22: the preceding attempt failed after preemption restart, with seven
ranks waiting for the first token-count reduction while rank zero had printed
WANDB_RUN but not completed the run-registration write. Commit 7d9849d3 bounds
startup metadata writes and starts diagnostics earlier. Attempt a04 was
submitted with explicit valid CoreWeave credentials; at 19:53 UTC it remains
capacity-pending in east. No GPU progress with this repair is yet verified.

October 22 regional relocation: the east pending root was canceled before handoff. Production resumed on eight RNO2A H100s with commit 990e3641. Independently verified local checkpoint step 5230 / 574,764,731 additional tokens, all 20 objects (50,489,547,803 bytes), tokenizer and eight optimizer shards at `s3://marin-rno2a/MarinFold/exp347_qwen_base_contacts/checkpoints/exp347-qwen35-4b-contacts_v1-corpus235m-100bt/step-5230`. All new checkpoints and periodic exports use marin-rno2a (verified physical region RNO2A). Source-corpus URIs and full optimizer/data/RNG state are preserved; subsequent recovery prefers local successors. No checkpoint objects were written to the east root after relocation.

October 30: a06 failed October 27 at 12:36 UTC because CoreWeave suspended RNO writes. Last observed step44316/6709343269 tokens; complete local checkpoint44293/6705498958 tokens independently verified (20 objects). Current regional PUT/GET and multipart upload probes pass. Replacement a07 submitted on the unchanged eight-H100 profile; preserve local resume state and100B target.
