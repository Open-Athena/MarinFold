---
marinfold_run:
  user: bizon
  launched_at: '2026-06-10T12:54:46Z'
  experiment: exp67_models_contacts_v1_1_5b
  kind: models
  short_description: 1.5B Llama on contacts-v1, unmasked next-token loss, shuffled,
    12k steps (~2.7 epochs), LR 3.5e-4, v5p-8 @ us-east5-a
  wandb:
    url: https://wandb.ai/open-athena/MarinFold/runs/protein-contacts-1_5b-3.5e-4-contacts-v1-unmasked-3b5cf2
    entity: open-athena
    project: MarinFold
    run_id: protein-contacts-1_5b-3.5e-4-contacts-v1-unmasked-3b5cf2
    run_name: protein-contacts-1_5b-3.5e-4-contacts-v1-unmasked
  git_sha: 5f925614076f9f3594a314182267b31841ce6064
  iris_job_ids:
  - /bizon/iris-run-job-20260610-124627
---

# 2026-06-10 · exp67_models_contacts_v1_1_5b · protein-contacts-1_5b-3.5e-4-contacts-v1-unmasked

- **Launched:** 2026-06-10T12:54:46Z by bizon
- **Kind:** models
- **Experiment:** exp67_models_contacts_v1_1_5b
- **W&B:** [protein-contacts-1_5b-3.5e-4-contacts-v1-unmasked](https://wandb.ai/open-athena/MarinFold/runs/protein-contacts-1_5b-3.5e-4-contacts-v1-unmasked-3b5cf2)
- **Git:** `5f925614`

## Description

1.5B Llama on contacts-v1, unmasked next-token loss, shuffled, 12k steps (~2.7 epochs), LR 3.5e-4, v5p-8 @ us-east5-a

## Outcome (✅ COMPLETED 2026-06-14)

Ran to `SUCCEEDED` — full 12,000 steps. **train loss 8.29→2.87** (min 2.85),
**eval/loss (full held-out val) 3.63→2.98**, no overfitting. 74.8 h wall-clock
(**17 preemptions, 0 failures** — all auto-resumed; two multi-hour overnight
pending stalls on v5p capacity). Artifacts under
`gs://marin-us-east5/protein-structure/MarinFold/exp67_contacts_v1_1_5b/checkpoints/protein-contacts-1_5b-3.5e-4-contacts-v1-unmasked-3b5cf2/`:
levanter `checkpoints/step-{2000,4000,6000,8000,10000,11999}` + auto HF exports
`hf/step-{…,11999}` (safetensors + tokenizer co-located). Final loadable model:
`hf/step-11999/`. Follow-up: downstream contact-recapitulation eval vs the prior
contacts-and-distances-v1 1.5B.

## Detailed plan

_(Why we ran this, what we expect to see, unusual parameters.)_

## Changes from previous runs

_(Bullet list of differences from the last run of this kind.)_

## Notes

- First MarinFold run to do a *fresh* tokenize→train on `marin-latest`
  (`0.99.dev20260529`); surfaced 7 distinct infra issues, all documented in the
  experiment README "Implementation notes".
- The launched run temporarily worked around marin bug #6008 at runtime.
  The repository does not retain that monkey-patch; future launches require a
  Marin build containing the upstream #6014 fix.
- Launched from branch `exp/67-contacts-v1-1_5b` (PR #70); also carries a
  repo-wide fix to `models/marinfold_models/defaults.py` (marin import drift).
- Healthy start: loss 8.29→6.21 over the first ~9 steps, MFU ~40%.
