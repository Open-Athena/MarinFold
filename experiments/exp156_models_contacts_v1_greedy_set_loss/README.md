---
marinfold_experiment:
  issue: 156
  title: 'exp: train contacts-v1 with greedy latent contact-set loss'
  kind: models
  branch: zack/greedy-contact-set-loss
---

# exp: train contacts-v1 with greedy latent contact-set loss

**Issue:** [#156](https://github.com/Open-Athena/MarinFold/issues/156) · **Kind:** `models` · **Branch:** `zack/greedy-contact-set-loss`

## Question

Can a contacts-v1 model train more data-efficiently if the loss treats contact-list ordering and pair orientation as latent, letting the model emit any remaining contact pair at each slot?

## Hypothesis

A greedy hard/Viterbi contact-set loss will let the model learn easy contacts first and use them as context for harder contacts, reducing wasted loss on arbitrary serialization order.

## Approach

Prototype a contacts-v1-specific greedy set loss and a custom Levanter training script that swaps the standard next-token CE for the new objective. Start with a smoke-sized run before any full-scale training.

For CoreWeave/Iris smoke runs, `train.py` defaults to:

- output prefix: `s3://marin-us-east-02a/marin/protein-structure/MarinFold/exp156_contacts_v1_greedy_set_loss/`
- resources: `ResourceConfig.with_gpu("H100", count=1, cpu=16, ram="128g", disk="200g")`
- warm start: `contacts-v1-exp120-1.5B` HF export, staged locally on the worker before Levanter init
- data globs: `s3://marin-us-east-02a/MarinFold/data/document_structures/contacts_v1/{train,val}/*.parquet`

The CoreWeave S3 mirror was verified with a temporary pod in namespace `iris`: train has 2067 shards, val has 22 shards. If using a different mirror, launch with explicit overrides, e.g.:

```bash
EXP156_ACCELERATOR=gpu \
EXP156_BASE_PREFIX=s3://marin-us-east-02a/marin/protein-structure/MarinFold \
EXP156_TRAIN_GLOB='s3://.../contacts-v1/train/*.parquet' \
EXP156_VAL_GLOB='s3://.../contacts-v1/val/*.parquet' \
EXP156_INITIALIZE_FROM_HF=contacts-v1-exp120-1.5B \
EXP156_STEPS=1 \
EXP156_TRAIN_BATCH_SIZE=1 \
python experiments/exp156_models_contacts_v1_greedy_set_loss/train.py --run
```

Loss selection:

- `EXP156_LOSS_KIND=next-token` runs stock Levanter autoregressive CE with the same data/resources/warm-start. This is the baseline arm.
- `EXP156_LOSS_KIND=greedy-set` is the target experimental arm, but the full JAX/Haliax `Trainer` loss is still being wired. The smoke/prototype loss works on real logits; the remaining work is to make the hard assignment and selected-token gather run inside Levanter's jitted training step.

Current caveat: the baseline config can launch; the greedy training arm is intentionally blocked until the JAX/Haliax loss replaces the current host-side prototype.

## Success criteria

- Custom loss target-selection logic is unit tested on toy contact sets.
- Custom training entrypoint can lower/build a Marin/Iris training job.
- A tiny smoke run starts and reports finite loss.

## Results

_(Fill in after the run completes.)_

## Conclusion

_(Fill in after results are in.)_
