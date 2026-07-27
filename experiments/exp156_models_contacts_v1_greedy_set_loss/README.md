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

This experiment adds a contacts-v1-specific greedy set loss and a custom Levanter training path. The same launcher can run two arms:

- `EXP156_LOSS_KIND=next-token`: stock Levanter autoregressive CE baseline.
- `EXP156_LOSS_KIND=greedy-set`: custom `Trainer` loop that parses the contacts-v1 target block and greedily matches each prediction slot against any remaining true contact pair, including either pair orientation.

The greedy arm now trains inside the jitted Levanter loop; it is no longer just a host-side smoke prototype. It also installs validation hooks for:

- `eval/<val-name>/greedy_set/loss`
- `eval/<val-name>/next_token/loss`, unless `EXP156_ENABLE_GREEDY_NEXT_TOKEN_EVAL=0`

The next-token validation hook is useful on single-device runs, but currently disabled for greedy multi-GPU runs because Levanter's fused next-token CE path can hit an axis-size mismatch when evaluating a greedy-trained model on an 8-way local mesh.

For CoreWeave/Iris runs, defaults are:

- output prefix: `s3://marin-us-east-02a/marin/protein-structure/MarinFold/exp156_contacts_v1_greedy_set_loss/`
- warm start: `contacts-v1-exp120-1.5B` HF export, staged locally on the worker before Levanter init
- data globs: `s3://marin-us-east-02a/MarinFold/data/document_structures/contacts_v1/{train,val}/*.parquet`
- optimizer LR: `EXP156_LEARNING_RATE` if set, otherwise `3.1623e-3`
- optional durable GPU telemetry: `EXP156_ENABLE_GPU_TELEMETRY=1`

Example smoke launch:

```bash
EXP156_ACCELERATOR=gpu \
EXP156_BASE_PREFIX=s3://marin-us-east-02a/marin/protein-structure/MarinFold \
EXP156_TRAIN_GLOB='s3://.../contacts-v1/train/*.parquet' \
EXP156_VAL_GLOB='s3://.../contacts-v1/val/*.parquet' \
EXP156_INITIALIZE_FROM_HF=contacts-v1-exp120-1.5B \
EXP156_LOSS_KIND=greedy-set \
EXP156_STEPS=75 \
EXP156_STEPS_PER_EVAL=50 \
EXP156_MAX_EVAL_BATCHES=16 \
EXP156_TRAIN_BATCH_SIZE=16 \
python experiments/exp156_models_contacts_v1_greedy_set_loss/train.py --run
```

## Results so far

### H100 single-GPU short runs

A 75-step greedy-set rerun with both validation hooks succeeded:

- step 50: greedy-set val loss `5.499`; next-token CE val loss `5.545`
- final step 74: greedy-set val loss `5.393`; next-token CE val loss `5.419`

Earlier 300-step stock next-token H100 run reached final next-token eval loss `4.19192`, so the short greedy run is not yet a fair same-step comparison.

### GB200 4-GPU runs

The stock next-token 4×GB200 run completed 2000 steps and exported `step-1999`:

- final next-token eval loss: `5.516`

The first greedy 4×GB200 run failed during its auxiliary next-token validation hook with a fused CE axis mismatch. The rerun disabled that auxiliary hook and completed 2000 steps:

- final greedy-set validation loss: `5.674`

Both high-LR 4×GB200 runs showed train loss collapsing near zero while validation losses became noisy/worse, consistent with overfitting or an overly aggressive learning rate on this smoke-sized cache. Follow-up 8×H100 runs are using a lower LR (`3e-4`).

## Current caveats

- Multi-GPU greedy runs currently report greedy-set validation only; next-token CE validation is disabled with `EXP156_ENABLE_GREEDY_NEXT_TOKEN_EVAL=0` until the CE eval path avoids the fused-kernel axis mismatch.
- The runs above used small cached datasets and short schedules; they are signal-gathering runs, not final model-quality comparisons.
- W&B is intentionally offline unless credentials are available, so logs and telemetry should be read from Iris/S3 artifacts.

## Success criteria

- Custom loss target-selection logic is unit tested on toy contact sets.
- Custom training entrypoint can lower/build a Marin/Iris training job.
- Greedy and next-token arms both produce finite train losses and validation metrics.
- Durable GPU telemetry is written for CoreWeave GPU runs.

## Conclusion

The greedy-set loss is trainable inside Levanter and can be evaluated directly with a greedy-set validation loss. Early results do not show a clear quality win yet; the next useful comparison is a same-hardware, lower-LR, longer 8×H100 run against the stock next-token baseline.
