---
marinfold_experiment:
  issue: 199
  title: 'exp: optimize 1.5B contacts-v1 AFDB + ESM models'
  kind: models
  branch: exp/199-optimize-contacts-v1
---

# exp: optimize 1.5B contacts-v1 AFDB + ESM models

**Issue:** [#199](https://github.com/Open-Athena/MarinFold/issues/199) · **Kind:** `models` · **Branch:** `exp/199-optimize-contacts-v1`

## Question

Which optimization settings improve the 1.5B no-crops contacts-v1 model trained on the AFDB and ESM-Atlas corpora?

## Hypothesis

The data mixture from #196 should benefit from a focused optimization sweep around the exp117 recipe inherited through #155. Before spending protein-training compute, a ten-step DCLM nano run should validate that MarinFold's package-only Marin environment can dispatch training through Iris on the requested TPU and region.

## Background

Issue #199 follows [#196](https://github.com/Open-Athena/MarinFold/issues/196), which removes crops and trains on contacts-v1 from AFDB plus ESM-Atlas, and [#155](https://github.com/Open-Athena/MarinFold/issues/155), which defines the 1.5B exp117-derived recipe. The first script here is an infrastructure smoke test, not an arm of the protein optimization sweep.

## Approach

- `train_dclm_nano.py` trains the tutorial-sized Llama model for ten steps on one `v6e-4` slice.
- The input is the completed Llama-3-tokenized DCLM cache at `gs://marin-eu-west4/tokenized/dclm_baseline-0206f1/`.
- The script uses `ArtifactStep.adopt`; it neither imports nor constructs a tokenization recipe. A missing cache therefore fails instead of rebuilding.
- The smoke test reads the cache as a continuous token stream; eager document packing would index all 2.9 billion rows before step 0.
- The outer Iris job is pinned to `europe-west4`, and `MARIN_PREFIX` points to the co-located `marin-eu-west4` bucket.
- Marin is installed from pinned published packages in this experiment's isolated `uv.lock`; the sibling Marin checkout is reference material only.

## Success criteria

1. The preflight confirms the regional DCLM cache is complete and was built with `meta-llama/Meta-Llama-3.1-8B`.
2. Iris schedules the training child on one `v6e-4` in `europe-west4` and the run reaches a training step.
3. No tokenize or cache-build job is present in the lowered plan or Iris job tree.
4. Checkpoints and W&B metadata are written under the exp199 regional prefix and the `open-athena/MarinFold` project.

## Runbook

Run from this experiment directory. The outer job is a small coordinator; the
training artifact requests the `v6e-4` child itself.

```bash
set -a
source /home/exedev/marin.env
set +a
export MARIN_PREFIX=gs://marin-eu-west4/protein-structure/MarinFold/exp199_models_optimize_contacts_v1_afdb_esm

uv run --extra tpu --frozen python verify_dclm_cache.py
uv run --extra tpu --frozen python train_dclm_nano.py --version 2026.08.06

uv run --extra tpu --frozen iris --cluster=marin job run \
  --user eczech \
  --job-name exp199-dclm-nano-v6e4-smoke-streaming \
  --no-wait \
  --priority interactive \
  --enable-extra-resources \
  --cpu 1 \
  --memory 16GB \
  --disk 16GB \
  --extra tpu \
  --region europe-west4 \
  -e MARIN_PREFIX "$MARIN_PREFIX" \
  -e HF_TOKEN "$HF_TOKEN" \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  -e WANDB_ENTITY "$WANDB_ENTITY" \
  -e WANDB_PROJECT "$WANDB_PROJECT" \
  -- python train_dclm_nano.py --version 2026.08.06 --run
```

## Results

- Preflight passed against `gs://marin-eu-west4/tokenized/dclm_baseline-0206f1/`: the ledger is complete, contains 1,024 shards and 2,918,356,905 rows, and identifies `meta-llama/Meta-Llama-3.1-8B` as the tokenizer.
- The lowered graph contained exactly the adopted DCLM artifact and the training artifact. It contained no tokenize or cache-build step.
- The initial attempt exposed a stale hard-coded W&B destination. A second attempt using `marin.env` authenticated correctly, but `pack=True` made this Levanter version eagerly index all 2.9 billion document offsets; it was stopped after a live stack trace confirmed the bottleneck.
- The continuous-stream run [`/eczech/exp199-dclm-nano-v6e4-smoke-streaming`](https://iris.oa.dev/#/job/%2Feczech%2Fexp199-dclm-nano-v6e4-smoke-streaming) and child `run_levanter_train_lm-60342421` both succeeded with no failures or preemptions.
- Runtime logged `device=TpuConfig(variant='v6e-4', kind='tpu')`, and [W&B](https://wandb.ai/eric-czech/marin/runs/exp199-dclm-nano-v6e4-smoke-streaming) recorded one host with four TPU v6 lite devices.
- Runtime logged `Overriding auto_build_caches to False`, then loaded `dclm_baseline-0206f1/train/shard_ledger.json` directly. The Iris job tree contained no tokenization child.
- The first batch loaded in 3.6 seconds, step 0 completed after 14.9 seconds including compilation, and all ten steps completed through global step 9.
- The final logged loss was 11.9435 over 163,840 tokens. Mean throughput was 579,847 tokens/s with 39.8% mean MFU.
- Step-9 Levanter and HF-compatible checkpoints were written under `gs://marin-eu-west4/protein-structure/MarinFold/exp199_models_optimize_contacts_v1_afdb_esm/checkpoints/exp199-dclm-nano-v6e4-smoke-streaming/2026.08.06/`.

## Conclusion

The MarinFold package-only path successfully trained a model through Iris on the requested regional `v6e-4`. The adopted-cache design enforced the no-retokenization invariant at planning and runtime, and continuous token streaming avoided an impractical full-corpus document-packing index. The infrastructure smoke milestone is complete; the protein optimization sweep can build on this launch pattern.
