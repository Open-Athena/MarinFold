---
marinfold_experiment:
  issue: 157
  title: 'exp: Replace learned residue location tokens with position embedding'
  kind: models
  branch: exp/157-fixed-position-embeddings
---

# exp: Replace learned residue location tokens with position embedding

**Issue:** [#157](https://github.com/Open-Athena/MarinFold/issues/157) · **Kind:** `models` · **Branch:** `exp/157-fixed-position-embeddings`

## Question

Should we use a positional embedding like RoPe (rather than learned embeddings) for residue position tokens?

## Hypothesis

Intuitively, absolute positions of residues is not as meaningful as relative positions (distances). If we use an embedding designed to make it easy to compute distance, rather than learned per-position embeddings, it could improve model efficiency for two reasons: (1) fewer parameters to learn, and (2) possibly a more flexible and re-usable embedding

## Approach

First implementation pass:

1. Treat contacts-v1 residue location tokens (`<p0>` ... `<p1999>`) as a contiguous fixed span in the tokenizer vocabulary; `residue_position_spec_from_tokenizer(...)` verifies that span before building the model config.
2. Remove that span from the trainable **input** embedding table, leaving a compact table for all non-position tokens.
3. At embed time, map non-position token ids into the compact table and synthesize position-token vectors from `token_id - p0_id` using a fixed RoPE/sinusoidal feature map.
4. Keep the LM head untied and full-vocabulary-sized so the model can still emit position tokens normally.

The first prototype lives in [`fixed_position_model.py`](fixed_position_model.py) as a `FixedResiduePositionLlamaConfig` / `FixedResiduePositionLlamaLMHeadModel` pair. The smoke test in [`test_fixed_position_model.py`](test_fixed_position_model.py) builds a tiny model, runs a few optimizer steps, and checks that learned parameters move while the fixed position feature map is unchanged and has no trainable rows.

Follow-up variant after the fixed-only run lagged the learned control:

- `EXP157_POSITION_MODE=rope_delta` keeps the full input embedding table shape from the learned baseline.
- Position-token rows are initialized to zero and used as a learned residual: `embedding(<pN>) = fixed_rope(N) + learned_delta[<pN>]`.
- This preserves a strong RoPE prior without increasing dimensionality or removing per-position trainable capacity.

## Success criteria

Initial setup is successful when:

- a tiny fixed-position model can run next-token loss and optimizer updates;
- non-position input embeddings and the LM head update under gradient descent;
- residue-position input vectors are deterministic before/after training steps;
- the fixed span is absent from the trainable input embedding table.

The later experiment-level criterion is whether this improves training efficiency and/or downstream contact prediction at matched token budget.

## Results

Setup in progress. Static syntax check passes:

```bash
python3 -m py_compile \
  experiments/exp157_models_fixed_position_embeddings/fixed_position_model.py \
  experiments/exp157_models_fixed_position_embeddings/test_fixed_position_model.py
```

Full `uv run pytest -q` was not runnable on this macOS x86_64 workspace because current JAX wheels required by `marin-levanter` are not published for this platform; the experiment `pyproject.toml` resolves for Linux workers.

CoreWeave/Iris smoke submission is wrapped by [`submit_smoke_coreweave.sh`](submit_smoke_coreweave.sh):

```bash
cd experiments/exp157_models_fixed_position_embeddings
./submit_smoke_coreweave.sh
```

Initial direct-CoreWeave submission from this workstation reached Iris/Kubernetes client setup, but the local CoreWeave kubeconfig returned HTTP 403. The wrapper now uses the standard path: submit through the main Iris controller with `--target-cluster cw-rno2a`. The first federated job (`/zack/exp157-fixed-position-smoke`) reached the CoreWeave worker but failed before tests because `pytest` was only in the local dev group; `pytest` is now a runtime dependency for the smoke job. The second job (`/zack/exp157-fixed-position-smoke-r2`) ran tests and exposed a test bug: the loss function returned a rank-0 `NamedArray` rather than its scalar `.array` to `eqx.filter_value_and_grad`.

The third federated smoke succeeded on CoreWeave:

```text
Job: /zack/exp157-fixed-position-smoke-r3
Dashboard: https://iris.oa.dev/#/job/%2Fzack%2Fexp157-fixed-position-smoke-r3
Result: 2 passed, 1 warning in 17.22s
```

After adding the learned residual variant, the CoreWeave unit smoke succeeded:

```text
Job: /zack/exp157-rope-delta-position-smoke-r1
Result: 4 passed, 1 warning in 31.49s
```

### Regular training smoke

[`train_fixed_position_smoke.py`](train_fixed_position_smoke.py) is the regular
next-token-loss training smoke, using the contacts-v1 tokenizer/corpus and the
control-style recipe (`lr=3.162e-3`, `wd=0.2`, cosine, 10% warmup). It dispatches
a batch-priority child job through Fray/Iris. By default it runs the 1.5B shape
on one 8xH100 node; for queue-friendly integration tests it also supports
`EXP157_MODEL_SIZE=tiny EXP157_GPU_COUNT=1`.

Submission wrapper:

```bash
cd experiments/exp157_models_fixed_position_embeddings
./submit_training_smoke_coreweave.sh
```

Status:

- r4 created W&B but failed before training while trying to build a fresh Zephyr
  token cache; the script now reuses the existing exp108 contacts-v1 cache.
- r5 launched the 1.5B 8xH100 smoke, created W&B, then was preempted before a
  train step; replacement task is waiting in the CoreWeave/Kueue scheduling gate.
- r7 launched a tiny 1xH100 regular-training fallback smoke and is also waiting
  in the CoreWeave/Kueue scheduling gate.
- `rope_delta` tiny Qwen3 training smoke r2 succeeded on `cw-us-east-02a`, wrote
  checkpoint/HF export at step 19, and logged `eval/contacts-v1-val/loss = 6.0809`.
- `rope_delta` control-matched Qwen3 1.5B run r1 launched on `cw-us-east-02a`:
  W&B `exp157-cv1-1_5b-e16-lr3em3-wd0p2-bs128-qwen3-rope_delta-position-controlmatch-r1-east02-h100x8`.
  It was cancelled after the GB200 switch request, before the first validation point.
- GB200 support now comes from the same runner via `EXP157_GPU_VARIANT=GB200`,
  `EXP157_GPU_COUNT=4`, and `EXP157_GPU_REPLICAS=8`. A one-node GB200 smoke
  completed successfully at global batch 128 with `throughput/tokens_per_second = 109,552`
  and `throughput/examples_per_second = 13.37`.
- The 8x4 GB200 throughput smoke completed on `cw-us-east-08a`:
  W&B `exp157-cv1-1_5b-e16-lr3em3-wd0p2-bs128-qwen3-rope_delta-position-throughput-bs128-gb200x4n8-r2`.
  It reached step 79 with `throughput/tokens_per_second = 748,219`,
  `throughput/examples_per_second = 91.34`, and `throughput/duration = 1.401s/step`.
- The full 8x4 GB200 control-matched `rope_delta` run completed on
  `cw-us-east-08a`:
  W&B `exp157-cv1-1_5b-e16-lr3em3-wd0p2-bs128-qwen3-rope_delta-position-controlmatch-r2-east08-gb200x4n8`.
  It used Qwen3 1.5B, global batch 128, 71,360 steps, full validation every
  2,230 steps, LR `3.1623e-3`, WD `0.2`, warmup `0.1`, BlockShuffle, and 32
  GB200 GPUs. Final losses at step 71,359: train `3.0170`, eval `3.0654`.
- The matched fixed-position/RoPE-only control completed with final losses at
  step 71,359: train `3.0313`, eval `3.0728`.
- A follow-up `rope_delta` variant adds an explicit zero-centered L2 prior on
  the learned residue-position deltas via `EXP157_POSITION_DELTA_L2_WEIGHT`.
  The prior is added only to reduced training loss; Levanter's unreduced eval
  path remains plain next-token CE, so validation curves stay comparable to the
  unregularized run. The first run used `1e-3` and completed with final losses
  at step 71,359: train `3.1494`, eval `3.1771`.

### FoldBench monomer rollout evaluation

The final checkpoints were scored with the exp245 FoldBench monomer
rollout+resample evaluator (100 rollouts/protein, contact multiplier 6, seed 0)
for the three-way comparison. These are the current decontaminated FoldBench
monomer evaluation sets for natural-protein claims and same-manifest baseline
comparisons:

- `eval-val`: 97 natural FoldBench monomers from the historical FoldBench-100,
  used as the free iteration/selection set.
- `eval-test`: 217 other natural FoldBench monomers, used here as the final
  held-out confirmation read for this experiment.
- `eval-denovo`: 19 designed FoldBench monomers, reported separately as a
  sanity check rather than pooled into a headline natural-protein number.

The true vanilla training-recipe baseline for this experiment is **not** the
fixed/RoPE-only arm. It is Eric's TPU-trained vanilla contacts-v1 Qwen3 1.5B run
with ordinary learned residue-position token embeddings and no fixed RoPE input
features:

```text
contacts-v1-exp117-1.5B
prot-exp117-cv1-s02-1_5b-e16-lr3p162e-3-wd0p2-bs256-europe-west4
```

That run uses the same model family, tokenizer revision, LR (`3.1623e-3`), weight
decay (`0.2`), 16-epoch token budget, shuffled contacts-v1 data recipe, and
plain next-token loss. Its batch/step schedule is the TPU equivalent of exp157's
CoreWeave schedule: exp117 is `35,680 × bs256`; exp157 is `71,360 × bs128`.
Therefore exp117, not #232 or #199, is the matched vanilla baseline for the
question "does replacing learned position-token embeddings with RoPE-style inputs
help?" The exp117 W&B CE (`eval/tokenized/contacts-v1-val/loss = 2.7037`) is
from the older TPU/cache logging path, not a direct rerun on the exp157 cache.
For an apples-to-apples CE check, we ran exp117 eval-only on exp157's exp108
validation cache:

```text
/zack/exp157-exp117-direct-val-loss-exp108-cache-r3
https://wandb.ai/open-athena/MarinFold/runs/exp157-exp117-direct-val-loss-exp108-cache-r3
eval/contacts-v1-val/loss = 3.0815000534057617
```

The combined results table below uses this corrected CE scale.

The r5 FoldBench run below scored the three exp157 position-embedding arms. A
follow-up `exp117-vanilla` r2 run scored the true vanilla baseline on the same
exp245 manifest. The #232 Decontam checkpoints, #199 cooldown, structure
baselines, and KNN rows are same-manifest context only; they are not matched
training-recipe controls for exp157.

| label | W&B run | HF export |
| --- | --- | --- |
| fixed/RoPE-only control | `exp157-cv1-1_5b-e16-lr3em3-wd0p2-bs128-qwen3-fixed-position-controlmatch-e16-r3-east02-h100x8` | `s3://marin-us-east-02a/MarinFold/exp157_fixed_position_embeddings/checkpoints/.../hf/step-71359` |
| learned RoPE delta | `exp157-cv1-1_5b-e16-lr3em3-wd0p2-bs128-qwen3-rope_delta-position-controlmatch-r2-east08-gb200x4n8` | `s3://marin-us-east-02a/MarinFold/exp157_fixed_position_embeddings/checkpoints/.../hf/step-71359` |
| learned RoPE delta + L2 prior | `exp157-cv1-1_5b-e16-lr3em3-wd0p2-bs128-qwen3-rope_delta-position-l21em3-controlmatch-r4-east08-gb200x4n8` | `s3://marin-us-east-02a/MarinFold/exp157_fixed_position_embeddings/checkpoints/.../hf/step-71359` |
| vanilla no-RoPE baseline | `prot-exp117-cv1-s02-1_5b-e16-lr3p162e-3-wd0p2-bs256-europe-west4` | `open-athena/marinfold-exp117:prot-exp117-cv1-s02-1_5b-e16-lr3p162e-3-wd0p2-bs256-europe-west4/hf/step-35679` |

The successful evaluation driver was:

```text
/zack/exp245-exp157-rope-rollout-exp157-rope-final-step71359-r5
```

It produced 33,300/33,300 usable rollouts per checkpoint, with zero unfinished
rollouts. The exp117 follow-up driver was
`/zack/exp245-exp117-vanilla-rollout-exp157-exp117-vanilla-step35679-r2`; it
also produced 33,300/33,300 usable rollouts with zero unfinished rollouts.
Results are under:

```text
s3://marin-us-east-02a/marin/protein-structure/MarinFold/exp245_foldbench_held_out_monomers/evals/rollout/exp157-rope-final-step71359-r5/results/
s3://marin-us-east-02a/marin/protein-structure/MarinFold/exp245_foldbench_held_out_monomers/evals/rollout/exp157-exp117-vanilla-step35679-r2/results/
```

Headline metrics for the controlled exp117/exp157 comparison:

| model | corrected val CE | eval-val AUC | eval-val R | eval-test AUC | eval-test R | eval-denovo AUC | eval-denovo R |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vanilla no-RoPE baseline | 3.0815 | 0.8920 | 0.3947 | 0.8894 | 0.3843 | 0.9585 | 0.5336 |
| fixed/RoPE-only arm | 3.0728 | 0.8805 | 0.3732 | 0.8805 | 0.3625 | 0.9563 | 0.5197 |
| learned RoPE delta | **3.0654** | **0.8983** | **0.4130** | **0.9027** | **0.4103** | 0.9574 | 0.5244 |
| learned RoPE delta + L2 `1e-3` | 3.1771 | 0.7966 | 0.1938 | 0.7891 | 0.1818 | 0.9084 | 0.3681 |

Same-manifest all-range R-precision context. The vanilla no-RoPE row is the
matched training-recipe baseline; the next three rows are exp157
position-embedding arms. The remaining rows are external context only.

| predictor | eval-val | eval-test | eval-denovo |
| --- | ---: | ---: | ---: |
| vanilla no-RoPE learned-token baseline (`contacts-v1-exp117-1.5B`) | 0.395 | 0.384 | 0.534 |
| exp157 learned RoPE delta | 0.413 | 0.410 | 0.524 |
| exp157 fixed/RoPE-only arm | 0.373 | 0.362 | 0.520 |
| exp157 RoPE delta + L2 `1e-3` | 0.194 | 0.182 | 0.368 |
| seq-KNN null, decontaminated corpus | 0.407 | 0.426 | 0.050 |
| #232 m1-p02, decontaminated training | 0.473 | 0.493 | 0.588 |
| #232 m2-p06, decontaminated training | 0.520 | 0.538 | 0.591 |
| #199 cooldown, contaminated training reference | 0.589 | 0.613 | 0.619 |
| Protenix-v2 single-sequence | 0.263 | 0.265 | 0.835 |
| ESMFold | 0.750 | 0.753 | 0.795 |
| ESMFold2 | 0.802 | 0.792 | 0.864 |
| Protenix-v2 + MSA | 0.846 | 0.845 | 0.844 |

The immutable checkpoint manifests are recorded as the `exp157-rope` and
`exp117-vanilla` suites in
`experiments/exp245_evals_foldbench_held_out_monomers/rollout/checkpoint_specs.py`.
These are this experiment's exp245 held-out test-set reads.

## Conclusion

The controlled result is positive on the headline natural-protein metrics.
Learned RoPE deltas improve corrected same-cache validation CE over the true
vanilla no-RoPE baseline (`3.0654` vs `3.0815`) and improve natural-monomer
R-precision over both the fixed/RoPE-only arm and vanilla (+0.018 eval-val,
+0.026 eval-test). On the small designed
`eval-denovo` split, vanilla is slightly ahead (+0.009 R-precision over
RoPE-delta), which should be treated as a sanity-check result rather than a
headline. After noticing loss spikes during RoPE-delta training, we tried adding
a zero-centered L2 prior on the learned deltas; the single `1e-3` strength we
ran caused a major regression in both logged CE and contact metrics, and we did
not experiment further. External same-manifest context should not be read as a
controlled recipe comparison: the RoPE-delta model roughly matches the
decontaminated-corpus
seq-KNN null and beats Protenix-v2 single-sequence on natural monomers, but
remains below the #232 Decontam MarinFold checkpoints, the #199 cooldown
reference, and ESMFold/Protenix+MSA structure baselines, all of which differ in
training data and/or method. If regularizing the learned delta remains
interesting, the next run should sweep much smaller weights rather than reusing
`1e-3`.
