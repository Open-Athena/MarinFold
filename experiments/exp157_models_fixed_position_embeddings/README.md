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

## Conclusion

_(Fill in after results are in.)_
