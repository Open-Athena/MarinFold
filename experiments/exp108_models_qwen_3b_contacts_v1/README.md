---
marinfold_experiment:
  issue: 108
  title: "exp: train a 3B qwen model contacts-v1 on coreweave GPUs"
  kind: models
  branch: claude/musing-euclid-fd5ea5
---

# exp: train a 3B qwen model contacts-v1 on coreweave GPUs

**Issue:** [#108](https://github.com/Open-Athena/MarinFold/issues/108) · **Kind:** `models` · **Branch:** `claude/musing-euclid-fd5ea5`

## Question

Can we train efficiently on the new coreweave GPU cluster? Do we do better than the best model Eric trained in #75 (eval loss 2.7) if we epoch for 16 epochs instead of 8 and switch to a 3B instead of 1.5B qwen model?

## Hypothesis

N/A

## Background

We have limited use of a large GPU cluster this week

## Approach

Get the data onto coreweave's S3 equivalent. Make sure all objects we create have an initial prefix (path component) of "MarinFold" so that they can be removed later.

Launch a model training sweep using a few variations focusing on ~3B parameter models and similar values for other parameters to what Eric tried in #75 

IMPORTANT: Always use iris "batch" priority for all of this.

## Success criteria

We trained a model to < 2.7 eval loss

## Files

| File | Purpose |
|------|---------|
| `train_qwen_3b_contacts_v1_sweep.py` | Sweep entry point: Qwen3-3B config + the recipe; submits one batch-priority job per LR. |
| `dispatch_train.py` | Direct batch-priority Fray dispatch (`build_on_pod_config` + `dispatch_training_run`) — the reason we bypass the executor. |
| `contacts_v1_train_common.py` | Shared constants (S3 prefix, tokenizer, GPU `ResourceConfig`) + HF-export helper. |
| `export_qwen_3b_contacts_v1.py` | HF export of one chosen sweep run (pick with `EXP108_EXPORT_LR`). |
| `stage_data_to_coreweave.py` | Copy the contacts-v1 parquet corpus into CoreWeave S3 under `MarinFold/`. |
| `pyproject.toml` | Fixed marin pins + `gpu`/`cpu` extras. **Set `marinfold-models` `rev` to the launch commit before submitting.** |

## Design (scale-up of #75)

Keep [#75](https://github.com/Open-Athena/MarinFold/issues/75)'s tuned recipe; change only **1.5B → ~3B** and **8 → 16 epochs**.

- **Model** — Qwen3 (levanter `Qwen3Config`) at **#75's exact 1.5B width** (hidden 2048, intermediate 8192, 32 heads / 8 KV, head_dim 64, `Llama3RotaryEmbeddingsConfig`, seq 8192) with **layers doubled 24 → 48** → ~2.9B params. Depth-only scaling keeps #75's width so its tuned LR/wd transfer directly. Vocab (~2845) from the contacts-v1 tokenizer at model-init.
  - *Alternative if you'd rather use an off-the-shelf 3B*: marin's `experiments/qwen3.py::qwen2_5_3b` (Qwen2.5-3B: 2048h / 36L / 11008ff / 16h / 2kv). Changes heads/kv/ff vs #75, so #75's LR transfers less cleanly — hence not the default.
- **Recipe (tracks #75)** — unmasked next-token loss, Feistel shuffle (`data_seed=0`), packed, full held-out-val eval. AdamW, cosine, **10% warmup**, **wd 0.2**. Batch 128 × seq 8192 → ~4,457 steps/epoch → **16 epochs ≈ 71,312 steps**.
- **Sweep** — peak **LR ∈ {5e-4, 1e-3, 2e-3}** at wd 0.2 (bracketing #75's winning 1e-3). Three separate single-node GPU jobs. Override via `EXP108_LRS`.
- **Device** — default **single 8×H100 node** per run (`replicas=1`), FSDP over the 8 GPUs.

**Benchmark:** #75's best was `prot-exp75-cv1-1_5b-e8-lr1e-3-wd0p2` at **eval loss 2.7566** (step-35679, 8 epochs). Target: **< 2.7**.

## Batch priority (#108 requirement — implemented via direct dispatch)

#108 requires **iris `batch` priority for all work**. The wrinkle: `--priority batch` on `iris job run` sets only the *driver* job's band, and the marin executor submits the actual training gangs via `fray.JobRequest` **without** a band (→ interactive). There's no executor/env knob for it.

**Solution (chosen):** `dispatch_train.py` bypasses the executor for training and submits each gang itself as a `fray.JobRequest(priority=3)` — grug-style direct dispatch (`PRIORITY_BAND_BATCH == 3`; fray maps `JobRequest.priority` int straight to the iris band). It still reuses marin's `run_levanter_train_lm` entrypoint and `_build_train_lm_config` (AdamW / TrainerConfig / mesh / checkpointer), so we keep the proven training internals; only the *submission* and the output-path/tokenize-cache resolution are ours. The data is tokenized **on the fly** from raw S3 parquet into a concrete S3 cache (`auto_build_caches=True`), so there's no separate executor tokenize step to worry about priority for.

Still pass `--priority batch` on the driver (the launch commands do). **HF export** (`export_qwen_3b_contacts_v1.py`) is a one-off CPU job that still uses the executor — run its driver with `--priority batch`; it's a single small post-hoc job, not GPU cluster time.

**Verify on the smoke run:** `uv run iris --cluster=cw-rno2a job status <training-job-id>` should report the **batch** band; `dispatch_train.py` also asserts at import that the resolved fray build exposes `JobRequest.priority` (the frozen `0.99.dev` build lacks it).

## Runbook

**0. Auth (done on Tim's workstation):** kubeconfig `~/.kube/coreweave-iris-rno2a` (verified), object-storage key in `~/.config/marin/cw-rno2a.env`. Source it first: `set -a; source ~/.config/marin/cw-rno2a.env; set +a`.

**1. Stage the corpus to CoreWeave S3** (workstation; GCS mirror → S3, byte-identical to the HF publish):
```bash
cd experiments/exp108_models_qwen_3b_contacts_v1
set -a; source ~/.config/marin/cw-rno2a.env; set +a
python stage_data_to_coreweave.py --splits train val     # ~12 GiB, idempotent
```
Lands at `s3://marin-us-east-02a/MarinFold/data/document_structures/contacts_v1/{train,val}/`.

**2. Pin `marinfold-models`** — commit this experiment + the `models/` `gpu`-extra change, push, and set `rev` in `pyproject.toml` to that commit. Local iteration: `uv sync --extra gpu && uv pip install --reinstall-package marinfold-models ../../models`.

**3. Smoke run** (one LR, ~50 steps — confirm batch fits + measure step time):
```bash
WANDB_API_KEY=<key> uv run iris --cluster=cw-rno2a job run --no-wait --priority batch \
    --enable-extra-resources --cpu=2 --memory=6GB --disk=16GB --extra gpu \
    -e WANDB_API_KEY <key> -e EXP108_LRS 1e-3 -e EXP108_MAX_STEPS 50 \
    -- python -m train_qwen_3b_contacts_v1_sweep
```
Follow: `uv run iris --cluster=cw-rno2a job logs <job-id> -f`.

**4. Full sweep** — same command without the `EXP108_MAX_STEPS`/`EXP108_LRS` smoke overrides (launches all 3 LRs). If the batch OOMs, drop it (`-e EXP108_TRAIN_BATCH 64`) or scale out (`-e EXP108_REPLICAS 2`).

**5. Export** the winner — set `EXP108_EXPORT_LR`, fill the `-<runid>` suffix in `export_qwen_3b_contacts_v1.py`, run with `--extra cpu --priority batch`.

## Implementation notes & open questions

- **First GPU/CoreWeave run in MarinFold, via direct dispatch.** We reuse marin's `run_levanter_train_lm` (device-agnostic; marin's `train_tiny_model` uses `with_gpu("H100", count=8)`) but submit it ourselves. **Not yet verified live — the smoke run exists to shake these out:**
  - the `TrainLmOnPodConfig` object graph cloudpickles across the Fray boundary (same graph the executor ships, but we now build it);
  - `auto_build_caches=True` tokenizes the S3 parquet on the workers into `<cache_dir>/{train,validation}` (the val component has empty `train_urls` by design — weight 0);
  - the submitted job reports the **batch** band;
  - checkpoints land under the explicit `output_path` (confirm the exact `step-{N}` / any run-id subdir from the S3 listing — the export script needs it);
  - **multi-node** (`replicas>1`) — start single-node; multi-host GPU gangs need coscheduling not exercised here.
- **Storage is S3, not GCS.** All artifacts under `s3://marin-us-east-02a/MarinFold/` (removable later, per #108). In-cluster, iris injects CoreWeave AWS creds + endpoint; the `check_gcs_paths_same_region` guard is a no-op (no GCS paths).
- **Data source.** The canonical HF publish is an HF *bucket* (`hf://buckets/open-athena/MarinFold/…`), not anon-listable via the normal API. The staging script defaults to the **byte-identical GCS mirror** exp53 wrote; `--source hf` needs bucket auth.
- **Overfitting.** 16 epochs = 16× repetition of a 4.7B-token corpus. Watch the train/val gap; wd is already 0.2 (#75's value). If val loss diverges, revisit epochs / regularization.

## Results

_(Fill in after the run completes.)_

## Conclusion

_(Fill in after results are in.)_
