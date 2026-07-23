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

- **Model** — Qwen3 (levanter `Qwen3Config`) **width-scaled from #75** (keep #75's 24 layers, widen hidden 2048→2816, ff 8192→11264, 44 heads / 11 KV, head_dim 64, GQA 4:1, `Llama3RotaryEmbeddingsConfig`, seq 8192) → **~2.78B params**, with **QK-norm on**. Vocab (~2845) from the contacts-v1 tokenizer at model-init.
  - *Why width, not depth:* the first attempt (`v2` runs) depth-doubled #75 (24→48 layers) on the theory that keeping the width lets #75's tuned LR transfer. **It diverged at all three LRs.** A config diff vs #75's stable run showed the *only* difference was `num_layers` (48 vs 24) — LR transfers under **width** scaling, not depth, so 48 layers lowered the stable-LR ceiling. Width-scaling (this version) preserves #75's LR/stability; QK-norm (off in both #75 and v2) is added for extra deep-attention stability. See #108.
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

**0. Workstation prereqs** (one-time, done on Tim's box):
- kubeconfig `~/.kube/coreweave-iris-rno2a` (context `marin-rn02a_RNO2A`).
- object-storage key in `~/.config/marin/cw-rno2a.env`; source it before any command: `set -a; source ~/.config/marin/cw-rno2a.env; set +a`.
- **`kubectl` on `PATH`** — the launcher shells out to it for the controller tunnel. Install a version matching the cluster (k8s v1.36.x). *Without it: `Could not connect to controller: … 'kubectl'`.*
- `marin-iris[controller]` (already in `pyproject.toml`) — needed for `CloudK8sService`. *Without it: `Install iris[controller] to use CloudK8sService`.*
- W&B key: pulled from `~/.netrc` (`python -c "import netrc; print(netrc.netrc().authenticators('api.wandb.ai')[2])"`).

**1. Stage the corpus to CoreWeave S3** (done — GCS mirror → S3, byte-identical to the HF publish; ~12 GiB, idempotent):
```bash
python stage_data_to_coreweave.py --splits train val
```
Lands at `s3://marin-us-east-02a/MarinFold/data/document_structures/contacts_v1/{train,val}/`.

**2. Pin `marinfold-models`** (done) — `rev` in `pyproject.toml` points at the `models/` refresh commit; `uv.lock` freezes marin 0.2.38. Local iteration: `uv sync && uv pip install --reinstall-package marinfold-models ../../models`.

**3. Launch — one SEPARATE, STAGGERED driver per LR** (see "Operational findings" for *why* not one driver):
```bash
set -a; source ~/.config/marin/cw-rno2a.env; set +a
WK=$(python -c "import netrc; print(netrc.netrc().authenticators('api.wandb.ai')[2])")
for lr in 5e-4 1e-3 2e-3; do
  uv run iris --cluster=cw-rno2a job run --no-wait --priority batch \
    --enable-extra-resources --cpu=2 --memory=6GB --disk=16GB --extra gpu \
    -e WANDB_API_KEY "$WK" -e EXP108_LRS "$lr" -e EXP108_REPLICAS 4 \
    -e EXP108_ATTN jax_flash -e EXP108_RUN_SUFFIX v2 --job-name exp108-v2-lr$lr \
    -- python -m train_qwen_3b_contacts_v1_sweep
  sleep 90   # stagger the bootstraps
done
```
Smoke first with `-e EXP108_MAX_STEPS 50` (single LR). Monitor: `uv run iris --cluster=cw-rno2a job list | grep exp108`; `… job logs <job> --max-lines N` (filter `zephyr|aiobotocore|cuda_vmm` noise; the gang's own line is `…wd0p2-<suffix> <state>` with a SPACE).

**4. Export** the winner — WIP (`export_qwen_3b_contacts_v1.py` is a stub; marin's HF-export API moved in 0.2.38 and needs porting; not needed until a checkpoint exists).

## Operational findings (validated live, 2026-07-07)

Everything the smoke run was meant to prove **works**: batch-priority dispatch, `TrainLmOnPodConfig` cloudpickle across Fray, `auto_build_caches` S3 tokenize (→ `<cache_dir>/{train,validation}`, 2067 shards), GPU training, held-out eval, S3 checkpoint. Plus:

- **Driver must WAIT on its gangs.** Gangs are *children* of the driver job; if the driver exits (`wait=False`), iris finalizes (kills) them. The sweep submits all, then blocks.
- **Node-count ceiling ≈ 4.** 1/2/4-node gangs bootstrap and train fine; **8-node fails** — the JAX multi-host coordination bootstrap aborts (~5 min in, `CoordinationServiceAgent::SetError`), reproduced on a single isolated 8-node gang. Likely fix (untried): the `NCCL_*` env grug forwards for >4-node bootstrap (we forward `XLA_FLAGS`/`NCCL_`/`JAX_` but set none). Chasing this unlocks 8+ nodes (~2-day runs).
- **Launch runs as SEPARATE staggered drivers.** 3 gangs from one driver fail fast (coscheduling/coordination collision); one gang per driver, ~90 s apart, matches the working single-gang case.
- **`CUDA_ERROR_NOT_PERMITTED` fabric warnings are benign** — the container lacks NVIDIA IMEX for fabric memory; XLA falls back. (Also a suspect for the ~15% MFU via non-NVLS collectives.)

## Throughput / scaling

- **~20 s/step at batch 128 × seq 8192 on one 8×H100 node (~52k tok/s, ~15% MFU).** Scales **near-linearly**: ~10 s/step (2 nodes), ~5 s/step (4 nodes) → full 16-epoch run ≈ **~4 days at 4 nodes**.
- **`gradient_checkpointing` is mandatory** — disabling it OOMs hard (692 GiB activations for 48 layers × seq 8192, even at 8 seq/GPU). The ~30% recompute tax is unavoidable for this shape.
- **Attention backend is irrelevant** here (only ~15% of FLOPs); `jax_flash` == the NVTE→reference fallback (Transformer Engine isn't in the `--extra gpu` env). Low MFU is systemic (memory-bound recompute + suboptimal FSDP collectives — params are gathered in f32; `p=bf16` would halve that).
- Node count is the wall-clock knob (`EXP108_REPLICAS`); the effective batch stays 128 seq regardless (data-parallel), so the LRs/steps are unchanged.

## Results

- **Smoke run** (`exp108-smoke3`, 50 steps, 2026-07-07): SUCCEEDED — full path validated end-to-end (see Operational findings).
- **`v2` sweep — DEPTH-scaled (24→48 layers), 3 LRs @ 4 nodes, 2026-07-07: FAILED (instability).** All three LRs went unstable within ~1.5 epochs (val bouncing 3.0–3.9, spiking; never approached the target). Diff vs #75 showed the *only* config difference was `num_layers` (48 vs 24) → depth lowered the stable-LR ceiling. Killed and superseded.
- **`v3` sweep — WIDTH-scaled (~2.78B, 24 layers, QK-norm on), 3 LRs @ 4 nodes, 2026-07-07:** relaunched with the fix (see Design). Runs `plm-exp108-cv1-3b-e16-lr{5e-4,1e-3,2e-3}-wd0p2-v3` on W&B `open-athena/MarinFold`; checkpoints under `s3://marin-us-east-02a/MarinFold/exp108_qwen_3b_contacts_v1/checkpoints/<run>/`. **Target: beat #75's 2.7566.**
- _(Fill in final val losses + comparison to #75 when the runs complete.)_

## Conclusion

_(Fill in after results are in.)_
