---
marinfold_experiment:
  issue: 112
  title: "exp: train a 3B qwen3 model on coreweave but use nvidia nemo framework for mfu comparison"
  kind: models
  branch: claude/exp112-nemo-mfu
---

# exp: 3B Qwen3 on CoreWeave via **NVIDIA NeMo** — MFU comparison vs Levanter

**Issue:** [#112](https://github.com/Open-Athena/MarinFold/issues/112) · **Kind:** `models` · **Branch:** `claude/exp112-nemo-mfu`

## Question

exp108 trained the 3B Qwen3 contacts-v1 model on the CoreWeave `cw-rno2a` H100
cluster with JAX/**Levanter** and got only **~15% MFU** (~20 s/step, ~52k tok/s,
batch 128 × seq 8192 on one 8×H100 node). Does the **NVIDIA NeMo / Megatron-Core**
stack do materially better on the *same* model / seq / batch / hardware?

## Hypothesis

NeMo/Megatron-Core on H100 typically lands **~30–40% MFU** for ~3–8B dense
models, so we expect a **2–3× throughput win** over exp108's Levanter run.

## Approach

**Benchmark first, then decide** (per issue scope): land a single-node MFU number,
and only invest in a full 16-epoch run if the win justifies it.

- **Same model as exp108** (so the comparison is apples-to-apples): Qwen3 ~2.9B —
  hidden 2048, ffn 8192, 32 heads / 8 KV groups (GQA), head_dim 64, **48 layers**,
  seq 8192; RMSNorm, SwiGLU, QK-layernorm, RoPE. Real contacts-v1 vocab (2845,
  Megatron-padded to 2944). Built as a Megatron-backed `llm.GPTConfig`.
- **Mock/synthetic data for the benchmark.** MFU is compute-bound and
  data-content-independent — this is exactly how NVIDIA publishes its perf tables.
  This removes the entire tokenize→`.bin/.idx` pipeline from the critical path.
  (Real data is a documented follow-up; see *Deferred*.)
- **Single 8×H100 node**, `torchrun --standalone --nproc-per-node=8`. Single-node
  sidesteps the one iris limitation for PyTorch: it exposes no torchrun rendezvous
  / coordinator API (only `IRIS_TASK_ID`/`IRIS_NUM_TASKS`), which JAX has built in.
- **Launch = direct batch-priority Fray dispatch of a foreign container.** iris
  lets a job override its image (`ResourceConfig.image` → `nvcr.io/nvidia/nemo`)
  and run a **binary** `torchrun` entrypoint (`Entrypoint.from_binary`). The NeMo
  container has no repo checkout, so `dispatch_nemo.py` **base64-inlines** the
  training script into the bootstrap bash. Batch band via `JobRequest.priority=3`
  (identical to exp108). See *Design* below.
- **Primary metric = throughput** (tokens/s, s/step) — formula-free and directly
  comparable to exp108. **MFU** (÷ 989 TFLOP/s H100 bf16 peak) is reported as a
  cross-check; both an analytical count *and* NeMo's native number where available.

## Success criteria

A trustworthy single-node **tokens/s + MFU** for NeMo on the exp112 Qwen3-3B, and
a head-to-head vs exp108's ~52k tok/s / ~15% MFU. (No eval-loss target here — that
belongs to the deferred full run.)

## Files

| File | Purpose |
|------|---------|
| `common.py` | Launcher constants: NeMo image tag, S3 prefix, model geometry, 8×H100 `ResourceConfig`, H100 peak. |
| `nemo_train_qwen3.py` | **In-container** (base64-inlined): builds the Qwen3-3B `GPTConfig`, mock data, Trainer; runs under `torchrun`; emits a throughput/MFU JSON summary. Self-contained (no `common.py` import). |
| `dispatch_nemo.py` | Direct **batch-priority** Fray dispatch: NeMo image + `torchrun` binary entrypoint; inlines the train script; best-effort uploads the summary to S3. |
| `pyproject.toml` / `uv.lock` | **Launcher-only** deps (`marin-iris[controller]`, `marin-fray`, boto3/s3fs). No training stack — NeMo lives in the container. |

## Design — batch-priority dispatch of a foreign NeMo container

The novelty vs exp108 is the *payload*, not the *submission*:

- exp108 dispatched a **levanter callable** into marin's default iris-task image.
- exp112 dispatches a **`bash -lc`/`torchrun` binary** into
  `nvcr.io/nvidia/nemo`, which has neither our repo nor the fray task harness.

`dispatch_nemo.build_request` sets:
- `resources = ResourceConfig.with_gpu("H100", count=8, replicas=1, image=<nemo>)`;
- `environment = create_environment(docker_image=<nemo>, env_vars=…)` — passing
  `docker_image` (not `workspace`) is essential: otherwise `create_environment`
  defaults `workspace` to the launcher dir and would try to `uv sync` our pyproject
  **inside** the NeMo image;
- `entrypoint = Entrypoint.from_binary("bash", ["-lc", BOOTSTRAP])`, where
  `BOOTSTRAP` base64-decodes the inlined `nemo_train_qwen3.py`, runs
  `torchrun --standalone --nnodes=1 --nproc-per-node=8`, and best-effort uploads
  the MFU summary JSON to S3 (creds injected by iris; the summary is also in logs);
- `priority=3` (iris **batch** band) + an import-time assert that this fray build
  actually has `JobRequest.priority`.

Everything lands under one removable prefix
`s3://marin-us-east-02a/MarinFold/exp112_qwen_3b_nemo_mfu/` (issue #108 convention).

## Runbook

**0. Prereqs** (same as exp108, already on Tim's box): kubeconfig
`~/.kube/coreweave-iris-rno2a`, object-storage key in `~/.config/marin/cw-rno2a.env`,
`kubectl` on PATH, `marin-iris[controller]` in this env.

**1. Local validation (no cluster):**
```bash
uv sync
python nemo_train_qwen3.py --help                    # argparse (NeMo import is lazy)
EXP112_DRY_RUN=1 python -m dispatch_nemo              # build the JobRequest, no submit
# API smoke against the NeMo container on a local GPU (tiny model):
docker run --rm --gpus all -v "$PWD:/w" -w /w nvcr.io/nvidia/nemo:25.04.02 \
    torchrun --standalone --nproc-per-node=1 nemo_train_qwen3.py --tiny --max-steps 5
```

**2. Cluster smoke (~50 steps, batch priority):**
```bash
set -a; source ~/.config/marin/cw-rno2a.env; set +a
WK=$(python -c "import netrc; print(netrc.netrc().authenticators('api.wandb.ai')[2])")
uv run iris --cluster=cw-rno2a job run --no-wait --priority batch \
    --cpu=2 --memory=6GB --disk=16GB \
    -e WANDB_API_KEY "$WK" -e EXP112_MAX_STEPS 50 -e EXP112_RUN_NAME plm-exp112-smoke \
    -- python -m dispatch_nemo
# Monitor: uv run iris --cluster=cw-rno2a job list | grep exp112
#          uv run iris --cluster=cw-rno2a job logs <job> --max-lines N
# Verify batch band: uv run iris --cluster=cw-rno2a job status <child-job-id>
```

**3. Benchmark (~300 steps):** as above with `-e EXP112_MAX_STEPS 300 -e EXP112_RUN_NAME plm-exp112-cv1-3b-nemo-bench`. Sweep `-e EXP112_MICRO_BATCH {1,2,4}` to find the MFU-max micro-batch.

## Watch-items / risks

- **Binary-entrypoint × foreign image on iris has no precedent here** — the smoke
  run must prove: nvcr.io image pull (a CoreWeave node **pull secret** may be
  needed — if `ImagePullBackOff`, mirror the image to a pullable registry), `bash`
  exec, batch band, injected S3 creds.
- **NeMo 2.0 API** (`llm.GPTConfig`/`GPTModel`/`MockDataModule`/`llm.train`) is
  validated against the pinned container tag by the local `--tiny` smoke; class/field
  names can drift across tags.
- **QK-norm / RoPE details** (`qk_layernorm`, `rotary_base`) match Qwen3 for the
  benchmark's FLOPs; reconcile bit-for-bit only before a real run.
- **FLOP-formula parity** — we lead with *throughput* to avoid it; treat the MFU
  ratio as indicative unless both frameworks' FLOP formulas are reconciled.

## Full 16-epoch training run (multi-node) — implemented

Because the MFU win was compelling, the deferred full run was built out and
**launched at 4 nodes (32 H100)**: `plm-exp112-cv1-3b-nemo-e16-lr1e-3-wd0p2`,
**71,312 steps** (= exp108, from 4,672,623,743 train tokens), LR 1e-3, wd 0.2,
10% warmup, global batch 128, real contacts-v1 data. Target: beat #75's eval
loss 2.7566. Run it with `EXP112_DATA=real EXP112_REPLICAS=4` (see the top of
`dispatch_nemo.py`); `prepare_megatron_data.py` produced the `.bin/.idx` corpus.

This required solving several **real multi-node / no-shared-FS** problems on
`cw-rno2a` (all validated live on a 2-node smoke incl. a kill→resume):

- **Data:** parquet `document` → JSONL → `preprocess_data.py` (HF tokenizer,
  `--append-eod` = `<eos>` boundary) → `.bin/.idx` on S3; `reset_attention_mask` +
  `reset_position_ids` for block cross-doc attention.
- **torchrun rendezvous:** iris has no coordinator API for torchrun, so rank-0
  publishes `IRIS_ADVERTISE_HOST` to S3 and peers poll (static rendezvous). Two
  gotchas: the injected LOTA endpoint negative-caches a cross-node
  poll-before-write → use the **consistent** `cwobject.com` endpoint for the
  rendezvous object; and a preemption-restart reuses the key, so the poller
  rejects a **stale** master IP by S3 `LastModified` freshness.
- **Gloo:** Megatron's CPU process groups + distributed optimizer need Gloo, whose
  cross-node full-mesh fails unless `GLOO_SOCKET_IFNAME` is pinned to the host-eth
  iface — found via Python stdlib `SIOCGIFADDR` (the NeMo container has no `ip`).
- **Dataset index (no shared FS):** Megatron builds the sample `.npy` on global
  rank-0 into a node-local dir; a monkeypatch makes `get_rank()` report the LOCAL
  rank *during the build* so each node's local-0 builds its own cache.
- **Checkpoint + resume:** Megatron writes a sharded (DCP) checkpoint; each node
  syncs its own shards to S3 (union = complete), rank-0 prunes + commits
  `latest.txt`, throttled to one upload per `checkpoint_every` steps. On (re)start
  each node mirrors the latest checkpoint back and `AutoResume` continues — batch
  priority is preemptible, so the gang auto-restarts (`max_retries_preemption`) and
  resumes; the CPU driver is non-preemptible and outlives it.

Deferred still: **HF export** of the final checkpoint (custom arch → needs an
exporter/state-dict remap) + the downstream contacts eval.

## Results

Single **8×H100** node on `cw-rno2a`, batch **128 × seq 8192** (= exp108),
**bf16**, gradient checkpointing **on** (as exp108), **mock** data, NeMo 2.3.2 /
`nvcr.io/nvidia/nemo:25.04.02`. The whole novel path worked first try on the
cluster: nvcr.io image pull (~4.5 min), binary `torchrun` entrypoint at **batch**
band, base64-inlined script, 8-GPU bringup, and S3 result upload. Model built with
the exact geometry (48L / 2048h / 32h·8kv GQA / QK-norm / RMSNorm / SwiGLU / RoPE,
vocab 2845). JSON summaries under `…/exp112_qwen_3b_nemo_mfu/results/`.

| run | micro-batch | s/step | tokens/s | tokens/s/GPU | model TFLOP/s/GPU | **MFU** |
|-----|:-----------:|:------:|:--------:|:------------:|:-----------------:|:-------:|
| NeMo (smoke, 40 steps) | 1 | 11.69 | 89,737 | 11,217 | 251 | 25.4% |
| NeMo (50 steps) | 2 | 11.34 | 92,465 | 11,558 | 259 | 26.2% |
| **NeMo (50 steps)** | **4** | **11.29** | **92,861** | **11,608** | **260** | **26.3%** |
| **exp108 Levanter** (baseline) | — | ~20 | ~52,000 | ~6,500 | — | **~15%** |

**NeMo ≈ 1.75–1.79× Levanter** on identical model / shape / hardware:
**~26% MFU vs ~15%**, **~93k vs ~52k tok/s**, **~11.3 vs ~20 s/step**. Micro-batch
barely moves it (memory-bound at this shape with grad checkpointing), so mbs 1 is a
fine default. MFU is computed as model-FLOPs ÷ (step_time × 8 × 989 TFLOP/s); the
*throughput* columns are formula-free and carry the comparison on their own.

### Full training run (4 nodes / 32 H100)

_Launched `plm-exp112-cv1-3b-nemo-e16-lr1e-3-wd0p2`, 71,312 steps. Checkpoints
under `…/exp112_qwen_3b_nemo_mfu/checkpoints/<run>/`, W&B `open-athena/MarinFold`.
Fill in final **val loss** vs #75's 2.7566, wall-clock, and 32-GPU throughput/MFU
when it completes._

## Conclusion

**Yes — NeMo/Megatron-Core is materially more efficient than Levanter here (~1.75×),
out of the box.** That directly answers #112. Two honest caveats: the number is
**~26% MFU, not** the ~35–40% one might hope for a 3B on H100 — the ceiling here is
set by the **mandatory activation recompute** (48L × seq 8192 doesn't fit without it,
same tax exp108 paid), **bf16** (no fp8), and **DP-replicated** params (no TP/FSDP).
Each is a known lever: **fp8** (Transformer-Engine, plausibly another ~1.3–1.8×),
**TP/FSDP** to drop grad checkpointing, and larger effective batch — all deferred.

**Recommendation:** the throughput win is real and large enough to justify moving the
**full 16-epoch training run to NeMo** if we want the trained model sooner — a 4-node
NeMo run should land in **~2–2.5 days vs exp108's ~4** at the same node count. Before
that: wire the real `.bin/.idx` data path (block cross-doc attention; verify Megatron
#2357) and solve multi-node `torchrun` rendezvous over `IRIS_TASK_ID` (both in
*Deferred*). Left to the humans to decide (issue stays open).
