# exp163 scale plan — 1M-protein rollout generation + TPU fine-tune

Recipe from the two recon passes. **Gated on the local MVP** (does refiner@K16 > @K0?)
and on human go for TPU cost. Not launched yet.

## A. Rollout generation (~1M distinct proteins × ~32 rollouts ≈ 32M rollouts)

**Targets (~941k distinct, one-per-cluster, no regen):** copy
`experiments/exp100_data_only_correct_documents_contacts_v1/select_round0_all.py`
into exp163 → every round-0 protein from the exp53 train corpus (941,028 train /
960,054 all-splits clusters). Emits the target schema the worker consumes.
- True >1M distinct needs `exp53.../selection.py --min-cluster-size 1` (recovers
  singleton clusters → up to 1.68M) **plus** exp53 Stage-B regen for the new
  entries. Heavier; treat as an optional stretch. Start with ~941k.
- Keep a light `L<=512` cap for cost (AFDB skews short, most survive).

**Worker knobs** (`experiments/exp98.../gen_rollouts_worker_vllm_tpu.py`, `gen_prompts.py`):
- rollouts/protein → **32** (both `gen_prompts.py -k 32` and worker `--n-rollouts 32`).
- **`--top-k -1`** (disable top-k; default is 50) — the #142 under-generation fix.
- **Drop `logprobs=1`** (worker SamplingParams ll.148,161) — exp163 uses only `pred`;
  logprobs make gen ~3.7x slower. Keep `T=1.0`, `top-p=0.95`, `tensor-parallel 4`.
- **REFACTOR before scaling:** worker writes 3 tiny files/protein → ~3M GCS objects
  at 1M scale (melts `publish_to_hf`). Write **per-shard aggregate parquets**
  (one per `--shard`), columns slimmed to `(entry_id, r, pred)`.

**Cost:** ~146 v5p-8-h base (L<=512, top-k off, no logprobs); ~150–300 v5p-8-h once
top-k lengthening / length mix included. Parallelize 16–32× v5p-8 → hours–~half-day.
Launch via iris (us-east5, `--tpu=v5p-8 --extra=vllm --extra=tpu`), bf16 model
`gs://marin-us-east5/checkpoints/prot-exp75-cv1-1_5b-e8-.../hf_bf16/step-35679`.
Publish to a NEW bucket `hf://buckets/open-athena/MarinFold/data/contacts-v1-rollouts-exp163`.

## B. GPU fine-tune on CoreWeave rno-2a (SUPERSEDES the TPU/exp120 plan)

> The original plan here was "copy exp120's `contacts_v1_ft_common.py` and run it on
> TPU via the marin executor". That is dead: exp120 is **executor-era marin**
> (`default_train` / `ExecutorStep` / `versioned` / `this_output_path`), and none of
> it exists in the marin exp163 resolves (0.2.57). Rollout generation already moved
> to CoreWeave rno-2a (section A), so training follows it there — same cluster, same
> S3 prefix, same batch band, no GCS→TPU staging at all.

**Code (all committed):**
- `refine_ft_common.py` — `MODEL_CONFIG` (Qwen3 1.47B: 8192 ctx / hidden 2048 /
  inter 8192 / 32 heads / 8 kv / 24 layers / llama3 rope, cross-checked against the
  S3 export's `config.json`), the data config, and `build_on_pod_config()` on top of
  `marinfold_models.build_train_lm_on_pod_config` (exp108's StepContext-free builder).
- `dispatch_refine_train.py` — env knobs → one batch-band `JobRequest(priority=3)`
  per LR → submit → wait. `EXP163_DRY_RUN=1` builds them without submitting.
- `loss_mask.py` + `tokenize_refinement_corpus.py` — the answer-span mask (below).

**Warm start — `initialize_from_hf` against the HF export already on S3**
(`s3://marin-us-east-02a/MarinFold/exp163/model/step-35679`, the artifact vLLM
loads). levanter's `HFCheckpointConverter` reads fsspec URLs, so nothing is staged.
Weights only: fresh optimizer / LR schedule / step 0 = a continue-train.
The Levanter-native E8 checkpoint does exist, but **not** at the path this document
used to give — it is missing a `protein/` segment; the real one is
`gs://marin-us-east5/checkpoints/protein/prot-exp75-cv1-1_5b-e8-lr1e-3-wd0p2-v1-bc3084/checkpoints/step-35679`
(`hf/` and `hf_bf16/` are siblings). Unused: cw pods have no GCS creds, so it would
have to be staged S3-ward to warm-start from the very same weights. Reachable via
`EXP163_INIT_CKPT` if a full-state restore is ever wanted.

**Loss mask — now materialized in the corpus, not hooked onto the component.**
levanter 1.2 removed `DatasetComponent.loss_weight_fn`; per-token weights only
arrive via `PrebuiltLmDatasetFormat(input_ids_key=…, loss_weights_key=…)`, i.e. out
of the cache. So `tokenize_refinement_corpus.py` tokenizes + greedily packs the raw
corpus into fixed 8192-token rows and writes `input_ids` + `loss_weights` alongside:
- weight 1.0 across each document's `<begin_statements> … <end>` span (so the model
  is trained to emit the true contacts **and** to stop), 0.0 on the sequence header,
  every `<CAND>` block, the trailing `<eos>` and the pad tail;
- ids resolved from the live tokenizer (`<begin_statements>`=9, `<end>`=10) rather
  than assumed, with a drift tripwire;
- cross-document attention still blocked — `PrebuiltLmDataset` derives segment ids
  from the `<eos>` the packer writes after each document.

Two blockers dissolve with it: nothing cloudpickles by module reference any more (the
GPU worker never imports exp163 code), and there is no `marinfold` / `transformers`
pin clash on the training worker.

Measured on the 10k validation corpus (built + staged to
`exp163/val10k/refinement_tokenized/`): 18,750 docs / 44.3M tokens → **6,569
sequences** of 8192, 82.3% packing density, 25.2% of tokens carry loss,
`EXP163_STEPS_PER_EPOCH = 52` at batch 128.

**Launch** (batch priority; see `dispatch_refine_train.py`'s docstring for the smoke
form and every env knob):

```bash
cd experiments/exp163_models_teach_contacts_v1_to_refine_a
set -a; source ~/.config/marin/cw-rno2a.env; set +a
WK=$(python -c "import netrc; print(netrc.netrc().authenticators('api.wandb.ai')[2])")
uv run iris --cluster=cw-rno2a job run --no-wait --priority batch \
    --enable-extra-resources --cpu=2 --memory=6GB --disk=16GB --extra gpu \
    -e WANDB_API_KEY "$WK" -e EXP163_STEPS_PER_EPOCH 52 \
    -- python -m dispatch_refine_train
```

## Sequencing
MVP go/no-go (done) → generate rollouts on cw-rno2a (10k validated; section A) →
`build_refinement_corpus.py` → `tokenize_refinement_corpus.py` (mask + packing) →
warm-start fine-tune (`dispatch_refine_train.py`) → HF-export the checkpoint →
eval refiner@K vs @K0 + consensus vote on exp89 held-out.
