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

## B. TPU fine-tune (the real run; template = exp120, NOT exp150/137)

Copy `experiments/exp120_models_regen_vs_reepoch_contacts_v1/contacts_v1_ft_common.py`
→ `refine_ft_common.py`. Keep:
- `INIT_CHECKPOINT = gs://marin-us-east5/checkpoints/prot-exp75-cv1-1_5b-e8-lr1e-3-wd0p2-v1-bc3084/checkpoints/step-35679`
  (E8, **Levanter-native** dir, via `SimpleTrainConfig.initialize_from_checkpoint_path` = warm restart).
- `MODEL_CONFIG = Qwen3Config(max_seq_len=8192, hidden 2048, inter 8192, 32 heads,
  8 kv, 24 layers)`; `pad_tokenizer_to_match_model=True` (2845→2848).
- Corpus = parquet `document` text column → `TextLmDatasetFormat(text_key="document")`;
  tokenizer = bare id `timodonnell/contacts-v1-tokenizer`; keep the
  `ArrayExemplarTextLmDatasetFormat` reader (cache-ledger bug workaround); new MARIN_PREFIX.
- Launcher modeled on `train_regen_vs_reepoch_sweep.py` (warm-start, low LR, 1-epoch cosine).

**Loss mask (the custom piece) — `loss_weight_fn` on the training DatasetComponent**
(model on `experiments/exp0.../protein_train_common.py:61-129`; per-position float
0/1, packing-safe pure-JAX, NOT `-100`):
- weight = 1.0 for positions inside each `<begin_statements> … <end>` span, 0.0 elsewhere
  (sequence + `<begin_candidate>` blocks + any `<think>` → 0 automatically).
- **Packing-aware:** `pack=True` puts many docs per 8192 seq → must re-arm per doc.
  Build ON-at-`<begin_statements>` / OFF-at-`<end>` cumulative indicator, not a single index.
- ids via tokenizer `convert_tokens_to_ids(BEGIN_STRUCTURE_TOKEN / END_TOKEN)`.

**Staging (required):** HF buckets are NOT levanter-addressable → mirror bucket→GCS
first (template `experiments/exp120.../mirror_regen_train.py`): `hf buckets cp` (system
hf ≥1.5) → gcsfs-globbable shards under `gs://marin-us-east5/protein-structure/MarinFold/exp163_.../data/`,
keep `entry_id`+`document`. Co-locate GCS with TPU zone.

**iris launch:** `uv sync --extra tpu`; `WANDB_API_KEY=… uv run iris --cluster marin job
run … --extra=tpu --zone=…`. Depend on **PyPI** `marin-iris` (not frozen github-latest)
for the 14-day client-freshness gate.

## Sequencing
MVP go/no-go → generate ~941k×32 rollouts → publish → mirror→GCS → build refinement
corpus (`build_refinement_corpus.py`, at scale) → mirror corpus→GCS → default_tokenize
→ warm-start fine-tune (masked) → eval refiner@K vs @K0 + consensus vote on exp89 held-out.
