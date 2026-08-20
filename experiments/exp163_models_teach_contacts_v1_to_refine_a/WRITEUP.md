# exp163 — teaching contacts-v1 to emit many candidate structures in one document

**Issue:** [#163](https://github.com/Open-Athena/MarinFold/issues/163) · **PR:** [#164](https://github.com/Open-Athena/MarinFold/pull/164)

**Result.** A 1.5B contacts-v1 model fine-tuned to write a *chain* of candidate contact
maps in a single generation. It emits ~15 near-disjoint candidates; the best of them beats
the base model by **+27% F1 (+26σ)**, making it the best contact predictor in this
experiment — and it gives up nothing on the ordinary contacts-v1 task
(R-precision 0.3374 vs base 0.3357, a statistical tie).

That is a usable starting point for best-of-N RL: several distinct candidates per rollout,
a large gap between the best and the last, and no regression on the base task.

**Published** (public, anonymous):
`open-athena/MarinFold` → `checkpoints/plm-exp163-refine-cv1-1_5b-lr1e-4-e1-cos-tpuF/hf/step-404`
· interactive Colab: [`exp163_multidraft_demo.ipynb`](exp163_multidraft_demo.ipynb)

---

## 1. The question

contacts-v1 samples badly. A single rollout from the tuned 1.5B (E8) recovers only ~12% of
a protein's true contacts, but sample it 16 times and *some* rollout is much better —
oracle best-of-16 nearly doubles it. The information is there, spread across samples.

The original framing was **refinement**: show the model K of its own noisy rollouts and
have it write a better contact set than any of them. The framing that actually worked is
**self-generated candidates**: have the model write many candidates in one document, and
select. The distinction turned out to matter enormously (§4.3).

## 2. The document format

contacts-v1 documents are `<contacts-v1> <begin_sequence>` + (position, residue) pairs in
**random order**, then `<begin_statements>` and a list of `<contact> <pi> <pj>` triples,
closed by `<end>`. The shuffle is deliberate — the model must use the `<pN>` position
tokens rather than adjacency in the prompt. Contacts use `MIN_SEP = 6`.

The multi-draft format changes exactly two things:

* **`<begin_statements>` may repeat.** Each occurrence means "discard the previous
  candidate, here is a new one". It is the *native* contacts-v1 separator, reused — not a
  new marker. Only the last section is closed by `<end>`; drafts are superseded by the
  next `<begin_statements>`.
* **A mode token.** Multi-draft documents open with `<contacts-v1.multi>` instead of
  `<contacts-v1>`. This is vocab id **7 renamed in place** (`make_multi_tokenizer.py`) —
  vocab size stays 2,845 and every existing id is untouched, so there is no embedding
  resize and no id drift.

Training documents pair K noisy drafts with the true answer:

| span | content | size |
|---|---|---|
| header | sequence, shuffled | — |
| draft × K | a real E8 rollout, subsampled `Uniform[1, cap]` | **53.7** contacts, ~12% precision |
| final | ground truth | **199** contacts |

K is uniform 0–16 (mean 8.5). The ground-truth section is **199 contacts at every K**
(corr(K, |final|) = +0.000) — drafts never eat its budget.

**Corpus:** 100,000 multi-draft documents (50,000 proteins × 2) + 99,996 plain contacts-v1
documents mixed in as rehearsal, packed to 8192 → **51,724 sequences, 405 steps/epoch**.

## 3. What makes multi-draft generation work

This is the central mechanical finding, and it is entirely about **which token predicts a
section boundary**.

`weight[i]` supervises predicting `token[i+1]`. So the slot that teaches *"emit another
`<begin_statements>`"* is the last token of a **draft** (weight `w_draft`), while the slot
that teaches *"emit `<end>`"* is the last token of the **final** (weight `w_final`).
Measured over 600 real documents:

| profile (header/draft/final) | w on → `<begin_statements>` | w on → `<end>` | ratio | sections emitted |
|---|---|---|---|---|
| 0 / 0 / 1 | **0.000** | 1.000 | 0.00 | **1** |
| 0.1 / 0.1 / 1 | 0.100 | 1.000 | 0.10 | **1** |
| 0.1 / 0.3 / 1 | 0.300 | 1.000 | 0.30 | **1** |
| **E** — 0.1 / 1.0 / 2.0 | 1.000 | 2.000 | 0.50 | **~15** |
| **F** — 0.1 / 1.0 / 1.0 | 1.000 | 1.000 | 1.00 | **~15** |

With `w_draft = 0` the "continue" transition receives *exactly zero gradient* — the model
was never taught to do it, so one section is the only behaviour its objective describes.
At 0.1 and 0.3 it is supervised 10× and 3.3× more weakly than stopping, which is not
enough. E and F are the first profiles where continuing competes with stopping.

The aggregate view says the same thing: finals are ~4× longer than drafts, so they
dominate the weighted-token budget unless drafts carry comparable per-token weight
(drafts get 0% / 18% / 39% of the signal at `w_draft` 0 / 0.1 / 0.3, versus ~52% for E and
~68% for F).

The base model also emits exactly one section — contacts-v1 documents *have* one, so this
is not something fine-tuning destroyed. It has to be trained for.

`reweight_corpus.py` derives a new arm from an existing tokenized corpus (a weight profile
changes only `loss_weights`, never `input_ids`), so an arm costs one CPU pass rather than
a re-tokenization.

## 4. Results

### 4.1 Multi-draft generation — the headline

553 held-out proteins × 4 rollouts, vLLM on TPU:

| model | `n_sections` | Jaccard | first | **best** | last | improving | finished |
|---|---|---|---|---|---|---|---|
| base E8 | 1.00 | — | 0.2373 | 0.2373 | 0.2373 | — | 98% |
| single-section fine-tune | 1.00 | — | 0.2379 | 0.2379 | 0.2379 | — | 98% |
| **arm E** | 14.86 | 0.071 | 0.1814 | **0.3012** | 0.2470 | 0.552 | 57% |
| **arm F** | 14.99 | 0.071 | 0.1840 | **0.3025** | 0.2493 | 0.552 | 56% |

Paired on identical (protein, rollout), n = 2216:

| comparison | Δ F1 |
|---|---|
| arm F best-of-~15 vs **base E8** | **+0.0652 ± 0.0025** (+26.5σ, win 72.6%) |
| arm F best-of-~15 vs single-section fine-tune | +0.0646 ± 0.0026 (+24.7σ, win 70.9%) |
| arm F *last* section vs single-section fine-tune | +0.0114 ± 0.0029 (+4.0σ) |

Per-protein best over 4 rollouts: **0.3565** vs 0.3148.

Three properties matter for RL:

* **~15 candidates per generation** — best-of-N is scoreable *inside one rollout*.
* **Jaccard 0.071** — the candidates are nearly disjoint. The "model copies its own draft"
  failure mode, which would flatten a best-of-N reward regardless of accuracy, does not
  occur.
* **best 0.3025 > last 0.2493 > first 0.1840**, with `frac_improving` 0.552 — a mild but
  real refinement trend. The *spread* is worth far more than the trend: **reward the best
  section, not the last.**

E and F are statistically indistinguishable, so the 2× final weighting bought nothing —
**F (uniform) is the simpler choice** and is the published model.

### 4.2 The base task is not degraded

Teacher-forced **R-precision**, plain `<contacts-v1>` mode (no mode token, no drafts),
553 proteins:

| model | R-prec (all) | short | medium | long |
|---|---|---|---|---|
| base E8 | 0.3357 | 0.4720 | 0.3751 | 0.2694 |
| **arm F** | **0.3374** | 0.4726 | 0.3755 | 0.2724 |

Paired: **+0.0017 ± 0.0010 (+1.7σ)**, long band +0.0029 ± 0.0017. A statistical tie —
multi-draft training cost nothing on the ordinary task. Free generation agrees: arm F
0.2481 vs base 0.2373 (+0.0108 ± 0.0024, +4.6σ). Levanter's own val loss agrees too:
**3.139** for both arms against E8's **3.169**.

One behavioural note: arm F emits ~2.94 sections even under the plain sentinel, so the
multi-draft habit leaks into base mode. Accuracy does not suffer, but the mode token is
not a clean on/off switch.

### 4.3 Conditioning on *external* drafts hurts

Pasting other rollouts into the prompt, with draft sizes matched to training's
`Uniform[1, cap]`:

| k drafts shown | union recall available | F1 | paired Δ vs k=0 |
|---|---|---|---|
| 0 | — | 0.2379 | — |
| 4 | 0.134 | 0.2028 | −0.0351 ± 0.0031 |
| 8 | 0.220 | 0.1852 | −0.0527 ± 0.0031 |
| 16 | 0.332 | 0.1214 | −0.1165 ± 0.0036 |

Monotonic degradation, *even though the drafts contain steadily more of the true answer*.
The base model collapses far harder (0.2373 → 0.0015 at k=4) because it stops almost
immediately — 4.5 contacts instead of 160 — so format training does buy large robustness
here; it just does not turn external drafts into a gain.

**The contrast is the useful result.** Conditioning on someone else's noisy candidates
degrades prediction; generating its *own* chain and taking the best is the best predictor
we have. For RL that argues for on-policy self-generated candidates rather than a fixed
offline draft pool — and it means filtering the offline corpus for higher-quality drafts
is probably not the lever.

Why external drafts carry so little: they are ~12% precise, each recovering ~4% of the
truth, and the union of all ~8.5 covers only 21%. A training-free probe earlier in this
experiment showed that conditioning on *true* partial contacts lifts R-precision
0.145 → 0.556 — the joint signal is real and large, but real rollouts are too noisy and
too low-coverage to carry it.

## 5. Using it

```python
from huggingface_hub import HfApi
BUCKET = "open-athena/MarinFold"
PREFIX = "checkpoints/plm-exp163-refine-cv1-1_5b-lr1e-4-e1-cos-tpuF/hf/step-404"
api = HfApi()
paths = [p.path for p in api.list_bucket_tree(BUCKET) if p.path.startswith(PREFIX)]
api.download_bucket_files(BUCKET, [(p, DEST / p.rsplit("/", 1)[-1]) for p in paths],
                          token=False)      # anonymous
```

Prompt with `<contacts-v1.multi>` at position 0 and end the prompt on
`<begin_statements>`; the model writes candidates from there.
[`exp163_multidraft_demo.ipynb`](exp163_multidraft_demo.ipynb) does all of this and plots
every candidate map plus the section-vs-section Jaccard matrix.

**Cap the sections.** Only ~56% of generations emit `<end>`; the rest run to the context
limit. `--max-sections` bounds both the token budget (sized from the measured ~220-contact
section, so ~668 tokens each) and the parsed output:

| `--max-sections 8` | |
|---|---|
| `n_sections` after cap | mean 7.55, max 8 |
| `n_sections_raw` before cap | mean 17.03, max 42 |
| rollouts where the cap bound | 89% |
| tokens generated | 4,219 (flat in protein length) |

## 6. Training recipe

| | |
|---|---|
| model | Qwen3 1.47B — hidden 2048, inter 8192, 32 heads / 8 KV, 24 layers, Llama3 rope, vocab 2845 |
| warm start | E8 `prot-exp75-...-bc3084` step-35679 via `initialize_from_hf` (weights only) |
| schedule | 1 epoch cosine, peak LR 1e-4 → `min_lr_ratio` 0.1, warmup 10%, AdamW wd 0.2 |
| batch / seq | 128 × 8192, `per_device_parallelism=-1`, **no microbatching** |
| steps | 405 (51,724 packed sequences ÷ 128) |
| loss | per-token `loss_weights` via `PrebuiltLmDatasetFormat`; plain docs weighted 1.0 throughout |
| hardware | marin iris **v5p-16**, us-central1, interactive band, ~80 min |

**No microbatching, deliberately.** Levanter re-normalises per-token loss weights *per
microbatch*; with drafts and finals carrying different weights that would change the
effective objective and break comparability between arms. Gradient accumulation is
therefore not a free memory lever here.

## 7. Practical notes worth keeping

**The rope config trap — the most expensive thing in this experiment.** Levanter's HF
export writes the Llama3 rope under the newer `rope_parameters` key and leaves top-level
`rope_theta` / `rope_scaling` **null**. Any reader older than transformers 5.x (marin's
vLLM; the transformers 4.53.1 on the CUDA eval image) sees null and silently falls back to
**default rope** — a 50× wrong base frequency whose error grows with sequence distance. It
produced an entire round of plausible-looking but wrong evaluations, and the tell that went
unexamined for days was that the *long* band degraded hardest. Training is unaffected
(levanter uses its own config, not the exported JSON). `stage_v3_to_gcs.py` now translates
the key at the single point every checkpoint passes through, and `publish_to_hf_bucket.py`
refuses to publish a checkpoint without a top-level `rope_theta`.

**Ship the tokenizer you trained with.** Levanter exports the *published* contacts-v1
tokenizer, where id 7 still spells `<contacts-and-distances-v1>`. A notebook writing the
literal `<contacts-v1.multi>` would tokenize to garbage. The renamed tokenizer is
co-located with the published weights.

**Packing and `<eos>`.** Documents terminate `... <end> <eos>` so levanter can derive
segment ids, so a packed row splits on `<eos>`, not `<end>`. The weight slot *on* a
document's `<eos>` would supervise the first token of the next document — cross-document
leakage that is invisible whenever `w_header = 0`. It is explicitly zeroed.

**Cluster notes.** CoreWeave's batch band produced nothing across five submissions and
three days (~10h queued unplaced, one mass preemption sweep, one OOM); everything here ran
on marin TPU. A parent iris job reads `running` while its child gang is still `building`
and unplaced — always check per-task state. Pin the **region** when a locality guard
demands it, never the **zone** (zone-pinning starved three separate jobs). `with_tpu`
leaves `regions` unset and the scheduler may pick one with no v5p at all. A multi-region
job needs its *data* mirrored, not just its constraint widened. TPU work goes in the
**interactive** band — the opposite of the CoreWeave rule — because the v5p pool is
interactive-dominated and a batch job there never schedules.

**R-precision without a GPU.** `rprec_worker_tpu.py` computes the teacher-forced metric on
vLLM/TPU using next-token logprobs, since vLLM cannot do a raw forward pass but does return
the full distribution. Validated per protein against the CUDA reference: **corr 0.9992**,
mean |Δ| 0.0024, and the base model reproduces its known R-precision to 0.0002 at full
scale.

**Small generative samples are treacherous.** A 40-protein probe of the same comparison
gave best-of-N +0.048 where the full 553 gave +0.065, and an 8-protein probe suggested a
2.4× long-band gain that did not survive at all. Label them provisional.

## 8. Where to go next

* **RL.** Best-of-N over self-generated candidates, capping sections in the sampler.
  Reward the best section; the last-vs-best gap (0.2493 → 0.3025) is the headroom.
* **Termination.** ~56% of generations stop on their own. Worth teaching explicitly rather
  than only capping, if RL turns out to be sensitive to truncated final candidates.
* **Diversity vs quality.** Temperature trades spread against per-section accuracy and has
  not been swept; the notebook exposes the knob.

## Appendix — reproduction

```bash
cd experiments/exp163_models_teach_contacts_v1_to_refine_a

# corpus: multi-draft documents + plain rehearsal, then a weight profile
python -m make_multi_tokenizer                      # rename vocab id 7 in place
python -m build_refinement_corpus --format multi-draft --mix-plain 0.5 \
    --draft-order random --out corpus_v3.parquet
python -m tokenize_refinement_corpus --format multi-draft \
    --w-header 0 --w-draft 0 --w-final 1.0          # prints steps/epoch = 405
uv run python -m reweight_corpus --selftest         # always; guards the packing logic
uv run iris --cluster=marin job run --no-wait --enable-extra-resources \
    --cpu=8 --memory=32GB --disk=32GB --extra cpu \
    -- python -m reweight_corpus --src "s3://.../v3/tok_mix50/*.parquet" \
       --dst gs://.../v3/tok_mix50_F --w-header 0.1 --w-draft 1.0 --w-final 1.0

# train on marin TPU (interactive band; region pinned to where the data lives)
uv run iris --cluster=marin job run --no-wait --enable-extra-resources \
    --cpu=2 --memory=6GB --disk=16GB -e WANDB_API_KEY "$WK" \
    -e EXP163_DEVICE tpu -e EXP163_TPU_TYPE v5p-16 -e EXP163_TPU_REGIONS us-central1 \
    -e EXP163_LRS 1e-4 -e EXP163_STEPS_PER_EPOCH 405 -e EXP163_RUN_SUFFIX tpuF \
    -e EXP163_S3_PREFIX gs://marin-us-central1/MarinFold/exp163 \
    -e EXP163_CORPUS "gs://.../v3/tok_mix50_F/*.parquet" \
    -e EXP163_VAL "gs://.../contacts_v1/val/*.parquet" -e EXP163_INIT_HF "$E8" \
    -- python -m dispatch_refine_train

# stage to bf16 (required by vLLM/TPU; also repairs the rope key)
uv run iris --cluster=marin job run --no-wait --enable-extra-resources \
    --cpu=8 --memory=48GB --disk=48GB --extra cpu \
    -- python -m stage_v3_to_gcs --model-src gs://.../hf/step-404 \
       --model-dst gs://.../tpu/tpuF-bf16/step-404

# multi-draft generation. NO --zone (pinning starves), NOT --priority batch.
uv run iris --cluster=marin job run --no-wait --enable-extra-resources \
    --tpu=v5p-8 --cpu=16 --memory=64GB --disk=64GB --extra=vllm --extra=tpu \
    -- python gen_rollouts_worker_exp163.py --model gs://.../tpuF-bf16/step-404 \
       --targets gs://.../targets.parquet --prompts gs://.../prompts \
       --out gs://.../gen_tpuF --shard 0/1 --n-rollouts 4 --mode-id 7 \
       --format multi-draft --max-sections 8 --tensor-parallel-size 4

# teacher-forced R-precision, no GPU needed
uv run iris --cluster=marin job run --no-wait --enable-extra-resources \
    --tpu=v5p-8 --cpu=16 --memory=64GB --disk=64GB --extra=vllm --extra=tpu \
    -- python rprec_worker_tpu.py --model gs://.../tpuF-bf16/step-404 \
       --targets gs://.../targets.parquet --prompts gs://.../prompts \
       --out gs://.../rprec_tpuF --shard 0/2 --tensor-parallel-size 4

# publish (refuses a checkpoint with a broken rope config)
uv run iris --cluster=marin job run --no-wait --enable-extra-resources \
    --cpu=8 --memory=32GB --disk=32GB --extra cpu -e HF_TOKEN "$OA_TOKEN" \
    -- python -m publish_to_hf_bucket --src gs://.../tpuF-bf16/step-404 \
       --dest-prefix checkpoints/plm-exp163-refine-cv1-1_5b-lr1e-4-e1-cos-tpuF/hf/step-404
```

**Key files.** `build_refinement_corpus.py` · `tokenize_refinement_corpus.py` ·
`reweight_corpus.py` · `loss_mask.py` · `make_multi_tokenizer.py` · `refine_ft_common.py` ·
`dispatch_refine_train.py` · `gen_rollouts_worker_exp163.py` · `rollout_metrics.py` ·
`eval_refiner_worker.py` · `rprec_worker_tpu.py` · `stage_v3_to_gcs.py` ·
`publish_to_hf_bucket.py` · `exp163_multidraft_demo.ipynb`

**Data.** Corpora and eval set on CoreWeave S3 `s3://marin-us-east-02a/MarinFold/exp163/`
and marin GCS `gs://marin-us-east5/MarinFold/exp163/`; per-protein CSVs in `data_md/`;
the published checkpoint on the `open-athena/MarinFold` bucket.
