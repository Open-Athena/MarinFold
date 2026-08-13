---
marinfold_experiment:
  issue: 230
  title: 'exp: fine-tune contacts-v1-exp199-1.5B into a clean <contacts-v1.multi> multi-draft model (on-policy drafts, PDB + AFDB, 30%-decontaminated)'
  kind: models
  branch: claude/finetune-contacts-v1-multi-b46f9b
---

# exp: fine-tune contacts-v1-exp199-1.5B into a clean <contacts-v1.multi> multi-draft model (on-policy drafts, PDB + AFDB, 30%-decontaminated)

**Issue:** [#230](https://github.com/Open-Athena/MarinFold/issues/230) · **Kind:** `models` · **Branch:** `claude/finetune-contacts-v1-multi-b46f9b`

## Question

Can we port [#163](https://github.com/Open-Athena/MarinFold/issues/163)'s `<contacts-v1.multi>` multi-draft format onto the current best base model (`contacts-v1-exp199-1.5B`) — with **on-policy** drafts, an **experimental-PDB** data component, and a protein pool **decontaminated at 30% identity** against the eval set — such that the resulting checkpoint (a) writes many diverse candidate contact maps under `<contacts-v1.multi>`, (b) is a **clean single-document decoder** under plain `<contacts-v1>`, and (c) gives up nothing on plain-mode R-precision?

This is the SFT starting point for best-of-N RL ([#200](https://github.com/Open-Athena/MarinFold/issues/200) / [#208](https://github.com/Open-Athena/MarinFold/issues/208)), which is a **separate** experiment and explicitly out of scope here.

## Hypothesis

**H1 — the format transfers to a stronger base.** #163 established that multi-draft generation is a loss-weight-profile question, not a capacity question: `w_draft` has to compete with `w_final` for the section-boundary transition, and arm **F** (header 0.1 / draft 1.0 / final 1.0) plus a **50% plain-rehearsal mix** is the configuration that both emitted ~15 candidates and held the base task (R-prec 0.3374 vs base 0.3357 — a tie). Nothing in that mechanism is specific to E8, so it should reproduce from exp199.

**H2 — the mode leak is an under-training artifact, and steps are the lever.** #163's arm F emits **~2.94 sections under the plain `<contacts-v1>` sentinel** — the leak this issue is chartered to fix. It was trained for **405 steps**. [#175](https://github.com/Open-Athena/MarinFold/issues/175) got a *completely* clean token-0 mode switch out of the same kind of marker on the same kind of 50:50 marked mixture — **0.1 vs 42.0 retracts per rollout on one checkpoint, prompt-selected** — after **2,070 steps**. The two runs differ by ~5x in optimization, not in mechanism. Predicted: mean sections in plain mode falls to ~1.0 somewhere between 405 and ~2,000 steps. **Intermediate checkpoints make this falsifiable rather than assumed** (see Approach, Phase 3).

**H3 — drafts must be on-policy, and the risk is diversity, not quality.** #163's drafts were E8 rollouts at ~12% per-contact precision. exp199 is a far stronger sampler (R-prec **0.6103** under exp82's reference worker, [#209](https://github.com/Open-Athena/MarinFold/issues/209)), and RL will sample from *this* model, so the training drafts must come from it. The preregistered risk is the opposite of the obvious one: a stronger sampler is a *more consistent* one, so candidate diversity may shrink below #163's mean Jaccard of 0.071 — and best-of-N reward is paid for by spread, not by mean quality. **Jaccard is a reported gate, not an afterthought.**

**H4 — experimental PDB is worth including here.** #222's corpora are the first contacts-v1 documents whose `<begin_statements>` section is a *measurement* rather than a prediction. The multi-draft document is exactly the place that distinction lands: drafts are model opinion, the final section is truth, and the model is being taught to tell them apart.

## Background

| Prior work | What this run takes from it |
|---|---|
| [#163](https://github.com/Open-Athena/MarinFold/issues/163) ([WRITEUP](https://github.com/Open-Athena/MarinFold/blob/main/experiments/exp163_models_teach_contacts_v1_to_refine_a/WRITEUP.md)) | The format (repeated `<begin_statements>`; only the final section closed by `<end>`), the id-7 rename, arm F's weight profile, the 50% rehearsal mix, `build_refinement_corpus.py` / `tokenize_refinement_corpus.py` / `reweight_corpus.py` / `refine_ft_common.py` / `dispatch_refine_train.py` / `gen_rollouts_worker_exp163.py` / `rprec_worker_tpu.py` |
| [#175](https://github.com/Open-Athena/MarinFold/issues/175) | The evidence that a token-0 mode marker becomes a clean on/off switch given enough steps on a marked 50:50 mixture |
| [#199](https://github.com/Open-Athena/MarinFold/issues/199) / [#204](https://github.com/Open-Athena/MarinFold/pull/204) | The base model |
| [#209](https://github.com/Open-Athena/MarinFold/issues/209) | exp82's worker is the **reference scorer**; exp199 reads 0.6103/0.5639, not the published 0.5873/0.5422 |
| [#222](https://github.com/Open-Athena/MarinFold/issues/222) / [#223](https://github.com/Open-Athena/MarinFold/pull/223) | The PDB corpora on the bucket |
| [#225](https://github.com/Open-Athena/MarinFold/issues/225) / [#229](https://github.com/Open-Athena/MarinFold/pull/229) | The Tier-A decontamination rule and its machinery (`sequence_droplist.py`, `decontam_lib.py`) |
| [#213](https://github.com/Open-Athena/MarinFold/issues/213) / [#226](https://github.com/Open-Athena/MarinFold/issues/226) | The 70.9M-sequence MMseqs2 training DB, `sequence_from_document`, and the eval2 query set |
| [#200](https://github.com/Open-Athena/MarinFold/issues/200) / [#208](https://github.com/Open-Athena/MarinFold/issues/208) | The consumer. exp200's whole RL stack already assumes **token id 7 = `<contacts-v1.multi>`** — matching that is a deliberate compatibility constraint |

#163 also left a falsification worth carrying: its **v2 100k-document sweep collapsed the base task by −44%** across four weight profiles (mdA–mdD, all with `w_draft` ≤ 0.3 and *no plain rehearsal*). The published arm F run differs in exactly two ways — `w_draft = 1.0` and the 50:50 plain mix — and is a tie with base. **Both are load-bearing; neither is optional.**

## Approach

### Phase 0 — protein pool and decontamination

Three sources, all already published and anonymously readable on `hf://buckets/open-athena/MarinFold`:

- `data/document_structures/contacts_v1/train` — AFDB (#53), 4.13M docs
- `data/document_structures/contacts_v1_esm_atlas/train` — ESM-Atlas (#139), 66.8M docs
- `data/document_structures/contacts_v1_pdb_deduped_monomers/documents` — **experimental PDB** (#222), 41,661 docs, one representative per RCSB 40% cluster, pre-shuffled

Draw the pool so the plain-rehearsal half stays close to exp199's own pretraining mixture (AFDB + ESM-Atlas) while PDB carries real weight: **all surviving PDB monomers**, plus AFDB and ESM-Atlas draws to fill. Target ≈100k distinct proteins.

**Decontamination — Tier A at 30%**, which is what "no ≥30% sequence similarity to the eval set" means operationally, and is #225's Tier A rule verbatim:

> drop any training row with **identity ≥ 30% over ≥ 50% query coverage**, **or** E ≤ 1e-3, against any eval query. MMseqs2 at `-s 7.5` (the #65/#94/#213/#225 convention — keep it so the numbers stay comparable), `--max-seqs 5000` (#226: exp213's *published* table used 5000, not the 2000 in its argparse default).

- **Queries: eval2's 776**, `data/contacts-v1-eval2-exp226/eval2.fasta` — a strict superset of the 554-protein #89 benchmark we gate on. Costs nothing extra and protects the benchmark we are moving to.
- **AFDB / ESM-Atlas arms:** reuse `/data/exp213_overlap/targetDB` directly. Its headers are `{arm}|{shard:05d}_{row}_{entry_id}`, so a hit names the corpus row and inverting hits into a drop list needs no join.
- **PDB arm:** build a fresh 41,661-sequence DB. Cheap. Note #222 already excludes the eval set's 552 PDB entries *by PDB id*, but measured **50.2% of eval entries still have a 40% homolog** in the corpus — id-exclusion is not identity-exclusion, and this is the arm where contamination is most likely.
- Sequences come from `contacts_v1.read.sequence_from_document` (inverts the generator; currently only on [#216](https://github.com/Open-Athena/MarinFold/pull/216)'s branch — land it or vendor it, don't rewrite it). Neither published corpus carries a `sequence` column.
- **Report the drop rate per arm.** Expect ~1.9% for AFDB and ~1.6% for ESM-Atlas (#225 Tier A); PDB unknown and probably higher.

### Phase 1 — on-policy rollouts from exp199

Source model: `checkpoints/prot-exp199-cw-cv1-s02-m1-p06-aug/hf/step-145199` on the bucket (verified: `vocab_size` 2845, `rope_theta` 500000 present at top level so #198's repair is in place, id 7 = `<contacts-and-distances-v1>`).

Sampling is exp82/exp142's settled recipe, **not** exp98's: `T=1.0`, `top_p=0.95`, **top-k disabled** (`-1` in vLLM), budget `6L+128`. 24–32 rollouts per protein. Drop `logprobs` (~3.7x faster; only `pred` is used).

Fan-out: CoreWeave `cw-rno2a` single-H100 shards at **batch priority** (the standing rule), via #163's `dispatch_rollouts.py` — already an image-override + inlined-worker recipe that needs no repo checkout in the container. Write per-shard aggregate parquets (`entry_id`, `r`, `pred`), never per-protein files.

### Phase 2 — corpus

`build_refinement_corpus.py --format multi-draft --mix-plain 0.5 --draft-order random`, unchanged from #163:

| span | content |
|---|---|
| header | sequence, shuffled, fresh position numbering per document |
| draft × K | a real **exp199** rollout, subsampled `Uniform[1, cap]`, K ~ `Uniform{0..16}` |
| final | ground truth, closed by `<end>` |

Plain rehearsal documents are drawn from the **same decontaminated pool**, so PDB appears in both halves.

Tokenize with the **2845-token renamed tokenizer** from `make_multi_tokenizer.py` (id 7 renamed in place; vocab size and every other id untouched, so no embedding resize and no id drift). Weight profile **F** — header 0.1 / draft 1.0 / final 1.0, plain documents 1.0 throughout, the `<eos>` slot explicitly zeroed. Run `reweight_corpus.py --selftest` (it guards the packing logic).

Size for **~2x #163's protein breadth**, so that 2 epochs lands at **~2,000–2,500 steps** at batch 128 — ~5x #163's optimization, which is the H2 lever.

### Phase 3 — training

Warm start from exp199 via `initialize_from_hf` (weights only: fresh optimizer, fresh schedule, step 0). #163's `refine_ft_common.py` `MODEL_CONFIG` **matches exp199's `config.json` exactly** — hidden 2048, intermediate 8192, 32 heads / 8 KV, 24 layers, llama3 rope at 500000, ctx 8192, vocab 2845 — so the training code transfers 1:1.

- lr **1e-4** cosine to `min_lr_ratio` 0.1, warmup 10%, AdamW wd 0.2 (#163 swept this: 3e-4 fit the multi-draft objective 0.3% better while degrading base-task retention by **+7.4% bpb**)
- batch 128 × 8192, `per_device_parallelism=-1`, **no microbatching** — levanter re-normalises per-token loss weights *per microbatch*, which would silently change the objective
- 2 epochs, **checkpoint every ~250 steps**, so the leak-vs-steps curve in H2 is measured rather than assumed and the earliest clean checkpoint can be selected

### Phase 4 — evaluation

Two gates and one report. Everything scored with **exp82's rollout+resample worker** and **exp89's `compute_metrics.py`** — and the **base model re-scored in the same batch**, because #209 showed exp199's published 0.5873 understates its own weights by 0.023 under the reference scorer.

- **Gate A — plain-mode accuracy is not lost** (the requirement this run exists to protect). Teacher-forced/rollout R-precision under plain `<contacts-v1>`, 554-protein #89 benchmark, paired against the base.
- **Gate B — no mode leak.** Free generation under plain `<contacts-v1>`: section count per rollout. This is the behaviour #163 shipped broken.
- **Report — multi mode works.** At `--max-sections 8`: `n_sections`, mean pairwise Jaccard, first/best/last section F1, `frac_improving`, termination rate, contacts per section.
- **Secondary:** eval2-natural (78 proteins at <40% id, #226) — the honest low-homology readout. Lead with it separately, never pooled: eval2 is 75% de novo design.

## Success criteria

**Gate A — plain `<contacts-v1>` accuracy (paired vs base, same batch, same scorer):**
- R-precision (all) ≥ base − **0.005** (#204's four-replicate noise span is 0.0023; this is 2x it)
- R-precision (long) ≥ base − **0.010**

**Gate B — clean mode switch under plain `<contacts-v1>`:**
- mean sections per rollout ≤ **1.05** (#163 arm F: 2.94)
- ≥ **95%** of rollouts emit exactly one section

**Multi mode is usable for RL (reported; blocks the RL hand-off, not this run):**
- ≥ 8 sections at `--max-sections 8` (it uses the budget)
- mean pairwise Jaccard ≤ **0.30** (exp200's diversity-collapse kill criterion)
- best-of-N section F1 exceeds last-section F1 by a margin outside noise

**Deliverable:** checkpoint published to `hf://buckets/open-athena/MarinFold/checkpoints/<run>/hf/step-N` with the **renamed tokenizer co-located**, rope key verified at publish time, ready to be consumed by #200/#208.

**Kill criteria:**
- plain-mode R-precision Δ < **−0.02** → this is #163's v1/v2 forgetting failure reappearing; stop and diagnose rather than tune
- mean contacts per section < 60% of base → degenerate short sections

## Run book

Every stage writes its own provenance JSON into `data/`. Paths below are the
defaults each script carries.

```bash
# 0. stage bucket shards + write the shard manifest (needs huggingface_hub>=1.5,
#    which cannot share an interpreter with marinfold -- see stage.py)
/home/bizon/anaconda3/bin/python stage.py --work /data/exp230_multi --arm pdb
/home/bizon/anaconda3/bin/python stage.py --work /data/exp230_multi --arm afdb --n-shards 44
/home/bizon/anaconda3/bin/python stage.py --work /data/exp230_multi --arm esm_atlas --n-shards 10

# 1. Tier-A/30% drop list against #226's 776 eval queries
python decontam.py --work /data/exp230_multi

# 2. the decontaminated protein pool
python select_targets.py --work /data/exp230_multi --n-afdb 40000 --n-esm 40000 --n-pdb 0

# 3. on-policy rollouts from exp199 (marin TPU; see dispatch_rollouts.py for the
#    client-vs-workspace split and why this is not CoreWeave)
python dispatch_rollouts.py --num-shards 32 --tpu v6e-4 --zone us-east5-b

# 4. multi-draft + plain-rehearsal corpus, then profile F
python build_corpus.py --targets /data/exp230_multi/targets.parquet \
    --rollouts /data/exp230_multi/rollouts --out /data/exp230_multi/corpus
python make_multi_tokenizer.py --source /data/exp208_replication/model/C_bf16 \
    --out /data/exp230_multi/tokenizer_multi
python tokenize_corpus.py --in /data/exp230_multi/corpus \
    --out /data/exp230_multi/tokenized --tokenizer /data/exp230_multi/tokenizer_multi
#   ^ prints STEPS_PER_EPOCH, which the training dispatcher needs

# 5. fine-tune (train_common.py holds the recipe)
# 6. evaluate
python build_eval_targets.py --work /data/exp230_multi          # 554 units
#   Gate A  -- exp82's score_rollout_worker.py + exp89's compute_metrics.py,
#              with the exp199 BASE re-scored in the same batch (#209: exp82's
#              worker is the reference scorer, not #199's own pipeline)
#   Gate B  -- eval_modes_worker.py --mode plain
#   report  -- eval_modes_worker.py --mode multi --max-sections 8
python summarize_modes.py --rollouts gs://.../eval/step-N --label step-N
```

### Environments

Three interpreters, and they cannot be merged:

| what | interpreter | why |
|---|---|---|
| bucket staging | system python (`huggingface_hub` 1.5) | the bucket API does not exist below 1.5; `snapshot_download` cannot see buckets at all |
| corpus / tokenizer / local generation | `/data/exp208_replication/venv` | has `marinfold` + vLLM + a transformers that pins `huggingface_hub<1` |
| iris submission | `/home/bizon/git/marin-freshiris/.venv` | iris rejects a client more than 14 days old |

## Results

_(Fill in after the run completes.)_

## Conclusion

_(Fill in after results are in.)_
