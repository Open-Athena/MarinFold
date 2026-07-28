---
marinfold_experiment:
  issue: 160
  title: "exp: does training on backtracking traces let contacts-v1 self-correct at inference?"
  kind: models
  branch: exp160-backtracking-training
---

# exp: does training on backtracking traces let contacts-v1 self-correct at inference?

**Issue:** [#160](https://github.com/Open-Athena/MarinFold/issues/160) · **Kind:** `models` · **Branch:** `exp160-backtracking-training` (stacked on #159)

## Question

Does continue-training on the backtracking corpus (1) preserve standard contact-prediction accuracy, and (2) let retraction-enabled rollouts beat the settled resample+vote recipe at matched inference compute?

## Hypothesis

A model trained on self-correction traces can walk back a bad early commitment mid-rollout. **The bar is high:** #82's resample + pairwise-tiebreak already recovers much of what a single bad rollout loses, so backtracking must beat *that*, not greedy decoding. A null result — with the diagnostics below to explain it — is a clean answer either way.

## Background

Depends on #158 (`<retract>` + `read.py` fold) and #159 (the corpus). Base model / control: `contacts-v1-exp120-1.5B`. Reuses the #120 matched-budget harness and the #89 benchmark (`compute_metrics`).

## Approach

### Status: trained and evaluated. See Results.

**`retraction_diagnostics.py` (done).** Pure, model-free; computes from a document's edit list + its GT pairs:

- **Is retraction discriminative?** — the experiment's real pass/fail. A 2x2 of {retracted, kept} x {false positive, true contact}, reported as `precision = P(FP | retracted)`, `recall = P(retracted | FP)`, and the headline **`enrichment = precision / P(FP)`** (1.0 = no signal; scale-free, so comparable across proteins with different FP base rates).
- **retract rate** — a model that never retracts has ignored the mechanism.
- **retraction distance** — if it only ever retracts the statement it just emitted, the long-delay signal did not transfer.
- **recovery** — after retracting, does it emit a *true* contact on one of the freed residues?

10 unit tests pin the metric definitions on hand-built edit lists (perfect discrimination -> enrichment = 1/base_rate; retracting at the base rate -> exactly 1.0; no-retraction -> NaN not 0; orientation canonicalisation; re-emission; malformed retracts).

**Smoke test on the #159 370-doc corpus** (`diagnose_corpus.py`) — the reference the trained model is measured against:

```
documents:            370 (97.3% contain a retraction)
mean contacts/doc:    83.5      mean retracts/doc: 31.6
FP base rate:         0.360  (of 30,004 emitted pairs)
retract precision:    0.983     retract recall: 1.000
ENRICHMENT:           2.73x     (ceiling 1/0.360 = 2.78x -> 98% of max)
true contacts retracted: 190    (the deliberate 5% noise retractions)
retract distance:     mean 22.1 / median 10 statements (0.1% immediate)
recovery rate:        0.688
```

Note `recall = 1.000` is **by construction** — the corpus engine's correctness flush retracts every surviving false positive. A trained model has no flush, so its recall will be lower; **precision and enrichment are the meaningful comparisons.**

### Evaluation harness

Both questions are answered from **one** inference pass per arm — the rollouts
that vote on contacts are the same rollouts whose edit lists carry the
retractions, and re-sampling for the second question would measure a different
set of rollouts than the first.

| script | env | what it does |
|---|---|---|
| `score_backtracking_worker.py` | eval pod | the rollout worker; 12 unit tests pin its readout |
| `stage_to_cw.py` | marin CPU pod | assets GCS → CoreWeave S3 cloud-side (2.7 GiB in 22 s) |
| `dispatch_eval_cw.py` | workstation | **what actually ran** — both arms over 12x1 H100, batch band |
| `dispatch_eval_tpu.py` | marin checkout | the v5p-8 twin, kept: same measurement, whichever accelerator has room |
| `score_eval.py` | standalone `uv run --no-project` | exp89 metrics + retraction diagnostics |
| `prepare_eval_model.py` / `verify_eval_model.py` | marin venv / exp159 venv | fp32 → bf16 + transformers-4.x config, then read back under a real 4.57 |
| `export_trained_to_hf.py` | marin venv | levanter/orbax → HF; **not needed in the end** — the run writes its own `hf/step-2058` |

**Where it ran, and why not TPU.** Every marin TPU family refused this eval at
interactive band on 2026-07-28 — `v5p-8` in both `us-central1-a` and
`us-east5-a`, `v5p-16`, `v6e-4`, all `Insufficient TPUs (need 4, available 0)` —
while `cw-rno2a` had zero pending jobs. Two notes for next time: the autoscaler's
`ready` column counts *booted* slices, not free ones, and its `demand` column is
autoscaler demand, not queue depth (`us-central1-a` read "84 ready, 0 demand"
while admitting nothing) — read the scheduler's own reason string instead. And
on **CUDA** vLLM the bf16 recast is unnecessary: fp32 is cast at load, and
bf16-on-disk was only ever a TPU ragged-paged-attention requirement. That turned
model preparation into three small JSON files instead of 2.75 GB.

Two departures from exp82's worker, both load-bearing:

- **Votes come from the #158 fold, not a `<contact>` regex.** A pair the rollout
  later took back must not vote. For a model that never retracts the two agree
  exactly, which is what keeps the control comparable.
- **Each rollout's ordered edit list is kept.** Rollouts are sampled, so nothing
  downstream can reconstruct *which* pairs were retracted and *when*.

The **exp120 control is re-measured with this same worker**, not quoted from
#82/#169: those ran `contact_mult=6` and a readout that predates `<retract>`, so
quoting them would confound the training effect with the harness. Both arms run
`contact_mult=8` — retraction lengthens documents, and a budget that truncates
one arm and not the other is not a fair comparison.

**Universe.** Truth is defined only where #89 defines it: both residues resolved
and `|i-j| >= 6`. The primary diagnostic scores in-universe statements; the
unrestricted variant is reported beside it, since the #159 corpus reference was
computed with no restriction (its proteins are predicted structures, so every
residue is resolved).

### Remaining
1. **Decisive eval** — resample+vote baseline vs retraction rollouts vs the retract-probe, at matched **token** compute (retraction lengthens documents, so equal rollout *count* would hand the arm more compute).
2. **Mixing-ratio sweep** — 100:0 control and 75:25, against the 50:50 run here.

### Vocab / checkpoint compatibility (verified)

The exp120 checkpoint predates two appended tokens. Checked directly against
its `tokenizer.json`:

| | |
|---|---|
| exp120 checkpoint vocab | 2,845 (`vocab_size` = 2845) |
| current contacts-v1 tokenizer | 2,847 |
| **id mismatches on the checkpoint's 2,845 tokens** | **0** |
| new tokens needing embedding rows | `<contacts-v1.sequence_only>` (2845), `<retract>` (2846) |

So continue-training is a **+2 embedding resize, not a remap** — every existing
embedding keeps its meaning. This is the append-only vocab discipline from #158
paying off; had `<retract>` been inserted into `NATIVE_TOKENS` instead, all
2,838 shared contacts-and-distances tokens would have shifted and the
checkpoint would have been unusable.

Note the published `timodonnell/contacts-v1-tokenizer` repo (used by the exp120
training path) is the **pre-retract** tokenizer. Training on the backtracking
corpus needs the tokenizer published *with the corpus*
(`.../contacts_v1_backtracking/tokenizer/`), or the crops/ccoord superset if
training a mixture.

## Setup decisions (settled 2026-07-27)

**Base model: `contacts-v1-exp120-1.5B`** (val 2.7213, ppl 15.20). Eric's #117
sweep has better numbers — best **2.7037** on `...wd0p2-bs256-europe-west4` —
but **every exp117 checkpoint has been deleted**: the 23 surviving GCS dirs are
crashed runs holding only `.executor_info`/`.executor_status`, and none are on
the HF bucket. exp120 is the best contacts-v1 model that still loads, and it is
also the model that generated the #159 corpus, so this fine-tunes a model on
corrections of its own mistakes. (Note the true #117 best is **bs256**, not the
bs128/2.7112 run exp137 cites.) *Caveat:* exp120 reports
`eval/contacts-v1-val-orig/loss` while #117 reports
`eval/tokenized/contacts-v1-val/loss` — probably the same split, unproven, so
treat the 0.018 gap as approximate.

**GPU, not TPU.** At decision time the marin TPU cluster was fully subscribed
(0 chips free, 39 jobs pending on "Insufficient TPUs") while `cw-rno2a` had
~239 free H100s. The usual objection — CoreWeave cannot read GCS — is handled
by staging (below); #159 already proved CoreWeave pulls model + corpus from the
HF bucket at 48-worker scale.

**Full fine-tune** (not LoRA), **50:50** backtracking:clean, **superset
tokenizer**.

### Verified before launch

| check | result |
|---|---|
| exp120 ids under the superset tokenizer | **0 mismatches** on its 2,845 tokens → +1004 embedding rows, not a remap |
| superset tokenizer on 200 real backtracking docs | exact token counts, **0 UNKs**, exact round-trip |
| mix composition | 2,047,994 docs, **exactly 50.0%** backtracking, 2.17B tokens |
| mix protein overlap between halves | **0** (asserted) |
| staged Levanter checkpoint in CoreWeave S3 | 13 objects, **17.66 GB**, matches source |

### Data / artifact locations

- mix: `hf://buckets/open-athena/MarinFold/data/document_structures/contacts_v1_backtracking_mix50/` (`train/` + `tokenizer/`)
- init checkpoint: `s3://marin-us-east-02a/protein-structure/MarinFold/exp160_backtracking_training/init/exp120-step-1005`

### Staging notes (cost several attempts)

The Levanter **training-state** checkpoint is required — the HF export is not
enough for `initialize_from_checkpoint_path`. Routing it through the HF bucket
**failed**: that uploader panics on multi-GB files (`is not fully completed:
2257162240/2261618688 bytes`) and this checkpoint has a 2.26 GB shard. It now
goes GCS → CoreWeave S3 directly from a GCS-local marin pod via boto3
multipart — one hop, and training reads CoreWeave-local storage. Credentials
are passed as job env vars only; nothing secret is committed.

### BLOCKER: `marinfold_models` is stale against current marin/levanter

The training launch reached the worker and then failed here:

```
File ".../marinfold_models/defaults.py", line 35
    from levanter.data.text import LmDataConfig
ImportError: cannot import name 'LmDataConfig' from 'levanter.data.text'
```

This is the documented **marin 0.2.57 / levanter 1.2** API move: `levanter.data.text`
and `levanter.optim` became lazy plugin registries with empty `__init__`s, so
imports must name the defining submodule:

| was | now |
|---|---|
| `levanter.data.text.{LmDataConfig, DatasetComponent, UrlDatasetSourceConfig}` | `levanter.data.text.datasets` |
| `levanter.data.text.{TextLmDatasetFormat, PrebuiltLmDatasetFormat}` | `levanter.data.text.formats` |
| `levanter.optim.{OptimizerConfig, AdamConfig}` | `levanter.optim.config` |

`train_backtracking.py` already uses the new paths; **`models/marinfold_models/defaults.py`
does not**. Fixing it is a small library change that unblocks any experiment
re-locking to a recent marin, not just this one.

A second, related constraint forced the env split already committed:
`marin-core` requires `transformers>=5.5.3` while `marinfold` pins `<5`, so the
**training** env carries `marinfold-models` (+ marin stack) and *not* `marinfold`.
Diagnostics that need `marinfold.read` (`diagnose_corpus.py`) run under exp159's
env instead.

**Status: everything else is staged and verified; training is blocked only on
this import fix.**

## Success criteria

- **No regression** on #89 R-precision / AUC vs the matched-budget control.
- **Retraction is used** — non-trivial `<retract>` rate on held-out proteins.
- **Retraction is discriminative** — enrichment clearly above 1.0 (corpus reference: 2.73x). *The real pass/fail.*
- **Headline** — retraction-enabled inference beats resample+vote at matched token compute on long-range R-precision.

## Results

**Run:** `exp160-cv1-1_5b-bt50-lr3e-4-e1-cos-v3`, 2,059 steps on a v5p-32 in
`us-east5-a`, ~3h07m, **0 preemptions, 0 failures**. Final checkpoint and HF
export at **step-2058** (2,059 steps, 0-indexed — the export levanter writes
itself, so no separate conversion was needed).

Both arms scored through the **same worker** and exp89's metric implementation,
554 proteins x 100 rollouts each (55,400 rollouts per arm), on 12 CoreWeave
H100s at batch priority.

### 1. Is retraction discriminative? — yes, but at half the achievable strength

| | trained model | #159 corpus |
|---|---|---|
| rollouts containing a retraction | 43.0% (96.6% of proteins) | 79.9% of documents |
| retracts / rollout | 24.0 | 33.4 |
| contacts / rollout | 180.7 | 186.0 |
| P(FP \| retracted) | **0.902** | 0.974 |
| P(FP) base rate | **0.796** | 0.166 |
| **enrichment** | **1.134x** [1.111, 1.161] | 5.86x |
| ceiling (1 / base rate) | 1.26x | 6.02x |
| **headroom captured** | **52%** | 97% |
| P(retracted \| FP) recall | 0.137 | 1.000 (by construction) |
| retraction delay | **mean 19.0 / median 9** (0.1% immediate) | mean 18.0 / median 9 (0.2%) |
| recovery rate | 0.217 | 0.778 |

**The mechanism is used and it is not noise.** The bootstrap CI over proteins
excludes 1.0 by a wide margin, and of the contacts it takes back, 90% really
were wrong.

**The raw enrichment is not comparable to the corpus's 5.85x, and quoting it
that way would be wrong.** Enrichment is bounded by `1 / P(FP)`: the corpus's
documents are 83% correct, so it has room to reach 6x, while the model's own
rollouts on experimental structures are only 20% correct, capping it at 1.26x.
Normalised by achievable headroom the comparison is 52% against 97%.

**The timing signal transferred almost exactly.** Delay mean 19.0 / median 9
statements against the corpus's 18.0 / 9, with 0.1% immediate retractions
against 0.2%. This was the transfer most at risk — a model that had only learned
"emit `<retract>` sometimes" would retract what it just wrote — and it is the
part that worked. What did not transfer is *recall* (0.137 against the corpus's
1.000, which is by construction: the corpus engine has a ground-truth flush that
retracts every surviving false positive, and the model has no such thing) and
*recovery* (0.217 against 0.778).

Also measured: 8,110 malformed retracts (0.15/rollout, retracting a pair that
was not live) and 191,778 re-emissions (3.5/rollout, re-asserting a pair it had
retracted).

### 2. Did it cost contact accuracy? — yes, small but unambiguous

Mean over the same 554 proteins, exp89 `compute_metrics`:

| model | R all | R long | AUC all | P@L all |
|---|---|---|---|---|
| `exp120-base` (control) | **0.4357** | **0.3787** | **0.9057** | **0.3929** |
| `exp160-bt50` | 0.4158 | 0.3562 | 0.8975 | 0.3750 |

Paired per-protein differences (`exp160-bt50` − `exp120-base`), which is the
right statistic here — both arms score the same proteins and the between-protein
spread of R-precision is ~0.3, an order of magnitude larger than the effect:

| metric | Δ | 95% CI | wins |
|---|---|---|---|
| R-precision (all) | **−0.0199** | [−0.0239, −0.0158] | 29% |
| R-precision (long) | **−0.0225** | [−0.0283, −0.0167] | 28% |
| AUC (all) | −0.0081 | [−0.0102, −0.0061] | 30% |
| P@L (all) | −0.0178 | [−0.0215, −0.0142] | 27% |

Every interval excludes zero. And the backtracking arm spent **1.24x the tokens**
for the same 100 rollouts (624 vs 502 tokens/rollout) — it loses while being
given more compute, not less. Truncation at `contact_mult=8` was 2.82% of
rollouts against 0.00% for the control.

**The control does not isolate backtracking.** `exp120-base` is the *base model*,
not a matched-budget clean-only fine-tune. `exp160-bt50` additionally saw 2.17B
tokens, half of them ESM-Atlas documents, while the eval set is experimental
structures. So the −0.02 confounds "backtracking hurts" with "this fine-tune's
domain shift hurts", and the 100:0 clean-only arm is exactly the run that
separates them. Treat the regression as an upper bound on backtracking's cost.

### Harness validation

The control's numbers land where the loss ordering says they should, which is
the check that licenses everything above: exp120 (val 2.7213) scores
0.436 / 0.379 / 0.906 between #75 (2.7566 -> 0.425 / 0.366 / 0.901) and #117 best
(2.7037 -> 0.535 / 0.485 / 0.932). The #159 corpus reference was also recomputed
with *this* experiment's `retraction_diagnostics` rather than quoted
(`data/corpus_reference.txt`): 5.86x on a published shard against the 5.85x its
README reports.

## Conclusion

**Backtracking installs a real but weak self-correction signal, and on this
evidence it is not free.**

Training on #159's traces gives a model that uses `<retract>` on its own
initiative (24 per rollout), takes back contacts that really are wrong 90% of
the time, and — the part that most easily could have failed — retracts at
*roughly the right delay*, matching the corpus's distribution to within a
statement. The pass/fail criterion of this experiment is met: retraction is
discriminative, with a CI that excludes noise.

But it captures only half the achievable discrimination, it catches only 14% of
its own false positives, and folded contact prediction gets **worse** by ~0.02
R-precision at 1.24x the token cost. A mechanism that identifies 14% of your
mistakes at 90% precision is not yet worth what it costs to run.

The two things that would change the picture are both cheap and both listed
above as remaining work: the **100:0 clean-only control**, which decides how much
of the −0.02 is backtracking at all, and a **token-matched head-to-head** that
gives the backtracking arm fewer rollouts instead of more tokens.

