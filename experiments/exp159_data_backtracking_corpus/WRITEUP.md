# Teaching contacts-v1 to take back its mistakes

**A 1,023,997-document backtracking corpus, generated with the base model in the loop.**

Issues [#158](https://github.com/Open-Athena/MarinFold/issues/158) (format) ·
[#159](https://github.com/Open-Athena/MarinFold/issues/159) (corpus) ·
[#160](https://github.com/Open-Athena/MarinFold/issues/160) (training)
· Slides: [`plots/summary.pdf`](plots/summary.pdf)

---

## 1. The problem

A contacts-v1 document is **append-only**. Every `<contact> <pX> <pY>` asserts a
residue-residue contact; absence means "no contact"; and nothing can be taken
back. A rollout that commits to a wrong contact early carries that error to
`<end>`, and it costs precision for every statement that follows.

That is not a cosmetic limitation. #142 established that contacts-v1's apparent
"under-generation" is a difficulty symptom rather than a decoding bug — the
model stops early on hard proteins, not because its stopping rule is broken.
What it cannot do is **revise**.

## 2. The format change (#158)

One new token:

```
<retract> <pX> <pY>
```

It takes back a previously emitted contact. The structure section stops being an
unordered set and becomes an ordered **edit list**: `<contact>` adds,
`<retract>` removes, and the document's answer is whatever is **live at
`<end>`**. A retract may reference a contact emitted arbitrarily far back.

Two properties made this safe to add to a live codebase:

- **Append-only vocab.** `<retract>` is the last token, so every pre-existing id
  is unchanged and existing checkpoints grow their embedding by exactly one row.
- **Byte-identical generation.** `generate.py` was left untouched — the
  retraction documents are synthesised by this experiment's engine, not by the
  document builder — so every existing corpus reproduces exactly.

The canonical semantics live in `marinfold/.../contacts_v1/read.py`
(`live_contacts` / `fold_statements`), and the rollout readout in `inference.py`
folds retractions before voting, so a retraction-trained model is never scored
on contacts it explicitly took back.

*A hazard worth recording:* the coordinate and crops formats **inherit**
contacts-v1's vocab and append their own tokens after it. Naively appending
`<retract>` shifted every coordinate/crop token id and would have broken their
published checkpoints. Both now freeze their inherited block and re-append
`<retract>` last — which additionally makes those superset tokenizers
retraction-capable, so retraction documents can enter a **mixture**.

## 3. Where the decoys come from

To train a model to retract, we need documents that demonstrate self-correction:
a wrong contact, then later a retraction of it.

The decoys are **the base model's own false positives** — not synthetic
negatives. They are exactly the mistakes `contacts-v1-exp120-1.5B` is tempted to
make, so the context leading to each one is on-distribution in a way an invented
negative never is.

## 4. The central design choice: re-condition, don't splice

The obvious construction is to sample a rollout, label its contacts against
ground truth, and splice `<retract>` in after each wrong one.

**That is wrong**, and it is worth being precise about why: it leaves the entire
tail of the document conditioned on a context that still contains the mistake
and never contains the correction. Every contact after the splice point was
generated in a world where the error was still asserted. The document looks like
self-correction but does not contain any.

So we generate with the model **in the loop**, maintaining two streams:

| stream | contents |
|---|---|
| **output document** | the full ordered edit list — `<contact>`, `<retract>`, re-emitted `<contact>` |
| **conditioning prompt** | a *clean* contacts-v1 prefix with only the currently-**live** contacts, and **no** `<retract>` (the base model has never seen one) |

After every retraction the prompt is rebuilt from the corrected live set and
generation continues. Every emitted contact is therefore conditioned on a
coherent set, and the only synthetic tokens in the corpus are the retract
statements themselves — precisely the behaviour we want to install.

This leans on contacts-v1's deliberate order-invariance (the model was trained
on shuffled contacts), which is what makes "rebuild the prompt from the live
set" an in-distribution operation.

## 5. When to retract: the model's own collapsing posterior

If retraction timing is driven by ground truth, we teach the model to retract on
information it will not have at inference. If it is driven by a random delay,
the timing carries no information at all and the model can learn *that* it
retracts but never *when*.

So timing comes from the base model's **leave-one-out belief** in each queued
contact. For a queued pair `c = (i, j)`, scored against the *committed* set
(live contacts minus the queue):

```
lp1[i]   = log P(<p_i> | prompt, <contact>)
lp2[i,j] = log P(<p_j> | prompt, <contact>, <p_i>)
s(c)     = exp(lp1[i] + lp2[i,j]) + exp(lp1[j] + lp2[j,i])
```

`c` must be absent from the conditioning set — contacts-v1 trains the model not
to repeat a contact already in its prompt, so a still-present `c` scores ≈ 0 and
the number is meaningless.

We retract when that belief **collapses**: below `tau ×` its peak, below an
absolute floor, or out of the top-R of the predicted map. The mechanism
self-sorts, because a false positive becomes inconsistent with the emerging fold
as true contacts accumulate, while a true contact does not.

**Ground truth never decides timing.** It is used only for the correctness
guarantee below.

## 6. Correctness is structural, not hoped-for

The main loop only continues while it can still afford a closing flush —
retract every live non-ground-truth pair, emit every missing one. So whatever
the trigger does, the final live set equals ground truth **exactly**. A
badly-tuned trigger degrades *realism*, never *correctness*.

Verified on the finished corpus by re-deriving the fold per document rather than
trusting the generator's bookkeeping:

**1,023,997 / 1,023,997 documents fold to exactly their ground truth. 0 truncated.**

## 7. The corpus

| | |
|---|---|
| documents | **1,023,997** |
| tokens | 1,076,910,057 |
| mean protein length | 193.7 residues |
| mean contacts / document | 186.3 |
| **mean retracts / document** | **33.1** |
| documents containing ≥1 retraction | 79.7% |
| documents failing to fold to GT | **0** |
| truncated | **0** |

Proteins and ground truth come from the ESMFold2-Atlas distillation set via
exp139's saved raw pyconfind contacts — no pyconfind at generation time. Those
contacts are *raw*, so the contacts-v1 document filters
(`min_seq_separation=6`, `min_contact_degree=0.001`) are applied when deriving
ground truth; without them roughly two-thirds of the "ground truth" would be
weak or trivially-local pairs the base model was never trained to emit, and the
engine would have scored every one of them as a false positive.

## 8. Does retraction actually track wrongness?

This is the pass/fail for the whole idea. Across the **32,849,569** false
positives the base model emitted:

| | |
|---|---|
| retracted by the posterior trigger | **25,105,853 (76.4%)** |
| P(false positive \| retracted) | **0.974** |
| false-positive base rate | 0.166 |
| **enrichment** | **5.85x** (ceiling 6.02x → **97% of max**) |
| true contacts retracted by the trigger | **0** |

Enrichment is the number to quote because it is base-rate-normalised and
therefore comparable across corpora. On a separate AFDB-derived corpus the same
engine scored 2.73x against a 2.78x ceiling — a lower absolute figure, but the
same ~98% of what was achievable there. The trigger is not noise; it is a sharp,
ground-truth-free wrongness detector.

Stability across scale (nothing drifted as the run progressed):

| checked at | 5k docs | 30k docs | 1.02M docs |
|---|---|---|---|
| fold == GT | 100% | 100% | 100% |
| retract precision | 0.975 | 0.974 | 0.974 |
| enrichment | 5.37x | 5.92x | 5.85x |
| mean retract distance | 17.8 | 18.1 | 17.9 |

## 9. And the retractions are *delayed*

Mean **17.9** statements between a contact and its retraction (median 9); only
**0.1%** are immediate.

That spread is the entire point. A mistake retracted immediately teaches a
trivial local pattern. A mistake retracted eighteen statements later teaches the
model to revise once the accumulating picture turns against an earlier call —
the long-range self-correction the format exists to enable.

Recovery rate is **0.778**: after retracting a pair, the model goes on to emit a
true contact involving one of the freed residues most of the time.

## 10. Making it fast enough to matter

The first working loop cost **14.7 s/document**. Profiling attributed ~86% to
`propose` (about 50 calls at ~220 ms) and ~14% to `score` — the model was
running at **batch size 1**.

Three changes:

1. **The engine became a generator.** It now *yields* `ProposeRequest` /
   `ScoreRequest` instead of calling the model directly, so a scheduler can
   advance many proteins in lockstep and serve every pending proposal in one
   padded GPU batch. The synchronous single-protein driver remains as a thin
   wrapper, so the tests and pilot were unaffected.
2. **The decode budget dropped 12 → 6 tokens.** A contact statement is 3 tokens.
   A duplicate proposal is returned rather than treated as EOS, since the engine
   already skips live pairs — so this cannot truncate a document early.
3. **`score` computes only the tails its targets need**, instead of one tail per
   residue to fill an [L, L] map we only read a handful of entries from.
   Numerically identical, with an equivalence test pinning it.

| batch | 1 | 8 | **24** | 48 | 96 |
|---|---|---|---|---|---|
| s/doc (A5000) | 14.7 | 2.18 | **1.68** | 2.09 | 1.83 |

**8.8x**, and 1.09 s/doc on an H100 — which turned scale from an algorithmic
problem into a fan-out problem. (Batches beyond 24 lose to padding waste from
ragged prompt lengths.)

## 11. Running it at scale

48 × 1 H100 on CoreWeave `cw-rno2a` at **batch priority**, ~4.5 hours wall-clock,
**0 worker failures**.

Two failures during the first attempt are worth recording, because both are
easy to repeat:

- **Sizing.** 480 of the cluster's 512 GPUs were already in use; requesting 64
  workers meant ~29 failed admission. Size to *free* capacity, not nominal.
- **Checkpoint granularity.** Workers originally wrote one parquet per 4,000-doc
  shard, so nothing landed for ~66 minutes and a preemption on the (preemptible)
  batch band discarded up to a full shard of GPU work. Workers now write a part
  every **250 documents** and resume by skipping existing parts.
- **`max_task_failures` is a separate field defaulting to 0**, distinct from
  `max_retries_failure`. GPU reclamation arrives as a SIGTERM recorded as a
  *failure*, not a preemption — so `max_retries_preemption=100` never applied and
  every reclaimed worker died outright, bleeding the fleet 48 → 28 in twenty
  minutes. With it set, the second run had zero failures.

## 12. Published

```
hf://buckets/open-athena/MarinFold/data/document_structures/contacts_v1_backtracking/
    train/shard-*.parquet
    tokenizer/
    README.md
```

Public and anonymously readable, tokenizer co-located. `<retract>` is the final
token of the contacts-v1 vocab; the coordinate/crops superset tokenizers carry
it too (at a different id), so retraction documents can be mixed with the crops
and ESM-Atlas corpora under a single tokenizer.

## 13. What this does *not* establish

The corpus is **authored** to be discriminative. Whether a *trained* model
reproduces that behaviour is #160, and the bar there is high: #82's
resample + pairwise-tiebreak recipe already recovers much of what any single bad
rollout loses, so backtracking must beat **that**, not greedy decoding. A null
result — with diagnostics explaining why — is a legitimate outcome.

Two honest caveats for whoever runs #160:

- The corpus's **recall = 1.000 is an artifact** of the correctness flush, which
  retracts every surviving false positive. A trained model has no flush, so its
  recall will be lower; **precision and enrichment are the comparable numbers.**
- The **10% single-retract "probe" class was deferred.** The cheap inference-time
  probe (run a clean rollout, append one `<retract>`, read which contact the
  model wants to take back) has no on-distribution training support in this
  corpus, so that evaluation arm would be testing a capability the data does not
  teach.

## Code

| file | role |
|---|---|
| `backtrack_engine.py` | the pure state machine (generator; no torch) |
| `backtrack_adapter.py` | exp120 Proposer/Scorer + seq↔position mapping |
| `batch_runner.py` | cross-protein batched scheduler |
| `esm_atlas_source.py` | proteins + ground truth from exp139's saved contacts |
| `gen_esm_atlas_worker.py` | sharded, resumable generation worker |
| `dispatch_coreweave.py` | batch-priority fan-out over CoreWeave H100s |
| `consolidate_esm_atlas.py` | corpus QA (re-derives the invariant) |
| `publish_to_hf.py` | stage from S3 → publish to the HF bucket |
| `../exp160_.../retraction_diagnostics.py` | the #160 discrimination metrics |
