---
marinfold_experiment:
  issue: 299
  title: "Train compact delta-stream contact documents"
  kind: models
  branch: exp299/delta-stream-contacts
---

# Train compact delta-stream contact documents

## Question

Can a compact contact document retain ordinary next-token cross-entropy training
while conditioning every contact prediction on the complete protein sequence?

## V2 sequence-prefix delta stream

The active format is **not** the earlier interleaved prototype. Every document
puts the complete sequence before any contact target:

```text
DOC_START AA_0 AA_1 ... AA_(L-1) CONTACTS_BEGIN
DELTA(0 -> j)* STOP
DELTA(1 -> j)* STOP
...
DELTA((L-1) -> j)* STOP
DOC_END
```

`AA_0` etc. are notation only: the stream contains ordinary amino-acid tokens.
The contact section has one signed-delta segment per residue in sequence order;
a zero-contact residue emits only `STOP`. Deltas within a segment are sorted by
signed offset. `DOC_START`, `CONTACTS_BEGIN`, and `DOC_END` are reserved tokens
in the previously unused IDs 21–23; AA IDs remain 1–20, `STOP=0`, signed delta
tokens start at 32, and the vocabulary remains 2080 tokens.

This is deliberately analogous to contacts-v1: an evaluator can provide the
complete sequence prefix, then sample only contact structure. It fixes the
causal defect in the V1 interleaved stream, where contacts after residue `i`
were predicted before the model had seen residues `i+1..L-1`.

## V1 prototype: withdrawn

The first implementation serialized `AA_i DELTA* STOP` repeatedly. Its
full-corpus conversion and 12k-step training run are retained as infrastructure
history only; they are not the format for subsequent sweeps.

The earlier reported delta-stream R-precision values are **withdrawn**. They
used an ad-hoc next-token log-probability scorer rather than MarinFold's
reference contact metric. The reference metric lives in
[`exp82`](../exp82_evals_contacts_v1_contact_prediction/): it samples a panel
of contact rollouts, ranks pairs by occurrence count, and computes metrics with
the exp89 candidate universe/metric code. Future V2 evaluation must:

1. sample 100 sequence-conditioned delta-stream contact rollouts;
2. parse deltas into undirected pair-vote matrices;
3. use the unchanged `exp82/build_rollout_rows.py` metric implementation; and
4. validate the harness against a published contacts-v1 rollout checkpoint
   before any delta-vs-contacts-v1 comparison.

No claim about V1 relative R-precision should be used to select a larger run.

## Canonical rollout comparison

We evaluated the available V2 checkpoints and the available exp177 contacts-v1
contact-order-augmentation controls. Every row uses the same 554-protein exp89
resolved-residue universe and the published exp82 recipe: 100
sequence-conditioned rollouts at temperature 1.0 / top-p 0.95, one undirected
vote per pair per rollout, followed by the unchanged `build_rollout_rows.py`
metrics.

V2 uses approximately 10% more tokens per protein. With the same batch size and
sequence length, its protein-count-equivalent contacts-v1 step is therefore
approximately `V2 step / 1.1`.

| checkpoint | approximate contacts-v1-equivalent step | coverage | R (all) | R (long) | AUC (all) | AUC (long) |
|---|---:|---:|---:|---:|---:|---:|
| V2 step 2,000 | 1,818 | 554/554 | 0.0238 | 0.0192 | 0.5660 | 0.5455 |
| V2 step 4,000 | 3,636 | 554/554 | 0.0540 | 0.0345 | 0.6146 | 0.5835 |
| V2 step 6,000 | 5,455 | 554/554 | 0.0785 | 0.0477 | 0.6520 | 0.6085 |
| V2 step 8,000 | 7,273 | 554/554 | **0.1306** | **0.0877** | **0.7083** | **0.6567** |
| exp177 contacts-v1 step 10,000 | 10,000 | 554/554 | 0.0239 | 0.0199 | 0.5863 | 0.5622 |
| exp177 contacts-v1 step 71,359 | 71,359 | 554/554 | 0.5113 | 0.4595 | 0.9246 | 0.9033 |

There is no exact protein-count-matched exp177 export for the available V2
checkpoints: the earliest retained exp177 HF export is step 10,000, equivalent
to roughly V2 step 11,000. The comparison above is nevertheless conservative
for V2: V2 step 8,000 has seen only about 73% as many proteins as exp177 step
10,000, yet leads it by +0.1066 all-range and +0.0677 long-range R-precision.
The exp177 final checkpoint is retained only as a mature-model reference and is
not a training-progress-matched control.

As in the canonical contacts-v1 worker, decoding is permissive: valid in-range
pairs vote while malformed or out-of-universe statements are ignored rather
than causing the whole rollout to be discarded. All four checkpoints produced
nonzero votes for all 554 targets. Full per-protein rows and aggregate metrics
are in `data/rprecision_comparison_rows.csv.gz` and
`data/rprecision_comparison_summary.csv`.

## Storage

V2 conversion and cache creation run as a federated root job on
`cw-us-east-02a`, co-located with their existing
`s3://marin-us-east-02a/` source and training artifacts. A Marin-local Zephyr
job has no credentials for this private S3 store and must not be used for this
pipeline.

## Next steps

1. Measure whether strict grammar-constrained decoding improves V2 further.
2. For future head-to-heads, retain contacts-v1 exports at the exact
   protein-count-equivalent steps rather than comparing against final weights.

## Files

- `compute_contacts_delta_stream_documents.py` — V2 document conversion.
- `build_packed_delta_stream_cache.py` — fixed-length packed LM cache builder.
- `dispatch_delta_stream_full.py` — distributed training dispatcher.
- `build_summary.py` / `summary_narrative.md` — experiment summary deck.
