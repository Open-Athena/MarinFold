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

We evaluated V2 checkpoints at steps 2,000, 4,000, and 6,000 against the final
exp177 contacts-v1 contact-order-augmentation checkpoint (step 71,359). Every
row uses the same 554-protein exp89 resolved-residue universe and the published
exp82 recipe: 100 sequence-conditioned rollouts at temperature 1.0 / top-p 0.95,
one undirected vote per pair per rollout, followed by the unchanged
`build_rollout_rows.py` metrics.

| checkpoint | coverage | R (all) | R (long) | AUC (all) | AUC (long) |
|---|---:|---:|---:|---:|---:|
| V2 step 2,000 | 554/554 | 0.0238 | 0.0192 | 0.5660 | 0.5455 |
| V2 step 4,000 | 554/554 | 0.0540 | 0.0345 | 0.6146 | 0.5835 |
| V2 step 6,000 | 554/554 | 0.0785 | 0.0477 | 0.6520 | 0.6085 |
| exp177 contacts-v1 step 71,359 | 554/554 | **0.5113** | **0.4595** | **0.9246** | **0.9033** |

V2 improves monotonically over these checkpoints, but step 6,000 remains far
behind the matched contacts-v1 control: −0.4328 all-range R-precision and
−0.4118 long-range R-precision. The result does not support replacing
contacts-v1 with this delta serialization at the evaluated training stages.

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

1. Evaluate later durable checkpoints from the continuing 12,000-step run.
2. Measure whether strict grammar-constrained decoding improves V2 enough to
   alter the conclusion from the canonical permissive rollout comparison.
3. Prefer contacts-v1 for subsequent training unless later V2 checkpoints close
   the large observed gap.

## Files

- `compute_contacts_delta_stream_documents.py` — V2 document conversion.
- `build_packed_delta_stream_cache.py` — fixed-length packed LM cache builder.
- `dispatch_delta_stream_full.py` — distributed training dispatcher.
- `build_summary.py` / `summary_narrative.md` — experiment summary deck.
