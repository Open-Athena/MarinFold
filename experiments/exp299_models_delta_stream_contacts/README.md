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

## Storage

V2 conversion and cache defaults use `gs://marin-us-central1/`, matching the
pinned `us-central1` Zephyr workers. A CoreWeave training cache must not stream
from that bucket: if it proves larger than 10 GB, a one-time GCS-to-S3 mirror
requires explicit human approval before it is made.

## Next steps

1. Build the V2 sequence-prefix corpus and its fixed-8192 packed cache.
2. Add a V2 constrained-contact rollout worker that supplies the full sequence
   prefix and samples only the contact suffix.
3. Validate its vote matrices through the canonical exp82/exp89 metric path.
4. Use the exp288 configuration/prepare/launch/runtime pattern for the first
   production-scale V2 training sweep.

## Files

- `compute_contacts_delta_stream_documents.py` — V2 document conversion.
- `build_packed_delta_stream_cache.py` — fixed-length packed LM cache builder.
- `dispatch_delta_stream_full.py` — distributed training dispatcher.
- `build_summary.py` / `summary_narrative.md` — experiment summary deck.
