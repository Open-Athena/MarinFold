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

## Sequence-prefix delta stream

Every document puts the complete sequence before any contact target:

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
complete sequence prefix, then sample only contact structure.

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
| V2 step 8,000 | 7,273 | 554/554 | 0.1306 | 0.0877 | 0.7083 | 0.6567 |
| V2 step 11,999 (pilot final) | 10,908 | 554/554 | 0.1579 | 0.1054 | 0.7365 | 0.6866 |
| exp177 contacts-v1 step 10,000 | 10,000 | 554/554 | 0.0239 | 0.0199 | 0.5863 | 0.5622 |
| V2 step 22,000 (28% schedule) | 20,000 | 554/554 | **0.4691** | **0.4028** | **0.8886** | **0.8549** |
| exp177 contacts-v1 step 20,000 (28% schedule) | 20,000 | 554/554 | 0.0250 | 0.0223 | 0.5995 | 0.5751 |
| V2 step 30,000 (38% schedule) | 27,273 | 554/554 | **0.5091** | **0.4390** | 0.9054 | 0.8775 |
| exp177 contacts-v1 step 71,359 | 71,359 | 554/554 | **0.5113** | **0.4595** | **0.9246** | **0.9033** |

### Conclusion at 28%: large matched-training win

V2 step 22,000 and exp177 step 20,000 are matched in both estimated protein
exposure and schedule progress: each is approximately 28% through its full
cosine schedule. On the legacy 554-protein universe, V2 leads by +0.4441
all-range and +0.3806 long-range R-precision (18.8× and 18.1× respectively).
The AUC gains are +0.2890 all-range and +0.2798 long-range. V2 step 22,000 has
already reached 92% of exp177-final all-range R-precision and 88% of its
long-range R-precision, despite only 28% of the matched protein exposure.

A later progress check at V2 step 30,000 strengthens the result. At only 38.2%
of the full matched schedule and approximately 27,273 contacts-v1-equivalent
steps, it reaches 0.5091 R/all and 0.4390 R/long: 99.6% and 95.5% of the
exp177-final values. Its AUCs remain lower than exp177 final, at 0.9054 all-range
and 0.8775 long-range versus 0.9246 and 0.9033.

The same conclusion holds on the exp277 split added for these checkpoints:

| subset | proteins | model | R (all) | R (long) | AUC (all) | AUC (long) |
|---|---:|---|---:|---:|---:|---:|
| eval-val | 97 | V2 step 22,000 | **0.3321** | **0.2942** | **0.8467** | **0.8176** |
| eval-val | 97 | exp177 step 20,000 | 0.0219 | 0.0197 | 0.5880 | 0.5604 |
| eval-denovo | 19 | V2 step 22,000 | **0.4383** | **0.3935** | **0.8830** | **0.8427** |
| eval-denovo | 19 | exp177 step 20,000 | 0.0284 | 0.0340 | 0.5866 | 0.5620 |
| eval-val | 97 | V2 step 30,000 | **0.3870** | **0.3435** | **0.8747** | **0.8521** |
| eval-denovo | 19 | V2 step 30,000 | **0.4881** | **0.4224** | **0.9000** | **0.8634** |

All 670 proteins were scored at both V2 checkpoints. Step 30,000 emitted no
malformed rollouts among 67,000 samples. Step 22,000 likewise emitted no
malformed rollouts; one de-novo protein (`8qaf_A`) received an all-zero vote
matrix. The contacts-v1 control had five length-truncated samples, whose valid
prefixes were retained under the same permissive rollout semantics.

As in the canonical contacts-v1 worker, decoding is permissive: valid in-range
pairs vote while malformed or out-of-universe statements are ignored rather
than causing the whole rollout to be discarded. All evaluated V2 checkpoints
produced nonzero votes for every legacy target. Full per-protein rows and aggregate metrics
for that curve are in `data/rprecision_comparison_rows.csv.gz` and
`data/rprecision_comparison_summary.csv`. The 670-protein matched comparison is
in `data/rprecision_step22000_u670_rows.csv.gz` and
`data/rprecision_step22000_u670_subset_summary.csv`. The corresponding step-30,000
files use `step30000` in their names; per-protein runtime and worker metadata
are in the `data/timings_*_u670.csv` files.

## Storage

V2 conversion and cache creation run as a federated root job on
`cw-us-east-02a`, co-located with their existing
`s3://marin-us-east-02a/` source and training artifacts. A Marin-local Zephyr
job has no credentials for this private S3 store and must not be used for this
pipeline.

## Next steps

1. Continue the 78,500-step V2 run and evaluate later matched checkpoints.
2. Measure whether strict grammar-constrained decoding improves V2 further.

## Files

- `compute_contacts_delta_stream_documents.py` — V2 document conversion.
- `build_packed_delta_stream_cache.py` — fixed-length packed LM cache builder.
- `dispatch_delta_stream_full.py` — distributed training dispatcher.
- `build_summary.py` / `summary_narrative.md` — experiment summary deck.
