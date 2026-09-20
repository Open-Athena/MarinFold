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
| V2 step 30,000 (38% schedule) | 27,273 | 554/554 | 0.5091 | 0.4390 | 0.9054 | 0.8775 |
| V2 step 36,000 (46% schedule) | 32,727 | 554/554 | **0.5130** | 0.4432 | 0.9046 | 0.8750 |
| V2 step 42,000 (54% schedule) | 38,182 | 554/554 | 0.4761 | 0.4055 | 0.8855 | 0.8526 |
| exp177 contacts-v1 step 71,359 | 71,359 | 554/554 | 0.5113 | **0.4595** | **0.9246** | **0.9033** |

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
and 0.8775 long-range versus 0.9246 and 0.9033. At step 36,000, V2 narrowly
passes exp177 final on all-range R-precision, 0.5130 versus 0.5113, despite only
approximately 32,727 contacts-v1-equivalent steps. This all-range crossover is
a narrow point-estimate lead: the paired protein-bootstrap 95% interval for
the difference is [-0.0067, 0.0103]. It has not passed the contacts-v1 reference
on long-range R-precision or either AUC: R/long is 0.4432 versus 0.4595, with a
paired-bootstrap difference interval of [-0.0260, -0.0066]. Step 42,000 then
regresses to 0.4761 R/all and 0.4055 R/long despite lower validation CE. A full
independent 100-rollout rerun reproduced the drop (0.4756 and 0.4058), ruling
out a single rollout sample as its cause. Relative to step 36,000, paired
protein-bootstrap difference intervals are [-0.0475, -0.0269] for R/all and
[-0.0490, -0.0268] for R/long.

The same conclusion holds on the exp277 split added for these checkpoints:

| subset | proteins | model | R (all) | R (long) | AUC (all) | AUC (long) |
|---|---:|---|---:|---:|---:|---:|
| eval-val | 97 | V2 step 22,000 | **0.3321** | **0.2942** | **0.8467** | **0.8176** |
| eval-val | 97 | exp177 step 20,000 | 0.0219 | 0.0197 | 0.5880 | 0.5604 |
| eval-denovo | 19 | V2 step 22,000 | **0.4383** | **0.3935** | **0.8830** | **0.8427** |
| eval-denovo | 19 | exp177 step 20,000 | 0.0284 | 0.0340 | 0.5866 | 0.5620 |
| eval-val | 97 | V2 step 30,000 | **0.3870** | **0.3435** | **0.8747** | **0.8521** |
| eval-denovo | 19 | V2 step 30,000 | **0.4881** | **0.4224** | **0.9000** | **0.8634** |
| eval-val | 97 | V2 step 36,000 | **0.4080** | **0.3773** | **0.8827** | **0.8607** |
| eval-denovo | 19 | V2 step 36,000 | **0.5261** | **0.4626** | **0.9092** | **0.8631** |
| eval-val | 97 | V2 step 42,000 | 0.4122 | 0.3744 | 0.8800 | 0.8587 |
| eval-denovo | 19 | V2 step 42,000 | 0.5087 | 0.4307 | 0.9055 | 0.8521 |

All 670 proteins were scored at all four V2 checkpoints. Step 42,000 emitted no
malformed rollouts among 67,000 samples, as did its independent rerun. Earlier
checkpoints likewise emitted no malformed rollouts; one de-novo protein
(`8qaf_A`) received an all-zero vote matrix at steps 22,000 and 36,000. The
contacts-v1 control had five length-truncated samples, whose valid prefixes
were retained under the same permissive rollout semantics.

As in the canonical contacts-v1 worker, decoding is permissive: valid in-range
pairs vote while malformed or out-of-universe statements are ignored rather
than causing the whole rollout to be discarded. All evaluated V2 checkpoints
produced nonzero votes for every legacy target. Full per-protein rows and aggregate metrics
for that curve are in `data/rprecision_comparison_rows.csv.gz` and
`data/rprecision_comparison_summary.csv`. The 670-protein matched comparison is
in `data/rprecision_step22000_u670_rows.csv.gz` and
`data/rprecision_step22000_u670_subset_summary.csv`. The corresponding
step-30,000, step-36,000, and step-42,000 files use `step30000`, `step36000`, and
`step42000` in their names; per-protein runtime and worker metadata are in the
`data/timings_*_u670.csv` files.

## Paired teacher-forced CE by document phase

To separate serialization difficulty from contact quality, we serialized the
same 670 proteins used by the canonical R-precision evaluation in each model's
native format, then scored every next-token target. Rows are paired exactly by
`(dataset, stem)`:

| checkpoint | proteins | AA tokens | contact tokens | AA CE | contact CE |
|---|---:|---:|---:|---:|---:|
| exp177 contacts-v1 final (step 71,359) | 670 | 131,874 | 356,820 | 2.6024 | 2.2301 |
| V2 step 32,000 | 670 | 131,874 | 369,754 | **2.2575** | **0.4086** |

The paired result confirms that contact prediction has lower token CE than
amino-acid prediction in both formats, with a much larger gap for V2. The V2
contact suffix is 1.849 nats/token below its AA prefix, versus 0.372 for
contacts-v1. This establishes that easier serialization contributes
substantially to V2's low aggregate CE. It does not explain away contact
learning: the canonical rollout result independently shows high R-precision.

Contact token counts differ because the formats encode the same contact graph
differently: contacts-v1 uses one three-token statement per undirected contact,
while V2 emits directed deltas plus one `STOP` per residue. Protein-level raw
log-probability sums therefore need to be compared with the appropriate token
normalization rather than as interchangeable sequence likelihoods.

The notebook-ready per-protein table is
`data/teacher_forced_ce_paired_u670.csv.gz`; it includes phase token counts,
mean CEs, and summed log probabilities for both models. Aggregate subset results
are in `data/teacher_forced_ce_paired_u670_summary.csv`. Full ragged per-token
arrays (`target_token_ids`, `role_ids`, and `token_ce`) are under
`s3://marin-us-east-02a/protein-structure/MarinFold/exp299_contacts_delta_stream_v2_sequence_prefix/eval/teacher_forced_ce/paired-u670-v1/`.

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
