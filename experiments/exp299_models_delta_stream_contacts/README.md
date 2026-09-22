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
| V2 full-run step 4,000 | 3,636 | 554/554 | 0.0823 | 0.0520 | 0.6639 | 0.6181 |
| V2 full-run step 10,000 | 9,091 | 554/554 | 0.3251 | 0.2486 | 0.8313 | 0.7848 |
| V2 full-run step 16,000 | 14,545 | 554/554 | 0.4525 | 0.3812 | 0.8806 | 0.8436 |
| V2 full-run step 20,000 | 18,182 | 554/554 | 0.4685 | 0.3920 | 0.8934 | 0.8591 |
| exp177 contacts-v1 step 10,000 | 10,000 | 554/554 | 0.0239 | 0.0199 | 0.5863 | 0.5622 |
| V2 step 22,000 (28% schedule) | 20,000 | 554/554 | **0.4691** | **0.4028** | **0.8886** | **0.8549** |
| exp177 contacts-v1 step 20,000 (28% schedule) | 20,000 | 554/554 | 0.0250 | 0.0223 | 0.5995 | 0.5751 |
| V2 step 30,000 (38% schedule) | 27,273 | 554/554 | 0.5091 | 0.4390 | 0.9054 | 0.8775 |
| V2 step 36,000 (46% schedule) | 32,727 | 554/554 | **0.5130** | 0.4432 | 0.9046 | 0.8750 |
| V2 step 42,000 (54% schedule) | 38,182 | 554/554 | 0.4761 | 0.4055 | 0.8855 | 0.8526 |
| V2 step 48,000 (61% schedule) | 43,636 | 554/554 | **0.5418** | **0.4748** | 0.9131 | 0.8878 |
| V2 step 54,000 (69% schedule) | 49,091 | 554/554 | **0.5449** | **0.4799** | 0.9149 | 0.8888 |
| V2 step 60,000 (76% schedule) | 54,545 | 554/554 | **0.5464** | **0.4804** | 0.9142 | 0.8894 |
| V2 step 66,000 (84% schedule) | 60,000 | 554/554 | **0.5611** | **0.4963** | 0.9212 | 0.8986 |
| V2 step 72,000 (92% schedule) | 65,455 | 554/554 | **0.5679** | **0.5053** | 0.9217 | 0.8987 |
| V2 step 78,000 (99% schedule) | 70,909 | 554/554 | **0.5755** | **0.5130** | 0.9227 | 0.9010 |
| V2 step 78,499 (final) | 71,363 | 554/554 | **0.5763** | **0.5160** | 0.9245 | 0.9028 |
| exp177 contacts-v1 step 71,359 | 71,359 | 554/554 | 0.5113 | 0.4595 | **0.9246** | **0.9033** |

### Notebook-ready training curves

All retained exp177 HF checkpoints have now been evaluated on `legacy_554`:

| exp177 step | R (all) | R (long) | AUC (all) | AUC (long) |
|---:|---:|---:|---:|---:|
| 10,000 | 0.0239 | 0.0199 | 0.5863 | 0.5622 |
| 20,000 | 0.0250 | 0.0223 | 0.5995 | 0.5751 |
| 30,000 | 0.4112 | 0.3467 | 0.8936 | 0.8640 |
| 40,000 | 0.4538 | 0.3955 | 0.9066 | 0.8808 |
| 50,000 | 0.0193 | 0.0175 | 0.5670 | 0.5519 |
| 60,000 | 0.4903 | 0.4359 | 0.9163 | 0.8927 |
| 70,000 | 0.5113 | 0.4609 | 0.9229 | 0.9014 |
| 71,359 | 0.5113 | 0.4595 | 0.9246 | 0.9033 |

The step-50,000 collapse is present in the exported checkpoint and was reproduced
by a complete independent second 100-rollout evaluation; it is not a plotting
or aggregation error. The canonical run had 113 length-capped rollouts, whose
valid prefixes were retained.

Notebook-ready local files are:

- `data/legacy554_rprecision_training_curve_wide.csv`: one row per checkpoint.
- `data/legacy554_rprecision_training_curve.csv`: tidy aggregate metrics.
- `data/legacy554_rprecision_training_curve_rows.csv.gz`: per-protein values for
  error bars and paired analyses.
- `data/eval_denovo_rprecision_training_curve_wide.csv`: matching one-row-per-
  checkpoint curve on the 19-protein exp277 `eval-denovo` set.
- `data/eval_denovo_rprecision_training_curve.csv`: tidy `eval-denovo`
  aggregates with SEM and protein counts.
- `data/eval_denovo_rprecision_training_curve_rows.csv.gz`: per-protein
  `eval-denovo` values.
- `data/timings_eval_denovo_training_curve.csv`: per-protein timing and worker
  provenance for every checkpoint in the `eval-denovo` curve.

The delta V2 pilot and full 78,500-step run have separate `run` labels so a
notebook cannot accidentally draw them as one continuous schedule. The full-run
curve now contains steps 4k, 10k, 16k, 20k, 22k, 30k, 36k, 42k, 48k, 54k,
60k, 66k, 72k, 78k, and final step 78,499. The `eval-denovo` curve contains
those same full-run checkpoints and all exp177
checkpoints in the legacy curve (10k, 20k, 30k, 40k, 50k, 60k, 70k, and 71,359).

### Final training-run statistics

The fresh mature run completed successfully at step 78,499 on 2026-09-22. It
trained for 112.9 wall-clock hours on 2 nodes × 4 GB200 GPUs with global batch
128 and sequence length 8,192. W&B reports 82.31 billion packed tokens and mean
throughput 205,396 tokens/s (5.105 seconds/step). This corresponds to about
94.73 million protein-document exposures, 1.42 passes over the packed cache,
and approximately 71,363 contacts-v1-equivalent steps—effectively identical to
exp177 final at step 71,359. Final train loss was 0.9190 and final/best native
validation loss was 0.91425. The learning rate ended at 1.0e-4 after peaking at
1.0e-3.

### Exp277-scale follow-up

The next run applies the successful V2 serialization to the complete exp277
first-epoch corpus: 232,090,905 decontaminated native and ProteinMPNN-redesigned
protein documents. `convert_exp277_caches_to_delta_stream.py` reads the four
existing token caches in `marin-us-east-02a`, recovers each document's canonical
N-to-C sequence and undirected contact set, and writes V2 token IDs without
re-fetching structures or recomputing contacts. Conversion is fail-loud and can
strictly parse every emitted V2 document back to the same sequence/contact set.

The original 2,080-token V2 vocabulary represents offsets through ±1,024. The
exp277 source format permits chains through 2,000 residues, so the follow-up
extends V2 by appending tokens for offsets ±1,025 through ±1,999. All original
IDs remain unchanged; no source contact is dropped. A full census and fresh
packed-example count will determine the one-epoch optimizer-step count. The
training target is one finite pass over all source documents with exp277's
batch-128, 8,192-context, LR-1e-3, WD-0.2 WSD recipe, rather than blindly
copying exp277's contacts-v1 step count.

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
[-0.0490, -0.0268] for R/long. Step 48,000 then recovers to a new high of
0.5418 R/all and 0.4748 R/long, exceeding exp177 final on both metrics. The
paired-bootstrap V2-minus-exp177 intervals are [0.0207, 0.0407] for R/all and
[0.0044, 0.0265] for R/long. Steps 54,000 and 60,000 hold and slightly extend
that recovery: step 60,000 reaches 0.5464 R/all and 0.4804 R/long. Performance
then improves through the end of training. Final step 78,499 reaches 0.5763
R/all and 0.5160 R/long, exceeding exp177 final by 0.0651 and 0.0565. Paired
protein-bootstrap 95% intervals are [0.0547, 0.0757] and [0.0450, 0.0684]. Its
AUCs close the remaining gap: 0.9245 all-range and 0.9028 long-range versus
0.9246 and 0.9033 for exp177 final; paired intervals include zero for both.

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
| eval-val | 97 | V2 step 48,000 | **0.4537** | **0.4197** | **0.8969** | **0.8778** |
| eval-denovo | 19 | V2 step 48,000 | **0.5272** | **0.4595** | **0.9282** | **0.8936** |
| eval-val | 97 | V2 step 54,000 | 0.4555 | 0.4225 | 0.8964 | 0.8744 |
| eval-denovo | 19 | V2 step 54,000 | 0.5116 | 0.4303 | **0.9299** | 0.8714 |
| eval-val | 97 | V2 step 60,000 | **0.4665** | **0.4345** | **0.8969** | **0.8772** |
| eval-denovo | 19 | V2 step 60,000 | 0.5186 | **0.4758** | 0.9211 | **0.8924** |
| eval-val | 97 | V2 step 66,000 | 0.4940 | 0.4621 | 0.9042 | 0.8868 |
| eval-denovo | 19 | V2 step 66,000 | 0.5519 | 0.4594 | 0.9400 | 0.8852 |
| eval-val | 97 | V2 step 72,000 | 0.5008 | 0.4699 | 0.9110 | 0.8953 |
| eval-denovo | 19 | V2 step 72,000 | **0.5534** | **0.4871** | 0.9391 | 0.8901 |
| eval-val | 97 | V2 step 78,000 | **0.5150** | **0.4887** | **0.9134** | **0.8982** |
| eval-denovo | 19 | V2 step 78,000 | 0.5422 | 0.4756 | **0.9427** | **0.9055** |
| eval-val | 97 | V2 step 78,499 (final) | 0.5147 | 0.4842 | 0.9107 | 0.8928 |
| eval-denovo | 19 | V2 step 78,499 (final) | 0.5494 | 0.4779 | 0.9376 | 0.8933 |

All 670 proteins were scored at every listed V2 checkpoint. Steps 42,000 and
48,000 emitted no malformed rollouts among 67,000 samples, as did the
step-42,000 independent rerun. Earlier checkpoints likewise emitted no
malformed rollouts; one de-novo protein
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
later checkpoint files follow the same `rprecision_step<N>_u670_*` naming;
per-protein runtime and worker metadata are in the `data/timings_*_u670.csv`
files.

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
