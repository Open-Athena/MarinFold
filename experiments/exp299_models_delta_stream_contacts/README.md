---
marinfold_experiment:
  issue: 299
  title: "Train compact delta-stream contact documents"
  kind: models
  branch: exp299/delta-stream-contacts
---

# Train compact delta-stream contact documents

## Question

Can we train residue-level contact predictors as ordinary next-token language
models without a separate structured contact head, and still get useful contact
rankings early in training?

## Delta-stream contact documents

This experiment was split out from the earlier soft-target/contact-slot work
once the compact delta-stream representation became the active direction. Unlike
explicit packed contact-slot targets, delta-stream documents train with ordinary
next-token cross entropy and do not require a separate structured contact head.

For each residue `i`, the document contains the amino-acid token followed by the
signed sequence offsets for residues in contact with `i`, terminated by a stop
token:

```text
AA_i  DELTA(i -> j_1)  DELTA(i -> j_2) ... STOP
```

Residues with no contacts are simply:

```text
AA_i STOP
```

Contacts are ordered by signed delta, left-to-right. The format is deterministic
and compact enough for the full training corpus.

Vocab/layout used for the current run:

- `STOP_TOKEN_ID = 0`
- amino-acid tokens start at `1`
- delta tokens start at `32`
- `MAX_ABS_DELTA = 1024`
- `vocab_size = 2080`
- full dataset maximum protein length is `1000`

The pilot conversion was launched before this work was split into issue #299,
so the existing large artifacts still live under the original
`exp177_contacts_delta_stream_v1` object-store prefix. Future reruns from this
experiment's scripts default to an `exp299_contacts_delta_stream_v1` prefix.

The full delta-stream document conversion produced:

- documents: `s3://marin-us-east-02a/protein-structure/MarinFold/exp177_contacts_delta_stream_v1/documents/2026.09.15.1/`
- rows: `66,759,922`
- token count max: `6080`
- no row groups over `8096` tokens

For Levanter training, the variable-length documents are packed into fixed
`8192`-token examples with `input_ids` and `loss_weights`:

- packed cache: `s3://marin-us-east-02a/protein-structure/MarinFold/exp177_contacts_delta_stream_v1/packed_cache/2026.09.15.1`
- train examples: `7,056,450`
- train valid tokens: `57,792,595,992`
- validation examples: `2,107`
- validation valid tokens: `17,253,388`

## Training run

The first full delta-stream run is a 1.5B Qwen-style LM trained with ordinary
next-token CE on the packed cache:

- W&B/run name: [`protein-delta-stream-1_5b-1e-3-2x4gb200-a09-packed`](https://wandb.ai/open-athena/MarinFold/runs/protein-delta-stream-1_5b-1e-3-2x4gb200-a09-packed)
- global batch: `128`
- sequence length: `8192`
- planned steps: `12,000`
- latest active resume driver: `/zack/protein-delta-stream-1_5b-1e-3-2x4gb200-a10-resume-driver`

The run reached roughly step 3.25k before a checkpoint `PutObject` timeout in
CoreWeave object storage, then resumed successfully from step 3049 after bucket
cleanup. Early validation loss continued improving after resume.

## Contact scoring / R-precision

R-precision is computed offline from next-token log-probabilities rather than
from decoded contact sets. For a candidate pair `(i, j)`, the delta-stream LM
provides two directed pieces of evidence:

- `p(i -> j) = P(delta = j - i after residue i)`
- `p(j -> i) = P(delta = i - j after residue j)`

We currently report two undirected reductions:

- **mean/geomean readout** (`score_mean`): rewards bidirectional agreement.
- **max/either-side readout** (`score_max`): counts one-sided evidence if either
  residue strongly points to the other.

The preliminary comparison below uses the delta-stream native checkpoint at
`step-4000` and the closest readily scoreable early contacts-v1 baseline found
for this experiment family, `exp177-next-token-cw-r60-4x8-bs256-10pct` at
`step-3567`. The two runs do not have exactly identical token counts, but they
are both early checkpoints and much closer than the mature 70k-step contacts-v1
exports.

Scores are on the 533-protein subset common to both eval outputs. The
delta-stream scorer skipped 21 proteins containing `X`/unknown residues not in
its vocabulary; the contacts-v1 baseline scored all 554.

| model/readout | all R | short R | medium R | long R |
|---|---:|---:|---:|---:|
| delta-stream step-4000, mean/geomean | **0.0473** | 0.0683 | **0.0372** | **0.0300** |
| delta-stream step-4000, max/either-side | 0.0416 | **0.0969** | 0.0299 | 0.0272 |
| contacts-v1 r60 step-3567 baseline | 0.0297 | 0.0292 | 0.0232 | 0.0222 |

Interpretation:

- The delta-stream checkpoint is already ahead of the comparable early
  contacts-v1 baseline on all/medium/long contacts under the mean/geomean
  readout.
- The max/either-side readout helps short-range contacts substantially but does
  not improve all/medium/long. This suggests the early long-range weakness is
  not just a failure to combine one-sided evidence; bidirectional agreement is
  the better readout at this checkpoint.
- Absolute R-precision is still far below mature contacts-v1 runs, so the main
  open question is whether delta-stream continues scaling with training.

Small summary CSV: [`data/rprecision_preliminary.csv`](data/rprecision_preliminary.csv).

Score artifacts:

- delta step-4000 mean/max scores: `s3://marin-us-east-02a/protein-structure/MarinFold/exp177_contacts_delta_stream_v1/evals/rprecision/delta_step4000_meanmax_h100_rno2a_a01/scores`
- contacts-v1 r60 step-3567 scores: `s3://marin-us-east-02a/protein-structure/MarinFold/exp177_contacts_delta_stream_v1/evals/rprecision/exp177_r60_step3567_h100_rno2a_a01/scores`

## Files

- `compute_contacts_delta_stream_documents.py` — build deterministic
  delta-stream documents.
- `build_packed_delta_stream_cache.py` — convert variable-length documents into
  fixed-length packed LM examples.
- `train_delta_stream_full.py` / `dispatch_delta_stream_full.py` — full
  distributed training entry points.
- `score_delta_stream_checkpoint.py` — log-probability R-precision scorer for
  delta-stream checkpoints, emitting `score_mean` and `score_max`.
- `score_contacts_v1_baseline_vllm.py` — vLLM scorer for contacts-v1 HF exports.
- `compare_rprecision.py` — aggregate per-protein scores into R-precision
  tables.

## Next steps

1. Let the active 12k-step delta-stream training run finish.
2. Score later delta-stream checkpoints with both readouts.
3. Compare against later contacts-v1 checkpoints where the training-token budget
   is closer, including rope-delta contacts-v1 controls.
4. Decide whether to keep only delta-stream or revisit a structured contact head
   after seeing the full training curve.
