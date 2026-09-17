## Sequence before contacts

The delta-stream format places the whole amino-acid sequence in a causal prefix before any contact target:

`DOC_START AA_0 ... AA_(L-1) CONTACTS_BEGIN DELTA* STOP ... DOC_END`.

The contact suffix contains one signed-delta segment per residue, in sequence order. The model can therefore see every amino acid before predicting any contact, matching the conditioning structure used by contacts-v1.

## Evaluation

The reference metric is exp82's rollout-vote evaluation: sample 100 sequence-conditioned contact suffixes, vote emitted pairs, and run the unchanged exp82/exp89 candidate-universe and R-precision code.

## Preliminary canonical results

The V2 corpus, packed cache, production run, HF exports, and rollout evaluator
are now operational. We evaluated steps 2,000, 4,000, 6,000, and 8,000 over all
554 exp89 targets with the canonical exp82 recipe: 100 sequence-conditioned
rollouts, one undirected vote per pair per rollout, and unchanged exp89 metric
semantics.

| checkpoint | contacts-v1-equivalent protein exposure | R/all | R/long | AUC/all |
|---|---:|---:|---:|---:|
| V2 step 2,000 | step 1,818 | 0.0238 | 0.0192 | 0.5660 |
| V2 step 4,000 | step 3,636 | 0.0540 | 0.0345 | 0.6146 |
| V2 step 6,000 | step 5,455 | 0.0785 | 0.0477 | 0.6520 |
| V2 step 8,000 | step 7,273 | **0.1306** | **0.0877** | **0.7083** |
| exp177 contacts-v1 step 10,000 | step 10,000 | 0.0239 | 0.0199 | 0.5863 |

V2 uses about 10% more tokens per protein, hence the exposure adjustment above.
The earliest retained exp177 export is step 10,000, so there is no exact matched
control for these V2 checkpoints. The mismatch favors exp177: its step-10,000
checkpoint has seen about 38% more proteins than V2 step 8,000. Even so, V2 is
5.5× higher on all-range R-precision and 4.4× higher on long-range R-precision.
This is strong preliminary evidence for an early-training sample-efficiency win.

The fully trained exp177 step-71,359 reference reaches 0.5113 R/all, but has seen
roughly 9.8× as many proteins as V2 step 8,000 and is not a matched comparison.
The current result therefore supports continuing the V2 format; it does not yet
establish the mature-training crossover.

## Next steps

Measure strict grammar-constrained decoding and retain exact
protein-count-matched contacts-v1 exports in future head-to-head runs.
