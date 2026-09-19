## Sequence before contacts

The delta-stream format places the whole amino-acid sequence in a causal prefix before any contact target:

`DOC_START AA_0 ... AA_(L-1) CONTACTS_BEGIN DELTA* STOP ... DOC_END`.

The contact suffix contains one signed-delta segment per residue, in sequence order. The model can therefore see every amino acid before predicting any contact, matching the conditioning structure used by contacts-v1.

## Evaluation

The reference metric is exp82's rollout-vote evaluation: sample 100 sequence-conditioned contact suffixes, vote emitted pairs, and run the unchanged exp82/exp89 candidate-universe and R-precision code.

## Canonical result at 28% schedule progress

We evaluated V2 step 22,000 and exp177 contacts-v1 step 20,000 with 100
sequence-conditioned rollouts and unchanged exp82/exp89 metrics. V2 uses about
10% more tokens per protein, so these checkpoints match both estimated protein
exposure and schedule progress: each is approximately 28% through its full
cosine schedule.

| subset | proteins | model | R/all | R/long | AUC/all |
|---|---:|---|---:|---:|---:|
| legacy | 554 | V2 step 22,000 | **0.4691** | **0.4028** | **0.8886** |
| legacy | 554 | exp177 step 20,000 | 0.0250 | 0.0223 | 0.5995 |
| eval-val | 97 | V2 step 22,000 | **0.3321** | **0.2942** | **0.8467** |
| eval-val | 97 | exp177 step 20,000 | 0.0219 | 0.0197 | 0.5880 |
| eval-denovo | 19 | V2 step 22,000 | **0.4383** | **0.3935** | **0.8830** |
| eval-denovo | 19 | exp177 step 20,000 | 0.0284 | 0.0340 | 0.5866 |

On the legacy universe, the matched V2 checkpoint is 18.8× higher on all-range
R-precision and 18.1× higher on long-range R-precision. It has already reached
92% of exp177-final all-range R-precision at only 28% of matched exposure. The
large advantage also transfers to the 19 held-out de-novo proteins, so the
result is not confined to the older evaluation universe.

All 670 proteins were scored. The V2 run produced no malformed rollouts among
67,000 samples.

## Next steps

Continue evaluating later matched checkpoints and measure strict
grammar-constrained decoding separately.
