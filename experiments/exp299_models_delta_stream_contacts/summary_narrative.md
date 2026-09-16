## V2: sequence before contacts

The active delta-stream format now places the whole amino-acid sequence in a causal prefix before any contact target:

`DOC_START AA_0 ... AA_(L-1) CONTACTS_BEGIN DELTA* STOP ... DOC_END`.

The contact suffix contains one signed-delta segment per residue, in sequence order. The model can therefore see every amino acid before predicting any contact, matching the conditioning structure used by contacts-v1.

## Why V1 is not a production format

The earlier V1 stream interleaved each residue's amino-acid token and contact deltas. A causal model predicted early-residue contacts before it had seen the future amino acids.

Its conversion and training run remain infrastructure history only. They do not justify a production sweep or a contact-accuracy claim.

## Evaluation reset

The earlier delta R-precision values used an ad-hoc next-token log-probability probe and are withdrawn.

The reference metric is exp82's rollout-vote evaluation: sample 100 sequence-conditioned contact suffixes, vote emitted pairs, and run the unchanged exp82/exp89 candidate-universe and R-precision code.

## Next steps

Build the V2 corpus/cache, implement the V2 contact-suffix rollout worker, validate its score matrices through the canonical reference metric, then launch production-scale V2 sweeps using the exp288 pattern.
