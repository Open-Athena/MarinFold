# exp218 — contacts-v1 as a bidirectional protein language model

**Issue:** [#218](https://github.com/Open-Athena/MarinFold/issues/218) · **Kind:** `evals`

Every contacts-v1 document opens with the protein's residues as a *randomly
ordered* list of `<pN> <AA>` statements. Prompt the model with every residue but
one and it returns a distribution over the one left out — the same conditional
ESM-1v and ESM-2 compute with a mask token, and the object ProteinGym's
zero-shot DMS benchmark scores.

This experiment asks whether that conditional is any good, and reports the
answer on the community-standard scale.

## Why this is not a party trick

Roughly **half of contacts-v1 pretraining is sequence modelling**. From exp53:
4,213,203 documents, ~4.67 B train tokens, mean 1,131 tokens/doc, mean ~200
contacts/doc. Contacts cost 3 tokens each and the frame is 4, so the sequence
section is ≈527 tokens — **~47% of all training tokens**, half of which are the
`<AA>` targets. exp199's 152 B-token run therefore spent on the order of **35 B
tokens predicting amino acids from other amino acids**.

Two further facts make the readout exactly on-distribution rather than a
reinterpretation:

- **No loss masking.** `marinfold_models` trains a plain LM loss over packed
  documents; the sequence section is inside it.
- **The default model was trained as an any-order model on purpose.** exp199's
  winner is its `-aug` arm, and the exp166 sweep defines that as *"every packed
  example receives a fresh deterministic re-permutation of the two-token
  `<pN> <AA>` sequence statements"*, ramped from probability 0 to 1 over
  training. Under a uniform shuffle any statement can be last, so asking for
  `P(residue i | all the others)` asks something the model saw directly.

## The estimator: one pass per ordering, not one per residue

A document is an ordinary causal-LM sequence, so a single teacher-forced pass
yields the conditional at *every* residue at once: the slot holding a `<pN>`
token already carries `P(amino acid at N | every statement before it)`. An
`L`-residue protein costs one forward pass per ordering rather than `L`.

What each residue is conditioned on varies with where its statement landed in
the shuffle, so the readout also returns a per-slot **context size**. That turns
what would be a nuisance into two knobs no masked LM has: how many orderings to
ensemble, and how much context to require.

The whole 212-assay benchmark is ~29 M tokens at K=200 — under two hours on one
A5000. There is no cost argument for scoring a subset, and a strong
methodological argument against it.

## Preflight: three ways this could have produced confident nonsense

1. **The aggregation is exact.** ProteinGym's headline is not a mean over
   assays — it is mean within UniProt id, then within function category, then
   over the five categories. Our implementation reproduces every published
   leaderboard number from their own per-assay file to ±0.0005.
2. **Mutation indexing is right.** Across 212 assays and 2,438,361 variants, the
   stated wild-type letter matches `target_seq` at the stated position every
   time — zero mismatches.
3. **The rope config survives the load.** The bucket copy of exp199 carries both
   transformers-4 and transformers-5 rope, so the silent theta-10000 fallback
   (worth 0.76 nats/token) does not fire.

## Predictions, recorded before the run

Headline ~0.35 average Spearman (80% interval 0.22–0.45), which is the
ESM2-150M-to-650M band. Below 0.19 means the readout is broken, not the model.
The per-category profile should tilt toward Stability the way a structure-trained
model's does. Both knobs should help monotonically.

## Status

Readout primitive landed with tests. Harness built and preflighted. Phase 0 and
the Phase 1 sweep are in flight.

## Results

_(pending)_
