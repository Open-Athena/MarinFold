## exp222 — an experimental-PDB contacts-v1 corpus

Every contacts-v1 corpus so far is **predicted** structure: exp53/exp105/exp132
are AFDB (AlphaFold2), exp139/exp155 are ESM-Atlas (ESMFold2 distillation). The
model has never trained on a measurement, and it has never seen a **complex** —
contacts-v1 has been single-chain since the format was written.

This experiment closes both gaps with one curation pass over the PDB, following
Protenix/AF3's training-data recipe with a date cutoff of **2021-09-30**.

## Two corpora

| Corpus | One document per | Analyzed as |
| --- | --- | --- |
| `contacts_v1_pdb_monomers` | protein chain of the asymmetric unit | that chain **alone** |
| `contacts_v1_pdb_multimers` | entry whose assembly 1 has ≥ 2 protein chains | the **whole assembly** |

Pulling a chain out and analyzing it in isolation is not a shortcut — it is how
the AFDB corpus and the exp74/exp89 eval ground truth are built. Analyzing it in
place would bury its interface and change which residues pyconfind calls
solvent-accessible, making a PDB monomer document incomparable to an AFDB one.

The subsets are disjoint, so "with multimers" is just the union of both prefixes.

## The format extension

`SPEC.md` has anticipated multi-chain documents since the format was written. The
implementation: *k* chains share the one 2000-index wrap-around ring. Shuffle them
into a random ring order, split the leftover slack into *k* random gaps, draw a
global rotation offset, and walk. Every chain gets one unbroken, disjoint run of
indices and its own `<n-term>` / `<c-term>` pair.

Two decisions carry the weight:

- **`min_seq_separation` is intra-chain only.** "How far apart in the chain" is
  undefined across a chain break. Applying the ≥ 6 rule there would delete most of
  the interface — the entire content of a multimer document.
- **No new doc-type token.** The chain count is fully visible in the prompt, so
  unlike #175's retraction mode there is nothing for the model to marginalize over.

Single-chain documents are byte-identical to the pre-change generator: with *k* = 1
the shuffle and the gap draw are skipped and the only random number is the same
`randrange(2000)` that used to pick the n-terminal index.

## Results

_(Filled in when the full run lands.)_

## Conclusion

_(Filled in when the full run lands.)_
