<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide.
     The renderer is textwrap.fill on plain text: markdown tables, bold
     markers and bullet lists all render literally, and lines within a
     paragraph get reflowed. So keep this file plain prose, one idea per
     paragraph. -->

## exp222 — an experimental-PDB contacts-v1 corpus

Every contacts-v1 corpus so far is PREDICTED structure: exp53/exp105/exp132 are AFDB (AlphaFold2), exp139/exp155 are ESM-Atlas (ESMFold2 distillation). The model has never trained on a measurement, and it has never seen a complex — contacts-v1 has been single-chain since the format was written.

This experiment closes both gaps with one curation pass over the PDB, following Protenix/AF3's training-data recipe with a date cutoff of 2021-09-30.

## What came out: 688,519 documents, 995 M tokens

Monomers: 602,859 documents, 695 M tokens, median 205 residues, one chain each.

Multimers: 85,660 documents, 300 M tokens, median 562 residues, mean 2.93 chains (max 60), and 9.19 M contacts — 15.1% of the total — that cross a chain boundary.

Those 9.19 M inter-chain contacts are a class of contact no previous MarinFold corpus contained at all. The multimer set is dimer-dominated (62% two chains), so it teaches interfaces rather than large-assembly organisation.

Fine-tuning scale by design: the AFDB corpus of exp105 is 138 B tokens.

## Two corpora, and why the monomer one is built the way it is

The monomer corpus is one document per protein chain of the asymmetric unit, with that chain analyzed ALONE. The multimer corpus is one document per entry whose biological assembly 1 holds two or more protein chains, analyzed as the whole assembly.

Pulling a chain out and analyzing it in isolation is not a shortcut — it is how the AFDB corpus and the exp74/exp89 eval ground truth are built. Analyzing it in place would bury its interface and change which residues pyconfind calls solvent-accessible, making a PDB monomer document incomparable to an AFDB one.

The two subsets are disjoint, so a training mixture that wants both just reads both prefixes.

## The format extension

SPEC.md has anticipated multi-chain documents since the format was written. The implementation: k chains share the one 2000-index wrap-around ring. Shuffle them into a random ring order, split the leftover slack into k random gaps, draw a global rotation offset, and walk. Every chain gets one unbroken, disjoint run of indices and its own n-term / c-term pair.

Decision one: the minimum sequence separation is intra-chain only. "How far apart in the chain" is undefined across a chain break, and applying the 6-residue rule there would delete most of the interface — the entire content of a multimer document.

Decision two: no new doc-type token. The chain count is fully visible in the prompt, so unlike #175's retraction mode there is nothing for the model to marginalize over, and the tokenizer is untouched.

Single-chain documents are byte-identical to the pre-change generator: with k = 1 the shuffle and the gap draw are both skipped and the only random number is the same randrange(2000) that used to pick the n-terminal index. Every existing checkpoint stays token-compatible.

## Nothing was dropped silently

195,858 entries became 177,710 selected and then 688,519 documents, and the ledger's error column is empty.

Every rejection is attributed to a named filter: 13,321 released after the cutoff, 763 below 9 A resolution, 3,881 with no protein entity, 183 held out as eval set; then at the chain level 28,775 non-protein, 2,851 all-unknown, 733 over 2000 residues, 354 with a CA break, 186 clashing, 55 too short; and 10,530 complexes too large for the index ring. The 339 chains (0.06%) that pass curation but still fail to serialize are counted too.

Validation on 36,505 sampled documents: all clean. Chains land on contiguous, disjoint index runs that exactly cover the assigned positions, with one terminus pair each. For 40 multimers, a fresh pyconfind run on the rebuilt assembly reproduces the document's interface contacts exactly.

## Leakage: no worse than what the model already trained on

The 552 eval PDB entries are excluded by id, and none appear in either corpus. Exact resolved-sequence matches from other entries: 44 in the monomer corpus (0.007%), 1 in the multimer corpus.

Residual homology, measured from the eval side so it is comparable to #213: 50.2% of eval entries have a 40%-identity homolog in the monomer corpus, against 58% for exp199's AFDB training data. Slightly LESS exposed than the status quo — and #213 established that this overlap does not inflate the score.

## Conclusion, and one trap to carry forward

Both corpora exist, round-trip cleanly, and are published to the public open-athena/MarinFold bucket with the tokenizer co-located.

The trap: global_plddt means the opposite thing here. It is the mean CA B-factor in both corpora, but for AFDB that IS pLDDT (higher is better, 0-100) and here it is a B-factor (lower is better). Any mixture that filters or weights on it must branch on the corpus.

Training is deliberately a separate experiment. Two follow-ups suggest themselves: fine-tune contacts-v1-exp199-1.5B on the monomer corpus (does experimental structure beat predicted?), and a multimer curriculum (can the model place an interface at all?). Each needs its own controls.
