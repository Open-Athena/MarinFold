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

## Two more corpora you can just train on

PDB redundancy is extreme: the largest 40%-identity group in the raw corpora holds 4,055 near-duplicate documents, so a uniform pass spends most of its gradient on a handful of over-crystallised proteins. Protenix/AF3 fix this with cluster-weighted sampling at training time.

Two deduplicated cuts bake that intent into the data instead, both one representative per cluster and pre-shuffled with a fixed seed so a sequential read is already in random order. No sampler, no subsampling logic.

contacts_v1_pdb_deduped: 72,524 documents / 128 M tokens (41,661 monomers + 30,863 multimers) — use this when you want the model to see interfaces.

contacts_v1_pdb_deduped_monomers: 41,661 documents / 37 M tokens, single-chain only — a drop-in replacement for an AFDB-style contacts-v1 corpus, changing only the provenance of the structures and not the document shape. Its rows are exactly the monomer rows of the mixed cut.

Grouping is by the chain's 40% cluster id for a monomer and by the SORTED TUPLE of its chains' cluster ids for a multimer, so composition and stoichiometry both count — a homodimer, a homotetramer and a heterodimer of the same protein are three different things to learn, not three copies of one. The representative is the best-resolution member.

One-per-cluster is the strictest reading of "deduplicated". build_deduped.py --max-per-cluster N regenerates at any size: cap 2 gives 112k docs / 202 M tokens, cap 5 gives 183k / 322 M, cap 10 gives 251k / 433 M. --from-subsets picks which source corpora feed it.

Monomer and multimer keys live in disjoint namespaces, which is what makes the monomers-only cut exactly the monomer rows of the mixed one. That held only after fixing the sequence-hash fallback to carry the chain count: the hash is over concatenated residues, so a 4-residue chain and two 2-residue chains spelling the same thing collided, and a complex displaced a monomer from the mixed corpus.

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

Validation on every corpus: all sampled documents clean. Chains land on contiguous, disjoint index runs that exactly cover the assigned positions, with one terminus pair each. For 40 multimers, a fresh pyconfind run on the rebuilt assembly reproduces the document's interface contacts exactly.

## Leakage: no worse than what the model already trained on

The 552 eval PDB entries are excluded by id, and none appear in either corpus. Exact resolved-sequence matches from other entries: 44 in the monomer corpus (0.007%), 1 in the multimer corpus.

Residual homology, measured from the eval side so it is comparable to #213: 50.2% of eval entries have a 40%-identity homolog in the monomer corpus, against 58% for exp199's AFDB training data. Slightly LESS exposed than the status quo — and #213 established that this overlap does not inflate the score.

## Conclusion, and one trap to carry forward

All four corpora exist, round-trip cleanly, and are published to the public open-athena/MarinFold bucket, each with the tokenizer co-located and a README rendered on the bucket's web view.

The trap, now defused: global_plddt is 0.0 here and carries no information. The library fills it with the mean CA B-factor, which for AFDB IS pLDDT (higher is better, 0-100) and here is a B-factor (lower is better) — same column, opposite sign, so a mixture filtering on it would have silently kept the good AFDB documents and the bad PDB ones. Zeroed rather than documented-around. Use resolution, which is stored separately.

Training is deliberately a separate experiment. Two follow-ups suggest themselves: fine-tune contacts-v1-exp199-1.5B on contacts_v1_pdb_deduped_monomers (does experimental structure beat predicted? -- that cut changes only the provenance, not the document shape), and a multimer curriculum (can the model place an interface at all?). Each needs its own controls.
