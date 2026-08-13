---
marinfold_experiment:
  issue: 222
  title: 'exp: curate an experimental-PDB contacts-v1 corpus (Protenix-style, <= 2021-09-30) — monomers and multimers'
  kind: data
  branch: claude/pdb-marinfold-curation-59086c
---

# exp: curate an experimental-PDB contacts-v1 corpus (Protenix-style, <= 2021-09-30) — monomers and multimers

**Issue:** [#222](https://github.com/Open-Athena/MarinFold/issues/222) · **Kind:** `data` · **Branch:** `claude/pdb-marinfold-curation-59086c`

## Question

Can we build an **experimental-structure** contacts-v1 corpus from the PDB — curated
the way Protenix/AF3 curate their training set — and use it to fine-tune our best
model (`contacts-v1-exp199-1.5B`, R-precision 0.587)?

Every contacts-v1 corpus so far is *predicted* structure: exp53/exp105/exp132 are
AFDB (AlphaFold2 predictions), exp139/exp155 are ESM-Atlas (ESMFold2 distillation).
The model has never seen an experimental structure in training, and it has never
seen a **complex** — contacts-v1 has been single-chain since day one. Both gaps are
addressable with the same curation pass.

## Hypothesis

This is a data experiment: the deliverable is the corpus, not a measurement. The
belief it rests on is that experimental structures and protein *interfaces* carry
signal the AFDB/ESM-Atlas corpora cannot, because predicted monomer structures are
(a) a model's opinion rather than a measurement and (b) silent about the entire
class of contacts that only exists between chains. Whether that signal transfers to
R-precision is a separate, later experiment.

## Approach

### Deliverables

Two corpora plus the format change that makes the second expressible.

| Corpus | One document per | Analyzed as |
| --- | --- | --- |
| `contacts_v1_pdb_monomers` | protein chain of the asymmetric unit | that chain **alone** (`assembly=None`) |
| `contacts_v1_pdb_multimers` | entry whose assembly 1 has ≥ 2 protein chains | the **whole assembly** |

The monomer convention is deliberate: pulling the chain out and analyzing it in
isolation is exactly how the AFDB training corpus and the exp74/exp89 eval ground
truth are built, so a PDB monomer document is directly comparable to an AFDB one.
Analyzing the chain *in place* would bury its interface and change which residues
pyconfind calls solvent-accessible.

The two subsets are disjoint by construction, so an "includes multimers" mixture is
just the union of the two prefixes.

**Non-protein molecules are ignored entirely** — ligands, nucleic acids, ions,
glycans, waters. contacts-v1 has no vocabulary for them.

### The multimer format extension

Implemented in the kind library (`marinfold/document_structures/contacts_v1/`, see
its `SPEC.md` § *Multiple protein chains*), not here. *k* chains share the one
2000-index wrap-around ring: shuffle them into a random ring order, split the
leftover slack into *k* random gaps (each ≥ `min_chain_gap`), draw a global rotation
offset, and walk. Each chain gets one unbroken, disjoint run of indices and its own
`<n-term>` / `<c-term>` pair.

Two decisions worth stating:

- **`min_seq_separation` is intra-chain only.** "How far apart in the chain" is
  undefined across a chain break; applying the ≥ 6 rule there would delete most of
  the interface, which is the entire content of a multimer document.
- **No new doc-type token.** The chain count is fully visible in the *prompt* (the
  sequence section carries every terminus before `<begin_statements>`), so unlike
  [#175](https://github.com/Open-Athena/MarinFold/issues/175)'s retraction mode
  there is nothing for the model to marginalize over. The tokenizer is untouched.

Single-chain documents are **byte-identical** to the pre-change generator: with
*k* = 1 the shuffle and the gap composition are skipped and the only RNG draw is the
same `rng.randrange(2000)` that used to pick the n-terminal index. Verified two
ways — a unit test on the RNG stream, and a 180-case fingerprint diff of
`build_document` against the pre-change code across five config variants and lengths
2 … 2000.

### Curation

Source: the local RCSB mmCIF mirror at `/data/tim/af3-db/mmcif_files` — 195,858
entries, ~234 GB, a snapshot from ~2022-09, so a superset of everything the cutoff
admits. All work lands under `/data/exp222_pdb_curation/`.

Filters follow [Protenix's `prepare_training_data.md`](https://github.com/bytedance/Protenix/blob/main/docs/prepare_training_data.md),
which follows the AF3 supplement:

| Filter | Setting | Where |
| --- | --- | --- |
| Release date | ≤ **2021-09-30** (initial release, revision-history ordinal 1) | entry |
| Resolution | < **9 Å**; entries reporting none (NMR) are kept | entry |
| Waters / hydrogens / element `X` / all non-protein | removed | entry |
| Chains entirely of unknown residues | removed | chain |
| Chains with no resolved residues | removed | chain |
| Adjacent-numbered Cα–Cα > **10 Å** | chain removed | chain |
| ≥ **1/3** of heavy atoms within **1.7 Å** of another chain | chain removed | chain |
| Residue count | 2 ≤ L ≤ 2000; multimers also need `sum(L) + k ≤ 2000` | chain / entry |

**Deliberate deviation from Protenix.** Their ">20 chains → keep the 20 nearest a
random interface atom" rule exists to cap a 5120-*token* AF3 crop. Our 2000-residue
index ring binds much earlier, so oversized complexes are **dropped, not cropped** —
a cropped complex would emit a document whose `<n-term>`/`<c-term>` set is a lie
about the assembly it claims to describe.

### Leakage control

The 552 PDB entries behind the [#74](https://github.com/Open-Athena/MarinFold/issues/74)/[#78](https://github.com/Open-Athena/MarinFold/issues/78)
contact eval set are excluded **by PDB id** (`data/eval_set_pdb_ids.csv`). On top of
that, `qc.py` measures what remains: exact resolved-sequence matches, and RCSB
40%-identity cluster overlap — the number directly comparable to
[#213](https://github.com/Open-Athena/MarinFold/issues/213), which found the eval
set is already 58% homologous to exp199's AFDB training data *without* the score
being homology-inflated. The measurement is the deliverable; no homology purge.

### Redundancy

Kept, not removed. PDB is enormously redundant, but each copy is a genuinely
different experimental structure. Every passing chain/assembly is emitted and the
metadata carries an RCSB 40% `cluster_ids` list plus a `resolved_seq_sha1`, so
downstream training can weight or subsample — what Protenix/AF3 do with their
`cluster_id` column.

### Pipeline

| Script | Stage |
| --- | --- |
| [`scan_metadata.py`](scan_metadata.py) | header scan of all 195,858 entries → `entries.parquet` (date, resolution, method, entity types, assembly-1 composition). No coordinates, ~40 ms/entry. |
| [`curate.py`](curate.py) | the chain-level filters, as a testable library |
| [`curate_and_generate.py`](curate_and_generate.py) | one coordinate pass: curate → pyconfind → contacts-v1 documents → parquet shards + per-entry ledger |
| [`qc.py`](qc.py) | funnel, corpus statistics, leakage audit, plots |
| [`publish_to_hf.py`](publish_to_hf.py) | push to the public `open-athena/MarinFold` bucket |

Every rejection is named and counted in the ledger, so the funnel from 195,858
entries to the final corpora is attributable line by line — no silent drops.

## Success criteria

- Both corpora generated with a **zero silent-drop** ledger (every rejected entry
  attributable to a named filter), published to the public HF bucket.
- Single-chain generation proven byte-identical to the pre-change generator.
- A multimer document round-trips: parsing it back recovers *k* chains with the
  right lengths and disjoint index ranges, and its inter-chain contacts match a
  direct pyconfind run on the assembly.

## Results

_(Fill in after the run completes.)_

## Conclusion

_(Fill in after results are in.)_
