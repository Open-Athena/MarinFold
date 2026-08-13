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

### One caveat for mixed-corpus training

The `global_plddt` column is filled, as always, with the mean Cα B-factor.
For AFDB that column really is pLDDT (0–100, **higher** is better); for a crystal
structure it is a B-factor (typically 10–100, **lower** is better). The column name
is kept for schema compatibility with the exp53/exp105/exp132 corpora, but the two
are not on the same scale or even the same sign. Any mixture that filters or weights
on `global_plddt` must branch on the corpus.

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

Both corpora are built, validated and published. The whole pipeline is 82 min of
header scan plus 161 min of generation on 60 cores.

### The corpora

| | `contacts_v1_pdb_monomers` | `contacts_v1_pdb_multimers` |
| --- | --- | --- |
| documents | **602,859** | **85,660** |
| distinct PDB entries | 177,468 | 85,660 |
| distinct resolved sequences | 247,675 | 70,756 |
| distinct 40% clusters | 38,572 | 24,075 |
| tokens | 695.5 M | 299.9 M |
| mean tokens / document | 1,154 | 3,501 |
| median / max residues | 205 / 1,997 | 562 / 1,996 |
| mean / max chains | 1.0 / 1 | 2.93 / 60 |
| contacts | 133.7 M | 61.0 M |
| **interface contacts** | — | **9.19 M (15.1%)** |
| truncated by the 8192-token budget | 0.04% | 7.1% |
| on disk (zstd parquet) | 1.2 GB | 529 MB |

Together **688,519 documents / 995 M tokens** — fine-tuning scale by design, next
to the 138 B-token AFDB corpus of exp105.

The multimer corpus is mostly small complexes: 62% dimers, 17% tetramers, 12%
trimers, and a 1,190-document tail at ≥ 10 chains. In 0.79% of documents *every*
emitted contact crosses a chain boundary — those are assemblies of short peptides
(e.g. `6wl1`, 52 copies of a 36-mer) where almost no intra-chain pair clears the
sequence-separation-6 rule. That is correct, not a defect: for an amyloid-like
fibril the interface really is the entire structure.

### The funnel, with nothing unaccounted for

195,858 entries → 177,710 selected → 688,519 documents, every drop named
([`data/funnel.csv`](data/funnel.csv)):

| Entry level | | Chain level (ASU) | |
| --- | --- | --- | --- |
| released after 2021-09-30 | 13,321 | not protein | 28,775 |
| resolution ≥ 9 Å | 763 | all unknown residues | 2,851 |
| no protein entity | 3,881 | > 2000 residues | 733 |
| eval-set holdout | 183 | Cα break > 10 Å | 354 |
| | | ≥ 1/3 atoms clashing | 186 |
| | | < 2 residues | 55 |

Of the 603,198 chains that survive, 339 (0.06%) still fail to serialize —
pyconfind counts "legal protein residues" slightly differently from gemmi, so a
handful fall below the 2-residue floor. They are counted, not swallowed.

Multimer outcomes over the same 177,710 entries: 85,660 documents, 80,304 not a
complex, 10,530 too large for the 2000-index ring, 1,210 with too many chains,
6 with no assembly 1. The `error` column is **empty**.

### Validation

[`validate.py`](validate.py) on a 36,505-document sample (19,794 monomer, 16,711
multimer — every shard sampled):

- **Structural round-trip: 36,505 / 36,505 clean.** Parsing each document back
  from its own text recovers the right residue count, one `<n-term>`/`<c-term>`
  pair per chain, chain runs that are contiguous, pairwise disjoint and exactly
  cover the assigned positions, and contacts referencing only assigned positions.
- **Geometry cross-check: 40 / 40 clean.** For 40 multimers, rebuilding the
  assembly from the mirror and re-running pyconfind reproduces the document's
  interface contacts exactly.

### Leakage

The 552 eval PDB entries are excluded by id, and **0** appear in either corpus.
What remains ([`data/leakage_audit.csv`](data/leakage_audit.csv)):

| | monomers | multimers |
| --- | --- | --- |
| exact resolved-sequence matches | 44 (0.007%) | 1 (0.001%) |
| corpus documents sharing a 40% cluster with the eval set | 8,400 (1.4%) | 2,410 (2.8%) |
| **eval entries with a 40% homolog in the corpus** | **50.2%** | 31.3% |

The last row is the one comparable to
[#213](https://github.com/Open-Athena/MarinFold/issues/213), which measured the
eval set as **58% homologous to exp199's AFDB training data**. This corpus is
therefore *slightly less* homologous to the benchmark than the data the current
best model already trained on — and #213 found that overlap does not inflate the
score (rank correlation ~0 against sequence identity). No homology purge is
warranted; the number is the deliverable.

### Published

`https://huggingface.co/buckets/open-athena/MarinFold` under
`data/document_structures/`:

- `contacts_v1_pdb_monomers/{documents,tokenizer}/`
- `contacts_v1_pdb_multimers/{documents,tokenizer}/`
- `contacts_v1_pdb_curation/{metadata,ledger}/` — the entry scan, the RCSB
  cluster file and the per-entry ledger, so the funnel can be re-derived and
  audited without the local mirror.

The tokenizer (2,848 tokens) is built from the library rather than pulled from a
pinned Hub revision, so it is provably the vocabulary the documents were
generated under.

## Conclusion

**Both corpora exist, round-trip cleanly, and are ready to fine-tune on.**
688,519 documents / 995 M tokens of experimental structure, of which 85,660
describe protein complexes and 9.19 M individual contacts cross a chain boundary
— a class of contact no previous MarinFold corpus contained at all.

The contacts-v1 format needed no new tokens to express a complex. Laying *k*
chains disjointly around the existing 2000-index ring, with one
`<n-term>`/`<c-term>` pair each, is enough, and single-chain documents come out
byte-identical to before — so every existing checkpoint stays token-compatible
and a mixed corpus trains under one tokenizer.

Three things worth carrying forward:

1. **The eval set is no more exposed than before.** 50.2% of eval entries have a
   40% homolog here against 58% for the AFDB data exp199 already trained on.
2. **`global_plddt` means the opposite thing here.** Same column, mean Cα
   B-factor in both corpora — but for AFDB that *is* pLDDT (higher is better) and
   here it is a B-factor (lower is better). Any mixture weighting on it must
   branch on the corpus.
3. **The multimer corpus is dimer-dominated.** 62% of it is two chains. Training
   on it teaches interfaces, but not large-assembly organisation; the 2000-residue
   ring drops 10,530 complexes that were otherwise fine.

Training on these corpora is deliberately **not** part of this experiment. Two
follow-ups suggest themselves: fine-tuning `contacts-v1-exp199-1.5B` on the
monomer corpus (does experimental structure beat predicted?), and a multimer
curriculum (can the model learn to place an interface at all?). Each needs its own
controls.
