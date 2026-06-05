# spec: contacts-v1

This document defines a spec for the *contacts-v1* document type. It is the input to a coding agent that will do the implementation.

## Example document

(Shown with the **actual emitted tokens**; the original design sketch used
`<pos-22>` / `<begin-sequence>` / `<begin-structure>` and lowercase amino
acids, but per the decisions below those are realized as the
contacts-and-distances-v1 tokens `<p22>` / `<begin_sequence>` /
`<begin_statements>` and uppercase `<ALA>` …)

```
<contacts-v1>
<begin_sequence>
<p22> <PHE>
<n-term> <p20>
<p21> <ALA>
<c-term> <p22>
<p20> <ALA>
<begin_statements>
<contact> <p20> <p21>
<contact> <p22> <p21>
<end>
```

## Details

We have two sections: a sequence section (starting with `<begin_sequence>`) and a structure section (starting with `<begin_statements>`).

### Sequence section
This consists of three kinds of statements.

`<pXXX> <RESIDUE>` indicates that position XXX is the given amino acid. The position tokens are contacts-and-distances-v1's `<p0>` through `<p1999>` (2000 indices total), reused rather than minted anew.

`<n-term>` `<pXXX>` indicates that position XXX is the N-terminus of a protein chain

`<c-term>` `<pXXX>` indicates that position XXX is the C-terminus of a protein chain.

The statements of the sequence section are given in random order. We define the amino acid for all residues exactly once. We define the N- and C-termini for each protein chain.

### Residue indexing
We support structures with up to 2000 residues

Rather than numbering residues in a protein as e.g. 0 to 1999, each time we generate a document we pick a random number n in [0, 2000) to be the n-terminal residue. We start indexing from this residue. Residue indices "wrap around", so the residue after `<p1999>` is `<p0>`. The motivation here is that we want the model to be experienced in using all residue indices. Since most proteins are way less than 2000 residues, if we always started the protein chain off at `<p0>` the model would only rarely see the higher value indices.

In the future we will support multiple protein chains, and we will just have multiple <n-term> and <c-term> statements for these. They will need to be spaced out enough to not overlap. For example we might have one protein that starts at index 1800 and continues until residue 100, and another that starts at residue 300 and continues until 800.

### Structure section
The structure section consists of statements of the form `<contact>` `<pXXX>` `<pYYY>`, which indicates that residues at index XXX and YYY are in contact.

Contacts are defined as contact degree > 0 where contact degree is implemented in [pyconfind](https://github.com/timodonnell/pyconfind). We run pyconfind in `native_only=True` mode, i.e. only consider the actual given amino acid at each position rather than all other possibilities.

We also require a minimum primary-sequence separation: a pair of residues at sequence positions i and j is only a contact if `|i - j| >= min_seq_separation` (default 6). Residues 5 or fewer positions apart in the chain are never contacts, regardless of geometry — this keeps trivial local / secondary-structure contacts out of the documents.

Before selecting which contacts to include, we discard any contact whose contact degree is below a minimum threshold (`min_contact_degree`, default 0.001). Contacts below this threshold are **never** written to a document, even if there is room for them. (pyconfind reports many very weak contacts — degrees down to ~1e-8 — and this threshold keeps those noise-level contacts out.)

From the contacts that pass this threshold, we include the N with the highest contact degree (the N strongest), where N is chosen so the document fills the 8192-token budget (see Document length). These N selected contacts are then listed in **random order** in the structure section — they are *not* sorted by degree. (We select by strength so that, when a protein has more above-threshold contacts than fit, the weakest are the ones dropped; but the order they appear in the document is randomized so the model does not learn a degree-sorted ordering.)

Note that the contact matrix is symmetric. We randomize the order that each contact pair is given in: if there is a contact between XXX and YYY, with 50% probability we output `<contact> <pXXX> <pYYY>` and the other half of the time we output `<contact> <pYYY> <pXXX>`. Each contact is only specified once, i.e. once we emit the contact between XXX and YYY we will never emit it again in either order.

### Document length
Our max document length is 8192 tokens. N (the number of contacts included) is the number of the strongest *above-threshold* contacts whose `<contact>` statements fit in the budget remaining after the sequence section. If the protein has more above-threshold contacts than fit, the weakest of them are dropped (truncation); if it has fewer, all above-threshold contacts are included and the document is shorter than the budget.

### Additional tokens
For the vocab, also include as additional tokens all the tokens in the contacts-and-distances-v1 [vocab](https://github.com/Open-Athena/MarinFold/blob/main/marinfold/marinfold/document_structures/contacts_and_distances_v1/vocab.py). We may fine tune on documents like that later. Also include this additional token: `<think>`

---

## Implementation notes & discrepancies

This section records where the implementation (in this directory:
`vocab.py`, `parse.py`, `generate.py`, `cli.py`) **diverges from** or
**pins down** the spec above. It is the source of truth for the
as-built behavior. **Keep it current** whenever the implementation or
the spec changes.

### Decisions that override the spec text

contacts-v1 **reuses contacts-and-distances-v1 tokens wherever an
equivalent already exists**, so the two structures share token IDs /
embeddings and a contacts-v1 model can be fine-tuned on
contacts-and-distances-v1 documents without a tokenizer change. This
overrides the spec's own token spellings:

- **Position tokens reuse `<p0>` … `<p1999>`.** The *Details* section's
  `<pXXX>` indices (sketched as `<pos-0000>` / `<pos-22>` in the original
  draft) are contacts-and-distances-v1's existing `<p0>`–`<p2700>`
  position tokens; contacts-v1 uses the first 2000 of them and mints no
  `<pos-N>` tokens. Note a contacts-v1 position is a *randomized
  wrap-around* index, whereas the same `<pX>` in contacts-and-distances-v1
  is a true chain position — the leading doc-type token distinguishes the
  two interpretations.
- **Section markers reuse `<begin_sequence>` / `<begin_statements>`.**
  The spec's `<begin-sequence>` / `<begin-structure>` are realized as
  contacts-and-distances-v1's underscore-spelled markers (so
  `<begin_statements>` opens the structure section).
- **Amino-acid tokens reuse the uppercase `<ALA>` … `<VAL>`.** The
  example's lowercase (`<phe>`, `<ala>`) is not used; reusing the
  uppercase AAs shares amino-acid embeddings with contacts-and-distances-v1.
- **`<end>` is the shared contacts-and-distances-v1 token** too.

The only tokens contacts-v1 mints itself are `<contacts-v1>`, `<n-term>`,
`<c-term>`, `<contact>`, and the (unused) `<think>` — five tokens with no
contacts-and-distances-v1 analog. (The original example's `<c-term> <pos
22>` space typo is emitted as `<c-term> <p22>`.)

### Points the spec left open (implementation choices)

- **Residue + contact source.** Both the residue sequence and the
  contacts come from a single `pyconfind.analyze(..., native_only=True)`
  call, read off pyconfind's ordered *position* list, so they are always
  mutually consistent. Contacts reference residues by 0-based sequence
  index with `seq_i < seq_j`.
- **pyconfind parameters.** confind/C++ defaults: `contact_distance=3.0`,
  `dcut=25.0`, `clash_distance=2.0`, plus `native_only=True`. We pass
  `assembly=None` by default, i.e. analyze the asymmetric unit / input
  structure as-is rather than implicitly expanding biological assembly 1.
  pyconfind returns every contact with degree > 0 (down to ~1e-8); the
  `min_contact_degree` filter above then decides which are eligible.
  These (including `--assembly` and `min_contact_degree`) are exposed as
  CLI knobs but default to the above.
- **Non-canonical residues.** pyconfind's "legal" protein residues
  beyond the standard 20 (HIS variants HSD/HSE/HSC/HSP/HIP, plus MSE,
  SEC, CSO, SEP, TPO, PTR) are canonicalized to their parent amino acid
  (e.g. MSE→MET, HSD→HIS, SEP→SER). Anything unexpected maps to `<UNK>`
  (reused from the contacts-and-distances-v1 vocab).
- **Single chain only.** Multi-chain support is future work per the spec.
  Structures with more than one protein chain are skipped (with a
  warning); exactly one `<n-term>` and one `<c-term>` are emitted.
- **Residue-count bounds.** "Up to 2000 residues" is enforced: chains
  with fewer than 2 residues or more than 2000 (can't be uniquely
  indexed under wrap-around) are skipped with a warning.
- **Minimum sequence separation.** A pair counts as a contact only if its
  residues are at least `config.min_seq_separation` (default 6) positions
  apart in the primary sequence (`seq_j - seq_i >= min_seq_separation`).
  This is *definitional* — closer pairs are filtered before anything is
  counted, so `contacts_pre_filter` (and the degree statistics) already
  exclude them. (`contacts-and-distances-v1` used the same minimum of 6 via
  its `short_range_sep`.)
- **Minimum degree filter.** Contacts with degree below
  `config.min_contact_degree` (default 0.001) are dropped before anything
  else and are never emitted, regardless of budget. pyconfind returns a
  long tail of near-zero-degree contacts (down to ~1e-8); this keeps that
  noise out of documents.
- **Contact selection & ordering.** From the above-threshold contacts,
  selection of the N strongest uses a *stable* sort by descending degree,
  so equal-degree contacts at the truncation boundary break ties by
  pyconfind's `(seq_i, seq_j)`-ascending order — deterministic. The
  selected N are then shuffled, so the order in the document is random
  (not degree-sorted).
- **Truncation.** The whole sequence section is kept when it fits; if the
  framing + sequence section alone exceeds the context budget, the protein
  is skipped with a warning. Otherwise the N strongest *above-threshold*
  contacts fill the remaining budget, `N = floor((8192 − frame − sequence)
  / 3)` capped at the number passing the filter. `truncated` means a
  budget overflow specifically — some above-threshold contact didn't fit
  (`contacts_passing_min_degree > contacts_emitted`) — not that the
  degree filter dropped something.
- **Determinism.** Seeded by `random.Random(int(sha1(entry_id)[:8], 16))`
  (same scheme as contacts-and-distances-v1). RNG draws happen in a
  fixed, load-bearing order: (1) the random n-terminal start index, then
  (2) the shuffle of the sequence-section statements, then (3) the
  shuffle of the N selected contacts (their in-document order), then
  (4) the per-contact pair-order coin flips.
- **Vocab order.** Load-bearing and append-only: the 5 native tokens
  (`<contacts-v1>`, `<n-term>`, `<c-term>`, `<contact>`, `<think>`), then
  the entire contacts-and-distances-v1 `all_domain_tokens()` list. The two
  groups are disjoint — contacts-v1 reuses c-and-d-v1 tokens by *emitting*
  them, not by re-minting — so no dedup is needed. Total domain vocab =
  2843 tokens (2845 with `<pad>`/`<eos>`).

### Metadata tracked (beyond the spec)

The spec does not define output metadata; the issue asks to track
protein-docs-style metadata. Per document we record: `entry_id`,
`seq_len`, `global_plddt` (mean CA B-factor), `start_index`,
`n_term_index`, `c_term_index`, `num_tokens`, `sha1` of the document,
and the following contact statistics:

- `contacts_pre_filter` — total contacts (degree > 0) the protein had.
- `contacts_passing_min_degree` — how many passed the `min_contact_degree`
  filter (the pool the strongest N are chosen from).
- `contacts_emitted` — how many were included in the document.
- `contacts_excluded` — how many did not make it in (`pre_filter −
  emitted`); counts both below-threshold and budget-truncated contacts.
- `truncated` — whether a budget overflow dropped any above-threshold
  contact (`contacts_passing_min_degree > contacts_emitted`).
- `highest_contact_degree` — max degree over all the protein's contacts.
- `lowest_nonzero_contact_degree` — min degree over all the protein's
  contacts (pyconfind only returns degree > 0, so this is the smallest
  positive degree, which may be below the filter threshold).
- `lowest_included_contact_degree` — min degree among the contacts that
  made it into the document (always ≥ `min_contact_degree`).

The whole-protein degree fields (`highest_contact_degree`,
`lowest_nonzero_contact_degree`) are null when the protein has no contacts
at all; `lowest_included_contact_degree` is null when nothing was emitted
(no contacts, or all below threshold). The local `--summary-out` JSON
additionally lists the full residue sequence and every emitted contact
with its contact-degree value and the residue numbers/names it connects.

### Not yet implemented

- Multiple protein chains (multiple `<n-term>`/`<c-term>`, spaced-out
  index ranges) — see *Residue indexing* above.
- Model inference / evaluation against this structure (there is no
  trained contacts-v1 model yet); only `generate` / `view` / `tokenizer`
  exist.
