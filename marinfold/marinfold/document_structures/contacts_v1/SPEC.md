# spec: contacts-v1

This document defines a spec for the *contacts-v1* document type. It is the input to a coding agent that will do the implementation.

## Example document

```
<contacts-v1>
<begin-sequence>
<pos-22> <phe>
<n-term> <pos-20>
<pos-21> <ala>
<c-term> <pos 22>
<pos-20> <ala>
<begin-structure>
<contact> <pos-20> <pos-21> 
<contact> <pos-22> <pos-21> 
<end>
```

## Details

We have two sections: a sequence section (starting with `<begin-sequence>`) and a structure section (starting with `<begin-structure>`).

### Sequence section
This consists of three kinds of statements.

`<POS-XXX> <RESIDUE>` indicates that position XXX is the given amino acid. We have indexing tokens `<pos-0000>` through `<pos-1999>` (2000 indices total).

`<n-term>` `<pos-XXX>` indicates that position XXX is the N-terminus of a protein chain

`<c-term>` `<pos-XXX>` indicates that position XXX is the C-terminus of a protein chain.

The statements of the sequence section are given in random order. We define the amino acid for all residues exactly once. We define the N- and C-termini for each protein chain.

### Residue indexing
We support structures with up to 2000 residues

Rather than numbering residues in a protein as e.g. 0 to 1999, each time we generate a document we pick a random number n in [0, 2000) to be the n-terminal residue. We start indexing from this residue. Residue indices "wrap around", so the residue after `<pos-1999>` is `<pos-0>`. The motivation here is that we want the model to be experienced in using all residue indices. Since most proteins are way less than 2000 residues, if we always started the protein chain off at `<pos-0>` the model would only rarely see the higher value indices.

In the future we will support multiple protein chains, and we will just have multiple <n-term> and <c-term> statements for these. They will need to be spaced out enough to not overlap. For example we might have one protein that starts at index 1800 and continues until residue 100, and another that starts at residue 300 and continues until 800.

### Structure section
The structure section consists of statements of the form `<contact>` `<pos-XXX>` `<pos-YYY>`, which indicates that residues at index XXX and YYY are in contact.

Contacts are defined as contact degree > 0 where contact degree is implemented in [pyconfind](https://github.com/timodonnell/pyconfind). We run pyconfind in `native_only=True` mode, i.e. only consider the actual given amino acid at each position rather than all other possibilities.

We include the N contacts with the highest contact degree (the N strongest), where N is chosen so the document fills the 8192-token budget (see Document length). These N selected contacts are then listed in **random order** in the structure section — they are *not* sorted by degree. (We select by strength so that, when a protein has more contacts than fit, the weakest are the ones dropped; but the order they appear in the document is randomized so the model does not learn a degree-sorted ordering.)

Note that the contact matrix is symmetric. We randomize the order that each contact pair is given in: if there is a contact between XXX and YYY, with 50% probability we output `<contact> <pos-XXX> <pos-YYY>` and the other half of the time we output `<contact> <pos-YYY> <pos-XXX>`. Each contact is only specified once, i.e. once we emit the contact between XXX and YYY we will never emit it again in either order.

### Document length
Our max document length is 8192 tokens. N (the number of contacts included) is the number of the strongest contacts whose `<contact>` statements fit in the budget remaining after the sequence section. If the protein has more contacts than fit, the weakest are dropped (truncation); if it has fewer, all are included and the document is shorter than the budget.

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

- **Position tokens are unpadded: `<pos-0>` … `<pos-1999>`.** The
  *Details* section above writes `<pos-0000>` through `<pos-1999>`
  (zero-padded), but the example writes `<pos-22>`. Per a maintainer
  decision the unpadded form (matching the example) is canonical. 2000
  tokens total.
- **Amino-acid tokens are the uppercase three-letter tokens reused from
  contacts-and-distances-v1: `<ALA>`, `<ARG>`, … `<VAL>`.** The example
  above shows lowercase (`<phe>`, `<ala>`), but per a maintainer
  decision contacts-v1 reuses the existing uppercase AA tokens so the
  two document structures share amino-acid embeddings (useful for the
  planned later fine-tuning). No lowercase `<ala>`-style tokens exist in
  the vocab.
- **The worked example is illustrative and has informal typos** that the
  implementation does not reproduce: `<c-term> <pos 22>` is always
  emitted as `<c-term> <pos-22>` (hyphenated), and the prose "we will
  never see `<contact> <pos-XXX> <pos-YYY>` or `<contact> <pos-YYY>
  <pos-XXX>`" means a contact pair is emitted exactly once in one of the
  two orders.

### Points the spec left open (implementation choices)

- **Residue + contact source.** Both the residue sequence and the
  contacts come from a single `pyconfind.analyze(..., native_only=True)`
  call, read off pyconfind's ordered *position* list, so they are always
  mutually consistent. Contacts reference residues by 0-based sequence
  index with `seq_i < seq_j`.
- **pyconfind parameters.** confind/C++ defaults: `contact_distance=3.0`,
  `dcut=25.0`, `clash_distance=2.0`, plus `native_only=True`. Every
  contact pyconfind returns with degree > 0 is eligible (including very
  weak contacts with degree ~1e-8). These are exposed as CLI knobs but
  default to the above.
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
- **Contact selection & ordering.** Selection of the N strongest uses a
  *stable* sort by descending degree, so equal-degree contacts at the
  truncation boundary break ties by pyconfind's `(seq_i, seq_j)`-ascending
  order — deterministic. The selected N are then shuffled, so the order
  in the document is random (not degree-sorted).
- **Truncation.** The whole sequence section is always kept (it fits for
  any chain ≤ 2000 residues); the N strongest contacts fill the remaining
  budget, `N = floor((8192 − frame − sequence) / 3)` capped at the number
  available. `truncated`, `contacts_pre_filter`, `contacts_emitted`, and
  `contacts_excluded` are recorded as metadata.
- **Determinism.** Seeded by `random.Random(int(sha1(entry_id)[:8], 16))`
  (same scheme as contacts-and-distances-v1). RNG draws happen in a
  fixed, load-bearing order: (1) the random n-terminal start index, then
  (2) the shuffle of the sequence-section statements, then (3) the
  shuffle of the N selected contacts (their in-document order), then
  (4) the per-contact pair-order coin flips.
- **Vocab order.** Load-bearing and append-only: contacts-v1 native
  control tokens, then `<pos-0>`…`<pos-1999>`, then `<think>`, then the
  contacts-and-distances-v1 `all_domain_tokens()` list deduplicated. The
  only token shared between the two groups is `<end>`, kept once (shared
  id). Total domain vocab = 4845 tokens (4847 with `<pad>`/`<eos>`).

### Metadata tracked (beyond the spec)

The spec does not define output metadata; the issue asks to track
protein-docs-style metadata. Per document we record: `entry_id`,
`seq_len`, `global_plddt` (mean CA B-factor), `start_index`,
`n_term_index`, `c_term_index`, `num_tokens`, `sha1` of the document,
and the following contact statistics:

- `contacts_pre_filter` — total contacts (degree > 0) the protein had.
- `contacts_emitted` — how many were included in the document.
- `contacts_excluded` — how many did not make it in (`pre_filter −
  emitted`); `truncated` is `contacts_excluded > 0`.
- `highest_contact_degree` — max degree over all the protein's contacts.
- `lowest_nonzero_contact_degree` — min degree over all the protein's
  contacts (pyconfind only returns degree > 0, so this is the smallest
  positive degree).
- `lowest_included_contact_degree` — min degree among the contacts that
  made it into the document (the truncation threshold). Equals
  `lowest_nonzero_contact_degree` when nothing is truncated.

The three degree fields are null when the protein has no contacts. The
local `--summary-out` JSON additionally lists the full residue sequence
and every emitted contact with its contact-degree value and the residue
numbers/names it connects.

### Not yet implemented

- Multiple protein chains (multiple `<n-term>`/`<c-term>`, spaced-out
  index ranges) — see *Residue indexing* above.
- Model inference / evaluation against this structure (there is no
  trained contacts-v1 model yet); only `generate` / `view` / `tokenizer`
  exist.
