# contacts-v1 MPNN-redesign corpus

**8 ProteinMPNN sequences for every backbone in the decontaminated AFDB
`contacts_v1` training set, with contacts recomputed for each.**

- Documents: **31,702,680** (3,962,835 backbones × 8 designs)
- Tokens: **35,320,841,292** (35.32 B; mean 1,114/doc)
- Size: 64.1 GB, 199 ZSTD parquet shards
- Source backbones: [`contacts_v1_decontam`](https://huggingface.co/buckets/open-athena/MarinFold/tree/main/data/document_structures/contacts_v1_decontam) train split ([#225](https://github.com/Open-Athena/MarinFold/issues/225))
- Document format: `contacts-v1`, unchanged — same vocabulary, same 8192-token budget
- Built by [#266](https://github.com/Open-Athena/MarinFold/issues/266) / [PR #267](https://github.com/Open-Athena/MarinFold/pull/267)

## What this is

Every fold in `contacts_v1` appears with exactly one sequence — the native one.
This corpus pairs each of those backbones with **eight ProteinMPNN designs**,
so the model can see many sequences over one geometry and separate "this
contact is geometry" from "this contact is this rotamer".

It is the cheap null for a larger idea (generate *novel* backbones with a
structure generator and inverse-fold them): it changes only the sequences, not
the folds, and so isolates the part that would have to pay off anyway.

## How it was built

1. **Keep-list** — `entry_id`s read straight out of the published
   decontaminated corpus, so decontamination is inherited rather than
   re-derived. A redesigned sequence cannot reintroduce an eval *sequence*
   because it is a new sequence.
2. **Backbones staged** from AFDB mmCIF as N/CA/C/O + per-residue pLDDT, with
   coordinates as **int32 milli-ångströms** (exact for AFDB's 3-decimal
   `Cartn_x`, so the round-trip is lossless).
3. **ProteinMPNN `v_48_020`**, 8 designs per backbone on a temperature ladder
   `{0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.5, 0.5}`, recorded per row.
4. **pyconfind re-run per (backbone, sequence)** — the contact label is
   genuinely sequence-dependent (a sequence shuffle on an identical backbone
   keeps only ~43–54 % of native contacts), so labels cannot be copied from the
   parent document.

**Same contact operator as `contacts_v1`.** confind rebuilds side chains from
the Dunbrack rotamer library rather than reading them, so a backbone plus a
residue-name assignment is a complete input. Verified end-to-end: regenerating
published corpus documents from the staged backbones reproduces them
**byte-for-byte** (200/200 sha1 match on a mid-length production shard).

## Columns

Every `contacts_v1` metadata column, plus:

| column | meaning |
|---|---|
| `design_index` | 0–7, which slot of the ladder |
| `mpnn_temperature` | sampling temperature for this design |
| `mpnn_score` | mean per-residue NLL of the sampled sequence |
| `identity_to_native` | fraction of positions matching the native AA |
| `native_sha1` | sha1 of the parent `contacts_v1_decontam` document |
| `native_contacts_emitted` | parent document's contact count |

`entry_id`, `struct_cluster_id`, `seq_cluster_id`, `split` and `round` carry
through from the parent, so a redesigned document can always be traced back.

`global_plddt` is the parent AFDB structure's — it describes confidence in the
**backbone**, which is what was reused, and says nothing about whether the
designed sequence folds there.

## What to know before training on it

- **Contact density is essentially unchanged.** Designed documents carry
  contacts/residue at a ratio of **1.016** to native (measured on 400
  length-representative backbones), approaching 1 as length grows. The feared
  "MPNN's alanine bias shortens the documents" artifact does not occur.
- **Composition does shift**: P +4.33, A +4.16, S −3.25, E +2.98, L +2.59,
  Q −2.17 percentage points against the native sequences.
- **The temperature ladder is narrow.** 0.1 → 0.5 moves identity-to-native only
  0.356 → 0.326 and density not at all, so subsetting on `mpnn_temperature`
  buys little.
- **Sequence recovery on AFDB is 0.356** at T=0.1, well below ProteinMPNN's
  published ~50 % on crystal structures — predicted models are harder to
  recover.
- **The designs were not refolded before inclusion.** Whether a design folds to
  its backbone is measured separately on a sample (see the issue); it is not a
  filter applied to this corpus.

## Verification

Anything that re-derives documents from these backbones **must install
`pyconfind[fast]`**. The numba and pure-python backends disagree marginally at
the `min_contact_degree = 0.001` cut — enough to change a document's `sha1` —
and the corpora are defined by `[fast]`.
