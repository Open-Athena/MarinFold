# contacts-v1 ESM-Atlas MPNN-redesign corpus

**2 ProteinMPNN sequences for every backbone in the decontaminated ESM-Atlas
`contacts_v1` training set, with contacts recomputed for each.**

- Documents: **130,872,044** (65,436,022 backbones × 2 designs)
- Tokens: **138,624,482,815** (138.62 B; mean 1,059/doc)
- Size: 256.0 GB, 3,338 ZSTD parquet shards
- Source backbones: [`contacts_v1_esm_atlas_decontam`](https://huggingface.co/buckets/open-athena/MarinFold/tree/main/data/document_structures/contacts_v1_esm_atlas_decontam) train split ([#225](https://github.com/Open-Athena/MarinFold/issues/225))
- Document format: `contacts-v1`, unchanged — same vocabulary, same 8192-token budget
- Built by [#266](https://github.com/Open-Athena/MarinFold/issues/266) / [PR #267](https://github.com/Open-Athena/MarinFold/pull/267)

Companion to
[`contacts_v1_mpnn_redesign`](https://huggingface.co/buckets/open-athena/MarinFold/tree/main/data/document_structures/contacts_v1_mpnn_redesign),
which does the same thing to the AFDB corpus at 8 designs per backbone. This
one is ~3.9× larger in tokens because ESM-Atlas has ~16× the backbones.

## What this is

Every fold in `contacts_v1_esm_atlas` appears with exactly one sequence — the
native one. This corpus pairs each of those backbones with **two ProteinMPNN
designs**, so the model can see more than one sequence over a given geometry
and separate "this contact is geometry" from "this contact is this rotamer".

**Two designs, not eight, on purpose.** The AFDB arm shipped an 8-slot
temperature ladder `{0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.5, 0.5}` so a training
experiment could subset near-native ↔ diverse without regenerating, and then
measured that the ladder spans almost nothing: identity-to-native moves only
0.373 → 0.345 across it, contact density not at all, and T=0.5 refolds *worse*.
ESM-Atlas's distinctive value is 65 M non-redundant backbones, so the budget
went into backbones rather than into more sequences per backbone. Only
`T = 0.1` and `T = 0.2` are used here.

## How it was built

One pass, no staging — unlike the AFDB arm, which stages backbones because
AFDB's bucket is requester-pays and reachable only from GCP.

1. **Source read directly.** ESM-Atlas structures are inline `cif_content` in
   a public HF bucket, so a CoreWeave pod reads them itself.
2. **No coordinate encoding.** The AFDB arm stores staged coordinates as int32
   milli-ångströms, which is exact for AFDB's 3-decimal `Cartn_x`. ESM-Atlas
   cifs carry up to **11 decimals**, and rounding them to 1/1000 Å changed
   **114 of 200** documents — same contact counts, different contact sets at
   the margin. So the structure never leaves float64: parsed once, used for
   both ProteinMPNN and pyconfind, discarded.
3. **ProteinMPNN `v_48_020`**, 2 designs per backbone at `T ∈ {0.1, 0.2}`,
   recorded per row.
4. **pyconfind re-run per (backbone, sequence)** — the contact label is
   genuinely sequence-dependent (a sequence shuffle on an identical backbone
   keeps only ~43–54 % of native contacts), so labels cannot be copied from
   the parent document.

**Same contact operator as `contacts_v1`.** confind rebuilds side chains from
the Dunbrack rotamer library rather than reading them, so a backbone plus a
residue-name assignment is a complete input.

## Completeness

Verified two independent ways, which agree exactly:

| check | result |
|---|---|
| shard index coverage | **3,338 / 3,338** |
| kept backbones with no document | 117,156 of 65,553,178 (**0.179 %**) |
| documents whose backbone is not in the parent corpus | **0** |
| documents vs 2 × kept backbones | delta **−234,312** = 2 × 117,156 |
| designs per backbone (6 shards read in full) | 2 for every backbone, none duplicated |

The 0.179 % are `contacts-v1` declining a structure: multimers, non-canonical
residues, non-finite coordinates, and degenerate geometry that makes pyconfind
produce NaN rotamer positions. They are spread evenly — every one of the 3,338
shards loses a few, worst case 59 of ~19,600 — which is what a uniform filter
looks like rather than a coverage hole.

## Columns

Every `contacts_v1` metadata column, plus:

| column | meaning |
|---|---|
| `design_index` | 0 or 1 |
| `mpnn_temperature` | 0.1 or 0.2 |
| `mpnn_score` | mean per-residue NLL of the sampled sequence |
| `identity_to_native` | fraction of positions matching the native AA |
| `native_sha1` | sha1 of the parent `contacts_v1_esm_atlas_decontam` document |
| `native_contacts_emitted` | parent document's contact count |
| `parent_seq_len` | parent document's sequence length |

`entry_id`, `seq_cluster_id`, `split`, `cluster_size`, `ptm` and
`global_plddt` carry through from the parent, so a redesigned document can
always be traced back. There is no `struct_cluster_id` or `round` — both are
AFDB-only.

`global_plddt` and `ptm` describe the **parent ESMFold2 prediction**. They
speak to confidence in the backbone, which is what was reused, and say nothing
about whether the designed sequence folds there.

## What to know before training on it

- **Sequence lengths run 60–1000 residues**, inherited from the parent corpus —
  a narrower band than the AFDB arm's 30–1998.
- **The designs were not refolded**, and self-consistency was **not measured on
  this arm**. On the AFDB arm, ESMFold2 at one diffusion sample put designs at
  **79 % of a native control's sub-2 Å rate and 91 % of its same-fold rate**;
  whether that transfers to ESM-Atlas backbones — which are themselves
  predictions, from a different predictor — is untested.
- **Composition shifts**, as ProteinMPNN always does. Measured on the AFDB
  arm: P +3.60, A +2.72, S −2.47, E +2.19, Q −2.14, L +2.09 percentage points
  against native.
- **This corpus shares its backbones with `contacts_v1_esm_atlas_decontam`.**
  Mixing both means seeing each geometry three times with three sequences, not
  three independent structures.

## Verification

Anything that re-derives documents from these backbones **must install
`pyconfind[fast]`**. The numba and pure-python backends disagree marginally at
the `min_contact_degree = 0.001` cut — enough to change a document's `sha1` —
and the corpora are defined by `[fast]`.
