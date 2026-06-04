# contacts-v1

A protein-document format: each single-chain protein is serialized as a
**sequence section** (the residues) followed by a **structure section**
(side-chain *contacts* scored by [pyconfind](https://github.com/timodonnell/pyconfind)).
The full format spec plus the as-built "Implementation notes &
discrepancies" live in [`SPEC.md`](SPEC.md).

## Document structure

A document is a single space-separated token string:

```
<contacts-v1> <begin_sequence>
    [ <pN> <AA> | <n-term> <pN> | <c-term> <pN> ]   # one per residue + the two termini, shuffled
<begin_statements>
    [ <contact> <pX> <pY> ]                          # strongest contacts, random order
<end>
```

- **Sequence section** — one `<pN> <AA>` statement per residue, plus one
  `<n-term>` and one `<c-term>` marker, all in random order. Residues are
  numbered from a random n-terminal index that wraps around 2000 position
  indices (so the model sees the whole index range, not just low values).
- **Structure section** — pyconfind side-chain contacts with contact
  degree ≥ `min_contact_degree` (default 0.001). We take the strongest N
  that fill the 8192-token budget and list them in random order; each
  `<contact>` pair's order is coin-flipped.
- **Token reuse** — positions (`<pN>`), section markers (`<begin_sequence>`
  / `<begin_statements>`), amino acids (`<ALA>`…), `<UNK>` and `<end>` are
  reused from `contacts-and-distances-v1`; only `<contacts-v1>`, `<n-term>`,
  `<c-term>`, `<contact>` and `<think>` are minted here. See [`SPEC.md`](SPEC.md).
- **Deterministic** per `entry_id` (the RNG seed), so the same structure +
  id always yields the same document.

A tiny worked example — a 4-residue chain `MET-ALA-GLY-PHE` numbered from
a random start index of 20. It's a single space-separated line in a file;
shown one statement per line here:

```
<contacts-v1>
<begin_sequence>
<p23> <PHE>
<n-term> <p20>
<p21> <ALA>
<c-term> <p23>
<p20> <MET>
<p22> <GLY>
<begin_statements>
<contact> <p20> <p23>
<contact> <p20> <p22>
<end>
```

All six sequence statements — the four `<pN> <AA>` residues and the two
terminus markers — appear in **one random order**, so `<n-term>` /
`<c-term>` are interleaved with the residues rather than grouped up front.
The contacts are **also in random order** (selected strongest-first to fill
the budget, but not *listed* by degree — here the degree-0.92 `<p20> <p22>`
contact comes second), each pair's two positions coin-flipped.

Eyeball a real one (prints the document + a contact table):

```bash
cd marinfold
uv sync --extra contacts-v1
uv run contacts-v1 view --input tests/data/1QYS.cif
```

## Output + metadata

`generate` writes one row per protein — the `document` token string plus
metadata columns mirroring the published `protein-docs` datasets:

| column | meaning |
|---|---|
| `entry_id` | structure id (and generation seed) |
| `seq_len` | residue count |
| `global_plddt` | mean CA B-factor (pLDDT for AFDB inputs) |
| `start_index`, `n_term_index`, `c_term_index` | residue-numbering offsets |
| `contacts_pre_filter` | all contacts with degree > 0 |
| `contacts_passing_min_degree` | how many passed the min-degree filter |
| `contacts_emitted` / `contacts_excluded` | included / not included |
| `truncated` | whether a budget overflow dropped an eligible contact |
| `highest_contact_degree`, `lowest_nonzero_contact_degree`, `lowest_included_contact_degree` | degree stats |
| `num_tokens`, `sha1` | document length + checksum |

`--summary-out FILE.json` additionally writes a rich per-protein summary
(full residue sequence + every emitted contact with its degree).

## Generate documents

### From local structure files

```bash
# a .pdb/.cif(.gz) file or a directory of them; .jsonl works too
uv run contacts-v1 generate \
    --input /path/to/structures/ \
    --out docs.parquet \
    --summary-out summary.json
```

### From afdb-24M (~1000 documents)

[`afdb-24M`](https://huggingface.co/datasets/timodonnell/afdb-24M) is
sharded Parquet (2,000 structures/shard) with the raw mmCIF in a
`cif_content` column — point `--input` straight at a shard:

```bash
# one shard ≈ 2,000 AFDB structures, ~100 MB
hf download timodonnell/afdb-24M \
    shard_000000-009999/shard_000000.parquet \
    --repo-type dataset --local-dir afdb

uv run contacts-v1 generate \
    --input afdb/shard_000000-009999/shard_000000.parquet \
    --num-docs 1000 \
    --out contacts_v1_docs.parquet \
    --summary-out contacts_v1_summary.json
```

Roughly <1 s per structure (a few minutes for 1,000); installing
`pyconfind[fast]` (numba) speeds up the contact backend. The `entry_id`
column is used as each document's seed, so a document is reproducible from
its AFDB id. Inspect the result:

```bash
python -c "import pyarrow.parquet as pq; t=pq.read_table('contacts_v1_docs.parquet'); \
print(t.num_rows, 'docs'); print(t.column_names)"
```

Sample stats (illustrative — first few of shard 0):

```
entry_id           len  plddt  contacts(emit/found)  truncated  tokens
AF-E9I562-F1       156   87.5         159/238          False      797
AF-A0A2N8S3N2-F1   616   89.4         939/1299         False     4057
AF-A0A7G8TSG9-F1    49   77.3          67/83           False      307
```

Use `--num-docs` to cap output; a whole shard's 2,000 structures works
too (just drop `--num-docs`). For a custom schema use `--cif-column` /
`--id-column`.

## Build the tokenizer

```bash
uv run contacts-v1 tokenizer --save-local ./contacts-v1-tokenizer
# or: uv run contacts-v1 tokenizer --push open-athena/contacts-v1-tokenizer
```

The tokenizer subcommand needs no extra (pyconfind is only used by
`generate` / `view`).

## Algorithm knobs

`--min-contact-degree` (0.001), `--context-length` (8192), `--assembly`
(`none` = asymmetric unit as-is), and the pyconfind geometry knobs
`--contact-distance` / `--dcut` / `--clash-distance`. See
`contacts-v1 generate --help` for the full set.

## Library interface

For the future zephyr data job (or any script), import the package:

```python
from marinfold.document_structures.contacts_v1 import (
    generate_documents,   # over a file / dir / parquet shard(s) -> Iterator[GenerationResult]
    generate_document,    # one structure (path or gemmi.Structure) -> GenerationResult | None
    build_document,       # pure builder from (entry_id, residues, contacts)
    GenerationConfig,
)

for result in generate_documents("afdb/shard_000000.parquet", num_docs=1000):
    row = result.metadata_row()        # flat dict: document + metadata columns
    # ... write row to your sink
```

`generate_document` accepts a `gemmi.Structure` directly (no temp files),
and you can pass a pre-loaded `rotamer_library` (from
`pyconfind.load_library`) to skip re-parsing it across calls.
