# contacts-v1.sequence_only protein-document dataset

**Sequence-only** training documents in the contacts-v1 token space, generated
from [`LiteFold/UniRef50`](https://huggingface.co/datasets/LiteFold/UniRef50) by
calling MarinFold's `generate_sequence_only_document` (no re-implementation).
Each document is the **contacts-v1 sequence section only** — same random
wrap-around `<pX> <AA>` indexing, `<n-term>`/`<c-term>` markers, and shuffled
statement order — under a new doc type `<contacts-v1.sequence_only>` and with
**no structure section** (no contacts):

```
<contacts-v1.sequence_only> <begin_sequence> <p976> <GLY> <p572> <ASN> … <n-term> <p336> … <c-term> <p813> … <end>
```

That shared representation is the point: this corpus can be mixed with the
[`contacts_v1`](../contacts_v1) corpus under **one tokenizer** to (hypothesis,
tested later) improve the contacts-v1 eval. See the format spec:
[`marinfold/.../document_structures/contacts_v1`](https://github.com/Open-Athena/MarinFold/tree/main/marinfold/marinfold/document_structures/contacts_v1)
(the "Sequence-only variant" section of `SPEC.md`).

Produced by experiment [exp64](https://github.com/Open-Athena/MarinFold/issues/64)
(`marinfold` @ `8e6249a`). The sequence section is **byte-identical** to what
`<contacts-v1>` emits for the same `entry_id` — only the leading doc-type token
and the absent structure section differ.

## Splits

**Arbitrary** train/val/test, independent of the contacts-v1 splits (issue #64
allows this): `bucket = sha1(entry_id) % 1000`; `[0,5)` -> `test`, `[5,10)` ->
`val`, the rest -> `train` (≈ 99 / 0.5 / 0.5%). Hashing on `entry_id` keeps each
split length-balanced.

## Layout

```
<split>/uniref50-<shard>-<chunk>.parquet   # ≤200k rows/file; <shard> = source UniRef50 shard (0-60)
tokenizer/                                  # unified tokenizer (2846 tokens; see below)
```

**Ordering caveat.** UniRef50's 61 source shards are globally **sorted by
length, longest first**, and documents are written in that order, so the
published files are **length-banded** (low `<shard>` numbers = longer
sequences). Shuffle at training time (shuffle file order + a shuffle buffer)
rather than reading the shards in order.

## Tokenizer

The unified contacts-v1 tokenizer: contacts-v1's 2845 tokens **plus** the single
`<contacts-v1.sequence_only>` doc-type token appended **last** (id **2845**), so
**every pre-existing contacts-v1 token id is unchanged**. A model can train on
this corpus and the contacts-v1 corpus together with no tokenizer change.

## Counts

**60,004,535 documents — ~32.98 B tokens** from 60,315,044 UniRef50 sequences
(0 generation failures; **310,509** sequences = 0.51% dropped for falling
outside the `[2, 2000]`-residue serializable range — almost all the
>2000-residue giants in source shards 0-1). Mean ~550 `num_tokens`/doc
(≈271 residues).

| split | documents | tokens | files |
|---|--:|--:|--:|
| train | 59,403,434 | 32,653,114,680 | 301 |
| val | 300,982 | 165,485,468 | 61 |
| test | 300,119 | 164,819,873 | 61 |
| **total** | **60,004,535** | **32,983,420,021** | **423** |

## Columns

`document` (token string) · `structure` (`"contacts-v1.sequence_only"`) ·
`entry_id` (UniRef50 accession, e.g. `UniRef50_P00350`) · `seq_len` ·
`start_index` / `n_term_index` / `c_term_index` · `num_tokens` · `sha1` · `split`.

The contact-statistics columns of the contacts-v1 corpus
(`contacts_emitted`, `highest_contact_degree`, …) are **omitted** — there is no
structure section. `num_tokens == 2 * seq_len + 7` for every row.

`sha1` = sha1 of `document`, so byte-equality with the MarinFold generator is a
single-column compare.
