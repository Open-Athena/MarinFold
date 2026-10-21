# contacts-v1 predicted-complex corpus

**3,410,738 contacts-v1 documents / 11.15 B tokens of protein–protein
complexes**, each document one whole complex with every chain laid out together.

- **261,833,754 interface contacts** (12.4% of 2.11 B total) — contacts that
  cross a chain boundary
- **611,656 heterodimers** and 2,799,082 homodimers
- 19 GB, 171 ZSTD parquet shards
- Decontaminated against the 577-unit **eval2** set, every subunit checked
- Built by [MarinFold #294](https://github.com/Open-Athena/MarinFold/issues/294)

## What this is

Two arms, distinguished by `source_arm` and never mixed silently:

| arm | documents | what it is |
| --- | --- | --- |
| `afcdb` | 2,982,517 | **predicted** dimers from NVIDIA's AlphaFold complex release, selectively extracted from 48.8 TB of archives |
| `pinder` | 428,221 | **experimental** heterodimer interfaces from PINDER 2024-02 |

The AFCDB arm is the bulk and is where the "predicted" in the name comes from.
The PINDER arm exists because AFCDB's heterodimer pool is genuinely exhausted at
183,435 usable models — it is not rationed, and no threshold change produces
more. PINDER supplies 385,354 systems on 32,939 UniProt pairs that appear
nowhere in AFCDB, which is what takes heterodimers past 500k.

## Layout

```
train/shard-NNNNN-of-00171.parquet   the corpus
manifest_natural.parquet             sampling manifest: the mixture as-is
manifest_balanced.parquet            sampling manifest: cluster-balanced weights
corpus_stats.json                    the numbers on this page, machine-readable
tokenizer/                           contacts-v1 tokenizer (2,848 tokens)
```

Read anonymously with `huggingface_hub>=1.5`. `snapshot_download` does **not**
see bucket contents — use the bucket API:

```python
from huggingface_hub import list_bucket_tree, download_bucket_files

prefix = "data/document_structures/contacts_v1_complex/train/"
shards = [f.path for f in list_bucket_tree(
    "open-athena/MarinFold", prefix, recursive=True, token=False)]
download_bucket_files("open-athena/MarinFold",
    files=[(shards[0], "shard-00000.parquet")], token=False)
```

## What one document is

One complex, all chains on a shared 2000-index wrap-around ring, in the
multi-chain `contacts-v1` format introduced by
[#222](https://github.com/Open-Athena/MarinFold/issues/222) — **no new tokens,
same tokenizer**. Chains are shuffled into a random ring order with random gaps,
so chain identity is recoverable from the `<n-term>`/`<c-term>` statements and
the contiguity of each chain's index run, not from document order. The minimum
sequence separation of 6 is **intra-chain only**: two residues on different
chains are a candidate contact at any index distance.

Full spec: [SPEC.md § Multiple protein
chains](https://github.com/Open-Athena/MarinFold/blob/main/marinfold/marinfold/document_structures/contacts_v1/SPEC.md).

Sizes: mean 700 residues, median 630, range 26–1998. Mean 3,270 tokens, capped
at 8,192; 82,756 documents (2.4%) hit that cap and carry `truncated = true`.

## Curation

**AFCDB.** 29,025,020 source models were ranked by
`quality_ratio = min(ipSAE/0.6, pDockQ2/0.23)` and filtered on: at least one
reported interaction, at most 10 backbone clashes, `sum(chain lengths) + k ≤
2000`, a UniProt sequence whose length matches the model's residue count, and a
distinct sequence pair. 3,000,000 models were selected; **every one of the
29,025,020 carries exactly one terminal ledger reason**, and the 3,000,000
resolve to 2,982,771 generated, 16,856 whose member was not in the tar the
metadata named, and 373 in 28 tars that no longer exist upstream.

`confidence_tier` splits the selection: **A** (1,748,907, mean quality ratio
1.33) passes the original hard gate; **B** (1,233,610, mean 0.67) is the relaxed
tier needed to reach 3M — the 0.5 floor misses it on arithmetic alone. Tier B is
kept and labelled rather than dropped, so training can choose.

**PINDER.** 2,319,564 systems reduced to 449,835: heterodimers only, PINDER's
own `invalid` verdict honoured, same 2000-residue ring budget, a UniProt mapping
required. Coordinates were fetched by byte offset out of a 35 GB zip rather than
downloading it.

**Both arms carry `contacts_emitted_inter_chain ≥ 1`.** 254 AFCDB models cleared
the ipSAE/pDockQ2 gate without a single chain–chain contact; a complex whose
chains never touch is not an interface example, so they are dropped.

## Decontamination

Reference: **eval2**, all 577 units
([#226](https://github.com/Open-Athena/MarinFold/issues/226)) — not the legacy
554. The [#225](https://github.com/Open-Athena/MarinFold/issues/225) rule applied
to **every subunit** of every complex: drop if `identity ≥ 30%` over `≥ 50%`
query coverage, or `E ≤ 1e-3`. MMseqs2 at `-s 7.5`, `--max-seqs 1000000`,
search `-e 1000`, uncensored prefilter.

- AFCDB: 1,071,475 of 21,030,179 UniProt accessions dropped (5.1%), applied
  before selection. A complex is dropped if **either** subunit matches.
- PINDER: searched against the **structure** sequences rather than UniProt,
  because a PINDER system is a crystallised fragment and its UniProt entry is
  not what the document contains. 16,430 of 445,206 systems dropped (3.7%).

There is **no TM ≥ 0.5 fold purge** — the policy is sequence homology plus
structural near-duplicate protection, deliberately not a fold-level purge.

PINDER's **own val and test splits are also excluded** (555 systems). The eval2
drop list only knows about MarinFold's eval; shipping PINDER's benchmark inside a
training corpus would contaminate anyone who evaluates on it.

## The two manifests

Redundancy is **preserved in the corpus and controlled in sampling**. Both
manifests cover the same 3,410,738 documents and differ only in weight.

`cluster_key` is a real interface cluster for PINDER (`pinder:<id>`) and the
sequence pair for AFCDB (`afcdb_pair:<a>|<b>`), labelled so the two are never
conflated. AFCDB models are one-per-sequence-pair, so every AFCDB cluster has
size 1; PINDER averages 18.7 structures per interface cluster and reaches 15,274
in its largest.

- **`manifest_natural`** — weight 1 everywhere, the mixture the sources have.
- **`manifest_balanced`** — `sampling_weight = 1/√(cluster_size)`, with a cap at
  0.1% of total probability per cluster.

| clusters, ranked by sampling mass | share of documents | share of sampling |
| --- | --- | --- |
| top 1 | 0.448% | **0.0041%** |
| top 10 | 2.579% | 0.0299% |
| top 100 | 5.268% | 0.1216% |
| top 1% (30,054) | 12.764% | 2.020% |

**The cap never binds here.** Inverse-√ alone brings the largest cluster —
15,274 experimental structures of one interface — from 0.448% of the documents
down to 0.0041% of sampling mass, a 110× reduction, well inside the 0.1% budget.
The cap stays as a guard rail for a future arm with heavier redundancy.

This is deliberately *not* exp145's policy, which collapsed to one representative
per coarse chain-pair cluster and shrank 2.01M candidates to 17,083. Several
structures of one interface are signal, not duplication.

## Schema notes

- `source_arm` — `afcdb` or `pinder`. Check it before using any score column.
- `quality_ratio`, `ipsae_score`, `pdockq2_score` — AFCDB only, `NULL` for
  PINDER. `pdb_id`, `resolution`, `interface_cluster_id` — PINDER only.
- `confidence_tier` — `A`/`B` for AFCDB, `experimental` for PINDER.
- `partner_a`/`partner_b` — UniProt accessions (AFCDB) or the receptor/ligand
  UniProt ids (PINDER).
- `pair_also_in_afcdb` — PINDER only: whether this UniProt pair also appears in
  the AFCDB arm. `False` for 385,354 of the 428,221.
- `document_id` — `<complex_type>|<modelEntityId>` for AFCDB, the PINDER
  `system_id` otherwise. **AFCDB's `modelEntityId` is not unique across its two
  source tables** (130 ids are both a homodimer and a heterodimer), which is why
  the type is part of the id.
- `num_tokens` is exact against the co-located tokenizer, verified by
  re-encoding.
- All 3,410,738 `sha1` values are distinct: no document appears twice.
