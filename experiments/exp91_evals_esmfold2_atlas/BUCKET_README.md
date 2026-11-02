# ESM Atlas (ESMFold2) distilled structures for MarinFold training

A quality-filtered, novelty-deduplicated subset of the **ESM Atlas** (the
"ESMFold2 Atlas", ~1.1B predicted monomer structures from Biohub's
*Language Modeling Materializes a World Model of Protein Biology*,
[bioRxiv 2026.06.03.729735](https://www.biorxiv.org/content/10.1101/2026.06.03.729735v1)),
materialized to mmCIF for MarinFold training-set expansion.

Each row is one **cluster representative**: its decoded 3D structure (mmCIF text,
per-residue pLDDT in the B-factor column), its sequence, and the selection metadata
that put it here. Built for MarinFold
[issue #91](https://github.com/Open-Athena/MarinFold/issues/91). The column layout is aligned to
[`timodonnell/afdb-24M`](https://huggingface.co/datasets/timodonnell/afdb-24M) so
the two datasets union cleanly into one training corpus.

- **Representatives:** 66,759,963 (one per structural cluster, MMseqs2 linclust @ 40% id)
- **Size:** ~1.9 TiB across **3,338 Parquet parts** of 20,000 structures each
  (the last part holds 19,963)
- **Content:** ESMFold2 monomer structures only (one chain per row)

## Contents

```
structures/parts/part_NNNNN.parquet   # 3,338 parts, 20k structures each; the payload
selected_manifest.parquet              # 66,759,963 rows: the authoritative rep list + scores
README.md                              # this file
```

`selected_manifest.parquet` is the funnel's deliverable (columns `cluster_id,
protein_hash, seq_len, mean_plddt, ptm, plddt_std, cluster_size`); it joins to the
parts on `protein_hash` == `entry_id` if you want to reconcile the selection with
the materialized structures.

### Schema (per part)

Names align with `timodonnell/afdb-24M` where the field exists (so a union is a
straight concat); the rest are Atlas-specific extras.

| Column | Type | Meaning |
|---|---|---|
| `entry_id` | string | `protein_hash` — the Atlas protein id (afdb-24M id column) |
| `cif_content` | string | Decoded mmCIF text; per-residue pLDDT in the B-factor column |
| `sequence` | string | Amino-acid sequence read back from the decoded structure |
| `seq_ok` | bool | Integrity check: decoded sequence == Atlas `sequence` column |
| `split` | string | Constant `"train"` (see caveat) |
| `source` | string | Constant `"esm-atlas-v1"` provenance tag |
| `seq_len` | int | Residue count |
| `global_plddt` | double | Mean pLDDT, **0-1 scale** (afdb-24M name; renamed from Atlas `mean_plddt`) |
| `ptm` | double | Predicted TM-score |
| `plddt_std` | double | Std-dev of per-residue pLDDT |
| `seq_cluster_id` | string | This rep's linclust @ 40%-id cluster (afdb-24M name; renamed from `cluster_id`) |
| `cluster_size` | int | Members in that sequence cluster |

**Caveats for anyone unioning this with afdb-24M or training on it:**

- **pLDDT scale.** `global_plddt` here is **0-1** (Atlas convention). afdb-24M's
  same-named column may be **0-100**. Rescale one side before training on the union.
- **`split` is a placeholder.** Everything is `"train"`. A real train/val/test
  holdout is **deferred** to a later decision coordinated with MarinFold's (unbuilt)
  eval set and its leakage dedup — do not treat this as a validated split.
- **afdb-only columns are absent, not fabricated.** `uniprot_accession`, `tax_id`,
  `organism_name`, `struct_cluster_id`, `gcs_uri` do not exist for Atlas rows and
  are omitted rather than filled with nulls/guesses.
- `seq_ok` is essentially all `True` (~1 mismatch per 20k spot-checked — a trace
  integrity signal, not a filter that was applied).

## Provenance

Derived from `s3://esm-protein-atlas/v1/folds/folds_1B.lance` (1,095,530,880 rows;
structures predicted with `esmfold2-exp-2026-03`, 3 recycles / 22 diffusion steps).
The 1.1B Atlas monomers were reduced to these 66.76M representatives by a five-stage
funnel:

| Stage | Result |
|---|---|
| scan | 383,988,806 survivors (mean pLDDT ≥ 0.70, pTM ≥ 0.50, length 60-1000) |
| novelty | −220,802,158 redundant vs afdb-24M @ 40% id (57.5%) |
| leakage | −41,517 vs the eval reference → **163,144,153** kept |
| cluster | 163M → **66,759,963** clusters (MMseqs2 linclust @ 40% id) |
| select | one rep per cluster (longest, then lowest pLDDT std) |

Then each selected rep's `structure_blob` was decoded (brotli + msgpack atom37 →
mmCIF) and written to the parts here.

- **Source dataset:** ESM Atlas v1 `folds_1B.lance` (AWS Open Data, `us-west-2`)
- **Novelty/leakage reference:** afdb-24M cluster reps via the exp41 foldseek DB
  ([`silterra/afdb-24M-foldseek-train-reps`](https://huggingface.co/buckets/silterra/afdb-24M-foldseek-train-reps))
- **Builder:** [`experiments/exp91_evals_esmfold2_atlas/`](https://github.com/jsilter/MarinFold/tree/exp/91-evals-esmfold2-atlas)
  in the MarinFold repo (`pipeline.py` funnel → `materialize.py` decode →
  `upload_to_hf.py` publish), run on AWS `us-west-2`, 2026-07.
- **Issue / discussion:** [Open-Athena/MarinFold#91](https://github.com/Open-Athena/MarinFold/issues/91)

## Use

```python
import pyarrow.dataset as ds

# Stream the parts directly from the bucket (or `hf buckets sync` them local first).
data = ds.dataset("hf://buckets/<owner>/<bucket>/structures/parts", format="parquet")
batch = data.head(4)
cif = batch.column("cif_content")[0].as_py()   # parse with gemmi / biotite
```

This is the input format `marinfold contacts-v1 generate` reads (`entry_id` +
`cif_content`), so it feeds document generation unchanged.

## License & attribution

Released under [**CC BY 4.0**](https://creativecommons.org/licenses/by/4.0/): free
to use, share, and adapt with attribution.

This is a **modified, derived** dataset — a quality-filtered, novelty-deduplicated,
re-clustered, and re-materialized subset of the ESM Atlas (see [Provenance](#provenance)).
It is **not** the original Atlas data.

**Attribution for the source data** (required by the Atlas's CC BY license):

> ESM Atlas (ESMFold2), Biohub. *Language Modeling Materializes a World Model of
> Protein Biology.* bioRxiv 2026.06.03.729735.
> <https://doi.org/10.1101/2026.06.03.729735>. Licensed under
> [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

The paper states CC BY; the AWS Open Data registry lists CC BY-SA. We treat the
paper's CC BY as authoritative for this release.

**If you use this dataset**, please attribute both the source above and this derived
dataset (`open-athena/esm-atlas-esmfold2-distill`, MarinFold exp91).
