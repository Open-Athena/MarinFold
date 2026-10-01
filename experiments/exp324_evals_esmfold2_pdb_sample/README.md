---
marinfold_experiment:
  issue: 324
  title: "exp: analyze MarinFold vs ESMFold2 performance across PDB-deduped monomers"
  kind: evals
  branch: exp/324-esmfold2-pilot
---

# exp: analyze MarinFold vs ESMFold2 performance across PDB-deduped monomers

**Issue:** [#324](https://github.com/Open-Athena/MarinFold/issues/324) · **Kind:** `evals` · **Branch:** `exp/324-esmfold2-pilot`

## Question

Given matched ESMFold2 and MarinFold contact scores on the same 10k
PDB-deduped experimental monomers, what biology/structure/provenance features
explain the MarinFold-vs-ESMFold2 gap? Which protein subgroups does MarinFold
perform relatively well or poorly on?

## Hypothesis

ESMFold2 will remain the strongest aggregate predictor, but MarinFold will close
the gap on identifiable protein slices where language-model priors are useful or
single-sequence structure prediction is less dominant: synthetic/designed
proteins, viral proteins, rare experimental modalities such as Solution NMR, and
specific GO/CATH structural-function groups. Conversely, highly represented
natural structural classes and some complex cellular machinery should be
ESMFold2-favorable.

## Approach

1. Sample 100 single-chain deduped PDB monomer documents from the public HF
   bucket corpus, filtering to a tractable pilot length range.
2. Run ESMFold2 single-sequence predictions on CoreWeave H100.
3. Persist one predicted mmCIF and timing/provenance JSON per protein.
4. Score predicted structures in contact space with the same pyconfind/R-precision
   machinery used for MarinFold comparisons.

The comparison target is contact-space ranking, not raw coordinate RMSD:

```text
sequence -> ESMFold2 predicted structure -> pyconfind contact_degree matrix
sequence -> MarinFold pairwise score matrix
both -> same GT experimental-PDB pyconfind contacts -> same R-precision/AUC metrics
```

## Success criteria

- Produce a fixed 10k manifest and reproducible scoring pipeline for ESMFold2
  `n_samples=100` and MarinFold 100-rollout checkpoints.
- Compile per-protein R-precision/AUC metrics with compact feature annotations
  for taxonomy, GO-slim, CATH, experimental method, synthetic status, and a
  UniRef depth proxy.
- Identify interpretable feature groups where MarinFold closes the ESMFold2 gap,
  and distinguish direction (MarinFold-favorable vs ESMFold2-favorable) from
  feature importance.
- Keep generated statistics reproducible from tracked scripts and durable remote
  result prefixes rather than checking large generated outputs into git.

## Outputs

Working CoreWeave S3 prefix:

```text
s3://marin-us-east-02a/protein-structure/MarinFold/exp324_esmfold2_pdb_sample/
```

Local pilot manifest:

```text
data/sample_100_manifest.csv
```

## Results

Pilot status as of 2026-09-23:

- Sampled 100 proteins from `contacts_v1_pdb_deduped_monomers`, filtering to
  canonical sequences with `50 <= L <= 512` for this first smoke.
- Smoke-folded one protein successfully on CoreWeave H100.
- Launched the 100-protein ESMFold2 `n_samples=1` pilot as four independent
  CoreWeave root jobs:
  - `/zack/exp324-esmfold2-pilot100-s000-of-004`
  - `/zack/exp324-esmfold2-pilot100-s001-of-004`
  - `/zack/exp324-esmfold2-pilot100-s002-of-004`
  - `/zack/exp324-esmfold2-pilot100-s003-of-004`
- All four jobs succeeded.

Prediction artifacts are under:

```text
s3://marin-us-east-02a/protein-structure/MarinFold/exp324_esmfold2_pdb_sample/pilot_100_n1/
```

Each protein writes:

```text
structures/<stem>/structure.cif
structures/<stem>/provenance.json
structures/<stem>/timings.json
```

Next step: score these predicted structures with pyconfind against the original
experimental-PDB contacts and compute the same R-precision/AUC metrics used for
MarinFold.

## 10k run

Sampled a 10,000-protein set for the feature/delta table:

```text
uv run --no-project --with 'huggingface_hub>=1.5' --with pyarrow --with pandas \
  python experiments/exp324_evals_esmfold2_pdb_sample/sample_pdb_deduped_monomers.py \
  --n 10000 --seed 32410 --min-len 50 --max-len 1024 --order length_desc \
  --out experiments/exp324_evals_esmfold2_pdb_sample/data/sample_10000_manifest.csv
```

This filters out tiny peptides and the longest 194 chains in the corpus for this
first large pass, leaving 30,366 eligible rows. The sampled set has median length
205 and max length 1024. The manifest is length-descending so modulo sharding
balances long proteins across workers.

Launched ESMFold2 `n_samples=1` over the 10k manifest as 32 CoreWeave H100 root
jobs:

```text
/zack/exp324-esmfold2-10k-s000-of-032
...
/zack/exp324-esmfold2-10k-s031-of-032
```

Output prefix:

```text
s3://marin-us-east-02a/protein-structure/MarinFold/exp324_esmfold2_pdb_sample/pdb_deduped_10k_n1/
```

The first long proteins are slow but healthy: ~5–7.5 minutes for 900–1024 aa
chains on H100; shorter chains in the pilot ran in a few seconds each. The worker
records row-level failures and continues, so one bad/too-large protein should not
kill a shard.

All 32 ESMFold2 prediction jobs succeeded in about 1.5 hours wall-clock.

Scoring against the emitted contacts-v1 PDB contacts completed for all 10,000
proteins. Scoring outputs:

```text
s3://marin-us-east-02a/protein-structure/MarinFold/exp324_esmfold2_pdb_sample/pdb_deduped_10k_n1_scored/
  contact_precision.csv
  contact_eval_meta.csv
  contact_precision_summary.csv
  failures.csv
  score_manifest.json
```

Headline ESMFold2 `n_samples=1` contact metrics, ranking pyconfind contact
degrees from the predicted structures against emitted experimental-PDB contact
pairs:

| range | cut | n scored | mean precision | median precision | mean AUC | median AUC |
|---|---:|---:|---:|---:|---:|---:|
| all | R | 9,945 | 0.743849 | 0.816568 | 0.906478 | 0.945437 |
| short | R | 9,938 | 0.744265 | 0.810811 | 0.907663 | 0.944027 |
| medium | R | 9,838 | 0.762723 | 0.833333 | 0.917205 | 0.953215 |
| long | R | 9,746 | 0.738873 | 0.812500 | 0.905676 | 0.944504 |

`n scored` is below 10,000 for R-precision because proteins/ranges with zero GT
contacts have undefined precision@R. All 10,000 proteins scored successfully;
`failures.csv` is empty.


## 10k apples-to-apples ESMFold2 vs MarinFold comparison

We scaled the pilot into an apples-to-apples 10k-protein contact-space
comparison. All rows share the same sequence, same experimental-PDB pyconfind GT
contacts, same resolved-residue/candidate-pair universe, and the same
R-precision/AUC scoring. ESMFold2 is scored from `n_samples=100` predicted
structures converted back into pyconfind contact-degree rankings; MarinFold
checkpoints are scored from 100 contacts-v1 rollouts with the exp82-style
rollout+resample recipe.

Generated feature/performance table (not checked into git; regenerate locally):

```text
data/sample_10000_feature_performance_table.parquet
data/sample_10000_feature_performance_table.csv
```

Durable remote result prefixes used by the compiler:

```text
s3://marin-us-east-02a/protein-structure/MarinFold/exp324_esmfold2_pdb_sample/pdb_deduped_10k_n100_scored/score_shards/
s3://marin-us-east-02a/protein-structure/MarinFold/exp324_esmfold2_pdb_sample/pdb_deduped_10k_n100_scored_rescue_lesshalf_20260924/score_shards/
s3://marin-us-east-02a/protein-structure/MarinFold/exp324_esmfold2_pdb_sample/marinfold_10k_rollout_scores/
```

Coverage:

```text
rows:                         10,000
ESMFold2 n=100 metric stems:   9,998
MarinFold metric stems:       10,000
stems with UniRef50 proxy:     9,484
```

Headline all-range R-precision on the shared 10k sample:

| predictor | all-range R-precision |
| --- | ---: |
| ESMFold2 `n_samples=100` | 0.7224 |
| exp277 full-epoch contacts-v1, step 266344, 100 rollouts | 0.5338 |
| contacts-v2 / delta-stream, step 78499, 100 rollouts | 0.4909 |
| exp117 old-recipe control, step 35679, 100 rollouts | 0.3953 |
| exp177 contacts-v1 control, step 71359, 100 rollouts | 0.3499 |
| exp157 RoPE-delta diagnostic, step 71359, 100 rollouts | 0.0390 |

The 10k table also joins compact biology/metadata features:

- NCBI taxonomy lineage (`lineage_level_1`, `lineage_level_2`)
- GO-slim function/process/location tags
- CATH structural class/topology tags
- synthetic/protein-engineering flags
- experimental method and resolution
- UniRef100/90/50 cluster sizes as a cheap homolog-depth proxy

## Feature-importance findings

We tested which features explain the paired gap
`MarinFold R-precision - ESMFold2 R-precision`, using adjusted OLS/ANOVA models
with covariates for sequence length, emitted-contact count, resolution, and
UniRef50 depth. Importance is reported as partial eta-squared; direction is the
adjusted effect sign. Positive direction means the MarinFold checkpoint closes
the ESMFold2 gap, while negative direction means the feature is ESMFold2-favorable.

For **contacts-v2 / delta-stream vs ESMFold2**, the highest-importance features
were:

| feature family | feature/tag | direction | partial eta² | interpretation |
| --- | --- | ---: | ---: | --- |
| taxonomy | `lineage_level_2 = synthetic_construct` | +0.202 | 0.0537 | contacts-v2-favorable |
| taxonomy | `lineage_level_1 = synthetic` | +0.188 | 0.0529 | contacts-v2-favorable |
| GO-slim | `organelle` | -0.090 | 0.0365 | ESMFold2-favorable |
| CATH class | `alpha_beta` | +0.066 | 0.0213 | contacts-v2-favorable |
| GO-slim | `ribosome` | -0.151 | 0.0160 | ESMFold2-favorable |
| method | `SOLUTION NMR` | +0.113 | 0.0149 | contacts-v2-favorable |
| CATH topology | `3.40.50` | +0.079 | 0.0148 | contacts-v2-favorable |
| GO-slim | `structural molecule activity` | -0.116 | 0.0141 | ESMFold2-favorable |

For **exp277 vs ESMFold2**, the top features were similar:

| feature family | feature/tag | direction | partial eta² | interpretation |
| --- | --- | ---: | ---: | --- |
| taxonomy | `lineage_level_2 = synthetic_construct` | +0.202 | 0.0303 | exp277-favorable |
| taxonomy | `lineage_level_1 = synthetic` | +0.170 | 0.0276 | exp277-favorable |
| GO-slim | `ribosome` | -0.162 | 0.0229 | ESMFold2-favorable |
| GO-slim | `organelle` | -0.062 | 0.0217 | ESMFold2-favorable |
| method | `SOLUTION NMR` | +0.135 | 0.0210 | exp277-favorable |
| GO-slim | `structural molecule activity` | -0.118 | 0.0180 | ESMFold2-favorable |
| CATH class | `alpha_beta` | +0.053 | 0.0170 | exp277-favorable |
| CATH topology | `3.40.50` | +0.059 | 0.0100 | exp277-favorable |

The main qualitative result is that synthetic/protein-engineering status,
Solution NMR, and some CATH structural groups are MarinFold-favorable slices,
while ribosome/organelle/structural-molecule GO tags are ESMFold2-favorable.

## Categories where MarinFold closes the ESMFold2 gap

The following table reports mean all-range R-precision by category, plus the
paired gap and per-protein win fraction. Most categories still have ESMFold2
ahead on mean R-precision; the intended claim is that MarinFold closes the gap
substantially in these slices.

| category | n | ESMFold2 R | contacts-v2 R | exp277 R | contacts-v2 gap | exp277 gap | contacts-v2 win % | exp277 win % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| All proteins | 9,943 | 0.722 | 0.491 | 0.534 | -0.231 | -0.189 | 9.1 | 11.1 |
| Synthetic | 92 | 0.727 | 0.595 | 0.637 | -0.132 | -0.090 | 26.1 | 32.6 |
| Solution NMR | 622 | 0.558 | 0.430 | 0.485 | -0.128 | -0.073 | 29.1 | 38.9 |
| Riboviria | 195 | 0.389 | 0.212 | 0.242 | -0.176 | -0.147 | 23.1 | 26.7 |
| Duplodnaviria | 175 | 0.553 | 0.419 | 0.448 | -0.133 | -0.105 | 20.0 | 24.0 |
| Varidnaviria | 53 | 0.524 | 0.356 | 0.368 | -0.169 | -0.156 | 28.3 | 22.6 |
| CATH topology 1.20.5 | 70 | 0.484 | 0.399 | 0.409 | -0.085 | -0.075 | 18.6 | 24.3 |
| GO: cell wall organization/biogenesis | 106 | 0.765 | 0.599 | 0.617 | -0.166 | -0.148 | 7.5 | 8.5 |

Notebook-ready analysis snippets for comparisons and ANOVA/OLS helpers are in:

```text
notebook_quick_analysis_snippets.md
```

Generated statistics are intentionally not tracked. Rebuild them from tracked
scripts plus the durable S3 outputs:

```bash
uv run --no-project --with pandas --with pyarrow --with fsspec --with s3fs \
  python experiments/exp324_evals_esmfold2_pdb_sample/compile_10k_table.py \
  --allow-incomplete-esmfold

uv run --no-project --with pandas --with pyarrow --with numpy --with scikit-learn \
  python experiments/exp324_evals_esmfold2_pdb_sample/run_fisher_lda.py \
  --table experiments/exp324_evals_esmfold2_pdb_sample/data/sample_10000_feature_performance_table.parquet \
  --out-dir experiments/exp324_evals_esmfold2_pdb_sample/data/fisher_lda_with_uniref \
  --margin 0.02 \
  --min-feature-count 50
```


## Conclusion

ESMFold2 is still substantially ahead on aggregate contact R-precision on this
10k PDB-deduped sample, but the per-protein table is useful for finding where
MarinFold is relatively strongest. Synthetic/protein-engineering status,
Solution NMR, and selected CATH structural groups are MarinFold-favorable slices;
ribosome/organelle/structural-molecule GO tags are ESMFold2-favorable. The
headline claim should be framed as **gap closing and higher win fraction in
specific subgroups**, not aggregate superiority over ESMFold2.
