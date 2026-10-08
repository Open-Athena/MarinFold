# AlphaFold3 sampling on five very shallow MSAs

User-requested diagnostic on `8ii8_A`, `8oxk_A`, `8qoh_A`, `8ux2_A`, and
`8wrx_A`. All five are natural FoldBench eval-test proteins, with archived MSA
depths 3, 6, 3, 6, and 4. These are raw A3M row counts including the query,
not effective evolutionary counts or estimates of training-set exposure.

## Protocol

The [frozen protocol](data/af3_sampling_protocol.json) specifies 100 fresh runs
per protein, extended to 1,000 for all five if any protein has no TM-score
at least 0.8 in the first 100. The threshold is a working definition of an
accurate prediction for this experiment; continuous scores and counts above
0.5 and 0.7 are also retained.

Each seed is a **complete independent AF3 forward run**, including fresh
featurisation and the trunk, followed by **one diffusion sample**. Seeds
10000–10999 form the fixed sampling order. Ten recycles, the exact archived
ColabFold MSA, protein chain only, no templates, no relaxation. This tests
additional sampling under the existing shared-MSA recipe; it does not test
AF3's full default database search or alternate MSA/assembly inputs.

Official AF3 source: `3c89cc7b89aa7042b72885af9016a45b262da008`. Parameters:
`params-20241113/af3.bin.zst`, SHA-256
`0dd0290af76eec0119f4c0c41f441fbce4ec18aa21a0a02cac4a3af01d841369`.
The adapter hash, per-protein sequences and MSA hashes are in the protocol.
Weights remain private in the existing Modal volume; no new weight transfer
was required. Inference uses up to 20 resident H100 workers in us-east.

**Highest full-precision pTM selects a prediction.** Official AF3
`ranking_score` selection is retained as a separate diagnostic. The standard
AF3 confidence JSON rounds values to two decimals; new runs additionally save
unrounded pTM and ranking_score directly from model metadata. Exact ties use
the earliest seed. The best TM-score uses reference structures only for
offline analysis and is explicitly an oracle, not an inference-time selector.

## Accuracy and original-run comparison

The scorer verifies predicted sequence/residue correspondence and matches
atoms by query position and name. TM-score uses matched CA atoms through
TM-align, normalized by the reference chain; GDT-TS, all-atom lDDT and RMSD
use the same code as the other exp325 AF3 comparisons. Helico metric revision:
`b10385d736673c81b10e70d1099962af6f2573c0`. No Helico model runs in this analysis.

The original AF3 baseline had **five full seeds × five diffusion samples**,
selected by official ranking_score. All 25 original candidates are scored
separately, never added to the new seed prefixes. Their historical pTM JSON
values have two-decimal precision. A regression check requires reproducing
the original five selected TM-scores within 1e-8.

The curves follow the prespecified ascending-seed order. They describe this
realized sample pool, not an expectation over random sampling orders. Five
proteins remain five biological examples; 5,000 draws do not enlarge the
benchmark population. A failure to observe TM ≥0.8 does not show that AF3 can
never generate an accurate structure.

## Context: the other predictors

The 8oxk_A recovery is substantial relative to AF3's original prediction, but
ESMFold2 already achieves TM **0.940** on this protein. AF3's pTM-selected
1,000-run result reaches **0.947**. This is AF3 catching up on this case.

Across these five proteins, mean TM is **0.603** for original AF3, **0.681**
for official-ranking selection from 1,000 runs, **0.719** for pTM selection,
and **0.780** for oracle best-of-1,000. The archived ESMFold2 baseline averages
**0.854**. At the prespecified TM ≥0.8 threshold, the original AF3 prediction
succeeds on 2/5 proteins, either 1,000-run confidence selector on 3/5, and
ESMFold2 on 4/5. These are descriptive results on five proteins, with wide
protein-bootstrap intervals retained in the prepared table.

| Protein | AF3 baseline: 25, official rank | AF3: 1,000, official rank | AF3: 1,000, pTM | ESMFold2 | Protenix-v2 + MSA |
|---|---:|---:|---:|---:|---:|
| 8ii8_A | 0.861 | 0.849 | 0.849 | 0.849 | 0.672 |
| 8oxk_A | 0.472 | 0.947 | 0.947 | 0.940 | 0.324 |
| 8qoh_A | 0.428 | 0.394 | 0.472 | 0.882 | 0.467 |
| 8ux2_A | 0.361 | 0.318 | 0.434 | 0.708 | 0.740 |
| 8wrx_A | 0.895 | 0.895 | 0.895 | 0.894 | 0.757 |

ESMFold2 reaches 0.882 on 8qoh_A, compared with AF3's best-of-1,000 of 0.502.
On 8ux2_A, Protenix-v2 + MSA reaches 0.740 and ESMFold2 0.708, versus AF3's
best-of-1,000 of 0.635. More AF3 sampling does not close these gaps here.
Its aggregate gain is driven mainly by 8oxk_A; confidence-selected accuracy
need not increase with budget, as the individual curves show.

![AF3 and other predictors on each low-depth protein](plots/01c_af3_context.png)

These ESMFold2 entries are the archived predictor baseline, **not** the mean,
median or oracle-selected member of the later 100-prediction decoy pool.
The figure also includes the current 248B-token MarinFold top-L contacts through
Helico. Helico with the oracle map and oracle best AF3 use experimental truth
and are explicitly diagnostic rows. Budgets and training data differ across
predictors; this is not a matched-compute comparison.

## Context: other MSA depths

The same **305 matched natural proteins** used in the main structural panels
are used here, with **TM-score throughout**. Each protein has equal weight.
The main panels' GDT-TS/lDDT numbers should not be directly compared with these
TM-scores. The four bins contain 5, 20, 60 and 220 proteins, respectively.

| Predictor / selector | <10 (n=5) | 10–99 (n=20) | 100–999 (n=60) | ≥1000 (n=220) |
|---|---:|---:|---:|---:|
| AF3 baseline: 25, official rank | 0.603 | 0.891 | 0.932 | 0.955 |
| AlphaFold2 + MSA | 0.457 | 0.889 | 0.921 | 0.954 |
| Boltz-2 + MSA | 0.535 | 0.886 | 0.928 | 0.955 |
| Protenix-v2 + MSA | 0.592 | 0.900 | 0.926 | 0.957 |
| ESMFold2 | 0.854 | 0.691 | 0.874 | 0.944 |
| ESMFold | 0.698 | 0.624 | 0.829 | 0.925 |
| Protenix-v2 single sequence | 0.504 | 0.545 | 0.489 | 0.451 |
| Helico + MarinFold (248B) | 0.545 | 0.570 | 0.648 | 0.796 |
| AF3: 1,000, official rank | 0.681 | not run | not run | not run |
| AF3: 1,000, pTM | 0.719 | not run | not run | not run |

ESMFold2 is strongest among the archived predictors in the <10 bin, while the
existing AF3 baseline has higher mean TM than ESMFold2 in all three deeper
bins. These bins contain different proteins: this is **not** evidence for a
causal MSA-depth effect or a controlled MSA downsampling experiment.
The 1,000-run AF3 study has **not been run at higher depths**.

![Predictor TM-score across the four MSA depth bins](plots/01d_af3_depth_context.png)

## What is standard AF3 inference here?

The pinned official implementation defaults to **ten recycles and five diffusion
samples per seed**. Seed count is supplied by the user; five full seeds is our
chosen baseline budget, not a universal CLI default. See the
[official runner at the evaluated revision](https://github.com/google-deepmind/alphafold3/blob/3c89cc7b89aa7042b72885af9016a45b262da008/run_alphafold.py).

| Setting | Native AF3 pipeline / inference defaults | Our archived baseline | Our expanded sampling |
|---|---|---|---|
| Full seeds | User specifies | 5 (42–46) | 100, then 1,000 (10000–10999) |
| Diffusion samples / seed | 5 | 5 | 1 |
| Predicted structures / protein | 5 × chosen seeds | 25 | 100, then 1,000 |
| Recycles | 10 | 10 | 10 |
| MSA and templates | Native searches when omitted | Fixed archived ColabFold A3M; no templates | Same fixed A3M; no templates |
| Final selection | Official ranking_score | Official ranking_score | pTM and official ranking_score reported separately |
| Evaluated population here | Full native pipeline not evaluated | All 305 matched natural proteins | Five depth <10 proteins only |

AF3 documents both automatic MSA/template searches and custom input overrides.
We explicitly supply `unpairedMsa`, set `pairedMsa` to an empty string and
`templates` to an empty list. Therefore neither set of our results measures
the complete native search pipeline. See the
[official input documentation](https://github.com/google-deepmind/alphafold3/blob/3c89cc7b89aa7042b72885af9016a45b262da008/docs/input.md).

For these single-chain inputs, official ranking_score is
`pTM + 0.5 × fraction_disordered − 100 × has_clash`, whereas pTM selection
uses only pTM. These are distinct selectors even on monomers, as the
[official confidence implementation](https://github.com/google-deepmind/alphafold3/blob/3c89cc7b89aa7042b72885af9016a45b262da008/src/alphafold3/model/confidences.py)
shows. The official-rank rows keep the selector consistent between 25 and
1,000 predictions; the pTM rows report the alternative requested for this post.
Oracle best TM uses the experimental reference and is not a deployable ranker.

The 1,000-run study generates **40× as many structures but 200× as many full
trunk evaluations** as the five-seed baseline. Neither ratio is a measured
runtime multiplier. We retained input evidence while spending more inference
on independent full seeds; this does not test additional MSA evidence, templates,
multimer context, or an equal-compute tradeoff between seeds and diffusion samples.

## Reproduce and trace each point

Run commands from this experiment directory:

```bash
AF_VARIANT=af3 uv run --project generation modal run generation/run_af3_sampling.py --budget 100
uv run --project /home/bizon/git/helico --no-sync --with scikit-learn python generation/score_af3_sampling.py --budget 100
uv run python prepare_af3_sampling.py
# If any first-100 pool has no TM >=0.8, extend all five:
AF_VARIANT=af3 uv run --project generation modal run generation/run_af3_sampling.py --budget 1000
uv run --project /home/bizon/git/helico --no-sync --with scikit-learn python generation/score_af3_sampling.py --budget 1000
uv run python prepare_af3_sampling.py
# Build the matched-predictor TM tables once (no new inference):
uv run python prepare_af3_context.py
# Aesthetics-only iteration, with no inference or structural scoring:
uv run python render_af3_context.py
uv run python render_af3_sampling.py
uv run python build_summary.py
uv run python publish_af3_sampling.py --upload
```

The runner verifies weights, MSA and protocol on workers, gates the pilot on
the longest protein, commits bounded seed chunks, and resumes completed seeds.
`--collect-only` recovers output without generating predictions. Local
structure-score caches are keyed by structure/reference hashes, sequence and
metric implementation. Parameter files are excluded from exported artifacts.

| Figure element | Prepared source | Analysis |
|---|---|---|
| One TM versus pTM dot | `data/af3_sampling_samples.csv`, keyed by `(stem, seed)` | `generation/score_af3_sampling.py`; CIF path and SHA-256 on the same row |
| Best / pTM-selected curve at budget N | `data/af3_sampling_curves.csv`, `(stem, budget)` | `prepare_af3_sampling.py`; selected seed IDs on the same row |
| 100 / 1,000 comparison and hit counts | `data/af3_sampling_summary.csv` | Fixed prefix rows from the same curves |
| Original selected baseline | `data/af3_sampling_original.csv` | Highest official ranking_score among the original 25 candidates |
| Predictor comparison cell / horizontal reference | `data/af3_context_rows.csv`, `(stem, method)` | `prepare_af3_context.py`; original file, zero-based row and column retained |
| MSA-tier mean / hover interval | `data/af3_context_summary.csv`, `(method, tier)` | Same script; contributing `context_rows`, protein bootstrap |
| Per-run timing and hardware | `data/af3_sampling_timings.csv` | Captured at inference; model setup is shared across seeds and reported separately |

Raw structures, confidences, timings, input MSAs and reference CIFs are
published in the [public artifact directory](https://huggingface.co/buckets/open-athena/MarinFold/tree/data/exp325-writeup-analysis/af3-sampling-2026-10-06).
`data/af3_sampling_publication.json` contains the file inventory and checksums.
Extract each `raw/<stem>.tar.gz` into `scratch/af3_sampling/results/` to restore
the paths in the per-sample CSV. The public package includes no weights.

## Results

All **5,000 new predictions completed**: 1,000 independent full runs for each protein. The first 100 results and their timing records are unchanged from the pilot. Every candidate matched the full resolved CA chain; the original five selected baseline TM-scores reproduce within 1e-8.

| Protein | Best TM, original 25 | Best TM, new 100 | Best TM, new 1,000 | pTM-selected TM, new 1,000 | TM ≥0.8 / 1,000 |
|---|---:|---:|---:|---:|---:|
| 8ii8_A | 0.874 | 0.876 | 0.881 | 0.849 | 978 |
| 8oxk_A | 0.516 | 0.818 | 0.947 | 0.947 | 20 |
| 8qoh_A | 0.471 | 0.473 | 0.502 | 0.472 | 0 |
| 8ux2_A | 0.432 | 0.505 | 0.635 | 0.434 | 0 |
| 8wrx_A | 0.918 | 0.919 | 0.936 | 0.895 | 1000 |

**Extra sampling rescues 8oxk_A.** The first TM ≥0.8 appears on draw 85
(seed 10084), and the best structure reaches TM 0.947 on draw 471 (seed 10470).
Highest pTM also selects seed 10470. Twenty of the 1,000 runs (2%) reach
TM ≥0.8. At a budget of 100, the best candidate's TM was 0.818 while the
pTM-selected candidate's TM was 0.643; increasing the pool improves both
candidate generation and final confidence selection in this case.

8ii8_A and 8wrx_A already fold well in the original baseline. 8qoh_A and
8ux2_A remain below TM 0.8 after 1,000 runs (maxima 0.502 and 0.635), and
neither reaches 0.7. This is a failure to find an accurate prediction within
this fixed budget and input recipe, not evidence that accuracy is impossible.

![AF3 sampling by protein](plots/01b_af3_sampling.png)

Per-protein PDFs: [8ii8_A](plots/01b_af3_sampling_8ii8_A.pdf) ·
[8oxk_A](plots/01b_af3_sampling_8oxk_A.pdf) ·
[8qoh_A](plots/01b_af3_sampling_8qoh_A.pdf) ·
[8ux2_A](plots/01b_af3_sampling_8ux2_A.pdf) ·
[8wrx_A](plots/01b_af3_sampling_8wrx_A.pdf).
The [complete slide deck](plots/summary.pdf) includes all five as individual pages.
