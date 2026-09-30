---
marinfold_experiment:
  issue: 345
  title: 'exp: qualitative contact-map errors on eval-val'
  kind: evals
  branch: main
---

# exp: qualitative contact-map errors on eval-val

**Issue:** [#345](https://github.com/Open-Athena/MarinFold/issues/345) · **Kind:** evals

[Open the self-contained HTML report](report.html), or download it from the
[public artifact directory](https://huggingface.co/buckets/open-athena/MarinFold/tree/data/exp345-qualitative-contact-map-errors/v1).
The HTML works offline in a modern browser. It contains an annotated inspector,
vote-frequency and error views, synchronized drag-to-zoom, per-pair hover details,
contact downloads, and side-by-side prediction/truth cards for **all 97 eval-val
proteins**. The gallery is sortable and searchable.

## Question

What qualitative errors distinguish MarinFold's predictions from the correct
contact maps, and which errors plausibly explain its limited accuracy?

## Hypothesis

Missing or misplaced long-range contact features, diffuse ensemble votes, and
local residue-register shifts may contribute differently. Aggregate precision
alone cannot distinguish them.

## Approach

Reuse exp277's saved epoch-2 evaluation: W&B run
[`contacts-v1-exp277-m2-p06-full-epoch2-from213072-1.5B`](https://wandb.ai/open-athena/MarinFold/runs/contacts-v1-exp277-m2-p06-full-epoch2-from213072-1.5B),
**step 479417**. The score source is
`s3://marin-us-east-02a/MarinFold/exp277_models_single_mpnn_pilot/evals/rollout-v2/2026-09-13/v2-02/dense_scores/exp277_full_epoch2_from213072_step479417/`.
The evaluation completed on 2026-09-19. This analysis uses the latest documented
saved exp277 checkpoint; it does not assume every MarinFold checkpoint behaves
identically.

The exp82 sampling recipe used 100 fresh document realizations per protein,
temperature 1.0, top-p 0.95, disabled top-k, and the 6L+128 token budget subject
to the model context limit. All eval-val rollouts terminated. Only their final
live contacts vote. Counts are marginal occurrence frequencies, not calibrated
probabilities or a single sampled structure.

Truth comes from exp245's frozen `gt_universe_scored.jsonl`, filtered to its
97 natural eval-val proteins. These are **pyconfind side-chain contacts with
degree ≥0.001 and sequence separation ≥6**, not Cα/Cβ distance-threshold maps.
Both endpoints must be resolved. Coordinates are positions in the input
sequence; report labels are one-based. Unresolved residues are gray bands,
never false negatives or true negatives. Both matrix halves are displayed but
unordered pairs are counted once.

The top-R map has exactly as many predicted pairs as the truth, making its
precision equal to recall. It uses the canonical stable vote-count ranking.
R is an evaluation oracle, not information available at deployment. Separately
compute top-L/5 precision, range-specific metrics, true-contact vote support,
and nearest-contact distances. All reported aggregate means weight proteins
equally.

Two deliberately different local-error diagnostics prevent an appealing but
misleading conclusion. Many-to-one proximity asks whether any true contact is
within ±d positions at both endpoints. The stricter oracle keeps exact true
positives, then maximally matches false positives to still-missed true contacts
one-to-one within that radius. It does not permit reusing an already-recovered
contact. Neither diagnostic produces a usable predictor.

## Success criteria and validation

- All 97 eval-val proteins are present, with source matrix SHA256 digests,
  truth and membership digests, the source scoring table, and sampling settings
  in `data/provenance.json`.
- All **1,940 canonical rows** (97 proteins × four ranges × five metrics)
  reproduce exp277's source scores and counts to absolute tolerance 1e−12.
- Browser-side top-R selections independently reproduce all 97 canonical
  per-protein precisions. All 194 gallery canvases render at desktop and mobile
  sizes; controls, filtering, sorting, downloads and layout pass browser checks.
- Independent fixtures distinguish nearby duplicate predictions from distinct
  shift-correctable contacts, guarding against many-to-one inflation.
- No fresh model inference or eval-test analysis was performed. No new W&B run
  or predictor timing file is needed for this reuse-only analysis.

## Results

### Two qualitatively different regimes

The mean all-range R-precision is **0.55750**, long-range R-precision **0.53935**,
and top-L/5 precision **0.81179**. The strongest handful of contacts are much
better than the complete size-matched map; top-L/5 also has much lower coverage.
Individual R-precision ranges from **0.06195 to 0.84337**.

In the lowest-accuracy quartile (25 proteins selected post hoc by precision),
long-range contacts make up **69.3% of truth but only 41.2% of predictions**.
Across all 97 proteins, those fractions are 67.6% and 59.0%. Global-top-R recall
is 64.6% for short-range, 59.8% for medium-range, and 52.4% for long-range truth
contacts. These range recalls use the global contact budget; they are distinct
from the canonical per-range R-precision metric.

The severe failures visibly put dense local contact clusters near the diagonal
where truth contains extended off-diagonal traces. Better cases often recover
the broad pattern but add and omit detailed contacts within those regions.
The illustrative cases below were chosen after map inspection, not as a random
sample; the all-protein atlas and aggregate statistics provide the context.

| Protein | Exact P@R | One-to-one ±2 oracle | Qualitative observation |
|---|---:|---:|---|
| 7wz5_A, croaker IFNi | 6.2% | 15.9% | Missing long-range traces; 91% long-range truth versus 28% predictions |
| 8arl_A, Plasmodium TRAg domain | 6.5% | 11.2% | Diamond-like off-diagonal truth replaced by a near-diagonal band |
| 8bux_A, human MiniBAR Rab-binding domain | 9.6% | 19.2% | N-terminal arc and C-terminal patterns absent or displaced |
| 8dmu_A, Drosophila macrodomain | 15.8% | 27.3% | Broad loss of distant contacts, including between sequence ends |
| 7xz3_A, CRISPR-associated Csh2 | 40.8% | 52.5% | Some traces correct; extensive C-terminal omissions and local clutter |
| 7uk8_A, E. coli YgiC | 52.9% | 70.5% | Recognizable broad map with substantial detailed-contact errors |
| 8b6a_A, Bacteroides BfrB | 63.1% | 73.3% | Several features align, but local overpopulation and missing traces remain |
| 7y8h_A, sDscam FNIII1 | 84.3% | 88.0% | Most major traces and detailed contacts recovered |

![Severe failures](plots/cases_1.png)

![Mixed results and positive control](plots/cases_2.png)

### Small shifts explain only part of the gap

On average, **60.8% of false positives** are within two positions at both
endpoints of some true contact. Many-to-one proximity would therefore call
**80.7%** of the selected predictions near-correct, versus 14.6% expected for a
uniformly chosen candidate pair. That is not a valid replacement for accuracy:
many predicted contacts can cluster around a true contact already recovered.

Fixing exact matches and enforcing one-to-one matching produces a much smaller
change:

| Allowed local displacement | Mean matched fraction |
|---|---:|
| Exact | 55.75% |
| ±1 at each endpoint | 60.52% |
| ±2 at each endpoint | 64.01% |
| ±5 at each endpoint | 71.83% |

The ±2 oracle gains **8.26 percentage points**, equivalent to about 18.7% of
the mean exact-error fraction. Even this truth-informed procedure leaves about
36% unmatched. For 8b6a_A, many-to-one ±2 proximity is 95.4%, but one-to-one
matching is only 73.3%. Visual resemblance is appreciably better than recovery
of the full set of specific contacts.

### Correct contacts are usually sampled, but often weakly

An average **93.9% of true contacts** occur at least once in the 100 rollouts;
**29.3% receive fewer than ten votes**. Even among contacts missed by top-R,
89.0% appear somewhere in the samples. The mean per-protein median vote count
is 62.0 for true positives, 38.7 for false positives, and 10.3 for missed true
contacts. These are means of within-protein medians, not pooled distributions.

Coverage falls to 84.0% in the worst precision quartile. For 8arl_A it is only
56.5%, with a median one vote on missed true contacts. The hardest maps exhibit
both missing support and bad ranking; elsewhere, merely observing a correct
contact is insufficient to rank it among the highest-vote contacts.

![Quantitative diagnostics](plots/diagnostics.png)

### Context and limits

The published decontaminated-AFDB sequence-KNN reference on eval-val is
**0.40715** (exp245/exp277). This reference was not rebuilt over the full
native-plus-redesigned training mixture. The checkpoint is above that reference,
but considerable error remains. Viral eval-val proteins score 0.47959 (n=6) versus 0.56264 for non-viral
proteins (n=91); the small viral group supports only an indicative comparison.
The frozen exp260 natural low-MSA-depth subset has **zero eval-val members**,
so this analysis cannot characterize that regime. No test proteins were added
to fill that gap.

Nearby errors may involve detailed geometry or residue register, but proximity
does not prove a particular structural cause. Marginal vote matrices cannot
reveal whether competing contacts co-occur within a rollout, whether individual
samples are physically consistent, or whether a correct whole fold is ever
sampled. This report does not establish insufficient training diversity,
memorization, a defective loss, or an architectural limitation as the cause.

## Reproduction and artifacts

From this directory, `uv run python analyze.py` downloads the public 2.1 MB
input bundle if absent and rebuilds the numeric tables. The committed
`data/source_reference.csv` is the original eval-val scoring subset.
`uv run python plot_results.py`, `uv run python build_report.py`, and
`uv run python build_summary.py` regenerate figures, HTML and slides.
`uv run python test_analysis.py` checks the local-correction fixtures.
`uv run python check_report.py --browser /path/to/chromium` runs browser checks;
omit `--browser` with a Playwright-managed Chromium installation.

`uv run python analyze.py --fetch-source` recollects source matrices via the
CoreWeave `cw` profile. Ordinary reproduction needs no private credentials.
`uv run python publish_to_hf.py` rebuilds and publishes the report, input bundle,
CSV tables, provenance, case notes and plots to the public artifact directory.
No model weights are downloaded. The source contains only 97 permitted proteins
in its public reproducibility bundle.

## Conclusion

The clearest qualitative defect in the weakest predictions is loss of the
long-range contact pattern, replaced by local clusters. In moderate and strong
predictions, the map can look broadly plausible while still selecting many
wrong detailed contacts. Simple local shifts explain a minority of the error
under a one-to-one accounting. Correct contacts are usually sampled somewhere,
but their vote support is often insufficient.

The next useful experiment is to connect these per-protein patterns to
**individual whole-map rollouts**: does a correct long-range pattern ever appear,
and can a truth-free geometric or learned selector recover it? That separates
sampling coverage, within-sample quality, and marginal-ranking failure. Detailed
local refinement should be evaluated separately from recovering missing global
features. Improving accuracy by showing more permissive proximity scores would
not resolve the underlying contact errors.
