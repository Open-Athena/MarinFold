---
marinfold_experiment:
  issue: 325
  title: 'exp: analyses for a writeup'
  kind: evals
  branch: exp/325-writeup-analysis
---

# exp: analyses for a writeup

[Issue #325](https://github.com/Open-Athena/MarinFold/issues/325) ·
[Branch](https://github.com/Open-Athena/MarinFold/tree/exp/325-writeup-analysis)

A figure-first analysis of FoldBench predictors, oracle-conditioned Helico,
MarinFold contact prediction, downstream folding, and sampling diversity.
[DRAFT.md](DRAFT.md) is deliberately terse placeholder prose. No pull request
or website modification is part of this work.

Start with [the interactive preview](site/index.html) or [the PDF](plots/summary.pdf).
[FIGURES.md](FIGURES.md) maps each visual element to source rows;
[RUNBOOK.md](RUNBOOK.md) gives reproduction and restyling commands.

## Model and scope

Every MarinFold result uses **`contacts-v1-exp277-m2-p06-full-epoch-1.5B`, step
266,344**: a scratch-trained 1.47B-parameter Qwen3, one epoch over 232,090,905
documents containing **248,583,762,834 raw tokens**. Native and ProteinMPNN
redesign sequences use the decontaminated AFDB/ESM source corpus. Raw tokens
are distinct from padded training slots. No older MarinFold predictions enter
these figures. The exact weights and tokenizer are pinned in
[`generation/checkpoint_manifest.json`](generation/checkpoint_manifest.json).

The user explicitly authorized the test split for this publication. The 333
scorable monomers comprise 97 natural validation proteins, 217 natural test
proteins, and 19 designs. Natural proteins lead; designs remain separate.
The checkpoint, sampling recipe, top-L cut and control target list were fixed
before looking at test results. See [the test-read ledger](../exp245_evals_foldbench_held_out_monomers/data/eval_test_reads.md).

## Approach

Predictor work lives in `generation/`. `prepare.py` performs joins, complete-case
selection and bootstraps once. `render.py` reads only cached tables and exports
PNG/SVG, native website Plotly JSON, mobile variants and an offline preview.
Change `theme.py` or `render.py` to restyle without rerunning any analysis.

Every plotted value retains its source file and zero-based row in
`data/figure_rows.csv`. `summary.csv` stores means, intervals, sample sizes and
protein IDs; `paired_deltas.csv` stores within-protein comparisons. The manifest
pins input/output hashes. `coverage.csv` lists exclusions; missing results are
never treated as zero.

MSA depth is the query-inclusive alignment sequence count, not Neff, binned at
10, 100 and 1,000. Only **five** natural proteins occupy the lowest tier.
The target table distinguishes full prompt length `L` from resolved MSA query
length. Intervals use 5,000 protein bootstraps, seed 325. These are descriptive
strata, not the causal effect of changing an individual protein's MSA depth.

Contact metrics use exp89's unchanged resolved-pair universe and R-precision
implementation. The KNN baseline indexes the native decontaminated corpus,
not the additional redesign sequences. External predictors provide benchmark
context; their training contamination controls are not assumed identical.

Helico receives the top L predicted contacts in verified residue coordinates.
Its usual filter discards pairs that become closer than six residues after
mapping; requested and effective counts are both saved. Other pairs remain
unknown. Confidence selects one of three diffusion samples at the fixed cut.
Benchmark entities, including non-protein entities where present, are retained.
GDT-TS and lDDT remain distinct metrics.

The oracle supplies the **full ground-truth three-state map**, including true
non-contacts. It is an information upper bound, not a deployable predictor or
a matched-count comparison with MarinFold. The random-map control separately
matches its known mask, positive/negative counts and diffusion budget.

## Results

- **Contacts:** natural R-precision **0.5609** (validation 0.5537; test 0.5641).
  MSA-tier means are **0.3453, 0.3428, 0.4486, 0.6167**, with n=5/21/62/226.
- **Confidence:** all **20/20 proteins** rank their oracle map above every
  uniform and separation-matched random map, under both ranking_score and
  pLDDT. This is 660 structures with equal per-map budgets. Selected oracle
  GDT-TS averages **0.9189**, versus 0.1269/0.1406 for the controls. Random
  maps are weak negatives; this does not establish ranking of plausible folds.
- **Sampling:** mean individual R-precision **0.4009**, consensus **0.5605**,
  oracle best-of-100 **0.5285** on 314 natural proteins. Paired oracle minus
  consensus is **−0.0321**, 95% interval **[−0.0384, −0.0254]**.
- **Diversity:** every protein has 100 distinct sampled maps; mean Jaccard is
  0.2774 and true-contact union recall 0.9432. 177/31,400 individual maps are
  validity-gated to zero. Distinct contact strings do not establish distinct folds.
- **Folding:** on 305 matched natural proteins, top-L conditioning raises
  Helico GDT-TS from **0.1500 to 0.5044** and lDDT from **0.3627 to 0.6368**.
  The paired GDT-TS gain is **0.3544**, 95% interval **[0.3228, 0.3873]**.
  Test-only results are GDT-TS **0.5040** and lDDT **0.6381** (210 matched
  proteins). Six test proteins fail the frozen coordinate mapping; one further
  protein lacks the external comparison. All exclusions remain explicit.

Confidence's observed 100% win rates yield degenerate nonparametric bootstrap
intervals; they do not establish perfect population discrimination. pLDDT is
measured on the ranking_score-selected structure, not independently maximized.

## Provenance

Archived inputs are exp250's external/oracle structural scores, exp277's
validation/design contact scores, exp311's fixed-cut folding outputs,
exp321's ordinary iid validation control, and exp245's published baselines.
Exp321's 16 development and 81 held-out units together mean **eval-val**;
its “held-out” label does not mean the 217-protein test split.

[New contact inference](https://modal.com/apps/open-athena/main/ap-gbKIepnA1Q3Ux4QEXaK3Hz)
used eight Modal H100 shards in `us-east`, vLLM 0.9.2 and the exp277 document
generator pinned at `d1bea417a64cc042ad931422200c3edeb873f2e0`.
All 21,700 new rollouts terminated; raw completions reproduce every saved vote.
[Confidence inference](https://modal.com/apps/open-athena/main/ap-Oqepg5fH2fUlnGdLMatCyK)
uses Helico source `b10385d736673c81b10e70d1099962af6f2573c0`,
`contacts-msafree-01-step-6000.pt`, and the exact weight SHA256 in its manifest.
[Test folding](https://modal.com/apps/open-athena/main/ap-jdNvAAh1Pkw58kB09fTmt2)
completed 1,266 structures: 211 proteins × two conditioning arms × three samples.

Main contact votes exclude unfinished rollouts as exp277 does. The sampling
diagnostic follows exp321: all parsed maps vote, while invalid individual maps
score zero. Both diagnostic selectors use the same pool within each protein.
Individual maps rank contacts by emission order; short maps retain denominator R.
This distinction is explicit in the manifest and figure captions.

Small tables and figures live on this branch. `publish_to_hf.py` packages raw
completions, votes, diffusion coordinates, conditioning maps, scores and timings
for the [public artifact prefix](https://huggingface.co/buckets/open-athena/MarinFold/tree/data/exp325-writeup-analysis/exp277-step266344/v1).
Twelve analysis checks pass, including source-row round trips and fixed
confidence selection. Desktop and mobile previews were checked in Chromium,
including the test-split menus; the PDF has three narrative and eleven plot pages.

## Conclusion

Low-depth proteins remain harder for the 248B-token model. Oracle maps show
headroom, and confidence recognizes them against the matched random controls.
Aggregation beats choosing a single contact map overall, even with oracle
selection. The evidence supports limited usefulness of individual samples,
not an absence of contact-set diversity. Better whole-map candidates,
inference-time search and post-training remain directions to investigate.
