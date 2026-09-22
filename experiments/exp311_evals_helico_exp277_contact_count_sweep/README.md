---
marinfold_experiment:
  issue: 311
  title: 'exp: Helico confidence selection across top-0 to top-L exp277 contacts'
  kind: evals
  branch: codex/helico-exp277-contact-sweep
---

# exp: Helico confidence selection across top-0 to top-L exp277 contacts

**Issue:** [#311](https://github.com/Open-Athena/MarinFold/issues/311) · **Kind:** `evals` · **Branch:** `codex/helico-exp277-contact-sweep`

## Question

For the registered latest MarinFold model, exp277 step 266344, how much does Helico structural accuracy vary as the number of supplied contacts changes from zero to top-L in steps of ten? How well does Helico's confidence ranking select among these predictions?

## Hypothesis

The top-L input is unlikely to be uniformly optimal across proteins. A sweep
may expose a better contact budget for an individual target, but Helico's
confidence head must identify the useful structure without seeing ground truth.
The practical quantity is therefore the gap between the oracle envelope and
the Helico-selected prediction, measured separately on natural eval-val and
designed eval-denovo targets.

## Approach

The contact source is the registered latest default, [exp277 step 266344](../../marinfold/marinfold/MODELS.yaml), scored with exp82's 100-rollout and resampling recipe. Its dense vote matrices are the existing [exp277 evaluation](../exp277_models_single_mpnn_pilot/README.md), so this experiment does not regenerate contacts. The structure model is Helico `contacts-msafree-01/final.pt`, step 6000, at Helico git revision `b10385d736673c81b10e70d1099962af6f2573c0`.

`prepare.py` reads Helico exp14's FoldBench targets, ground-truth structures and independently verified prompt-to-token map. It ranks resolved upper-triangle residue pairs at sequence separation ≥6 with a stable descending sort of the exp277 score matrix. It verifies the predicted contact precision at L, L/2 and L/5 against exp277's committed per-protein scores before emitting lists. All **345 reference cuts on 115 targets matched to 1e-9**. One eval-val target, `7pv5_A`, is excluded: Helico exp14 found its sequence alignment ambiguous around a modified cysteine, so its prompt-to-token map is not verified. This gives 96 eval-val and 19 eval-denovo proteins. The input hashes and explicit exclusion are in [input_manifest.json](data/input_manifest.json).

For a target with MarinFold prompt length L, the cuts are 0, 10, 20, …, `10×floor(L/10)`, and exact L when it is not a multiple of ten. There are 3,089 target-cut pairs. Helico runs six trunk recycles and three diffusion samples at every cut with seed 42 reset per cut. It uses single-sequence mode with no MSA. Zero contacts leaves every pair unknown. At each positive cut, the **first k** ranked pairs are mapped to Helico tokens; no pairs are marked absent. Every sample is scored through Helico's established atom matching and lDDT, TM-score, GDT-TS, and RMSD functions.

`analyze.py` first reduces predictions **within each target**, then averages target summaries so longer proteins with more cuts do not dominate. For each metric the oracle is the best of all cuts and samples (minimum for RMSD); mean and median are over all predictions for that target. The confidence-selected result is one structure per target, selected by Helico's highest `ranking_score` across every cut and sample. The top-L reference is the confidence-selected diffusion sample at the exact-L cut. Oracle best is selected separately for each metric, so its different metric entries can describe different structures.

Run from this directory:

```bash
uv run python prepare.py --helico-exp14 /path/to/helico/experiments/exp14_foldbench_held_out_monomers
HELICO_DRY_RUN=1 uv run python run_sweep.py
uv run python run_sweep.py
uv run python analyze.py
uv run python build_summary.py
```

The exp14 directory must first have run its `build_eval_sets.py` and `build_index_map.py`; exp277's CoreWeave score files are read through the `cw` profile in `~/.aws/credentials`. `run_sweep.py` packages the pinned Helico source and uses the existing Helico Modal checkpoint volume. The dry run estimates 16.8 H100-hours, **$66.54** at [$3.95/hour](https://modal.com/pricing), below Helico's $100 cost gate. Actual usage will be recorded after completion.

## Success criteria

- Every target has all specified contact counts and all three diffusion samples, or an explicit failure; no unexplained missing rows.
- Results include GDT-TS, lDDT, TM-score, RMSD and confidence score per prediction, plus per-target timings and hardware metadata.
- Report the oracle envelope, mean, median, confidence-selected prediction, and exact-L confidence reference separately for eval-val and eval-denovo.

## Results

The structural sweep is in progress. The input preparation and 345 contact-precision checks completed. `7pv5_A` is the single documented exclusion.

## Conclusion

Pending the complete Helico run.
