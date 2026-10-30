---
marinfold_experiment:
  issue: 373
  title: 'exp: ProteinMPNN-only prediction diversity and Neff dependence on eval-val'
  kind: evals
  branch: codex/exp373-diversity-neff
---

# exp: ProteinMPNN-only prediction diversity and Neff dependence on eval-val

**Issue:** [#373](https://github.com/Open-Athena/MarinFold/issues/373) · **Kind:** `evals` · **Branch:** `codex/exp373-diversity-neff`

## Question

Does the completed ProteinMPNN-only checkpoint produce qualitatively different predictions from the current default, particularly greater rollout diversity on eval-val and less degradation in contact accuracy at low effective MSA depth (Neff)? If it is more diverse, does that diversity capture useful alternatives or mainly additional errors?

## Hypothesis

1. **Rollout diversity.** Training without native-sequence documents produces a broader distribution of predicted contact patterns on natural proteins, potentially through reduced reliance on close native-sequence matches. This could improve discovery of correct contacts across repeated samples even when individual predictions are less accurate.
2. **Neff dependence.** The ProteinMPNN-only checkpoint has a flatter performance curve across Neff tiers on eval-val: its accuracy drops less from high-Neff to low-Neff proteins than the current default's accuracy does. This is a hypothesis about relative sensitivity to MSA depth; whether it also achieves higher absolute accuracy on low-Neff proteins is a separate question to report explicitly.

## Background

This follows the redesign-only training experiment #364 and checkpoint publication in PR #367. Pin these two checkpoints throughout:

| Role | Registry nickname | Checkpoint |
| --- | --- | ---: |
| ProteinMPNN-only | `contacts-v1-exp364-mpnn-only-1ep-lr5e-4-cooldown-1.5B` | 186884 |
| Current full-data default | `contacts-v1-exp277-m2-p06-full-epoch-1.5B` | 266344 |

- [Published redesign-only GPU checkpoint, tokenizer and provenance](https://huggingface.co/buckets/open-athena/MarinFold/tree/checkpoints/contacts-v1-exp364-mpnn-only-1ep-lr5e-4-cooldown-1.5B/hf/step-186884)
- [Published default checkpoint](https://huggingface.co/buckets/open-athena/MarinFold/tree/checkpoints/contacts-v1-exp277-m2-p06-full-epoch-1.5B/hf/step-266344)
- [Existing exp364 evaluation results](https://huggingface.co/buckets/open-athena/MarinFold/tree/data/exp364-models-mpnn-only-lr-sweep/contact-eval/2026-10-29/latest-01)

The selected GPU checkpoint is slightly ahead of its completed TPU counterpart on eval-val all-range R-precision (0.46301 versus 0.46018), a practical tie under the 0.005 threshold. The full-data default scores approximately 0.55510. Differences in accuracy must therefore be considered when interpreting diversity.

Both models use decontaminated training sources. Redesigns retain structural ancestry and may retain sequence similarity; excluding native documents alone does not establish an inability to retrieve or memorize related examples. Neither model receives an MSA at inference: Neff describes the evaluation protein's available evolutionary information, not a model input.

Reuse #245's eval-set definitions and #260's existing MSA-depth metadata and conventions. Neff is redundancy-weighted effective depth at 80% identity (`n_eff_0.8` / `msa_neff`), distinct from raw sequence count. Relevant source: `experiments/exp260_evals_msa_depth_stratified/data/msa_depth.csv`, its `build_depth_table.py`, and [the public exp260 analysis artifacts](https://huggingface.co/buckets/open-athena/MarinFold/tree/data/contacts-v1-msa-depth-exp260/v1-01/analysis).

## Approach

1. **Generate paired samples.** Use all 97 natural eval-val proteins as the primary set, with the 19 eval-denovo proteins reported separately as a secondary check. For each model and protein, generate 300 rollouts from one identical fixed prompt, divided into three sampling-seed blocks. Add 100 rollouts using a matched bank of randomized prompts to measure sensitivity to amino-acid ordering and position offsets. Pin the same inference implementation, dtype, context, stopping policy and sampling settings for both models; use the established temperature-1, top-p-0.95, no-top-k recipe as the primary condition.

2. **Measure contact diversity.** Convert each rollout into canonical residue-pair sets, correcting position offsets and removing ordering, orientation and duplicate effects. The primary diversity endpoint is mean pairwise Jaccard distance between contact sets. Also measure contact-inclusion entropy and unique-contact discovery, with long-range contacts reported separately. Control for output contact count and protein length, and retain empty or malformed outputs in diagnostics rather than silently dropping them.

3. **Separate diversity from errors.** Measure single-rollout precision/recall, discovery of true versus false contacts, and ensemble R-precision as rollouts accumulate. Compare diversity–accuracy curves using temperatures 0.7, 1.0 and 1.3 with otherwise identical inference settings. Use the existing ground truth, candidate-pair universe and rollout/resample scorer. Existing aggregate vote matrices do not preserve individual rollout diversity, so save the newly generated individual samples.

4. **Compare performance across Neff tiers on eval-val.** Join the same 97 proteins to the existing Neff metadata by validated protein identity; record source/version, missing values and coverage. Use the established fixed tiers **Neff <10, 10–<100, 100–<1000, and >=1000**, without choosing boundaries based on either model's scores. Plot each model's mean all-range R-precision across tiers, with long-range R-precision alongside it, protein counts, paired model differences and 95% confidence intervals. Use the standard 100-rollout matched randomized-prompt condition at temperature 1 for the main performance comparison; fixed-prompt results are a sensitivity check.

   Quantify the hypothesis with the difference between the models' high-to-low-Neff performance drops, when those tiers are populated, and with a continuous model-by-log10(Neff) interaction, equivalently the slope of the paired per-protein accuracy difference against log10(Neff). A smaller positive accuracy-versus-Neff slope for the redesign-only model is the predicted direction. Show absolute low-Neff performance as well as relative degradation: a flatter line caused only by worse performance on high-Neff proteins must not be described as improved low-Neff accuracy. Report raw MSA-depth tiers separately as a sensitivity analysis; do not substitute raw depth for Neff. Explore whether diversity or ensemble gains also vary with Neff. Report empty/small tiers and missing metadata explicitly; do not expand into eval-test to populate them. Include protein-length and viral-status sensitivity checks where sample size permits.

5. **Inspect qualitative differences.** Compare contact density, long-range coverage, recurring contact patterns, malformed outputs and termination behavior. Use a contact-shuffling control that preserves contact frequencies and rollout sizes to test whether apparent clusters reflect coordinated predictions. Show paired contact maps for a predefined representative panel; label additional striking examples as exploratory.

6. **Investigate the proposed mechanism.** Audit available sequence-similarity evidence against the actual native and redesign training corpora. Examine whether differences concentrate among proteins with fewer close training matches, alongside viral and low-MSA subsets restricted to eval-val. Document gaps in redesign-corpus search coverage. Neff, training-set similarity and raw MSA depth are related but distinct variables; none should silently stand in for another.

7. **Preserve reproducibility and uncertainty.** Save individual rollouts, prompts, seeds, canonical contacts, stopping diagnostics and per-protein timings. Use paired bootstrap intervals over proteins, with seed blocks checking sampling variability; rollouts from one protein are not independent biological replicates. Freeze primary endpoints and contrasts before examining new results, and distinguish secondary/exploratory analyses. Retain capped samples under the previously approved continuation policy, preserving prefixes within the 8192-token context. Eval-test remains untouched. Publish small source tables, plots, manifests and a readable experiment summary; preserve large raw outputs in durable storage.

## Success criteria

- **Diversity hypothesis:** a positive paired difference in the primary diversity endpoint whose 95% confidence interval excludes zero and which persists after prompt and contact-count controls. Report the effect size as well as uncertainty. Call it *useful diversity* only if it improves correct-contact coverage or ensemble performance at comparable prediction quality.
- **Neff hypothesis:** a smaller high-to-low-Neff accuracy drop and/or a shallower positive accuracy-versus-log10(Neff) relationship for the ProteinMPNN-only model, quantified with paired uncertainty. Report which prespecified contrasts support the hypothesis and whether sparse tiers make a contrast inconclusive. Separate relative robustness from absolute low-Neff improvement and show the full tiered curves, rather than reporting only a slope.
- Preserve reproducible per-protein data and representative qualitative examples. Null results, increased diversity driven mainly by errors, or no evidence of flatter Neff dependence are valid outcomes.

## Results

_(Fill in after the run completes.)_

## Conclusion

_(Fill in after results are in.)_
