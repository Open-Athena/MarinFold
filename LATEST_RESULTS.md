# Latest results

This page tracks major new MarinFold results. See the [README](README.md) for the project overview and the experiment reports for complete records.

## September 14, 2026: full-corpus training with ProteinMPNN redesigns

The default model in [`MODELS.yaml`](marinfold/marinfold/MODELS.yaml) is
`contacts-v1-exp277-m2-p06-full-epoch-1.5B` — a 1.47B Qwen3 trained from scratch for one complete epoch over the native and ProteinMPNN-redesigned corpus, from [#277](https://github.com/Open-Athena/MarinFold/issues/277). R-precision is **0.620** on legacy 554, **0.554** on eval-val, and **0.696** on eval-denovo. The [public checkpoint](https://huggingface.co/buckets/open-athena/MarinFold/tree/checkpoints/contacts-v1-exp277-m2-p06-full-epoch-1.5B/hf/step-266344) includes its tokenizer and publication manifest. The previous native-only model remains available as `contacts-v1-exp232-m2-p06-train-1.5B`.

The model trains on contact maps from about 70 million natural proteins, with ProteinMPNN redesigns expanding the corpus to 232 million documents. The current exp277 model adds ProteinMPNN-redesigned documents from the filtered native source structures, producing 232,090,905 documents and 248.584 billion raw tokens. It visits all 34,092,146 packed examples once, with a fresh global shuffle and no fixed 50:50 source weighting.

[Training run on W&B](https://wandb.ai/open-athena/MarinFold/runs/contacts-v1-exp277-m2-p06-full-epoch-1.5B).

## Contact prediction results

Here's the MarinFold prediction for a simple de novo designed protein called [Top7](https://www.rcsb.org/structure/1QYS). The panel uses the 92-residue Top7 sequence from our legacy benchmark and the new exp277 checkpoint's saved rollouts.

<img src="experiments/exp277_models_single_mpnn_pilot/plots/top7_maps.png" alt="Top7 (1QYS): experimental contacts and exp277 predictions from 100 rollouts on the 92-residue benchmark sequence" width="66%">

More quantitatively, we can compare MarinFold contact prediction accuracy to existing predictors on protein monomers from the [FoldBench](https://www.biorxiv.org/content/10.1101/2025.05.22.655600v1) benchmark. We define the R-precision for a protein with N ground truth contacts as the fraction of the model’s N highest-confidence predicted contacts that are present in the ground truth structure. For MarinFold, we rank contacts by how often they occur across 100 rollouts. For the baseline models, we score the single highest-confidence structure and rank contacts by ConFind contact degree. This is what that looks like:

<img src="experiments/exp277_models_single_mpnn_pilot/plots/rprecision_natural.png" alt="Contact R-precision on 97 natural eval-val monomers: exp277 0.554, previous exp232 0.552" width="49%"> <img src="experiments/exp277_models_single_mpnn_pilot/plots/rprecision_designed.png" alt="Contact R-precision on 19 de novo designs: exp277 0.696, previous exp232 0.610" width="49%">

The new default is **exp277 at step 266,344** (`contacts-v1-exp277-m2-p06-full-epoch-1.5B`), trained for one complete epoch over native and ProteinMPNN-redesigned documents. It improves over the previous native-only exp232 default on legacy 554 and on de novo designs, while natural eval-val is effectively tied. These contact plots use the **97 eval-val proteins and 19 eval-denovo designs**, with identical proteins for every predictor and 95% protein-bootstrap intervals. Exp277 has not been scored on eval-test. The sequence-KNN reference indexes the native decontaminated corpus; it does not include redesigns.

MarinFold outperforms Protenix-v2 single-sequence on natural eval-val, but still trails the structure predictors on these de novo designs. The design gain over exp232 is +0.086 (paired 95% interval +0.040 to +0.139); the natural eval-val delta is +0.002 (−0.008 to +0.013). This is one seed with different training exposure. [Results, source tables, and plot reproduction](experiments/exp277_models_single_mpnn_pilot/README.md).

We train on predicted structures and evaluate on experimentally determined structures. The native corpus was filtered to remove sequence matches to evaluation proteins at ≥30% identity over ≥50% of the shorter sequence; redesigns derive from those filtered source structures.

The paired all-range R-precision gain on legacy 554 is +0.015 (95% protein-bootstrap interval +0.008 to +0.022). The older contaminated exp199 cooldown still scores higher on that set. The redesign sequences have not been independently screened for sequence homology.

<img src="experiments/exp277_models_single_mpnn_pilot/plots/paired_comparison.png" alt="Paired per-protein R-precision comparison of exp277 and exp232 on legacy 554, natural eval-val, and eval-denovo" width="100%">

Helico/GDT-TS evaluation of exp277 is still pending; the structure plots in the README use exp232.

## Assets and reproduction

- [Experiment report and plot reproduction](experiments/exp277_models_single_mpnn_pilot/README.md)
- [Per-protein results, timings, and figure source data](https://huggingface.co/buckets/open-athena/MarinFold/tree/data/exp277-models-single-mpnn-pilot/evals/rollout-v2/2026-09-13/v2-01/results)
- [Figures and summary PDF](https://huggingface.co/buckets/open-athena/MarinFold/tree/data/exp277-models-single-mpnn-pilot/evals/rollout-v2/2026-09-13/v2-01/plots)
