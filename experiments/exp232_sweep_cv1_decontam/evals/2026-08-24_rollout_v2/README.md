# Exp232 2026-08-24 rollout-v2 evaluation

This dated snapshot evaluates the exp232 decontaminated-data training checkpoint at step 363,000 and the #75 E8 validation checkpoint. The older `evals/rollout_v2` snapshot is unchanged; future training, cooldown, or cross-phase evaluations can use additional date-prefixed sibling directories.

## Checkpoints

| Purpose | W&B run | Source checkpoint |
|---|---|---|
| #75 E8 validation | `prot-exp75-cv1-1_5b-e8-lr1e-3-wd0p2-v1-bc3084` | `s3://marin-us-east-02a/MarinFold/exp163/model/step-35679` |
| Exp232 m2-p06 training | `prot-exp232-trc-cv1-decontam-train-s01-m2-p06-srcpeak-augcont-lr005-us-east1` | `s3://marin-us-east-02a/MarinFold/exp232_sweep_cv1_decontam/checkpoints/protein/prot-exp232-trc-cv1-decontam-train-s01-m2-p06-srcpeak-augcont-lr005-us-east1/2026.08.21.1/checkpoints/step-363000` |

The exp232 Levanter checkpoint was converted to an evaluation-local HF directory by `/eczech/exp232-export-train-step363000-v2-01-r1`, entirely inside CoreWeave. No checkpoint was copied to an external Hugging Face repository or GCS. The evaluation-local export is at:

```text
s3://marin-us-east-02a/marin/protein-structure/MarinFold/
  exp232_sweep_cv1_decontam/evals/rollout-v2/2026-08-24/v2-01/
  models/exp232-decontam-train-m2-p06-step363000/hf/step-363000
```

## Recipe and execution

- 670 distinct `(dataset, stem)` units: legacy 554, `eval-val` 97, and `eval-denovo` 19; overlapping stems remain separate evaluation units.
- 100 rollouts per unit, temperature 1.0, top-p 0.95, top-k disabled, and token budget `min(8192 - prompt_tokens, 6L + 128)`.
- Occurrence-frequency voting with no pairwise tie-break.
- Twelve single-H100 shards per checkpoint, submitted at batch priority to `cw-us-east-02a` through Iris cluster `marin` as user `eczech`.
- `eval-test` was intentionally not scored.

The completed parent job was `/eczech/exp232-training-eval-v2-01`. It succeeded in 10m35s with no failures or preemptions. Both checkpoints completed all 670 units with 67,000/67,000 usable rollouts and zero unfinished samples.

```bash
source ~/marin.env
uv run python submit_coreweave.py --run-id v2-01 --suite training --seed 0
```

The authoritative CoreWeave output root is:

```text
s3://marin-us-east-02a/marin/protein-structure/MarinFold/
  exp232_sweep_cv1_decontam/evals/rollout-v2/2026-08-24/v2-01
```

## Results

Legacy-554 values, used for comparison with prior evaluations:

| Checkpoint | R (all) | R (long) | AUC (all) | AUC (long) |
|---|---:|---:|---:|---:|
| #75 E8 validation | 0.423524 | 0.363062 | 0.900892 | 0.873574 |
| #232 m2-p06 training | **0.605059** | **0.555022** | **0.945625** | **0.931401** |

Current evaluation-set values are retained independently of the legacy comparison:

| Split | n | #75 R (all) | #232 training R (all) | #75 R (long) | #232 training R (long) |
|---|---:|---:|---:|---:|---:|
| `eval-val` | 97 | 0.242980 | **0.551707** | 0.202094 | **0.535909** |
| `eval-denovo` | 19 | 0.458415 | **0.609832** | 0.383517 | **0.572282** |

The full `eval-val`, `eval-denovo`, viral, and nonviral aggregates are in `data/coreweave_results/eval_split_metrics.csv`. The compressed per-protein rows, split manifest, timing/rollout accounting, checkpoint manifests, and CoreWeave aggregates are retained under `data/coreweave_results/`, so these splits can be re-aggregated without rerunning inference.

## Validation and comparison

The #75 E8 checkpoint passed all four published legacy gates at tolerance 0.005; the largest absolute difference was 0.002553 (long-range R-precision). `build_comparison_data.py` reuses the prior #146, #166, exp232 sweep, CW m1-p06 augmentation/cooldown, and Protenix-v2 results and adds only the recomputed #75 slice and the new training slice. `plot_primary_comparison.py` renders `plots/final_checkpoint_rprecision.png`, sorts boxes by mean all-range R-precision, and fits the descriptive sigmoid to every MarinFold point with a finite contacts-v1 validation loss.
