# Summary slides — MarinFold 1.5B on FoldBench monomers

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide.
     Keep this current as the experiment progresses. -->

## What we're doing

Score the new MarinFold 1.5B checkpoint (`protein-contacts-1_5b-distance-masked-70f8f5/step-49999`) on the 100 FoldBench monomer subset already collected by exp12 / exp20. Compare head-to-head against three baselines on the same proteins with the same distogram-derived metrics:

- `marinfold_1b` — exp20's 1B checkpoint (the model we're trying to outscale)
- `protenix_single_seq` — Protenix v2 in single-sequence mode
- `protenix_msa` — Protenix v2 with MSA conditioning (gold standard)

Inference: zero-shot, CB-CB pair sweep with vLLM prefix cache, one trunk + N²/2 tails per protein. Atom convention matches Protenix's distogram representative atom (CB with CA fallback for GLY / UNK).

## Why

Direct rerun of exp20 with the new 1.5B checkpoint substituted for 1B. Question being asked: does scaling from 1B to 1.5B at the current training budget translate into better zero-shot distogram quality on FoldBench monomers — specifically, does it close the gap to Protenix-single-seq?

Pre-registered hypothesis (issue #26): 1.5B beats 1B on at least **3 of 4** headline metrics — `lddt_distogram_cb` (↑), `mae_distogram_cb_angstrom` (↓), `drmsd_distogram_cb_angstrom` (↓), `prec_long_L` (↑). The bar is intentionally low.

## Compute

iris on TRC, v5p-8 (matches the marin protein-eval convention; the protein-training-1b branch in marin uses the same `--extra vllm --extra tpu` launch shape). Full 100-protein run finished in a single ~4 h job. Outputs land per-protein on GCS at `gs://marin-us-east5/protein-structure/MarinFold/exp26/protein-contacts-1_5b-distance-masked-70f8f5-step-49999-foldbench-monomers/<stem>/{distogram.npz, provenance.json}` so partial progress would survive preemption.

vllm-tpu version pin worth flagging: we had to inline marin's `vllm` extra (vllm-tpu 0.19 / tpu-inference 0.19 / libtpu 0.0.39 as published in `marin-latest`) rather than pulling `marin[vllm]` transitively — uv source maps don't propagate the pytorch-cpu index registration through transitive extras, so the `torchvision==0.25.0+cpu` pin couldn't resolve.

## Headline result

Hypothesis **not supported (2 / 4 metrics).**

| metric | 1.5B | 1B | Δ vs 1B | direction |
|---|---:|---:|---:|---|
| `lddt_distogram_cb` ↑ | **0.288** | 0.272 | +0.016 | ✓ support |
| `prec_long_L` ↑ | **0.285** | 0.258 | +0.026 | ✓ support |
| `mae_disto_cb` (Å) ↓ | **5.695** | 5.642 | +0.053 | ✗ refute |
| `drmsd_disto_cb` (Å) ↓ | **7.202** | 7.143 | +0.059 | ✗ refute |

The aggregate Δs are small in both directions — neither model is meaningfully moving on this benchmark at this scale.

## Per-protein head-to-head

The aggregate hides a sharper picture: 1.5B wins on the majority of proteins for bin-pooled metrics (LDDT, contact precision) but loses on most for per-pair Å-distance metrics (MAE, dRMSD).

| metric | 1.5B beats 1B | 1.5B beats Protenix-SS | 1.5B beats Protenix-MSA |
|---|---:|---:|---:|
| `lddt_distogram_cb` ↑ | 74 / 100 | 11 / 100 | 0 / 100 |
| `prec_long_L` ↑ | 70 / 100 | 33 / 100 | 1 / 100 |
| `mae_disto_cb` ↓ | 40 / 100 | 6 / 100 | 0 / 100 |
| `drmsd_disto_cb` ↓ | 38 / 100 | 6 / 100 | 0 / 100 |

Across all 400 (protein × metric) cells vs Protenix-MSA, MarinFold 1.5B wins on exactly one.

## Inference cost

Per-protein wall-time scales O(N²) (the pair sweep) for both MarinFold models on log-log. Apples-to-apples comparison is muddled by hardware: 1.5B ran on TPU v5p-8 via iris, 1B on H100 via Modal (exp20), Protenix on H100 (exp12). At equivalent N, the 1.5B-on-TPU pair sweep is roughly an order of magnitude slower per protein than 1B-on-H100 — this is a hardware mix story, not an algorithmic regression. Protenix sits roughly flat at ~100 s/protein (trunk-dominated).

## Conclusion

**Did 1.5B move us closer to Protenix?** Slightly, in the same shape 1B already had. Small bump on bin-pooled contact-class metrics (LDDT +1.6 pp, prec_long_L +2.6 pp), no movement on the metrics you'd actually use for downstream structure work (per-pair Å error). Gap to Protenix-single-seq stays wide on every metric (LDDT 0.288 vs 0.432); gap to Protenix-MSA is ≈3×.

Headline read: scaling 1B → 1.5B at the current training budget is **not the lever that closes the gap**. Natural next move is on the training-side levers (more tokens, distance loss reweighting, MSA conditioning) rather than further parameter scaling alone.
