# ProteinMPNN-only contacts-v1, 5e-4, two epochs

This is `contacts-v1-exp364-mpnn-only-2ep-lr5e-4-1.5B-step-373768`,
the completed two-epoch checkpoint from
[MarinFold experiment #364](https://github.com/Open-Athena/MarinFold/issues/364).
It is an optional model; exp277 remains the default.

The 1.47-billion-parameter Qwen3 model was trained from scratch on
162,574,724 ProteinMPNN-redesigned protein documents, excluding native-sequence
training documents. Two complete redesign epochs covered 47,842,378 packed
examples in 373769 optimizer updates. The peak learning rate was 5e-4, with
the default warmup/stable/decay schedule scaled to two epochs: warmup 10%,
cooldown 20%, minimum learning-rate ratio 0.1. Native trainer, optimizer and
learning-rate counters agree at the final zero-based checkpoint 373768.

The model uses contacts-v1 documents, an 8192-token context and the co-located
2845-token tokenizer. All six model/tokenizer files are byte-identical to the
export evaluated on November 2. Weights remain float32 on disk; the evaluated
inference recipe uses bfloat16. The trained Llama3 RoPE is stated in both
Transformers 4.x and 5.x forms: theta 500000, scaling factor 8.
`publication_manifest.json` records source SHA-256 and public Xet hashes.

## Measured results

Final full language-model validation loss: **3.01328707**.

| Set | Proteins | All-range R-precision | Long-range R-precision |
| --- | ---: | ---: | ---: |
| eval-val (natural) | 97 | 0.52470326 | 0.50812801 |
| eval-denovo (designed) | 19 | 0.67384995 | 0.63708610 |

All-range 95% protein-bootstrap intervals are [0.48272, 0.56513] on eval-val
and [0.58766, 0.75350] on eval-denovo. The November 2 evaluation used 100
randomized-prompt rollouts per protein, temperature 1, top-p 0.95, no top-k
truncation, seed 0, an initial 6L+128-token budget and the established
rollout/resample live-contact voting recipe. All 11,600 rollouts completed;
none needed the approved prefix-preserving token-budget extension. Eval-test
was not read.

This checkpoint improves eval-val all-range R-precision by 0.06170 over the
completed one-epoch GPU fork. The current full-data default remains higher
at 0.55510 on eval-val and 0.69640 on eval-denovo. Different training exposures
and schedules limit causal attribution to removal of native sequences;
eval-denovo contains only 19 designs. A matched redesign-only sequence-KNN
null is unavailable. These accuracy measurements do not by themselves
establish greater prediction diversity or reduced dependence on homologs.

## Provenance

- [Training run](https://wandb.ai/open-athena/MarinFold/runs/contacts-v1-exp364-mpnn-only-2ep-lr5e-4-1.5B)
- [Evaluation results, intervals, timings and manifests](https://huggingface.co/buckets/open-athena/MarinFold/tree/data/exp364-models-mpnn-only-lr-sweep/contact-eval/2026-11-02/final-01)
- [Follow-up diversity and Neff study](https://github.com/Open-Athena/MarinFold/issues/373)
- [MarinFold source](https://github.com/Open-Athena/MarinFold)
