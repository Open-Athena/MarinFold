# ProteinMPNN-only contacts-v1, 5e-4 one-epoch GPU cooldown

This is `contacts-v1-exp364-mpnn-only-1ep-lr5e-4-cooldown-1.5B-step-186884`,
an experimental checkpoint from [MarinFold experiment #364](https://github.com/Open-Athena/MarinFold/issues/364).
It is registered as an optional model, not the default.

The Qwen3 architecture has 1.47 billion parameters and was trained from scratch
on 162,574,724 ProteinMPNN-redesigned protein documents, excluding native-sequence
training documents. The one-epoch fork restores the 5e-4 parent's full state at
step 139053 and cools down from step 149508 to the exact first-epoch endpoint:
186885 total updates, final zero-based checkpoint 186884. Native trainer,
optimizer and LR counters agree. The model uses the contacts-v1 document format,
8192-token context and the co-located 2845-token tokenizer.

The published files are byte-identical to the evaluated float32 export, including
both weight shards, config and tokenizer. Load weights as bfloat16 for the
evaluated inference recipe. The config already states the trained Llama3 RoPE
in both Transformers 4.x and 5.x forms: theta 500000, scaling factor 8.
`publication_manifest.json` records source SHA-256 and public Xet hashes.

## Measured results

Final language-model validation loss: **3.03571987**.

| Set | Proteins | All-range R-precision | Long-range R-precision |
| --- | ---: | ---: | ---: |
| eval-val (natural) | 97 | 0.46300678 | 0.44183671 |
| eval-denovo (designed) | 19 | 0.66515037 | 0.62680095 |

October 29 evaluation used 100 complete rollouts per protein, temperature 1,
top-p 0.95, no top-k truncation, the 6L+128 initial token budget and the established
rollout-and-resample contact-voting recipe. No sample needed a token-budget
extension. All 11,600 rollouts completed. Eval-test was not read.

The GPU checkpoint was selected over the independent TPU fork by its slightly
higher eval-val all-range point estimate (0.46301 versus 0.46018), appropriate
to the proposed eval-val diversity study. This difference is below the 0.005
practical tie threshold and does not establish GPU-training superiority.
The current full-data default has higher contact accuracy. Different training
exposures and schedules prevent isolating the effect of removing native sequences.
Redesign-only training does not prove absence of memorization or homologous
structural information. Diversity has not yet been measured.

## Provenance

- [Training run](https://wandb.ai/open-athena/MarinFold/runs/contacts-v1-exp364-mpnn-only-1ep-lr5e-4-cooldown-1.5B)
- [Evaluation data, uncertainty intervals, timings and manifests](https://huggingface.co/buckets/open-athena/MarinFold/tree/data/exp364-models-mpnn-only-lr-sweep/contact-eval/2026-10-29/latest-01)
- [MarinFold source](https://github.com/Open-Athena/MarinFold)
