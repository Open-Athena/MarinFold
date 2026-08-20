# contacts-v1-exp199-cooldown-1.5B — republished copy

MarinFold's current default contacts-v1 model, and the best contact predictor
we have.

## What this is

[#199](https://github.com/Open-Athena/MarinFold/issues/199)'s CoreWeave arm,
annealed: eric-czech's AFDB + ESM-Atlas mixture sweep, point `p06`, `aug`
variant, run to a cooldown. Qwen3 1.47B (vocab 2,845, `max_seq_len` 8192),
lr 1e-3 / wd 0.2 / global batch 128, a 50/50 AFDB + ESM-Atlas token mixture
with amino-acid sequence-statement augmentation.

Three runs produced it, and only the last one is this checkpoint:

| run | steps | what it did |
|---|---|---|
| `prot-exp199-cw-cv1-s02-m1-p06-aug` | 0 → 145,200 | trained from scratch, 16 H100 nodes (`cw-rno2a`), WSD |
| `prot-exp199-cw-cv1-cont-s02-m1-p06-srcaug-aug100` | 116,160 → 261,360 | continued it, 8 H100 nodes (`cw-us-east-02a`), `aug100` |
| **`prot-exp199-cw-cv1-p06-cool-s01`** | **261,360 → 290,400** | **restored step 261,360 and annealed linear-to-zero** |

304.5B cumulative tokens against the first run's 152.3B. Same recipe, twice the
tokens, plus a cooldown.

On the 554-protein contact benchmark, scored in
[#234](https://github.com/Open-Athena/MarinFold/pull/234) with the rollout
recipe (n=100, occurrence-frequency voting): R-precision **0.6307** (all
ranges) / **0.5837** (long) / 0.6948 (short) / 0.6775 (medium), AUC **0.9511**
(all) / 0.9383 (long).

Against the checkpoint it displaces, scored by the same harness in the same
batch:

| checkpoint | R-precision (all) | Δ |
|---|---:|---:|
| **this checkpoint** | **0.6307** | — |
| `p06-aug` step 145,199 (previous default) | 0.6088 | +0.0218 |
| Protenix-v2, single-sequence (reference) | 0.6032 | +0.0275 |

That harness also re-ran the historical #75 E8 checkpoint as an acceptance
test and returned 0.4246966 against the 0.4245291 exp82 published — 0.00017 —
which is what puts these numbers on the same axis as every earlier MarinFold
result. The gap above is an order of magnitude outside the 0.0023 span of
[#204](https://github.com/Open-Athena/MarinFold/issues/204)'s four evaluations
of one unchanged checkpoint.

On **eval2** ([#226](https://github.com/Open-Athena/MarinFold/issues/226)), the
homology-controlled eval set:

| cut | n | R (all) | R (long) | AUC (all) |
|---|---:|---:|---:|---:|
| eval2 natural, <40% identity to training | 78 | 0.3579 | 0.2998 | 0.8702 |
| eval2 natural, <30% | 61 | 0.3202 | 0.2538 | 0.8568 |
| eval2 pooled, <40% | 307 | 0.5539 | 0.4955 | 0.9328 |
| eval2 pooled, <30% | 275 | 0.5503 | 0.4896 | 0.9322 |

Read the **natural** cut first: the pooled set is ~75% designed proteins, which
every predictor finds easier. The checkpoint this one displaces scores 0.3372
on that cut, so the improvement survives homology control.

Its `eval/tokenized/contacts-v1-val/loss` is **2.9397**, which looks worse than
exp166's 2.6642 and is not. marin
[#7209](https://github.com/marin-community/marin/pull/7209) changed the
packed-LM objective to mask padding targets partway through this model line,
worth ~+0.38 nats; on exp166's scale this run is ~2.5580. Never compare a
contacts-v1 loss across that boundary without converting — see
`experiments/exp180_evals_contacts_v1_progress_over_time/README.md`, "The two
loss scales".

Source: `s3://marin-us-east-02a/marin/protein-structure/MarinFold/exp199_continue_contacts_v1_cw/checkpoints/protein/prot-exp199-cw-cv1-p06-cool-s01/2026.08.14.1/hf/step-290400`.
W&B: `eric-czech/marin/prot-exp199-cw-cv1-p06-cool-s01`.

## What was changed, and why

**The weights are byte-identical to the source.** Every object was verified
against the sizes and S3 ETags #234 recorded when it evaluated this checkpoint,
before anything was uploaded. Only `config.json` differs.

The original was exported by transformers 5.12.1, which states rope as a
`rope_parameters` block. transformers 4.x — which MarinFold pins, and which a
lot of downstream code uses — does not read that key. It does not error
either: it silently falls back to the architecture default, loading
**`rope_theta = 10000` where the model was trained with 500000**, and dropping
the Llama3 scaling entirely.

Measured cost on three real contacts-v1 documents, this checkpoint, repaired
vs as-published (mean NLL/token over the whole document, fp32,
transformers 4.57.6):

| document | tokens | repaired | as published | Δ |
|---|---:|---:|---:|---:|
| 1UBQ | 361 | 1.790 | 2.674 | +0.88 |
| 1QYS | 420 | 2.367 | 3.323 | +0.96 |
| 7BNY (chain A) | 683 | 2.646 | 4.705 | +2.06 |
| **mean** | | **2.268** | **3.567** | **+1.300** |

**1.300 nats/token** thrown away — three times what the same defect cost the
`p06-aug` checkpoint (0.437) on these same three documents, and the largest
figure any published MarinFold checkpoint has measured. The error grows with
sequence length, which is the signature of a wrong rope base. This is the
clearest case yet that the cost is a property of the individual checkpoint and
has to be measured rather than carried over: the extra 152B tokens of training
that made this model better also made it depend harder on getting position
right.

This copy's `config.json` carries **both** shapes: `rope_parameters` is
untouched and `rope_theta` / `rope_scaling` are added alongside. Verified to
read correctly under transformers 4.x (`rope_theta = 500000`, `rope_scaling`
llama3) with `rope_parameters` left intact for 5.x, so a plain
`AutoModelForCausalLM.from_pretrained` on this directory is correct without any
MarinFold code in the loop.

Regenerate with
`experiments/exp238_models_promote_exp199_cooldown_to_default/publish_cooldown.py`,
which runs `scripts/repair_checkpoint_config.py`'s repair by value. Background:
[#180](https://github.com/Open-Athena/MarinFold/issues/180),
[PR #184](https://github.com/Open-Athena/MarinFold/pull/184),
[#197](https://github.com/Open-Athena/MarinFold/issues/197),
[#238](https://github.com/Open-Athena/MarinFold/issues/238).

## Contents

`config.json` (repaired), `model-0000{1,2}-of-00002.safetensors`,
`model.safetensors.index.json`, and the contacts-v1 tokenizer
(`tokenizer.json`, `tokenizer_config.json`) co-located per MarinFold
convention.

sha256, as published:

```
03cd68867df12e95cb18cb689c0cbfc3adc96acd9b6eb8b344a28e7ab4e5fc0f  config.json
5de9ee7610d3f0ac79d19359b42a119169a7a6081ecbbe9abe0fa9b44a149a79  model-00001-of-00002.safetensors
613e93c5e9c338abc0e4c66cfe96ff39f2374e3b9f04c5e45a7772545666b4ad  model-00002-of-00002.safetensors
ab300748eea954731b85225d41fe452f12c5b812d30655d7741d83e9af43b587  model.safetensors.index.json
72a7023f5196052476fe761205b70288b3219c7462d92e33f708e7b75fb71f24  tokenizer.json
c1af07688e82d814a63c74ec5cf8712364baa21fd10178ca92f6151ae70a2ae5  tokenizer_config.json
```

The source `config.json` (before repair) was
`496ed32596524e5c7b997449644fee418508b17c4decea0124bd262c1edff739`.

The tokenizer files are the source's, unmodified. Their
`tokenizer_class: "TokenizersBackend"` is a levanter export name that
`AutoTokenizer` cannot resolve on its own — the same as every other published
contacts-v1 checkpoint. MarinFold's loader handles it
(`marinfold/inference/_tokenizer.py`); a bare `AutoTokenizer.from_pretrained`
needs the fallback that module implements.
