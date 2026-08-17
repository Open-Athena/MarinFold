# The published exp199 R-precision is understated by ~0.023

**Status:** cause localised to #199's evaluation pipeline; the specific mechanism
is not identified. Measured in
[#208](https://github.com/Open-Athena/MarinFold/issues/208); the consequences
belong to [#180](https://github.com/Open-Athena/MarinFold/issues/180) and
[#204](https://github.com/Open-Athena/MarinFold/issues/204).
**Date:** 2026-08-11.
*(This file was called `RPRECISION_STACK_DISCREPANCY.md` until the controlled
re-score below excluded the accelerator. It never was a stack effect.)*

## The result

The exp199 checkpoint (`prot-exp199-cw-cv1-s02-m1-p06-aug`, step 145199) scored
four ways on the same 554 proteins, same ground-truth universe, same n=100 rollout
recipe:

| # | inference stack | scorer | R-precision (all) | (long) |
|---|---|---|---|---|
| 1 | v5p (TPU vLLM fork), seed 0 | exp82 `score_rollout_worker.py` | 0.609926 | 0.563922 |
| 2 | v5p, seed 1 | exp82 `score_rollout_worker.py` | 0.611398 | 0.564085 |
| 3 | **CoreWeave H100 (CUDA vLLM)** | exp82 `score_rollout_worker.py` | **0.610286** | **0.563873** |
| 4 | CoreWeave H100 (CUDA vLLM) | **#199's own pipeline** | **0.587348** | 0.542181 |

Paired per protein:

| comparison | band | Δ | SE | σ |
|---|---|---|---|---|
| row 3 vs row 1 (same scorer, **different accelerator**) | all | **+0.000360** | 0.001151 | **+0.3** |
| row 3 vs row 1 | long | −0.000048 | 0.001542 | −0.0 |
| row 3 vs row 4 (**same accelerator**, different scorer) | all | **+0.022938** | 0.001427 | **+16.1** |
| row 3 vs row 4 | long | +0.021692 | 0.002604 | +8.3 |

**exp82's worker gives the same answer on both accelerators, to 0.0004.** Run on
the very hardware #199 used, it reproduces the v5p figure and not #199's. So the
gap is not the accelerator, and #169's premise — that the same worker bytes give
comparable numbers on either backend — is **vindicated**, not undermined.

The difference lives in **#199's evaluation pipeline**.

## What it is not

Each of these was checked rather than assumed.

**Not the metric.** `n_true`, `n_candidate` and `n_top` are identical on **100% of
rows**, so the two agree on the candidate universe, the range filter and the size
of the top-R cut. The two implementations — exp89's `compute_metrics` (via exp82's
`build_rollout_rows`) and #199's `analyze_contact_eval.metric_rows` — were read
side by side and are functionally the same code, independently written: same
`np.triu_indices` over resolved residues, same `degree >= 0.001` and `sep >= 6`
truth matrix, same `np.argsort(-scores, kind="mergesort")` stable sort, same
`min(target, n_candidate)`, same `roc_auc_score`.

**Not the sampling recipe.** #199's `eval_contact_checkpoint.py` and exp82's
`score_rollout_worker.py` were read end to end. Both build prompts with
`build_document(f"{stem}:r{k}", residues, [], config=GenerationConfig())`, cut at
`<begin_statements>`, use the same `(n_term + offset) % 2000` position map,
dedupe pairs within a rollout, apply `MIN_SEP` 6, and sample n = 100 at T = 1.0 /
top_p = 0.95 / top_k = -1 with `max_tokens = min(8192 - prompt, 6L + 128)` in
bfloat16 with engine-level seed 0.

**Not the weights.** Both evaluations resolve to the same export. The safetensors
are byte-identical in size (4,979,485,528 and 906,042,048) between
`open-athena/marinfold-exp199` @ `ed7103b` (what #199 evaluated) and the
open-athena bucket copy (what exp208 evaluated).

**Not a checkpoint that trained further since it was benchmarked.** A natural
reading of "the same checkpoint scores higher now" is that the artifact moved —
the run continued and the export at `step-145199` was overwritten. It did not:

* the source repo's shard-1 LFS sha256 is **`e8db3b66…` at both
  `ed7103bfd7da`** (the revision #199's manifest pins) **and at current `main`**,
  and shard 2 likewise. The weights have not been rewritten since #199 evaluated
  them.
* the repo's last weight-bearing commits are `e2b1e27d` / `84d842e5` at
  2026-08-10 12:39-12:55 UTC, while #199's eval manifest is stamped
  2026-08-10 14:25 UTC — the evaluation **postdates** the final weight upload.
* the bucket copy's `PROVENANCE.md` records that both shards were sha256-verified
  byte-identical to that source (`e8db3b66…`, `a7a38503…`), with **only
  `config.json` differing**. That identity is documented by the #198 republish
  rather than independently re-hashed here, but it names the same shard-1 hash
  this document verified against HF.

So all three copies in play — repo @ `ed7103b`, repo @ `main`, and the bucket —
carry the same weights.

**Not rope, despite the configs differing.** The two copies' `config.json` do
differ — the bucket copy carries #198's repair, stating rope as top-level
`rope_theta` + `rope_scaling` *and* the transformers-5 `rope_parameters` block,
while the model-repo copy states only `rope_parameters`. This looked like the
[transformers-5 rope export bug](https://github.com/Open-Athena/MarinFold/issues/180)
that has bitten this project before. It is not: **both stacks run transformers
5.12.1**, and loading the bucket config on exp208's stack resolves rope to
`rope_theta` 500000 inside the rope block — the same value #199's stack reads from
`rope_parameters`. Both evaluations used the correct rope.

## The exp117 control, and the puzzle it leaves

#199's pipeline also scored an **exp117 control**, and exp117 has an independent
measurement from #169 using exp82's worker on v5p. Those agree:

| checkpoint | exp82 worker | #199 pipeline | Δ |
|---|---|---|---|
| **#117 final** step 35679 | 0.534418 (#169, v5p) | 0.532888 | **−0.001530** (inside the 0.0023 span) |
| **#199 CW p06-aug** step 145199 | 0.610286 (CoreWeave) / 0.609926 (v5p) | 0.587348 | **−0.0229** |

So #199's pipeline agrees with exp82's worker on exp117 and disagrees by 0.023 on
exp199. Whatever the pipeline difference is, **it does not bite every
checkpoint** — which is why it went unnoticed: the control that was run to
validate the pipeline is one of the cases where the two agree.

That asymmetry is unexplained and is the open question this document leaves.

## What the shape of the difference says

exp208's score matrix is *better at the head of the ranking and worse over the
tail*:

| cut (all band) | exp208 | #199 | Δ |
|---|---|---|---|
| P@L | 0.55052 | 0.53230 | +0.01822 |
| P@L/2 | 0.71224 | 0.69249 | +0.01975 |
| P@L/5 | 0.81705 | 0.80202 | +0.01504 |
| P@R | 0.60993 | 0.58735 | +0.02258 |
| **AUC** | **0.94805** | **0.95153** | **−0.00348** |

AUC moves the *other way*, in both bands (long: 0.93415 vs 0.93920, −0.00505).
Higher precision at every cut with lower AUC is the signature of a **more
concentrated rollout ensemble**: votes pile onto fewer pairs, which sharpens the
top of the ranking and flattens the tail that AUC integrates over. So the two
runs are not drawing from the same effective sampling distribution, even though
both requested T = 1.0 / top_p = 0.95 / top_k off.

## The hypothesis I had, and why it is dead

exp199 looked numerically unusual: larger weights than exp163 arm F
(`input_layernorm` 5.20x, `mlp.up_proj` 2.75x, no non-finite entries anywhere),
which would plausibly make it sensitive to which bf16 kernels execute it.

Two measurements killed it. First, extending the weight comparison to **exp117**
— the checkpoint that reproduces — showed exp199 is only **1.27x** exp117 at the
median (worst family 2.29x) while exp163 arm F is **0.70x**: the 5.20x headline
was mostly exp163 arm F being small. Second, and decisively, **the accelerator is
excluded outright** — exp82's worker returns the same number on CUDA and on
TPU/JAX vLLM, so no bf16-kernel difference can be responsible.

## What is left to identify

Everything about the *artifact* and the *measurement* is now excluded: the metric
implementation, the sampling recipe, weight identity, rope, the checkpoint having
trained further, the accelerator, and within-stack sampling noise. Two known
uncontrolled differences remain between the two pipelines, neither of which should
bias an estimator and neither of which is yet tested:

| parameter | exp82 worker | #199 pipeline |
|---|---|---|
| `max_num_seqs` | 512 (default) | **128** (per #199's manifest) |
| per-request sampling seed | set per rollout on CUDA; unavailable on TPU | engine-level seed only |

Neither is an obvious mechanism — both pipelines draw 100 independent rollouts at
T=1.0/top_p=0.95 — and neither explains why exp117 agrees while exp199 does not.
The cheapest next probe would be exp82's worker at `--max-num-seqs 128` on one
accelerator: one variable, one job, and it either reproduces 0.587 or clears the
last named difference.

**This is now #180/#204's to close rather than #208's.** exp208 needed a trustworthy
baseline for its own arms and has one; the remaining question is about the
published number.

## A separate finding: the published exp199 export cannot be read by transformers 4.x

Independent of the R-precision question, and surfaced while staging the CoreWeave
re-score. MarinFold **pins transformers 4.x**, and the exp199 export as published
fails a 4.x reader twice over:

| file | as published | on transformers 4.x |
|---|---|---|
| `config.json` | rope only under `rope_parameters` (model repo) | **silent** fallback to `rope_theta` 10000 against a model trained at 500000 |
| `tokenizer_config.json` | `"tokenizer_class": "TokenizersBackend"` | **hard failure**: `Tokenizer class TokenizersBackend does not exist or is not currently imported` |

The bucket copy fixes the first via #198's repair but **still carries
`TokenizersBackend`**, so anything loading it through a plain
`AutoTokenizer.from_pretrained` on a 4.x stack breaks outright. Verified against
transformers 4.57.6: as-published fails; setting `tokenizer_class` to
`PreTrainedTokenizerFast` loads with **identical ids** (`<contact>` 5, `<p0>` 143,
vocab 2845) and leaves `tokenizer.json` — the actual vocabulary — untouched.

exp169 hit both halves and repaired them together in `prepare_hf_export.py`. The
lesson worth carrying: an export written by transformers 5 needs *two* repairs to
be portable, and only the rope one is currently applied when republishing to the
bucket. The one-key tokenizer fix is now applied to
`timodonnell/marinfold-contacts-v1-exp199-1_5b-step145199`; the open-athena bucket
copy still has the issue.

## Consequences

- **#180's frontier row for #199 is understated.** #117 and #166 come from
  exp82's worker; #199 comes from its own pipeline. Scored the way every other
  row was, #199 reads **0.6103 rather than 0.5873**, so the #166 → #199 step is
  roughly **0.048 rather than 0.026** — #199 is further ahead of #166 than
  recorded, and is the best contacts-v1 model by a wider margin.
- **#169's premise is vindicated.** `dispatch_eval_tpu.py` claims that running the
  same worker bytes on either backend makes the numbers comparable to published
  CoreWeave ones. Measured directly here: +0.0004 at σ +0.3. That claim was right,
  and this document originally asserted the opposite.
- **exp82's worker is the reference scorer.** It is the only implementation shown
  to agree with itself across accelerators and seeds (three runs within 0.0015).
  Frontier numbers should come from it unless there is a reason otherwise.
- **exp208 baselines against its own parity run** (0.609926 / 0.563922) rather
  than the published rows, since every arm is scored through the identical path.
  That decision is unaffected by how this resolves.
