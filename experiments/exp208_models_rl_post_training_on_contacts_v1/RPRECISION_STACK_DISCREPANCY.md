# A +0.023 R-precision discrepancy on one unchanged checkpoint

**Status:** unresolved. Measured in [#208](https://github.com/Open-Athena/MarinFold/issues/208);
the consequences belong to [#180](https://github.com/Open-Athena/MarinFold/issues/180)
and [#204](https://github.com/Open-Athena/MarinFold/issues/204).
**Date:** 2026-08-11.

## The observation

exp208 re-measured the **exp199 checkpoint** (`prot-exp199-cw-cv1-s02-m1-p06-aug`,
step 145199) as a parity gate before using it as an RL warm start. It read
**0.0226 higher** than the number published for the same checkpoint:

| band | exp208 (v5p) | published #199 (CoreWeave H100) | paired Δ | paired SE | σ |
|---|---|---|---|---|---|
| all (sep ≥ 6) | **0.609926** | 0.587348 | **+0.022578** | 0.001515 | +14.9 |
| long (sep ≥ 24) | **0.563922** | 0.542181 | **+0.021741** | 0.002699 | +8.1 |

n = 554 / 553 proteins, paired per protein. For scale, #180 records that **four
evaluations of one unchanged #117 checkpoint span 0.0023** — this gap is ten times
that, and larger than the #166 → #199 frontier step it sits next to.

Sources: exp208's parity run is `gs://marin-us-central1/protein-structure/MarinFold/exp208/phase0/scores/exp199_cw_p06_aug_step145199`;
the published rows are
[`../exp180_evals_contacts_v1_progress_over_time/data/exp199_cw_p06_aug_step145199_rows.csv.gz`](../exp180_evals_contacts_v1_progress_over_time/data/exp199_cw_p06_aug_step145199_rows.csv.gz).

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

## The control that refutes the obvious explanation

The obvious reading is "CoreWeave CUDA vLLM vs marin's TPU vLLM fork give
different results", and **exp208's README asserted exactly that before this
document was written. That claim was wrong, and is corrected here.**

#199's pipeline also evaluated an **exp117 control** on CoreWeave, and exp117 has
an independent TPU measurement from #169. Those agree:

| checkpoint | v5p (TPU) | CoreWeave H100 | Δ | within the 0.0023 repeat span? |
|---|---|---|---|---|
| **#117 final** step 35679 | 0.534418 (#169) | 0.532888 (#199 control) | **−0.001530** | **yes** |
| **#199 CW p06-aug** step 145199 | 0.609926 (exp208) | 0.587348 (#199) | **+0.022578** | **no** |

Both rows are the same two stacks. If the accelerator were the cause, exp117
would show a comparable gap; it does not. **Whatever this is, it is specific to
the exp199 checkpoint rather than a general property of either pipeline.**

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

## The hypothesis I had, and why it is now weak

exp199 looked numerically unusual. A per-tensor comparison against exp163 arm F
found **no non-finite entries in either** but much larger magnitudes in exp199 —
`input_layernorm.weight` 5.20x, `mlp.up_proj.weight` 2.75x,
`post_attention_layernorm.weight` 2.52x. A model sitting closer to bf16's
precision limits would plausibly be sensitive both to which bf16 kernels execute
it and to a bf16 backward pass, which would explain the cross-stack gap *and* the
first-step NaN with one cause.

**Extending the comparison to exp117 — the checkpoint that actually reproduces
across the two stacks — mostly dissolves that.** Job
`/bizon/exp208-compare-weights-3way`, 267 tensors shared by all three sources, no
non-finite entries anywhere:

| comparison | median max&#124;w&#124; ratio | range across families |
|---|---|---|
| exp199 / **exp117** | **1.27x** | 0.67x - 2.29x |
| exp163 arm F / **exp117** | **0.70x** | 0.31x - 1.72x |

The ordering is exp163F (0.70) < exp117 (1.00) < exp199 (1.27), so exp199 *is* the
largest of the three — but the 5.20x headline was mostly **exp163 arm F being
small**, not exp199 being large. Against the control the median gap is 1.27x and
the worst family is 2.29x, which sits within the ordinary spread between these
checkpoints. That is a thin basis for "exp199 is uniquely fragile in bf16 while
exp117 is not", so the hypothesis is **weakened, not supported**.

What it does not settle either way: exp199 remains the largest of the three, and
it remains both the one that disagrees across stacks and the one that NaNs in
levanter. The direction is consistent; the magnitude is not compelling.

## What would settle it

1. ~~Extend the weight comparison to exp117.~~ **Done — see above; it weakens
   the numerical-sensitivity hypothesis rather than supporting it**, which makes
   the remaining two load-bearing.
2. **Re-score exp199 on CoreWeave with exp82's worker.** exp82's
   `score_rollout_worker.py` runs unmodified on both backends (that is #169's
   whole premise). Same code, same weights, one variable — the accelerator.
3. ~~A within-stack replicate on v5p.~~ **Done — the v5p measurement reproduces.**
   A second full 554 x 100 run at engine seed 1 (the parity run used 0; with
   `--no-per-request-seed` the engine seed is the only randomness) gives:

   | band | v5p seed 0 | v5p seed 1 | paired Δ | SE | vs CoreWeave |
   |---|---|---|---|---|---|
   | all | 0.609926 | **0.611398** | +0.001472 | 0.001200 | **+0.024050** |
   | long | 0.563922 | **0.564085** | +0.000163 | 0.001579 | **+0.021904** |

   Both within-stack deltas sit **inside** #180's 0.0023 four-repeat span, so
   0.6099 was not an anomalous draw — the v5p figure is stable to ~0.0015 — while
   the gap to the published CoreWeave number is unchanged at ~+0.023. Whatever
   this is, it is reproducible on at least one side.

Until at least (2) is done, neither number should be treated as *the* value.

**(3) is done (above). (2) is BLOCKED as of 2026-08-11** — the CoreWeave
credentials in `~/.config/marin/cw-rno2a.env` (dated 2026-07-06) are rejected by
the object store with "The access key ID you provided does not exist in our
records", so neither the weight mirror nor the eval fan-out can run until they
are refreshed. The design, for when they are: The CoreWeave re-score mirrors the
*same artifact* exp208 evaluated (`timodonnell/marinfold-contacts-v1-exp199-1_5b-step145199`,
bf16, carrying the repaired config that states rope in both 4.x and 5.x terms) to
`s3://marin-us-east-02a/MarinFold/exp208_eval/model_exp199`, so the accelerator is
the only variable. That choice of config matters: the CoreWeave eval image is vLLM
0.9.2, whose transformers is 4.x, and #199's own model-repo config states rope
*only* under `rope_parameters` — using it there would silently fall back to
default rope and confound the comparison. The v5p replicate re-runs at engine seed
1 (the parity run used 0); with `--no-per-request-seed` the engine seed is the
only source of sampling randomness, so an unchanged seed would risk reproducing
the same 100 rollouts and measuring nothing.

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

## Consequences if it holds

- **#180's frontier table mixes measurements** it treats as comparable. #117 and
  #166 come from exp82/#169 (v5p), #199 from its own pipeline (CoreWeave). If the
  exp199 row is understated by ~0.023, the #166 → #199 step is roughly **0.048
  rather than 0.026**, and #199 is further ahead than recorded.
- **#169's premise needs qualification.** `dispatch_eval_tpu.py` states that
  running the same bytes on both backends "is what lets these numbers be compared
  to the published CoreWeave ones". The exp117 control supports that for exp117;
  exp199 shows it is not unconditional.
- **exp208 baselines against its own parity run** (0.609926 / 0.563922) rather
  than the published rows, since every arm is scored through the identical path.
  That decision is unaffected by how this resolves.
