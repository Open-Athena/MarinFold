---
marinfold_experiment:
  issue: 218
  title: 'exp: is contacts-v1 a competitive bidirectional protein language model? (ProteinGym zero-shot DMS)'
  kind: evals
  branch: claude/proteingym-marinfold-tasks-ce0cc6
---

# exp: is contacts-v1 a competitive bidirectional protein language model? (ProteinGym zero-shot DMS)

**Issue:** [#218](https://github.com/Open-Athena/MarinFold/issues/218) · **Kind:** `evals` · **Branch:** `claude/proteingym-marinfold-tasks-ce0cc6`

## Question

contacts-v1 documents open with a **randomly ordered** list of `<pN> <AA>` statements.
So `contacts-v1-exp199-1.5B` is not only a contact predictor — it is an **any-order
(permutation) autoregressive model over amino acids**. Prompt it with every residue of a
protein except one and it returns a distribution over the missing one, conditioned on all
the others *and* on their exact sequence positions. That is the same conditional ESM-1v /
ESM-2 compute with a mask token, and it is the object ProteinGym's zero-shot DMS benchmark
scores.

**Is that conditional any good?** Concretely: what does `contacts-v1-exp199-1.5B` score on
the ProteinGym v1.3 substitution benchmark (217 assays, ~2.47M variants), under the
standard masked-marginals protocol, and where does it sit against the published
leaderboard?

Two things make it more than a curiosity, and they are the actual reasons to run it:

1. **Ensembling over orderings is a knob no baseline on that leaderboard has.** A masked
   LM has one conditional per masked position. We have one per *permutation*, and can
   average.
2. **We can compute the exact joint for multi-mutants.** Every single-sequence PLM on the
   leaderboard scores a k-mutant by *summing k independent single-site log-ratios*, and
   every one of them falls off a cliff as depth grows (ESM2-650M: 0.422 → 0.248 → 0.205 →
   0.163 for depths 1→4). An any-order AR model gets the true joint
   `log p(mutant AAs at S | rest of sequence)` from the chain rule at the same cost. 69 of
   the 217 assays include multi-mutants; **1.77M of the 2.47M variants are multi-mutant.**

## Approach

One library primitive plus a benchmark harness.

**The readout** lives in the format, not in this experiment:
[`marinfold/document_structures/contacts_v1/sequence_likelihood.py`](../../marinfold/marinfold/document_structures/contacts_v1/sequence_likelihood.py).
A contacts-v1 document is an ordinary causal-LM sequence, so **one teacher-forced pass
yields the amino-acid conditional at every residue at once**: at the slot holding a
`<pN>` token, the next-token distribution is `P(amino acid at N | every statement before
it)`. An `L`-residue protein costs one forward pass per ordering, not `L`. It also
carries a new `Backend.teacher_forced_target_probs` primitive (transformers backend;
MLX and vLLM raise — vLLM's `prompt_logprobs` is top-k truncated, which would silently
zero exactly the low-probability amino acids a log-ratio needs).

What each residue is conditioned on varies with where its statement landed in the
shuffle, so the readout also returns a per-slot **context size**. That is what makes the
ordering ensemble and the context-fraction sweep two knobs rather than one nuisance.

**The pipeline**, in the order it runs:

| script | what it does | cost |
|---|---|---|
| [`proteingym.py`](proteingym.py) | fetch v1.3, decide scorability, implement the official aggregation | 43 MB download |
| [`phase0_conditional_sharpness.py`](phase0_conditional_sharpness.py) | go/no-go: is the conditional sharp, vs two controls | minutes |
| [`cache_conditionals.py`](cache_conditionals.py) | the only GPU step — `(K, L, 20)` log-probs per assay | see below |
| [`score.py`](score.py) | masked-marginals scoring from the cache (no model) | seconds |
| [`analyze.py`](analyze.py) | aggregate, compare, plot | seconds |

Caching the conditionals rather than scoring inline is deliberate: the tensors are the
reusable artifact, and every scoring rule — the K sweep, the context sweep, anything
invented later — is a cheap re-read.

## Success criteria

Predictions were recorded in [#218](https://github.com/Open-Athena/MarinFold/issues/218)
**before** the run, so the result is interpretable either way:

- **Headline**: ~0.35 average Spearman, 80% interval 0.22–0.45 (the ESM2-150M-to-650M band).
- **Broken-readout floor**: below 0.19 means the readout or the export is wrong, not that
  the model is weak.
- **Category profile**: tilted toward Stability relative to a sequence-only PLM.
- **Ordering ensemble**: monotone gain in K, and in conditioning fraction.

## Coverage

**212 of 217 assays.** contacts-v1 indexes residues into 2000 wrap-around position
tokens, so a longer chain cannot be uniquely numbered. Exactly five assays exceed it:
`A0A140D2T1_ZIKV_Sourisseau_2019` (3423), `BRCA2_HUMAN_Erwood_2022_HEK293T` (3418),
`POLG_HCVJF_Qi_2014` (3033), `POLG_CXB3N_Mattenberger_2021` (2185),
`SCN5A_HUMAN_Glazer_2019` (2016). The 8192-token context is not the binding constraint
(`2·2000 + 7 = 4007`). Every baseline is re-aggregated on the same 212 before comparison.

## Preflight checks (done)

Three things that would each have produced plausible-looking but wrong numbers:

1. **The aggregation is exact.** `proteingym.aggregate` reproduces every published
   leaderboard number from ProteinGym's own per-assay file to ±0.0005 — ESM2 (650M)
   0.4138 vs 0.414, ESM-1v (ens.) 0.4065 vs 0.407, GEMME 0.4547 vs 0.455, ESM-IF1 0.4224
   vs 0.422, Site-Independent 0.3594 vs 0.359, ESM2 (8M) 0.2257 vs 0.226. The headline is
   not a mean over assays: it is mean within UniProt id, then within function category,
   then over the five categories.
2. **Mutation indexing is right.** Across all 212 assays and **2,438,361 variants**,
   ProteinGym's stated wild-type letter matches `target_seq` at the stated (1-based)
   position **every time** — zero mismatches, zero out-of-range sites. No target sequence
   contains a non-canonical residue.
3. **The rope config survives the load.** The `MODELS.yaml` bucket copy of exp199 carries
   both `rope_parameters` (transformers-5) and `rope_scaling` (transformers-4), so the
   silent theta-10000 fallback documented in
   [`_config.py`](../../marinfold/marinfold/inference/_config.py) — worth 0.76 nats/token —
   does not fire.

**Baselines on our 212 assays** (re-aggregated, so directly comparable to whatever we score):

| model | type | avg Spearman (212) |
|---|---|---:|
| TranceptEVE L | MSA | 0.4569 |
| GEMME | MSA | 0.4552 |
| ESM-IF1 | structure | 0.4245 |
| **ESM2 (650M)** | single sequence | **0.4152** |
| ESM-1v (ensemble) | single sequence | 0.4099 |
| ESM2 (150M) | single sequence | 0.3892 |
| Site-Independent | MSA | 0.3591 |
| ESM2 (8M) | single sequence | 0.2272 |

## Status

Readout primitive landed with tests (18 tests, including an oracle-backend identity check
that recovers the input sequence from the conditionals — a deliberate non-tautological
check on the slot mapping, verified to fail under a one-statement shift). Harness built
and preflighted. Model weights and GPU runs pending.

## Results

_(Fill in after the run completes.)_

## Conclusion

_(Fill in after results are in.)_
