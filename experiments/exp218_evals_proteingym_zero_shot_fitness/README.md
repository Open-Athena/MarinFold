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

## Results

### Phase 0 — the conditional is real, and it is not degenerate. **GATE: PASS**

14 ProteinGym target proteins (L = 37–306, one per UniProt id, spread across the length
range), K = 8 orderings, `data/phase0_context_curve.csv`:

| context fraction | top-1 (model) | top-1 (scrambled) | perplexity (model) | perplexity (scrambled) | P(wt) | entropy |
|---|---:|---:|---:|---:|---:|---:|
| 0.0–0.2 | 0.103 | 0.094 | 17.43 | 17.81 | 0.064 | 2.86 |
| 0.2–0.4 | 0.165 | 0.098 | 14.46 | 17.77 | 0.101 | 2.69 |
| 0.4–0.6 | 0.259 | 0.099 | 10.63 | 18.29 | 0.171 | 2.42 |
| 0.6–0.8 | 0.317 | 0.097 | 9.02 | 18.31 | 0.206 | 2.26 |
| **0.8–1.0** | **0.345** | **0.088** | **8.15** | **18.72** | 0.231 | 2.15 |

Composition floor (always guess the protein's modal amino acid): **0.132**.

Three things worth taking from this:

- **It clears both controls by a wide margin** — 0.345 vs 0.132 (composition) and 0.088
  (scrambled). Perplexity 8.15 vs 18.72, where 20 is the uniform bound.
- **The scrambled arm is flat in context (0.094 → 0.088) while the model arm nearly
  triples (0.103 → 0.345).** This is the load-bearing observation. Composition is
  identical in both arms, so the only thing extra context can buy in the scrambled arm is
  composition, and it buys nothing. Everything the model gains from seeing more of the
  protein is real sequence structure.
- **The conditional is soft, not memorised.** P(wt) is only 0.231 at high context and the
  entropy is 2.15 nats against a 3.00-nat maximum — so the ranking of the other 19 amino
  acids, which is the only thing a variant-effect score reads, sits in a broad
  distribution rather than a thin tail under a spike. This resolves the over-confidence
  risk flagged when a ubiquitin smoke test returned top-1 = 0.98: **ubiquitin is a
  memorisation outlier, not the typical case.**

Per-protein spread is large — CALM1_HUMAN 0.729 (calmodulin, hyper-conserved) down to
R1AB_SARS2 0.097 (at its own composition floor). Expect that heterogeneity to reappear as
per-assay variance in Phase 1.

### Phase 1 — full benchmark: **0.2964**, about ESM2-35M

212 assays, K = 200 orderings, 29.1 M tokens, **54.5 min on one A5000** (8,912 tok/s —
the cost forecast in #218 said 29.1 M tokens, which is what it took).

| | average Spearman (212 assays) |
|---|---:|
| ESM2 (650M) | 0.4152 |
| ESM2 (150M) | 0.3892 |
| ESM2 (35M) | 0.3213 |
| **MarinFold contacts-v1-exp199-1.5B** | **0.2964** |
| ESM2 (8M) | 0.2272 |
| ProtGPT2 (broken-readout floor) | 0.188 |

**The prediction was ~0.35 with an 80% interval of 0.22–0.45. The result is inside the
interval and below the point estimate.** It ranks 88th of 97 published baselines, sits
between ESM2-8M and ESM2-35M, and is 0.119 behind ESM-2 650M. So: a structure-objective
model does acquire real, usable zero-shot fitness signal for free — worth roughly a 35M
-parameter dedicated PLM — but it is not competitive with a 650M one.

**Both any-order knobs work, and together they double the score.**

| | ctx≥0 | ctx≥0.5 | ctx≥0.75 | ctx≥0.9 |
|---|---:|---:|---:|---:|
| K=1 | 0.1471 | | | |
| K=4 | 0.2117 | | | |
| K=16 | 0.2250 | 0.2640 | 0.2823 | |
| K=64 | 0.2306 | 0.2684 | 0.2866 | 0.2953 |
| K=200 | 0.2315 | 0.2688 | 0.2870 | **0.2964** |

The ordering ensemble saturates fast — most of its gain is in by K=16 (0.147 → 0.225),
and K=64 → K=200 buys 0.0011. The context threshold keeps paying to the end. The
pre-registered primary rule (K=200, ctx≥0.9) also happens to be the best cell of the
grid, so nothing here rests on test-set selection.

**The readout approximation is not what costs us the gap.** At ctx≥0.9 a residue is
conditioned on ≥90% of the others, never on exactly all-but-one. The context increments
are diminishing (+0.038, +0.018, +0.009 across the four thresholds), so extrapolating to
true all-but-one is worth ≲0.01 — against a 0.119 gap to ESM-2. **The gap is model
quality, not estimator error.** (Constructing orderings that place each target strictly
last, instead of waiting for random shuffles to do it, would reach exact masked marginals
at the same cost and is the obvious next efficiency step; it will not change the ranking.)

### The category-profile prediction fails, once level is controlled

#218 predicted a structure-trained model would tilt toward Stability. It does — but so
does an equally-weak sequence model, which is the control that makes the prediction
falsifiable:

| model | avg | Activity | Binding | Expression | OrgFitness | **Stability** |
|---|---:|---:|---:|---:|---:|---:|
| MarinFold (0.296) | 1.00 | 0.90 | 0.91 | 1.12 | 0.71 | **1.36** |
| **ESM2 (35M)** (0.321) | 1.00 | 0.97 | 0.90 | 1.06 | 0.71 | **1.36** |
| ESM2 (650M) (0.415) | 1.00 | 1.02 | 0.81 | 1.00 | 0.90 | 1.26 |
| ESM-IF1 (0.424, structure) | 1.00 | 0.87 | 0.92 | 0.96 | 0.79 | **1.47** |

(Each row is that model's per-category Spearman divided by its own average, so the
comparison is shape, not level.) MarinFold's profile is **indistinguishable from
ESM2-35M's** — identical on Stability (1.36) and OrganismalFitness (0.71) — and not
ESM-IF1's. Whatever this model has learned, it presents as a small *sequence* model, not
as a structure model wearing a sequence readout. The Stability tilt is a property of weak
variant-effect predictors generally, not evidence of structural knowledge.

### Where it wins, and where it is worst

MarinFold beats ESM-2 650M on **34 of 212 assays (16%)**; median Δ −0.106.

| stratum | n | win rate | MarinFold | ESM-2 |
|---|---:|---:|---:|---:|
| Stability | 66 | **29%** | 0.403 | 0.523 |
| Expression | 18 | 22% | 0.331 | 0.415 |
| Binding | 13 | 15% | 0.258 | 0.327 |
| Activity | 43 | 9% | 0.275 | 0.436 |
| OrganismalFitness | 72 | 7% | 0.211 | 0.398 |
| Taxon: Virus | 28 | **7%** | **0.111** | 0.272 |
| MSA depth: Low | 35 | 23% | 0.209 | 0.357 |

**Viruses are the standout weakness (0.111).** That is worth flagging because the prior
ran the other way: exp199 trains on ESM-Atlas metagenomic data, so one might have
expected *better* generalisation to under-represented sequence space. It does not
materialise — and ESM-2 650M is itself weak on viruses (0.261 overall), so this is a
weakness on top of a known hard stratum.

### Mutational depth — the Phase 2 baseline

Additive masked-marginals for both (this is the *baseline* for the exact-joint comparison,
not yet the result):

| depth | MarinFold | vs own depth-1 | ESM2-650M | vs own depth-1 |
|---|---:|---:|---:|---:|
| 1 | 0.2795 | 1.00 | 0.422 | 1.00 |
| 2 | 0.2089 | **0.75** | 0.248 | **0.59** |
| 3 | 0.1241 | 0.44 | 0.205 | 0.49 |
| 4 | 0.1502 | 0.54 | 0.163 | 0.39 |
| 5+ | 0.1925 | 0.69 | 0.218 | 0.52 |

MarinFold degrades *relatively* less with depth at 2, 4 and 5+. Do not over-read this:
both arms use the additive approximation, and a lower depth-1 leaves less room to fall.
Phase 2 (exact joint scoring via the chain rule) is what turns this into a real test.

### Per-assay agreement with ESM-2

r = 0.696 across the 212 assays — substantially decorrelated, which is the precondition
for an ensemble adding something. **The actual ensemble test was not run**: it needs
ProteinGym's per-variant score archive (1.9 GB) and the workstation is at 100% disk with
7.8 GB free, with other sessions running. It is a one-command follow-up when there is
room.

## Status

Readout primitive landed with tests (18 tests, including an oracle-backend identity check
that recovers the input sequence from the conditionals — a deliberate non-tautological
check on the slot mapping, verified to fail under a one-statement shift). Harness built
and preflighted. **Phase 0 passed; Phase 1 complete.** Phases 2–4 (exact joint scoring,
checkpoint ablations, structure-conditioned scoring) not started.

## Conclusion

**contacts-v1 is a real but weak bidirectional protein language model.** Half of its
pretraining is sequence modelling and it shows: the conditional is genuinely contextual
(Phase 0 — the scrambled control is flat in context while the model nearly triples), and
it converts to 0.2964 zero-shot Spearman on ProteinGym, well clear of the 0.188
broken-readout floor. But that is ESM2-35M territory, 0.119 behind ESM-2 650M, and the
gap is model quality rather than readout approximation (worth ≲0.01).

The more interesting finding is the negative one. **On the axis that was supposed to
reveal structural knowledge — the function-category profile — MarinFold is
indistinguishable from a small sequence-only model** and unlike a genuine structure model.
Training on a structure objective bought sequence understanding roughly in proportion to
the sequence tokens seen, and nothing extra that this benchmark can detect.

Two things remain genuinely open, and both are cheap now that the harness exists:

1. **Exact joint scoring for multi-mutants (Phase 2).** 1.77 M of the 2.44 M variants are
   multi-mutant, every single-sequence baseline scores them additively, and an any-order
   model does not have to. The internal comparison (additive-MarinFold vs
   exact-joint-MarinFold) is informative regardless of our absolute level.
2. **The `p03-base` vs `p03-aug` ablation (Phase 3).** exp199's own sweep has matched
   pairs that tie on contact R-precision (0.0036, inside #204's 0.0023 noise floor).
   ProteinGym measures exactly what the sequence-statement order augmentation was for, and
   this run establishes the harness to measure it.

Readout primitive landed with tests (18 tests, including an oracle-backend identity check
that recovers the input sequence from the conditionals — a deliberate non-tautological
check on the slot mapping, verified to fail under a one-statement shift). Harness built
and preflighted. Phase 0 passed. Phase 1 caching in flight.

## Conclusion

_(Fill in after results are in.)_
