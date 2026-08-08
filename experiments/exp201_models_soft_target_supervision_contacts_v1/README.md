---
marinfold_experiment:
  issue: 201
  title: 'exp: order-marginalized (soft-target) supervision for contacts-v1'
  kind: models
  branch: claude/contacts-soft-target-supervision-cc25fd
---

# exp: order-marginalized (soft-target) supervision for contacts-v1

**Issue:** [#201](https://github.com/Open-Athena/MarinFold/issues/201) · **Kind:** `models` · **Branch:** `claude/contacts-soft-target-supervision-cc25fd`

## Question

contacts-v1 documents list an unordered **set** — the sequence statements and
the contacts — in a **uniformly random order**. One-hot next-token supervision
therefore spends most of its budget asking the model to predict a nuisance
permutation it cannot predict and that we do not want it to learn.

**Does replacing the one-hot target with the exact conditional marginal over the
next token — computable in closed form from the generation process — train
contacts-v1 materially faster, and give a validation loss that actually tracks
R-precision?**

## Hypothesis

This is a **Rao–Blackwellization** of the current loss, not a heuristic. Since

```
E_ordering[ CE(onehot(y_t), p_theta) ] = E_prefix[ CE(q_t, p_theta) ]
```

the soft-target loss has an **identical population objective** — same optimum,
same expected value, same expected gradient — and, by the law of total variance,
a **strictly lower-variance target**. It is a lower-variance *estimator* of the
same quantity, **not a smaller number**: both losses have the same floor `H(q)`,
and pointwise soft-CE = `KL(q||p_theta) + H(q)`. The interpretable,
zero-at-optimum quantity is `KL = CE - H(q)`, and `H(q)` is computable per
document from the tokens alone.

Put another way: 16 epochs of hard targets averages over 16 sampled orderings.
Soft targets average over all `N!*2^N` of them in a single pass, for ~0.3 % extra
FLOPs.

**Three predictions:**

1. **Most of the current loss is nuisance.** Permutation entropy per document is
   `log(N!) + N*log 2 + log((L+2)!)`. On the three real contacts-v1 documents in
   [`exp82/data/benchmark_docs.parquet`](https://github.com/Open-Athena/MarinFold/blob/main/experiments/exp82_evals_contacts_v1_contact_prediction/data/benchmark_docs.parquet):

   | protein | L | contacts | tokens | perm. entropy | nats/token | share of 2.71 |
   |---|---|---|---|---|---|---|
   | 1UBQ | 76 | 67 | 361 | 529 | 1.47 | 54 % |
   | 1QYS | 92 | 76 | 420 | 645 | 1.54 | 57 % |
   | 7BNY | 132 | 137 | 683 | 1161 | 1.70 | 63 % |

   Extrapolating with `N ~ L`, `tokens ~ 5L + 8` to the corpus mean document
   (4,676,753,425 tokens / 4,213,203 docs = **1,110 tokens/doc**, L ~ 220):
   **1.90 nats/token = 70 % of the 2.7112 val loss**, rising to ~82 % at L = 500.

2. **This explains #169 mechanistically.** "Val-loss early stopping bought
   nothing" and "matched loss != matched accuracy across sizes" is exactly what a
   metric that is 70 % nuisance produces. It also reframes #166: 2.7112 -> 2.6642
   is a **~6 % relative** gain in the informative part of the loss, not 1.7 %.

3. **Better per-token information, not just lower variance.** At a
   second-endpoint slot the model currently gets one of ~15 true partners while
   the other 14 correct answers are actively pushed down by the softmax. The
   soft target hands it the whole row.

## Background

- **#150 / #117** — the recipe and the control. 1.5B Qwen3, exp53 corpus,
  bs 128, seq 8192, block shuffle, `data_seed=0`, final val loss 2.7112. The
  control curve and per-epoch checkpoints already exist, so only the treatment
  arm needs TPU time.
- **#169** — three checkpoints within 0.008 nats that val loss ranked *wrong*
  against R-precision. A ready-made adversarial test for the metric claim.
- **#163** — conditioning on true partial contact maps lifts R-precision
  0.145 -> 0.556. The joint signal is real, large, and precision-gated.
- **#174** — Pass 1 (the coarse fold) is the bottleneck, not refinement.
- **#82 / #89** — the settled inference recipe (rollout + per-rollout resampling,
  n = 100) and the fixed 554-protein metric.

## Approach

### Phase 0 — offline accounting (local, ~1 day)

- `marinfold/marinfold/document_structures/contacts_v1/soft_targets.py`: pure
  NumPy reference — `soft_targets(token_ids)` and `permutation_entropy(...)` with
  a per-section breakdown. Stays in the jax-free package; it is a property of the
  document structure.
- **Monte-Carlo identity test:** for a small protein, sample K orderings and
  assert (a) empirical next-token frequencies match `q` (in the document's
  own sequence-index frame), and (b) `mean_orderings(hard CE) ==
  mean_orderings(soft CE)` within MC error. This is the test
  that makes every later phase trustworthy — a silently wrong target looks
  exactly like a working run.
- Entropy decomposition over the exp53 val split.
- **Gate:** nuisance share >= ~40 % (current estimate ~70 %).

### Phase 1 — decompose the loss on checkpoints we already have (~1 day, 1 GPU)

Score the val split with existing HF exports (#117 final, #117 early-stop, #146
3B, #166, plus an early-training checkpoint or two for spread), reporting KL
**per slot kind**: contact first endpoint, contact second endpoint, amino-acid
identity, statement order, `<contact>`-vs-`<end>` timing. Plot each component
against already-measured R-precision.

Note what this is *not*: swapping hard CE for soft CE cannot re-rank checkpoints
on a val split this large. The two agree in expectation, and the ordering noise
that separates them is ~1e-4 nats over tens of millions of val tokens — far
below the 0.008-nat spread in #169. Subtracting `H(q)` likewise only rescales,
since it is a constant of the val split.

What *can* re-rank is the **decomposition**: the aggregate 2.71 mixes several
sub-tasks (predicting amino acids, predicting when the contact list ends,
predicting endpoints), and only some of them relate to contact accuracy. If the
endpoint-slot KL orders the #169 checkpoints by R-precision where the aggregate
does not, that is a usable model-selection signal and it ships on its own.

### Phase 2 — the JAX loss (~2-3 days)

`models/marinfold_models/soft_targets.py` (target construction + loss) and
`soft_loss_model.py` (`Qwen3SoftTargetConfig` + LM-head subclass). Tests: JAX vs
the NumPy reference on real documents, packed multi-document windows, and the
Phase-0 MC identity.

### Phase 3 — the training A/B (the decisive experiment)

- Recipe = #150's #117 reproduction verbatim. Control curve already exists.
- **First** a 2-epoch, 3-point LR mini-sweep for the soft arm (1x, 2x, 3.16x of
  3.16e-3) — see risk 2.
- Then the winner to 4 epochs (17,840 steps), extending to 16 only if the
  4-epoch point is at or ahead of the control.
- **Primary endpoint:** R-precision on the fixed 554-protein #82/#89 eval at
  matched steps. **Secondary:** hard-CE val loss at matched steps.
- **Success:** control's R-precision at <= half the steps, or beating it at the
  full budget.

### Phase 4 — probe documents (own issue, only if Phase 3 lands)

Supervise `q(partner | X)` for **every** position X from the sequence alone, not
just the ~N rows a random ordering happens to visit. The naive form is
prohibitive — one document per X repeats the whole sequence section, ~2L^2 ~
180k tokens at L = 300, a 22x blowup. The fix is a shared prefix plus a
**canonical enumeration** of every position:

```
<contacts-v1.probe> <begin_sequence> ...L+2 shuffled statements...
<begin_statements> [optional m real contacts]
<contact> <p_s> <contact> <p_s+1> ... <contact> <p_s+L-1>
```

Supervise only the logits at each `<p_X>` slot with the row target (uniform over
X's remaining partners; all mass on `<end>` when X has none); loss-mask
everything else.

- **No custom attention mask is needed.** Enumerating *every* position in
  canonical wrap-around order makes the probe stream a deterministic function of
  L, so it leaks nothing about the contact map and plain causal attention is
  sound even though probe i attends to probes < i. (Enumerating only
  contact-bearing positions *would* leak and would force a prefix/tree mask.)
- **Cost ~4L + 4 tokens** — *less* than a normal document (~5L), while carrying
  the entire contact map instead of one ordering.
- **Mode marker:** new doc-type token `<contacts-v1.probe>`, appended **last** in
  `all_domain_tokens()` (the #158/#160 id-freeze discipline). #175 showed a mode
  marker cleanly separates behaviours within one checkpoint. Reuse
  `<contact>`/`<end>` rather than minting more tokens.
- **The `m`-prefix variant is the interesting one.** With m real contacts in the
  prefix (m ~ U[0, N]) the target becomes the *remaining* partner row — dense
  supervision of exactly the conditional #163 measured as the big lever.
- Probe documents are a **training-only construct**; they are not sampleable as
  coherent structures. Keep the long random-order documents in the mixture so
  self-conditioned rollout generation still trains.

This will be split into its own issue if Phase 3 lands, per the
"different hypothesis -> new issue" rule.

## Success criteria

_(Concrete metrics + thresholds.)_

## Results

_(Fill in after the run completes.)_

## Conclusion

_(Fill in after results are in.)_
