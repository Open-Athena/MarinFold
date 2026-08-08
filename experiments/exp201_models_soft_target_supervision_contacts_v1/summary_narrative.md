# Summary slides — exp: order-marginalized (soft-target) supervision for contacts-v1

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide.
     Keep this current as the experiment progresses. -->

## What we're doing

contacts-v1 documents list an unordered **set** — the sequence statements and
the contacts — in a **uniformly random order**. One-hot next-token supervision
therefore spends most of its budget asking the model to predict a nuisance
permutation it cannot predict and that we do not want it to learn.

**Does replacing the one-hot target with the exact conditional marginal over the
next token — computable in closed form from the generation process — train
contacts-v1 materially faster, and give a validation loss that actually tracks
R-precision?**

## Why

This is a **Rao–Blackwellization** of the current loss, not a heuristic. Since

```
E_ordering[ CE(onehot(y_t), p_theta) ] = E_prefix[ CE(q_t, p_theta) ]
```

the soft-target loss has an **identical population objective** (same optimum,
provably unbiased) and, by the law of total variance, **strictly lower gradient
variance**. The two losses differ by exactly the conditional entropy `H(q_t)`:
soft-CE = `KL(q||p_theta) + H(q)`.

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

## Results so far

_(Fill in as results come in.)_
