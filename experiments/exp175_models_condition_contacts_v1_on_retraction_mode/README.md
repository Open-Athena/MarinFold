---
marinfold_experiment:
  issue: 175
  title: "exp: condition contacts-v1 on retraction mode with a <contacts-v1.backtracking> doc-type token"
  kind: models
  branch: exp175-backtracking-mode-token
---

# exp: condition contacts-v1 on retraction mode with a `<contacts-v1.backtracking>` doc-type token

**Issue:** [#175](https://github.com/Open-Athena/MarinFold/issues/175) · **Kind:** `models` · **Branch:** `exp175-backtracking-mode-token` (stacked on #160)

## Question

Does telling the model **at token 0** whether it may retract recover the contact-prediction accuracy [#160](https://github.com/Open-Athena/MarinFold/issues/160) lost, and sharpen the retraction behaviour it learned?

## Hypothesis

**#160 trained on a mixture the model could not condition on.** Both halves of its corpus begin with the identical prefix:

```
backtracking : <contacts-v1> <begin_sequence> <p714>  ...
clean        : <contacts-v1> <begin_sequence> <p1021> ...
```

Worse, **208,363 of the 1,023,997 backtracking documents (20.4%) contain no `<retract>` at all** — measured on the published corpus, not estimated — so a fifth of that half is indistinguishable from clean data in prefix *and* body.

A model in that position must marginalise over "may I take this back later?" at every step. Three of #160's measurements are what that predicts:

| #160 measurement | value |
|---|---|
| rollouts containing a retraction | 43.0% — it *samples* the mode rather than being told |
| enrichment vs its ceiling | 1.134x of 1.26x = **52% of headroom** |
| **emission-quality regression** (retraction ignored) | **−0.0251** R-precision |

That last number is the interesting one, and it is the one this experiment attacks. In retraction mode the optimal emission policy is *more speculative* — you can walk a guess back. With no marker that speculativeness has nowhere to live but the shared policy, so it leaks into clean generation. **A model that cannot tell which mode it is in has to hedge in both.**

## Approach

### Format (`marinfold`, done)

`<contacts-v1.backtracking>` — a variant doc type swapped in as token 0 by `GenerationConfig(backtracking=True)`, following the existing `<contacts-v1.sequence_only>` idiom. Not a new statement type: the statements, sections and fold are untouched.

**Append-only, verified against the tokenizer #160 actually trained under:** 0 id mismatches across all 3,849 pre-existing tokens, `<retract>` still 3848, `<crop>` still 3847, new token at 3849.

⚠️ **The hazard worth naming.** Both coordinate supersets build their inherited block by *filtering* contacts-v1's trailing tokens out and re-appending them at the end. A new trailing token that is not added to that filter lands **inside** the inherited block and shoves the whole xyz/crop block up by one id — silently breaking every published crops/ccoord checkpoint and desyncing the two formats' xyz ids. Both filters are extended, and a test pins `<xyz-000>` at 2847.

### Corpus (`mark_corpus.py`, done)

A **one-token rewrite** of #160's mix — no regeneration. The marker is determined by which generator produced the document, so it is a string operation, not a re-run of the model-in-the-loop job. Runs on a marin CPU pod (GCS → GCS, zero workstation uplink).

**Marked by generator, not by content.** The silent 20.4% keep the marker: they teach the honest conditional *"in this mode, sometimes nothing needs retracting"*. Marking by content would instead teach *"this token implies a retraction follows"* — a different target that destroys the token's use as a mode switch, because at inference we want to enter the mode without promising the model it must use it.

Verified on the output: clean half byte-identical, backtracking tails byte-identical, `live_contacts` unchanged on every sampled document, 0 UNKs and identical token counts under the 3850-token tokenizer.

| | |
|---|---|
| documents | 2,047,994 |
| marked `<contacts-v1.backtracking>` | 1,023,997 |
| ...of which contain no `<retract>` | **208,363 (20.4%)** |
| clean `<contacts-v1>` | 1,023,997 |

### Training

**#160's recipe, unchanged** — full fine-tune of exp120, lr 3e-4, wd 0.2, bs 128, seq 8192, 1-epoch cosine, v5p-32 — so the *only* difference between the two runs is the marker. `dispatch_train.py` and `resize_init_vocab.py` are reused from #160 rather than copied: both are fully parameterized, and reusing them is what makes "same recipe" checkable rather than asserted.

Needs one more +1 offline vocab resize (2845 → 3850) of the **same exp120 base** #160 started from; levanter does not resize on warm start.

### Eval

With a marker the retraction on/off comparison becomes a **generation-time** experiment on one checkpoint, which is strictly more informative than #160's readout-time ablation:

- prompt `<contacts-v1>` → clean mode
- prompt `<contacts-v1.backtracking>` → retraction mode

on the same 554-protein set, through #160's `score_backtracking_worker.py` and exp89 `compute_metrics`.

## Success criteria

- **Format is append-only** — asserted in tests, not assumed. ✅
- **The marker is obeyed** — retract rate under the backtracking marker clearly above the rate under `<contacts-v1>` (#160's unconditioned model: 43%).
- **Emission cost recovered** — clean-mode R-precision closes most of #160's −0.0251 against `exp120-base`.
- **Retraction sharpens** — enrichment beyond #160's 52% of headroom.
- **Headline** — best-of-both-modes beats `exp120-base` on the #89 benchmark at matched token compute. The bar #160 did not clear.

### Artifacts

```
corpus + tokenizer  gs://marin-us-east5/protein-structure/MarinFold/exp175_backtracking_mode/corpus
resized init        .../exp175_backtracking_mode/init/exp120-step-1005-vocab3850
run output          .../exp175_backtracking_mode/runs/exp175-cv1-1_5b-mode50-lr3e-4-e1-cos
```

The resize ran **locally**, not on a pod: it is a CPU-only job that asks for a
TPU purely for host RAM, and on 2026-07-29 every v5p-8 in both zones reported
`Insufficient TPUs (need 4, available 0)`. This workstation has 503 GB, so it
took 37 min end-to-end (`resized embeddings (2845, 2048) -> (3850, 2048)`,
`verified rows 0..2844 are bit-identical`) rather than waiting on a queue.
Two smaller notes for anyone re-running it: `resize_init_vocab.py` imports
`marinfold_models` transitively, so it needs `PYTHONPATH=<repo>/models` outside
a pod; and exp160's documented `--cpu 200` no longer fits — the v5p VMs now
advertise 176 allocatable cores.

### Warm-start check (and a correction to #160)

#160's launch comment claimed the run "opens at ~2.45-2.50", below exp120's
converged 2.7213, and treated that as proof the warm start was intact. **That
was wrong** — 2.45 was the loss at step ~586. Pulled from W&B, exp160 actually
opened at **3.87** and did not cross 2.7213 until **step 35**.

The right test is the *rate*, not the opening value: a scrambled vocab or a
wrong rope starts near `log(3850) = 8.3` and needs thousands of steps, so
crossing a converged model's loss in ~35 steps is only reachable from a warm
start that kept its embeddings' meaning.

By that test exp175 is healthy — it converges onto its control within 50 steps:

| step | 2 | 10 | 20 | 30 | 40 | 50 |
|---|---|---|---|---|---|---|
| exp160 | 3.866 | 3.480 | 3.007 | 2.784 | 2.671 | 2.632 |
| exp175 | 5.212 | 3.768 | 3.191 | 2.894 | 2.717 | 2.657 |
| **delta** | +1.345 | +0.288 | +0.183 | +0.110 | +0.046 | **+0.025** |

The residual +1.3 nats at step 2 decaying to +0.025 by step 50 is what one
brand-new untrained token at the head of half the documents should cost: it is
unpredictable exactly once per marked document, and the model learns it almost
immediately.

## Results

_(Training running. The v5p-32 needs 4 co-scheduled workers and the marin TPU
fleet has been saturated since 2026-07-28 — the same capacity crunch that moved
#160's eval to CoreWeave. Pending jobs cost nothing, so it holds position.)_

## Conclusion

_(Fill in after the run.)_
