# Summary slides — exp: fine-tune contacts-v1-exp199-1.5B into a clean <contacts-v1.multi> multi-draft model (on-policy drafts, PDB + AFDB, 30%-decontaminated)

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide.
     Keep this current as the experiment progresses. -->

## What we're doing

Can we port [#163](https://github.com/Open-Athena/MarinFold/issues/163)'s `<contacts-v1.multi>` multi-draft format onto the current best base model (`contacts-v1-exp199-1.5B`) — with **on-policy** drafts, an **experimental-PDB** data component, and a protein pool **decontaminated at 30% identity** against the eval set — such that the resulting checkpoint (a) writes many diverse candidate contact maps under `<contacts-v1.multi>`, (b) is a **clean single-document decoder** under plain `<contacts-v1>`, and (c) gives up nothing on plain-mode R-precision?

This is the SFT starting point for best-of-N RL ([#200](https://github.com/Open-Athena/MarinFold/issues/200) / [#208](https://github.com/Open-Athena/MarinFold/issues/208)), which is a **separate** experiment and explicitly out of scope here.

## Why

**H1 — the format transfers to a stronger base.** #163 established that multi-draft generation is a loss-weight-profile question, not a capacity question: `w_draft` has to compete with `w_final` for the section-boundary transition, and arm **F** (header 0.1 / draft 1.0 / final 1.0) plus a **50% plain-rehearsal mix** is the configuration that both emitted ~15 candidates and held the base task (R-prec 0.3374 vs base 0.3357 — a tie). Nothing in that mechanism is specific to E8, so it should reproduce from exp199.

**H2 — the mode leak is an under-training artifact, and steps are the lever.** #163's arm F emits **~2.94 sections under the plain `<contacts-v1>` sentinel** — the leak this issue is chartered to fix. It was trained for **405 steps**. [#175](https://github.com/Open-Athena/MarinFold/issues/175) got a *completely* clean token-0 mode switch out of the same kind of marker on the same kind of 50:50 marked mixture — **0.1 vs 42.0 retracts per rollout on one checkpoint, prompt-selected** — after **2,070 steps**. The two runs differ by ~5x in optimization, not in mechanism. Predicted: mean sections in plain mode falls to ~1.0 somewhere between 405 and ~2,000 steps. **Intermediate checkpoints make this falsifiable rather than assumed** (see Approach, Phase 3).

**H3 — drafts must be on-policy, and the risk is diversity, not quality.** #163's drafts were E8 rollouts at ~12% per-contact precision. exp199 is a far stronger sampler (R-prec **0.6103** under exp82's reference worker, [#209](https://github.com/Open-Athena/MarinFold/issues/209)), and RL will sample from *this* model, so the training drafts must come from it. The preregistered risk is the opposite of the obvious one: a stronger sampler is a *more consistent* one, so candidate diversity may shrink below #163's mean Jaccard of 0.071 — and best-of-N reward is paid for by spread, not by mean quality. **Jaccard is a reported gate, not an afterthought.**

**H4 — experimental PDB is worth including here.** #222's corpora are the first contacts-v1 documents whose `<begin_statements>` section is a *measurement* rather than a prediction. The multi-draft document is exactly the place that distinction lands: drafts are model opinion, the final section is truth, and the model is being taught to tell them apart.

## Where it stands

Decontamination, rollouts and corpus are done. The single-epoch fine-tune is
running, at roughly 1,050 of 1,989 steps with zero errors.

Nothing has been scored yet, so there is no accuracy claim. The evaluation
tooling is built and tested but has not been run.

Done: Tier-A/30% decontamination against #226's 776 eval queries; 8.3 M
on-policy exp199 rollouts at 32 per protein and 100% coverage; a
519,998-document corpus with disjoint halves and one document per protein.

Pending: Gate A (plain-mode R-precision, paired against the base), Gate B (the
mode leak), and the multi-mode report for the RL hand-off.

## The corpus, measured

519,998 documents: 259,999 multi-draft and 259,999 plain rehearsal, an exact 1:1
by document. The two halves share ZERO proteins, drawn arm-stratified so their
length distributions still match to 0.03 residues of mean L. Exactly one document
per protein, so no protein is seen twice anywhere.

Drafts are exp199's own rollouts, measured over all 8,319,968 of them: precision
0.4095, recall 0.4142, F1 0.4090 — against roughly 0.12 for #163's E8 drafts.
Precision and recall land within 0.005 of each other, so the model predicts close
to the right NUMBER of contacts and errs on WHICH ones. That is exactly the error
a refinement format exists to fix.

Two things the measurement changed. First, 1:1 by document is 84/16 by gradient:
a multi document is about 5.8x longer, so plain rehearsal is only 15.6% of the
supervised loss weight. That is the first knob to turn if Gate A regresses.

Second, the K=32 ceiling is the rollout budget, not the context. 66.3% of multi
documents consume every available candidate and still have context left, so
longer documents would need more rollouts per protein rather than a bigger
context window.

## Base-task retention so far

The contacts-v1 validation loss falls at every checkpoint measured: 2.9818 at
step 250, 2.9788 at 500, 2.9778 at 750, and 2.9764 at step 1000.

Monotonically improving. #163's four un-rehearsed weight profiles each lost about
44% of the base task; that is not happening here at 15.6% rehearsal weight.

This is the weak form of the evidence. Cross-harness per-token loss is not
strictly comparable, and H1 and H3 are decided by R-precision, not by loss.

## Three defects found in the measurement path

Building the gates turned up three defects, none of them in the model.

Gate B was measuring the leak with a clipped count. It used n_sections, which the
worker caps at --max-sections (8) when decoding, so a model emitting 20 plain
sections would have reported 8. #163's arm F is quoted at 2.94 for comparison, a
figure that only means something if both sides measure the same thing. It now
uses the uncapped count.

The HF export carries the transformers-5 rope bug — rope_theta absent, with
rope_parameters alone. That is the shape that silently costs 0.76 nats per token
on a transformers-4.x reader, and it forced a retraction in #163. Gate A is NOT
compromised: both configs resolve to bit-identical inv_freq under the node's
transformers 5.15. The export is repaired before publishing regardless, for
older readers.

test_corpus.py had been failing rather than passing, calling build_multi with
arguments the power-law rewrite had removed. 21 tests now cover the corpus, the
Gate A reducer, the publisher and the mode counters.
