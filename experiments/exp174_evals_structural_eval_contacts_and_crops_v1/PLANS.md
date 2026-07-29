# exp174 Component 1 — inference plans: document → 3D coordinates

**Status: for discussion. Nothing here is implemented.** Issue
[#174](https://github.com/Open-Athena/MarinFold/issues/174) gates Component 1
behind a design conversation; this is the input to it. Component 2 (the
scoring harness) is built and its numbers are used below.

Five plans, most to least conservative, then a recommendation. The decisions
that are *the same* in every plan — how tokens become xyz, how repeat
observations combine, what to do with an atom that was never refined — are
factored out first so the plans differ only where they actually differ.

---

## 0. What the format can possibly score (measured, not estimated)

Component 2 is done, so we can price every plan against a real ceiling instead
of arguing from intuition. `run_baselines.py` degrades the **ground truth** to
each of the format's resolution tiers and scores it on all 554 eval proteins
(`data/baseline_ceiling.csv`). No model involved — this is what a *perfect*
model would score.

| ground truth degraded to | atom cov. | lDDT | lDDT-CA | TM-score | all-atom RMSD |
|---|---|---|---|---|---|
| nothing (identity check) | 1.00 | 1.000 | 1.000 | 1.000 | 0.00 |
| 0.1 Å (Pass-2 converged), all atoms | 1.00 | 1.000 | 1.000 | 1.000 | 0.05 |
| 10 Å box centers (Pass-1 only), all atoms | 1.00 | 0.323 | 0.327 | 0.511 | 4.99 |
| boxes everywhere + 50 % refined | 1.00 | 0.533 | 0.530 | 0.749 | 3.51 |
| boxes everywhere + 30 % refined | 1.00 | 0.419 | 0.418 | 0.651 | 4.16 |
| boxes everywhere + 15 % refined | 1.00 | 0.360 | 0.361 | 0.578 | 4.59 |
| **one realistic document** (65 % boxed, 25 % refined) | 0.65 | **0.167** | 0.166 | **0.406** | 4.29 |

Four things fall out of this table, and they drive every plan below.

1. **0.1 Å quantization is free.** `tenths` scores 1.000/1.000. All the loss is
   coverage and box-resolution, none of it is the digit vocabulary.
2. **Box-only atoms are worth surprisingly little.** A structure where *every*
   atom is placed at its correct 10 Å box center still scores lDDT 0.32 /
   TM 0.51. Pass 1 alone cannot produce a good structure, however perfect the
   model is at it.
3. **lDDT is roughly quadratic in coverage; TM-score is linear.** Halving
   coverage takes lDDT 1.00 → 0.25 but TM 1.00 → 0.50, because an lDDT contact
   needs *both* of its atoms and a TM residue needs only itself. Clustering the
   coverage into whole 10 Å boxes barely helps (0.259 vs 0.250 at 50 %): lDDT's
   15 Å inclusion radius spans several boxes, so most contacts cross a box
   boundary regardless.
4. **A single document tops out around lDDT 0.17 / TM 0.41.** That is the
   number Plan A is competing for. Everything above it has to come from
   spending more inference compute to raise the *fine* fraction.

> The 65 %/25 % row uses the SPEC's own coverage table for a 150–500-residue
> chain (30–70 % of atoms boxed, 12–40 % refined); our eval set's median length
> is 161. Re-measuring it from real generated documents rather than the SPEC's
> summary is a small, worthwhile follow-up.

---

## 1. Decisions common to every plan

These are not free choices per plan; they are the decode contract. Arguing
them separately keeps the plan comparison honest.

**Prompt construction.** Build the contacts-and-crops-v1 sequence section from
the input sequence with the format's own deterministic builder (the same
`build_document` path `contacts_v1.inference` uses for its prefix), truncated
at `<begin_statements>`. The residue start index and the statement shuffle are
seeded, so this is reproducible and — because they are nuisance variables the
model marginalizes over — re-seeding them is free test-time augmentation
(exp89's `ensemble_k`).

**Frames.** Every document places the protein in its own random rotated,
translated frame. Every metric in Component 2 is frame-invariant (RMSD and
TM-score superimpose; lDDT is superposition-free), so a **single-document**
plan needs no registration at all. Any plan that merges *across* documents
must first estimate a rigid transform between them from their shared atoms —
a real cost, and a real failure mode, that Plans A, C, D and E simply do not
have.

**Token → xyz.** Invert the SPEC's digit rule exactly. A Pass-1 mention
`<pXXX> <ATOM> <xyz-HHH> <xyz-TTT>` gives the atom's 10 Å cell
`(hundreds*10 + tens)` per axis. A Pass-2 statement inside a
`<crop> <xyz-HHH> <xyz-TTT>` header gives ones + tenths, and the header
supplies hundreds + tens, so the position is
`box_index * 10 + ones + tenths/10`. Reuse the format's own
`_digits` / `_cell` helpers rather than re-deriving them (the SPEC's
`180.2 / 0.1` warning is there for a reason).

**Aggregating repeat observations.** Precision-weighted mean, with the weights
the SPEC hands us:

* Pass-1 mention → box center, per-axis variance `10²/12 + 2²` (cell width plus
  the σ=2 Å box noise).
* Pass-2 statement on a box's `i`-th appearance → the decoded position, per-axis
  variance `(1/(i+1)²)²`. `i` is countable from the document: it is the number of
  earlier `<crop>` headers naming that same box.

Because σ falls as `1/(i+1)²`, the weights rise as `(i+1)⁴` and the last visit
dominates — but the weighted mean is the right estimator and costs nothing.
**This must be validated, not assumed:** the schedule is what the model was
*trained* on, and a sampled document is free to ignore it. Cheap check on a dev
split: bucket refined atoms by emitted visit index and plot the actual error
against `1/(i+1)²`. If the model does not sharpen with re-shows, drop to
"last visit wins" and say so.

**Atoms that were never refined.** Three options; we should take the first for
v1 and treat the second as a clearly-labelled variant:

1. **Box center.** The minimax-optimal point estimate given only a box, and
   what the ceiling table's `box10` row prices. No geometry knowledge injected,
   so the number is attributable to the format and the model.
2. **Ideal-geometry completion.** If a residue has refined atoms, the missing
   ones are determined to well under 1 Å by standard bond lengths and angles.
   This would likely beat option 1 substantially — but it is a structure prior
   doing the work, not the model, so it must be reported as a separate row.
3. **Omit.** Raises `lddt_*_covered`, sinks `lddt_all` and TM-score. Never the
   right choice given the coverage-penalized headline metrics.

**Atoms never mentioned at all** are simply absent from the output file. The
scorer counts them against the prediction (see Component 2's convention), which
is what makes coverage a first-class result rather than a footnote.

**Sampling temperature — the one genuinely surprising knob.** The training
coordinates are *deliberately noisy*: σ=2 Å in Pass 1, σ=1/(i+1)² in Pass 2, in
both cases zero-mean. So the model's next-token distribution over an
`<xyz-DDD>` token is (roughly) the true digit convolved with that noise, and
its **mode is the noise-free digit**. Sampling coordinate tokens reproduces the
training noise; taking the argmax *denoises* it. That argues for a hybrid:
**greedy on the coordinate tokens, sampled on the structural choices** (which
atom to mention, which box to crop), which is a two-line logits-processor given
the token classes are disjoint by vocabulary block. Pure greedy risks the usual
degenerate repetition on the structural choices; pure sampling throws away a
free accuracy win. Worth measuring all three.

**Cost model.** Qwen3 1.5B on one H100 under vLLM ran ~25,000 tok/s on
contacts-v1 rollouts (root `AGENTS.md`, exp82's fan-out). A full 8192-token
document for all 554 proteins is ~4.5 M tokens ≈ **3 minutes of decode on one
H100**, ~15 s across a 12-shard fan-out. Inference compute is *not* the binding
constraint here — which is the strongest argument for preferring an expensive
plan over a cheap one.

---

## 2. Plan A — one free-running document per protein

**Condition on:** sequence section only. **Generate:** contacts, Pass 1, Pass 2
to `<end>` or 8192 tokens.

Exactly what the model was trained to produce, with no intervention. Decode per
§1 and score.

* **Cost:** 1 × ~8k tokens/protein → ~3 GPU-minutes for the whole eval set.
* **Ceiling:** lDDT ≈ 0.17, TM ≈ 0.41 (measured, table above).
* **Main failure mode:** the ceiling itself. Even a perfect model scores 0.17
  lDDT here, so a mediocre result is uninformative — we would not be able to
  tell "the model can't fold" from "the format didn't get a chance".
* **Why it still has to be run:** it is the only plan that measures the model
  in its training distribution. Every other plan prompts the model with
  something it never saw, and Plan A is the control that tells us how much that
  costs.

## 3. Plan B — K independent documents, registered and merged

**Condition on:** sequence section, K times with different section seeds (and
therefore different frames). **Generate:** full documents. **Then:** pick the
document with the lowest NLL as the reference frame, estimate a rigid transform
from each other document onto it by Kabsch on the atoms they share at fine
resolution, transform, and precision-weighted-merge.

* **Cost:** K × Plan A. K=16 is ~45 GPU-minutes for the eval set.
* **Ceiling:** raises the *fine* fraction toward 1.0. With K=16 and ~25 % fine
  per document, union coverage saturates well above 50 % → lDDT ≳ 0.53,
  TM ≳ 0.75, if the documents agree.
* **Main failure mode:** **registration.** The transform is estimated from
  shared refined atoms; if two documents fold the protein differently, Kabsch
  returns a confident nonsense alignment and merging makes the answer *worse*
  than either input. Needs a guard (reject a document whose post-superposition
  RMSD to the reference exceeds a threshold) and that guard is itself a tuning
  knob. There is also no reason the model's fold should be consistent across
  independent samples — exp82 found contacts-v1's unconditional signal weak
  enough that iterative/structured inference could not bootstrap on it.
* **Note:** the same K documents give best-of-N for free (exp98 found
  best-of-N ≫ mean on contacts), so B subsumes a selection strategy.

## 4. Plan C — one document, then a forced crop tiling (re-prompting)

**Condition on:** sequence section, then the model's *own* Pass-1 output.
**Generate:** Pass 2 many times over, one crop at a time, with the crop headers
chosen by us rather than by the model.

Concretely: run Plan A's prefix through the end of Pass 1. Read the occupied
boxes off the emitted Pass-1 mentions. Then, for each occupied box `b`, continue
from the shared prefix with `<crop> <box(b)>` appended and decode only that
crop's body. Optionally repeat a box several times with the earlier visits in
context, to walk it up the σ=1/(i+1)² refinement schedule.

* **Key property:** every continuation shares one prefix and therefore **one
  frame**, so there is nothing to register. This is Plan B's coverage win
  without Plan B's registration risk.
* **Cost:** one long prefill (~6k tokens) plus `n_boxes` × ~100 decode tokens.
  With prefix caching that is roughly 2–3 × Plan A, not `n_boxes` × Plan A. A
  200-residue protein has on the order of 100–200 occupied boxes.
* **Ceiling:** every occupied box refined → the `tenths` row, lDDT → 1.0, with
  the realized number set by how well the model actually places atoms.
* **Main failure mode:** **out-of-distribution prompting.** The model never saw
  a document whose crops tile the structure in an order it did not choose, and
  the 8192-token budget means it never saw more than ~20 crops at all. Forcing
  a header for a box the model would not have picked may produce a confidently
  wrong crop body — and, worse, the membership rule is "atoms whose noised
  position falls in this box", so a bad header invites the model to invent
  members. Mitigation: only force boxes the model's own Pass 1 actually
  occupied (as above), and compare against Plan A on the same proteins to
  measure the OOD cost directly.

## 5. Plan D — constrained decoding (grammar + coverage)

Plan A or C plus an exp100-style logits processor that masks the vocabulary to
what the grammar permits at each step:

* statement shapes (a `<crop>` header is always exactly 3 tokens; a Pass-1
  mention exactly 4);
* `<pXXX>` restricted to positions this protein actually has, and `<ATOM>`
  restricted to atoms the named residue actually has (both derivable from the
  input sequence — the model should not be able to give an ALA a `CD1`);
* optionally, `<pXXX> <ATOM>` masked to atoms **not yet mentioned** in the
  current crop, which turns "coverage" from something we measure into something
  we guarantee.

* **Cost:** same tokens as the underlying plan, plus a cheap CPU-side mask per
  step. exp100 has this working on local GPU via HF `transformers`; the
  **iris-TPU `logits_processor` path is still open**, so this plan is
  GPU-for-now.
* **Ceiling:** removes wasted tokens (malformed statements, impossible atoms),
  which buys back budget — worth a few percent of coverage, not a step change,
  *unless* the forced-coverage variant is used.
* **Main failure mode:** forcing an atom the model has no opinion about
  produces a confident coordinate with no information in it. Grammar-only
  constraints are safe and should probably just be on by default; forced
  coverage is the aggressive part and needs its own row in the results table.

## 6. Plan E — teacher-forced diagnostics (upper bounds, not a predictor)

Two variants, both cheap, neither a legitimate headline number:

* **E1 — sequence + true contacts.** Teacher-force the ground-truth contact list
  into the contacts section, then generate coordinates. Separates "cannot fold"
  from "can fold, cannot emit coordinates".
* **E2 — sequence + true Pass 1.** Teacher-force the *correct* coarse boxes for
  every atom, generate only Pass 2. Isolates the refinement ability, which is
  the thing the crops format exists to test and the thing Plan C is betting on.

* **Cost:** ~1 × Plan A each.
* **Value:** E2 in particular tells us whether Plan C is worth building. If the
  model cannot refine a box even when handed the correct boxes, Plan C's whole
  premise is wrong and we should not spend the engineering.
* **Failure mode:** none, as long as they are reported as oracle rows and never
  compared against other predictors.

---

## 7. Recommendation

Run **E2 first** (a day of work, answers the load-bearing question), then
**A** as the in-distribution control, then **C** as the real predictor, with
**D**'s grammar-only constraints on throughout.

Reasoning:

* The ceiling table says a single document cannot exceed lDDT 0.17, so Plan A
  alone cannot answer "is contacts-and-crops-v1 structure-capable". We need a
  coverage-raising plan to say anything.
* Between the two coverage-raising plans, **C beats B on the axis that actually
  bites**: C keeps one frame and one document's worth of self-consistency,
  while B's cross-document registration is both an extra failure mode and
  dependent on a fold consistency we have no evidence for. B's advantage —
  giving best-of-N for free — can be recovered later by wrapping C in an
  N-sample outer loop.
* E2 is the cheap gate on C. If teacher-forced Pass 1 does not produce good
  crops, C is dead and B (or a format change) is the conversation instead.
* Report Plan A and Plan C side by side, always with coverage, and always with
  the ceiling row for the coverage each achieved. A model at lDDT 0.15 against
  a 0.17 ceiling is a *success*; the same 0.15 against a 0.53 ceiling is not,
  and the table has to make that legible.

**Open questions worth settling in the discussion, in rough priority order:**

1. Is Plan C's forced tiling acceptable, or does prompting the model outside
   its training distribution disqualify the number? (My view: acceptable if
   Plan A is reported next to it, since A measures exactly that gap.)
2. Ideal-geometry completion of box-only atoms — in scope for v1, or a separate
   row? (My view: separate row, off by default.)
3. Greedy vs sampled coordinate tokens: measure, or just pick greedy on the
   denoising argument above?
4. Does anything here change the case for a **contacts-and-crops-v2** with a
   larger fine reserve? The ceiling table says the fine fraction is the whole
   ballgame, which is a format-design finding as much as an inference one.
