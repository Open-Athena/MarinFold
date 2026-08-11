---
marinfold_experiment:
  issue: 211
  title: 'exp: is a contacts-v1 rollout a geometrically self-consistent contact set?'
  kind: evals
  branch: claude/contact-consistency-exp199-9d394e
---

# exp: is a contacts-v1 rollout a geometrically self-consistent contact set?

**Issue:** [#211](https://github.com/Open-Athena/MarinFold/issues/211) · **Kind:** `evals` · **Branch:** `claude/contact-consistency-exp199-9d394e`

## Question

When `contacts-v1-exp199-1.5B` emits a contact set in one rollout, is that set
**jointly realizable as a single 3D structure** — more so than a contact set with the
*same per-pair marginals* assembled from *different* rollouts?

Put plainly: does autoregressive generation produce a coherent structural hypothesis,
or a bag of independently-drawn contacts that happen to be emitted in one document?

## Hypothesis

The model generates a **coherent structural hypothesis**, not a bag of independently
drawn contacts — so a single rollout's contact set should be measurably closer to 3D
realizability than a set with the same per-pair marginals assembled across rollouts.
The prior is genuinely uncertain: #201 Phase 0 showed 77 % of the val loss is nuisance
permutation entropy (so next-token CE barely scores the joint) and #163 showed the joint
signal is large when *supplied*, but neither says the model uses it when generating.
A null here is a real result — it would say the joint has to be taught, not decoded.

## Status

| step | state |
|---|---|
| Phase 0 — is the metric viable at all? | **done** (in the issue; reproduced by `validate_phase0.py`) |
| A — calibrate bounds on 554 GT structures (`calibrate_bounds.py`) | **done** |
| B — calibration gate on ground truth (`run_gt_gate.py`) | **done — PASS** |
| B2 — sensitivity at the operating point (`power_check.py`) | **done** |
| 1 — per-rollout generation (`gen_rollouts_worker.py`) | written; needs the model on CoreWeave S3 |
| 1b — CoreWeave dispatch (`dispatch_rollouts_cw.py`) | written, not launched |
| 2–4 — arms, scoring, analysis | not started |

### The gate result (step B, all 554 proteins)

Ground truth beats a separation-matched random set of the same size and the same
`|i−j|` profile by **5.6× in median per-contact excess, on 89.6 % of proteins** —
95.4 % at L 100–200, 88.3 % at L 200–350, 100 % at L ≥ 350. The metric works.

Two of the issue's three stated criteria were wrong, and fixing them is part of the
result:

* **"GT must score ≈ 0" was incoherent against the bounds it was paired with.**
  `u_contact` is the p99.5 of real contact CA–CA distances, so ~0.5 % of real
  contacts exceed it *by construction*; the ground truth carries a structural
  nonzero floor and only reaches < 0.01 per contact on 33 % of proteins. The gate is
  now **relative**, which is the only form the arms need — every arm is scored under
  identical bounds, so the scale cancels.
* **Arm 7 (decoy protein) is not a floor.** A different real protein's contact map
  scores the same as the true one (0.0384 vs 0.0337; the truth wins on 49.6 %, a coin
  flip). Correct behaviour — the score sees the contact graph and never the sequence
  — but it bounds the claim: this experiment can detect *"not a fold at all"*, not
  *"wrong fold copied from a real one"*.

**Scope limit found here:** below L ≈ 100 the metric is nearly uninformative (GT
0.0000 vs random 0.0011, GT lower on only 69.7 %) — a short chain embeds almost
anything. Those 76 proteins are reported separately; power comes from the 394 at
L ≥ 100. 84 of 554 proteins (15 %) have a chain break and are scored but reported
apart.

### The sensitivity result (step B2)

The gate contrast is easy; the experiment's real contrast is a rollout against a
chimera built from the *same* rollouts with the *same* marginals. Sweeping corruption
across the band #199 occupies (R-precision ≈ 0.59 ⇒ ~40 % wrong), 60 proteins at
L ≥ 100:

| corruption step | sign consistency | Wilcoxon p | verdict |
|---|---|---|---|
| 0.05 | 53–65 % | 0.04–0.35 | **not reliably separable at n = 60** |
| 0.10 | 58.3 % | 0.006 | separable |
| 0.20 | 73.3 % | 2.3e−05 | strongly separable |

So the resolvable effect is bounded and the experiment's power has to come from its
scale — 394 proteins × 100 rollouts per arm, against 60 single draws here. And the
sweep is an **upper bound** on the real effect: the chimera keeps every pair the
model proposed and only breaks their co-occurrence, a gentler perturbation than
swapping pairs for random ones.

### What is already known (Phase 0, and why the design is what it is)

The obvious implementation — Floyd–Warshall bound smoothing over the distances implied
by the contacts and the CA(i)–CA(i+1) bond — **does not work**, and the reason is
structural rather than a tuning failure. Measured on 1QYS:

* the bounds do not separate. Contact CA–CA runs p50 7.21 / p99 11.71 / max 13.65 Å;
  non-contact CA–CA runs min 4.06 / p1 6.05. Across all 554 GT proteins
  (`calibrate_bounds.py`), **10.7 % of non-contact pairs sit closer than the contact
  p99.5**. pyconfind contacts are *side-chain* contacts, so CA–CA distance is only a
  proxy and no threshold pair separates the populations;
* and even with idealized non-overlapping bounds, triangle smoothing reports **0
  violations for the true contact set and 0 for a separation-matched random one**, at
  four different bound pairs. A violation needs a path of upper bounds summing below a
  lower bound; any 2-hop contact path is `2·U ≈ 20–24 Å` and a contact plus *k* backbone
  steps is `U + 3.8k`, so nothing reaches under a ~10 Å lower bound once
  `min_seq_separation = 6` has excluded the close-in-sequence pairs.

Triangle smoothing tests feasibility in an arbitrary **metric space**, which a contact
graph satisfies nearly for free. It does not test feasibility in **ℝ³**. It is kept as
step 1 of 3 — it is the bound-smoothing step of Crippen–Havel EMBED and the right
preconditioner for the embedding — but it is not the metric.

What does work is the **3D embedding residual**, and it is graded rather than binary
(corrupting 0 → 100 % of a true set moves the score 0.00 → 0.66 → 0.87 → 2.19 → 4.94 →
6.41 → 7.14), which matters because a rollout at R-precision ≈ 0.59 is a ~60/40 mixture,
not a clean arm. Per-instance variance is comparable to the signal in the 10–30 %
corruption band, so **no single contact set can be called inconsistent on its own** —
every claim is made on the paired aggregate over 554 proteins × 100 rollouts.

## Approach

### Step 1 — rollouts that keep their identity

Fork exp82's `score_rollout_worker.py` into a worker that writes one row per emitted
contact — `(dataset, stem, rollout_k, order, i, j)` — preserving **rollout identity and
emission order** instead of collapsing to votes. (exp102 did exactly this for an older
model, so the shape is proven.) Everything else identical to the settled recipe:

- Model **`contacts-v1-exp199-1.5B`** (`prot-exp199-cw-cv1-s02-m1-p06-aug/hf/step-145199`).
- The fixed **554-protein eval set** (#89) — median L 162, p90 316, max 761.
- n = 100 rollouts/protein, fresh document realization per rollout, `T = 1.0`,
  `top_p = 0.95`, **`top_k = -1`** (#142 removed the truncating `top_k = 50`),
  `max_new = 6L + 128`.
- 16 × single-H100 shards on **cw-rno2a at batch priority**, exp163's
  `dispatch_rollouts.py` recipe verbatim. ~30–60 min wall.
- Publish the rollouts to the public HF bucket — #200 / #208 want the same artifact.

### Step 2 — the consistency score (reference-free, three tiers)

- **T1 · packing** (O(L)): max contact partners per residue against the empirical ceiling
  measured on the 554 GT structures. Free; catches gross impossibilities.
- **T2 · Floyd–Warshall bound smoothing** (`scipy.sparse.csgraph.shortest_path`): tightest
  triangle-consistent upper bounds. Reported, but used as the **initializer** for T3.
- **T3 · 3D embeddability residual** — the headline. Minimize bond / contact /
  non-contact / steric violation over `x ∈ ℝ^{L×3}`, **min over M restarts** (this is an
  *existence* question, so `min` is the correct estimator and it removes optimizer-failure
  confounding). Score = contact excess normalized by contact count. Batched across a
  protein's 100 rollouts on one GPU.

**Bounds are calibrated on the 554 GT structures, not assumed** — exp174's
`prepare_gt_structures.py` already produces full-atom GT plus `gt_contacts.jsonl` for
every eval record, which is exactly the calibration set. Bounds are quantile-based and
therefore *statistical*, so the writeup must say "less geometrically consistent", never
"provably unrealizable".

### Step 3 — arms (all size-matched)

| # | arm | role |
|---|---|---|
| 1 | **GT** contact set | calibration ceiling — must score ≈ 0 |
| 2 | **GT subsampled** to the rollout's contact count | removes the count confound (#142: rollouts emit ~0.70× GT) |
| 3 | **Within-rollout** | the treatment |
| 4 | **Marginal-matched chimera** — draw \|C_r\| pairs from the pooled rollout contacts ∝ vote count | **the key null**: identical per-pair marginals, joint destroyed |
| 5 | **Splice chimera** — half of rollout *a* + half of rollout *b*, deduped, topped up | the literal "different rollouts" comparison; two coherent halves that disagree |
| 6 | Separation-matched random | floor |
| 7 | Contacts from a **different protein** of the same length | hard floor |

Arm 4 is the sharp test. Arms 3 and 4 come from the same model, same protein, same
marginals and same size — the *only* difference is whether the contacts were drawn
jointly or independently.

### Step 4 — analyses

- **Primary:** paired within-rollout (3) vs marginal-matched chimera (4) across 554
  proteins; Wilcoxon + paired bootstrap CI.
- **Reference-free selector (the payoff):** per-protein Spearman ρ(consistency,
  R-precision) across the 100 rollouts; then R-precision of the **top-consistency**
  rollout vs a random rollout, vs the vote-consensus baseline (#82's current recipe), vs
  the oracle-best rollout (#98's headroom).
- **False-positive coherence:** score TP-only, FP-only and TP ∪ FP subsets separately.
  Coherent FPs = the model is proposing a wrong-but-real fold; incoherent FPs = noise.
- **Stratify by L and by separation range** — #180's long-protein story predicts a
  growing effect with L.
- **Noise floor:** re-run rollouts under a different engine seed and report the replicate
  span, the analogue of #204's 0.0023 R-precision floor.

---

## Success criteria

- **Calibration gate (must pass, or the metric is broken).** GT (arm 1) scores ≈ 0
  residual on ≥ 95 % of the 554 proteins, and separation-matched random (arm 6) scores
  strictly worse than every rollout arm. If this fails, stop and fix the bounds.
- **Primary.** Within-rollout beats the marginal-matched chimera by a paired margin whose
  95 % bootstrap CI excludes 0, and that margin is larger than the GT-vs-GT-subsample gap
  (i.e. it is not a contact-count artifact).
- **Secondary.** Per-protein ρ(consistency, R-precision) > 0 on a majority of proteins,
  **and** top-consistency rollout selection beats random-rollout selection by ≥ 0.01
  R-precision — 4× #204's 0.0023 noise floor.
- **A null is a real result.** If within-rollout ≈ chimera, the model's generation is
  marginal-driven: the #163 joint-conditioning gain has to be **taught**, not decoded,
  which argues directly for #201 / #208 over any further inference-side work. That
  conclusion is worth the compute on its own.

---

## Results

_(Fill in after the run completes.)_

## Conclusion

_(Fill in after results are in.)_
