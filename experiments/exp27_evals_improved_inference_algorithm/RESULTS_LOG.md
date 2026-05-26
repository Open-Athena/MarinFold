# exp27 results log

Running log of inference-algorithm experiments on the 10-protein train
set. The +10% bar to clear is mean `lddt_distogram_cb` ≥ **0.2746**;
the wall-clock budget is **≤ 6922 s** (5× baseline).

Aggregate per-algorithm rows also live in `data/experiments.tsv`; this
file adds the design rationale, what was tried, and what we learned.

## Baseline references

| algorithm | mean LDDT | median | wall (s) | notes |
|---|---:|---:|---:|---|
| `baseline_naive` | 0.2496 | 0.2500 | 1384.5 | exp20 algorithm, A100/bf16. |
| `gt_filtered_naive` | 0.2496 | 0.2499 | 215.3 | Same readout, only pairs with GT < 15 Å. Same headline LDDT by construction; uses GT mask so it's a speedup, not a generalizing algorithm. Used as the per-pair-cost unit for every later experiment. |

## Diagnostic: soft vs hard LDDT (baseline_naive)

`lddt_distogram_cb_soft` − `lddt_distogram_cb` per protein:

| stem | hard | soft | Δ |
|---|---:|---:|---:|
| 8eb9_A | 0.1510 | 0.2006 | +0.0496 |
| 7y5j_A | 0.4485 | 0.4062 | −0.0423 |
| 7ykm_A | 0.3317 | 0.3571 | +0.0254 |
| 7ur2_A | 0.2633 | 0.2833 | +0.0200 |
| 8baq_A | 0.1872 | 0.2203 | +0.0331 |
| 8cba_A | 0.2045 | 0.2292 | +0.0247 |
| 7zs2_A | 0.2500 | 0.2697 | +0.0197 |
| 7xz3_A | 0.1913 | 0.2197 | +0.0284 |
| 7ylr_A | 0.2673 | 0.2781 | +0.0108 |
| 7uk8_A | 0.2010 | 0.2171 | +0.0161 |
| **mean** | **0.2496** | **0.2681** | **+0.0185** |

**Interpretation:** 9/10 proteins have probability mass within the LDDT
threshold window that the mean-of-distogram point estimate misses —
classic multimodal/flat-distribution failure mode. Distogram sharpening
(idea 5) is bounded above by the +0.0185 gap on average; that's
~74% of the +0.025 we need, so sharpening alone cannot clear the bar
but is a useful stacking component. 7y5j_A is anomalous (already
confident, soft < hard) and will *lose* a tiny bit from sharpening.

## Experiments

_(filled in as runs land)_

### idea 5: distogram sharpening (post-process)

Post-process the saved distogram with `p' = softmax(log(p+ε)/T)` per
(i,j) row; non-queried rows (gt_filtered fills only in-shell pairs) are
left as zero. No new inference. Sweep:

| T | mean LDDT | median | Δ% vs baseline |
|---:|---:|---:|---:|
| 1.0 (identity) | 0.2496 | 0.2499 | +0.00 |
| 0.7 | 0.2611 | 0.2587 | +4.61 |
| 0.5 | 0.2675 | 0.2634 | +7.17 |
| 0.3 | 0.2713 | 0.2658 | +8.69 |
| 0.2 | 0.2721 | 0.2658 | +9.01 |
| 0.1 | 0.2731 | 0.2671 | +9.41 |
| **0.05** | **0.2738** | **0.2678** | **+9.68** |
| 0.02 | 0.2737 | 0.2688 | +9.65 |
| 0.01 | 0.2737 | 0.2685 | +9.65 |
| 0.005 | 0.2736 | 0.2684 | +9.61 |
| 0.001 | 0.2736 | 0.2684 | +9.61 |

Saturates at T ≤ 0.05 around 0.2738. Pure argmax (T→0) gives 0.2736.
The gain (0.2496 → 0.2738) is +0.0242, which is consistent with the
+0.0185 soft − hard gap diagnosed at the top (sharpening goes
*beyond* the soft estimate by being more aggressive than the soft-LDDT
window's Σ p_bin within the threshold).

**Verdict: +9.68% from a free post-process.** Falls 0.0008 short of the
+10% bar alone (need 0.2746). Should stack additively with any
context-injection idea that gives at least +0.001.

Saved to `outputs/<stem>/distogram_sharp_T0.05.npz` (gitignored). Row
in `data/experiments.tsv`: `sharpen_T0.05_on_gt_filtered_naive`.

### idea 1: self-bootstrapped contact seeding

Pre-flight: re-ran `baseline_naive` end-to-end on the A100 (1386.7 s)
to produce full-pair-set distograms. Snapshotted each as
`outputs/<stem>/distogram_baseline_naive.npz` (gitignored). Seeds for
all idea-1 variants come from this snapshot.

**`seeded_contacts_kL1.0_min0.3`** — K=L most-confident contacts per
protein, filtered to contact prob ≥ 0.3, ordered long → medium → short.
Tail readout on LDDT-shell pairs. 392.1 s (5.0× the gt_filtered base
unit; cost scales with seed prefix length).

Per-protein:

| stem | n_seeds | prefix tok | baseline | seeded | Δ |
|---|---:|---:|---:|---:|---:|
| 7uk8_A (394) | 117 | 748 | 0.2010 | 0.2148 | +0.0138 |
| 7ur2_A (195) | 101 | 501 | 0.2633 | 0.2836 | +0.0203 |
| 7xz3_A (325) | 32 | 424 | 0.1913 | 0.2102 | +0.0189 |
| 7y5j_A (102) | 102 | 411 | 0.4485 | 0.5432 | **+0.0947** |
| 7ykm_A (105) | 50 | 258 | 0.3317 | 0.4020 | +0.0703 |
| 7ylr_A (330) | 286 | 1191 | 0.2673 | 0.3157 | +0.0484 |
| 7zs2_A (316) | 114 | 661 | 0.2500 | 0.2888 | +0.0388 |
| 8baq_A (208) | 12 | 247 | 0.1872 | 0.2014 | +0.0142 |
| 8cba_A (214) | 37 | 328 | 0.2045 | 0.1973 | **−0.0072** |
| 8eb9_A ( 95) | 4 | 110 | 0.1510 | 0.1593 | +0.0083 |
| **mean** | | | **0.2496** | **0.2816** | **+0.0321** |

**+12.83% over baseline. Headline bar (+10%) cleared on the first
attempt.** 9/10 proteins improved; 8cba_A regressed slightly (only 37
seeds for L=214). Pattern: proteins with denser confident contacts
gain more (7y5j_A and 7ykm_A both have ~1L seeds; 8eb9 / 8baq / 8cba
have ≤ L/4 seeds).

Stacked with sharpening:

| algorithm | T | mean LDDT | Δ% |
|---|---:|---:|---:|
| seeded_contacts_kL1.0_min0.3 (no sharpen) | 1.0 | 0.2816 | +12.83 |
| sharpen on top | 0.5 | 0.2871 | +15.04 |
| sharpen on top | 0.3 | 0.2880 | +15.40 |
| sharpen on top | 0.1 | 0.2891 | +15.84 |
| **sharpen on top** | **0.02** | **0.2894** | **+15.94** |

Sharpening adds another +0.008 on top of seeding (vs +0.024 on top of
naive — the seeded distogram is already sharper because the model is
more confident with seed context).

Chain wall-clock (final): 1386.7 (baseline prior) + 392.1 (seeded) +
30.5 (sharpen sweep) = **1809.3 s**, 1.31× baseline — well within 5×
budget.

_(Next: sweep min_contact_prob to find the sweet spot, and try idea 6
which avoids the baseline-prior cost entirely.)_

### idea 6: model-sampled contact prefix

**`sampled_contacts_n1_T0.7`** — single rollout, sample with T=0.7,
top_p=0.9, max_tokens=600, stop tokens `<distance>` / `<end>`. Parse
emitted contact statements, use as prefix, re-read distances.

The model stops at `<distance>` very early — 2-33 contacts per
protein (vs idea 1's K=L hundreds). Cost 244.6 s.

| stem | L | n_seeds | sample tok | baseline | sampled | Δ |
|---|---:|---:|---:|---:|---:|---:|
| 7uk8_A | 394 | 33 | 100 | 0.2010 | 0.2139 | +0.0129 |
| 7ur2_A | 195 |  8 |  25 | 0.2633 | 0.2765 | +0.0132 |
| 7xz3_A | 325 | 11 |  34 | 0.1913 | 0.2084 | +0.0171 |
| 7y5j_A | 102 |  8 |  25 | 0.4485 | 0.4482 | −0.0003 |
| 7ykm_A | 105 |  2 |   7 | 0.3317 | 0.3165 | −0.0152 |
| 7ylr_A | 330 | 33 | 100 | 0.2673 | 0.2898 | +0.0225 |
| 7zs2_A | 316 | 13 |  40 | 0.2500 | 0.2527 | +0.0027 |
| 8baq_A | 208 |  8 |  25 | 0.1872 | 0.1954 | +0.0082 |
| 8cba_A | 214 |  8 |  25 | 0.2045 | 0.2072 | +0.0027 |
| 8eb9_A | 95 |  2 |   7 | 0.1510 | 0.1599 | +0.0089 |
| **mean** | | | | **0.2496** | **0.2568** | **+0.0072** |

**+2.88%.** Far weaker than idea 1's +12.83%. The model commits very
few contacts before transitioning to `<distance>` — fewer seeds, less
conditioning, less lift. Compared to idea 1 which picks K=L seeds from
the model's marginal contact distribution, sampling under-utilises the
contact knowledge the model has.

**`sampled_contacts_n5_T0.7`** — same as above but average 5
independent rollouts (different seeds). Cost 1409.4 s (5× single, as
expected).

| stem | baseline | M=5 | Δ |
|---|---:|---:|---:|
| 7uk8_A | 0.2010 | 0.1929 | −0.0081 |
| 7ur2_A | 0.2633 | 0.2554 | −0.0079 |
| 7xz3_A | 0.1913 | 0.1858 | −0.0055 |
| 7y5j_A | 0.4485 | 0.5032 | +0.0547 |
| 7ykm_A | 0.3317 | 0.3183 | −0.0134 |
| 7ylr_A | 0.2673 | 0.2509 | −0.0164 |
| 7zs2_A | 0.2500 | 0.2465 | −0.0035 |
| 8baq_A | 0.1872 | 0.1774 | −0.0098 |
| 8cba_A | 0.2045 | 0.1969 | −0.0076 |
| 8eb9_A | 0.1510 | 0.1759 | +0.0249 |
| **mean** | **0.2496** | **0.2503** | **+0.0007** |

**+0.28%, *worse* than single rollout.** 8/10 proteins regress under
averaging. Probabilistic averaging blurs the per-rollout distograms,
making them flatter — and we already knew (idea 5 sharpening) that
flatter distograms are worse for LDDT. The rollout diversity is real
but the aggregation rule (mean of probs) is wrong: a hard-mode rule
(pick the bin where rollouts agree most) might be different. Not
chased further here — idea 1 already cleared the bar with much less
machinery.

**Verdict: idea 6 ranks far below idea 1 in this form.** The "honest"
no-prior path costs us most of idea 1's gain. Worth revisiting if
sampling without a `<distance>` stop token produces denser contact
emissions — TODO.

### idea 3: stochastic multi-rollout averaging

Covered by `sampled_contacts_n5_T0.7` above (M=5 averaging of sampled
prefixes). Result: averaging hurts. Not pursuing M=10 — the trend is
clear and the cost grows linearly.

### idea 2: iterative self-distillation

**`iterative_R2_kc1.0_kd1.0`** — Round 1 = K=L contacts from
baseline. Round 2 = K=L/2 contacts + K=L modal-distance commits
(sharpen T=0.1, min_modal_p=0.9). Distance commits become one-hot
rows in the saved distogram (E[d] = bin midpoint), so a wrong commit
zeroes that pair's LDDT.

  mean LDDT 0.2805 (median 0.2801), 1639.7 s. **+12.39%, slightly
  *worse* than plain seeded (0.2816, +12.83%).** Distance commits
  hurt: locking in one-hot rows is high-variance per pair, and the
  net LDDT loss from wrong modes outweighs the gain on right modes
  even at min_modal_p=0.9.

  Verdict on distance commits: kill them. Iterative should be
  contact-only.

## Update: +50% goal

After clearing +10% with sharpened seeded (0.2894, +15.94%), the user
raised the bar: **mean LDDT ≥ 0.3744 (+50%)**. Need another +0.085
absolute. Strategy pivots from knob-tuning to algorithm-shape search.

### GT oracle ceiling diagnostic (NOT a candidate algorithm)

`gt_oracle_seeded` — seed the model with TRUE contacts (every (i,j)
where GT CB-CB < 8 Å and |i-j| ≥ 6), then run the standard
gt_filtered readout under that prefix.

  **mean LDDT 0.7167 (median 0.7219), +187%.** Huge.

| stem | baseline | seeded | oracle | oracle − seeded |
|---|---:|---:|---:|---:|
| 7uk8_A | 0.2010 | 0.2148 | 0.6290 | +0.4142 |
| 7ur2_A | 0.2633 | 0.2836 | 0.7219 | +0.4383 |
| 7xz3_A | 0.1913 | 0.2102 | 0.6993 | +0.4891 |
| 7y5j_A | 0.4485 | 0.5432 | 0.8044 | +0.2612 |
| 7ykm_A | 0.3317 | 0.4020 | 0.8052 | +0.4032 |
| 7ylr_A | 0.2673 | 0.3157 | 0.6734 | +0.3577 |
| 7zs2_A | 0.2500 | 0.2888 | 0.6895 | +0.4007 |
| 8baq_A | 0.1872 | 0.2014 | 0.6870 | +0.4856 |
| 8cba_A | 0.2045 | 0.1973 | 0.7319 | +0.5346 |
| 8eb9_A | 0.1510 | 0.1593 | 0.7254 | +0.5661 |

**Implications:**
- The model is *very* capable with right contacts (median per-protein
  0.72). The +50% bar (0.3744) is well below the ceiling.
- The gap between honest seeded and oracle (+0.43 average) is
  contact-prediction quality. Anything that improves contact quality
  will translate to LDDT gains roughly along this axis.
- Sharpening the oracle's distogram doesn't help (best T=1.0, no
  change). The model's distribution is already sharp when context is
  good. Sharpening was previously rescuing high-entropy distributions
  caused by inadequate context; once context is right, sharpening
  has nothing to do. **This means later seeded variants benefit less
  from sharpening as they get better.**
- Proteins where seeded did poorly (8eb9, 8baq, 8cba, 7xz3) have the
  *largest* oracle gaps. Contact quality is the bottleneck across the
  board, but more so where the model has weak natural confidence.

### idea 2b: iterative contacts-only

Round 1 = seeded contacts from baseline. Round 2+ = pick fresh
contacts from the *previous round's* distogram (sharper because that
round had context). No distance commits — those hurt (see idea 2).

Knob sweep:

| algorithm | rounds | K | min_p | order | LDDT (raw) | + sharpen | Δ% |
|---|---:|---|---:|---|---:|---:|---:|
| iterative_contacts_R2_kc1.0 | 2 | L | 0.3 | long_med_short | 0.2916 | 0.2970 (T=0.3) | +19.0 |
| iterative_contacts_R3_kc1.0 | 3 | L | 0.3 | long_med_short | 0.2953 | 0.2999 (T=0.3) | +20.16 |
| iterative_contacts_R2_kc1.0_min0.1 | 2 | L | **0.1** | long_med_short | 0.3292 | 0.3331 (T=0.1) | +33.5 |
| iterative_contacts_R3_kc1.0_min0.1 | 3 | L | 0.1 | long_med_short | 0.3327 | **0.3376 (T=0.05)** | **+35.3** |
| iterative_contacts_R2_kc1.0_min0.1_byprob | 2 | L | 0.1 | by_prob | 0.3246 | 0.3248 (T=0.3) | +30.1 |
| iterative_contacts_R2_kc2.0_min0.1 | 2 | **2L** | 0.1 | long_med_short | 0.3209 | 0.3209 (T=1.0) | +28.6 |

**Findings:**
- Lowering `min_contact_prob` from 0.3 to 0.1 was the big lever
  (+0.04). Below 0.1 there's no effect (the K=L cap binds in either
  case for our proteins).
- R=2 → R=3 with low threshold added only +0.005 (saturating).
- `by_prob` ordering is *worse* than `long_med_short` (−0.008). The
  range-priority sort matters: long-range contacts as anchors first
  reflects the training-time document structure.
- K=2L (double density) *hurts* (−0.012 vs K=L). Adding lower-
  precision seeds in the second half of the prefix poisons the
  readout. **Precision > count.**
- Sharpening's gain shrinks as the underlying distogram improves
  (matches the GT-oracle finding). For the best iterative variant
  sharpening adds only +0.005.

**Per-protein status vs the +50% bar (0.3744):**

| stem | base | iter_R3_low | oracle | gap to bar |
|---|---:|---:|---:|---:|
| 7y5j_A | 0.4485 | 0.5583 | 0.8044 | **PASS** |
| 7ykm_A | 0.3317 | 0.4468 | 0.8052 | **PASS** |
| 7ur2_A | 0.2633 | 0.3682 | 0.7219 | +0.006 |
| 7zs2_A | 0.2500 | 0.3350 | 0.6895 | +0.039 |
| 7ylr_A | 0.2673 | 0.3041 | 0.6734 | +0.070 |
| 8eb9_A | 0.1510 | 0.2872 | 0.7254 | +0.087 |
| 7xz3_A | 0.1913 | 0.2647 | 0.6993 | +0.109 |
| 8cba_A | 0.2045 | 0.2619 | 0.7319 | +0.112 |
| 7uk8_A | 0.2010 | 0.2565 | 0.6290 | +0.117 |
| 8baq_A | 0.1872 | 0.2439 | 0.6870 | +0.130 |

3/10 already pass. 5 of the 7 misses need >+0.07; 4 need >+0.10. The
oracle ceiling is wide open for every miss — contact-prediction
quality on the weak proteins is the bottleneck.

### Distogram-mixture diagnostic

Tested averaging distograms across {seeded, iter_R2, iter_R3,
iter_R2_low, iter_R3_low}. Equal-weight mix: 0.3029. Top-3 mix:
0.3134. Mix of just (R2_low + R3_low): 0.3329. **Mixtures don't beat
the best single algorithm.** The component algorithms aren't
*differently* wrong — they're varying degrees of the same wrong, and
averaging just blends a good signal with worse ones.

### Mode-vs-mean precision on high-confidence pairs (diagnostic)

For pairs with sharpened max prob > 0.5, the modal-bin LDDT
preservation is *slightly* better than the mean-bin (~+0.01-0.07
absolute) on most proteins, but the gap is small and mixed-sign per
protein. Confirms: distance commits at the mode don't have much
headroom over the mean. Killing distance-commit variants entirely.

### Where we are

Best honest algorithm: **`iterative_contacts_R3_kc1.0_min0.1` +
sharpen T=0.05 → mean LDDT 0.3376, +35.3%.** Chain wall =
1387 (baseline prior) + 1953 (iter R=3) + 21 (sharpen sweep) =
3361 s, 2.4× baseline. Within 5× budget.

+50% bar (0.3744) remains uncleared. Need +0.037 more. The cheap
knobs (more rounds, higher K, lower threshold) have plateaued. Next
must be a different algorithm shape.

### Things tried beyond iteration

**Sampled UNION** (M=10 rollouts → union of unique contacts → single
readout). 0.3168 raw, best T=1.0. Sampling-based seed selection gets
mostly the same set as marginal top-K, plus some noise. Worse than
deterministic top-K. Killed.

**Mixture of distograms** across {seeded, iter_R2, iter_R3,
iter_R2_low, iter_R3_low}. Equal-weight 0.3029. Top-3 mix 0.3134. Best
of (R2_low + R3_low) 0.3329. All worse than best single algorithm.
Averaging blurs distributions.

**Per-protein best (cheats)**: picking the best individual algorithm
per protein on the train set gives mean 0.3365 — barely better than
single best (0.3376). The algorithms aren't differently wrong on
different proteins; they're all wrong in similar ways.

**Per-pair max-confidence (no GT)**: for each pair, pick the variant
with the highest peak probability. 0.3037 — *worse* than individuals.
Cross-algorithm confidence doesn't correlate with cross-algorithm
correctness.

**Iter on top of iter_R3_low** (effectively R=5 total: snapshot
R3_low, then run R=2 iter from it). 0.3364 raw, 0.3389 sharpened.
+35.78%. Tiny improvement over R=3. Confirms iteration saturates by
round ~3.

**Strict iteration from R3_low prior** (1 round, K=2L, min_prob=0.5
using the now-richer iter_R3_low's high-confidence contact set).
0.3333 raw, 0.3343 sharpened. *Worse* than growing-K. Strict
threshold loses too many seeds even after iteration enriches the
candidate pool.

**Growing K per round** — kc=[0.5L, 1.0L, 1.5L], R=3, min_prob=0.1.
Cautious-then-bold schedule. **0.3421 raw (T=1.0 best), +37.07%.**
New best. 4/10 proteins now pass the +50% bar individually
(7y5j, 7ykm, 7ur2 newly passes, growing K also raises 7zs2 close to
the bar). The remaining hard proteins (8baq, 7uk8, 8cba, 7xz3,
8eb9, 7ylr) still need +0.07 to +0.13.

**Plddt-token probe**: free-form T=0.7 sample from base prompt
produces messy mixed output — many `<*-range-contact><pi><pj>` but
also bare position triplets, lonely `<distance>` markers without
position/atom/bin completion, and `<distance><d_X.Y>` direct binding.
No `<plddt_*>` tokens emitted. Plddt path is dead — the model
doesn't naturally produce confidence tokens during generation.

### Where I am

| algorithm | LDDT | sharpened | Δ% |
|---|---:|---:|---:|
| baseline_naive | 0.2496 | 0.2738 | +9.68 (sharp) |
| seeded_contacts (K=L, min=0.3) | 0.2816 | 0.2894 | +15.94 |
| iterative_R3_kc1.0_min0.1 | 0.3327 | 0.3376 | +35.31 |
| **iter_R3_grow_kc05_10_15** | **0.3421** | 0.3418 | **+37.07** |
| gt_oracle_seeded (diagnostic) | 0.7167 | 0.7163 | +187 |
| **+10% bar (cleared)** | — | 0.2746 | — |
| **+50% bar (NOT cleared)** | — | 0.3744 | — |

Best honest algorithm: `iter_R3_grow_kc05_10_15` at 0.3421 — without
even needing sharpening on top. Wall-clock chain = 1387 (baseline
prior) + 1931 (iter) = 3318 s, 2.39× baseline, well within 5× budget.

The growing-K schedule beats fixed-K because each round adds *more*
confident contacts (iteration sharpens contact predictions, raising
many pairs above the threshold). Starting bold and staying bold (K=L
throughout) saturates because round 1's noisy bold picks limit
round 2's improvement.

### iter R=4 grow (kc=[0.5L, 1L, 1.5L, 2.5L])

**`iter_R4_grow_05_10_15_25`**: mean LDDT **0.3511, +40.66%**.
Sharpening doesn't help (T=1.0 wins).

| stem | base | R=3 grow | R=4 grow | oracle | pad |
|---|---:|---:|---:|---:|---:|
| 7uk8_A | 0.2010 | 0.2501 | 0.2515 | 0.6290 | +0.122 |
| 7ur2_A | 0.2633 | 0.3871 | 0.4008 | 0.7219 | PASS |
| 7xz3_A | 0.1913 | 0.2562 | 0.2618 | 0.6993 | +0.112 |
| 7y5j_A | 0.4485 | 0.5626 | 0.5659 | 0.8044 | PASS |
| 7ykm_A | 0.3317 | 0.5131 | 0.5178 | 0.8052 | PASS |
| 7ylr_A | 0.2673 | 0.3026 | 0.3169 | 0.6734 | +0.057 |
| 7zs2_A | 0.2500 | 0.3526 | 0.3619 | 0.6895 | +0.012 |
| 8baq_A | 0.1872 | 0.2503 | 0.2540 | 0.6870 | +0.120 |
| 8cba_A | 0.2045 | 0.2652 | 0.2735 | 0.7319 | +0.100 |
| 8eb9_A | 0.1510 | 0.2815 | 0.3071 | 0.7254 | +0.067 |

R=3 → R=4 added +0.009. Chain wall: 1387 + 2986 = 4373 s, 3.2× baseline.

### iter R=4 grow + strict distance commits

**`iter_R4_grow_kc_kd_strict`** — same grow schedule, plus
kd=[0, 0.1L, 0.3L, 0.5L] distance commits with min_modal_p=0.5 and
sharpen T=0.5 for mode selection.

  mean LDDT **0.3503**, basically identical to no-kd variant
  (−0.0008, within noise). 3770 s wall (1.3× the no-kd R=4 grow).

**Distance commits add no value** even with strict filtering and
substantial prefix density (197 commits in the final round for the
largest protein). The per-pair one-hot LDDT loss exactly offsets the
prefix-context benefit. Dropping distance commits permanently.

### iter R=5 grow — aborted, would have busted budget

Launched `iter_R5_grow_05_10_15_25_30` (kc=[0.5L, 1L, 1.5L, 2.5L,
3.0L]). First protein (7uk8_A, L=394) was still on round 4-5 after
60 minutes. Projected total wall ≥ 6000 s → chain wall ≥ 7400 s >
5× baseline (6920 s). Killed before completion.

Per-round growth was already saturating (R=3 → R=4 added +0.009 in
LDDT); R=5 was unlikely to clear the +50% bar even if completed,
and would have disqualified itself on wall-clock anyway. **Best
in-budget remains `iter_R4_grow_05_10_15_25` at 0.3511 (+40.68%).**

## Final standings

| algorithm | mean LDDT | Δ% | chain wall (s) | budget? |
|---|---:|---:|---:|---:|
| baseline_naive | 0.2496 | — | 1386.7 | — |
| **+10% bar** | **0.2746** | **+10** | — | — |
| sharpen_T0.05 (post-process only) | 0.2738 | +9.68 | ~1610 | ✓ |
| seeded_contacts | 0.2816 | +12.83 | ~1780 | ✓ |
| sharpen on seeded | 0.2894 | +15.94 | ~1810 | ✓ |
| iterative_R3_contacts (min=0.3) | 0.2999 (sharp) | +20.16 | ~2660 | ✓ |
| iterative_R3 (min=0.1) | 0.3376 (sharp) | +35.24 | ~3360 | ✓ |
| iter_R3_grow_05_10_15 | 0.3421 | +37.07 | ~3320 | ✓ |
| iter_R4_grow_05_10_15_25 | 0.3511 | **+40.68** | **4373** | ✓ |
| iter_R4_grow + strict kd | 0.3503 | +40.34 | 5160 | ✓ (no gain) |
| iter_R5_grow_05_10_15_25_30 | — | — | >7400 (projected) | ✗ killed |
| **+50% bar** | **0.3744** | **+50** | — | — |
| gt_oracle (diagnostic) | 0.7167 | +187.14 | — | (cheating) |

**Headline:** `iter_R4_grow_05_10_15_25` — iterative contact-only
seeding with growing K per round. Mean LDDT **0.3511 (+40.68%)**,
chain wall **4373 s (3.16× baseline)**.

**Clears the issue's original +10% bar by 4×. Falls short of the
mid-experiment-raised +50% bar by 0.023 LDDT (9 percentage points).**
The bottleneck is contact prediction quality on harder proteins — the
GT-oracle diagnostic shows the model is capable of ~0.7 mean LDDT with
right contacts, but for proteins where the model's marginal contact
predictions are sparse (8baq, 7uk8, 7xz3, 8cba, 8eb9, 7ylr), even
iteration can't honestly fill in the missing contacts beyond ~0.25-
0.32.

## Update: re-running sampled-prefix variants properly

User flagged that I had a bug in idea 6: the original "sampled
contacts" stopped at the first `<distance>` token. But training docs
stochastically interleave contacts and distances, so `<distance>`
isn't a meaningful boundary. With the stop in place each rollout
emitted only 2-33 contacts; the seed set was tiny and noisy.

**Fix: `ContactsOnlyLogitsProcessor`** — masks all non-contact tokens
during generation. State cycle (mod 3): contact-range → position →
position. Each rollout now spends its whole `max_tokens` budget
emitting contact statements.

### Range-token diagnostic (`probe_range_entropy.py`)

For each protein, after each contact-range token, measure entropy of
the model's next-token distribution over position tokens. Result:
all 3 ranges have similar information (~5.0-6.9 bits, vs max
6.6-8.6); short tends to be 0.1-0.5 bits sharper. **No range is
"flat".** But the model's prior over WHICH range to emit is
overwhelmingly biased to medium (~99% at T=0.7).

So the medium-bias is in the range-token *prior*, not in the
conditional knowledge. `range_strategy` knob added to the LP:
  - `model`: keep model's prior (the bug — heavily medium-biased)
  - `uniform`: overwrite range-token logits to 0 (1/3 of each)
  - `round_robin`: deterministic L,M,S,L,M,S,...

### Sampled-prefix results (with the LP fix)

| variant | strategy | mean LDDT | Δ% | wall (s) |
|---|---|---:|---:|---:|
| M=1 | model (buggy: medium-bias) | 0.2219 | −11.1 | 743 |
| M=1 | uniform | 0.2713 | +8.7 | 769 |
| M=5 union | uniform | 0.3142 | +25.9 | 2458 |
| M=10 union | uniform | 0.3213 | +28.7 | 4237 |

Sampled-uniform plateaus around 0.32. Diminishing returns past M=5.
But — **per-protein breakdown shows it gains on different proteins
than iter_R4_grow**. 8eb9_A: 0.151 (baseline) → 0.362 (sampled M=5
union uniform), gain +0.21 (vs +0.16 from iter_R4_grow). Sampling
delivers contacts that iteration can't extract from the marginal
distogram.

### Combined: sampled-uniform-M=5 prior + iter R=4 grow

This is the new headline.

**`iter_R4_grow_on_sampled_uniform_M5`**: take the M=5 union uniform
distogram as the prior, then run the same R=4 growing-K iteration
(kc=[0.5, 1, 1.5, 2.5], min_p=0.1) on top.

  mean LDDT **0.3564, +42.81%**.  median 0.3665.
  chain wall = 2458 (sampled prior) + 3155 (iter R=4) = **5613 s**,
  4.05× baseline. Within 5× budget.

  Sharpening sweep: T=1.0 wins (no sharpening helps), same pattern as
  the GT-oracle finding.

Per-protein vs +50% bar (0.3744):

| stem | base | iter_R4_grow | new headline | pad |
|---|---:|---:|---:|---:|
| 7y5j (102) | 0.4485 | 0.5659 | 0.5136 | PASS |
| 7ykm (105) | 0.3317 | 0.5178 | 0.4639 | PASS |
| 7ur2 (195) | 0.2633 | 0.4008 | 0.3665 | +0.008 |
| 7ylr (330) | 0.2673 | 0.3169 | **0.3991** | **PASS (newly)** |
| 7zs2 (316) | 0.2500 | 0.3619 | **0.3778** | **PASS (newly)** |
| 8eb9 (95)  | 0.1510 | 0.3071 | 0.3455 | +0.029 |
| 8cba (214) | 0.2045 | 0.2735 | 0.3028 | +0.071 |
| 7uk8 (394) | 0.2010 | 0.2515 | 0.2657 | +0.108 |
| 7xz3 (325) | 0.1913 | 0.2618 | 0.2727 | +0.101 |
| 8baq (208) | 0.1872 | 0.2540 | 0.2567 | +0.117 |

**5 / 10 proteins now pass the +50% bar** (up from 4 with iter_R4_grow
alone). 7ylr and 7zs2 cross the bar with the sampled prior; 7ur2 is
within 0.01.

Tried `iter_R3_grow_on_sampled_uniform_M10`: 0.3401 (worse — adding
more sampling rollouts and dropping one iteration round doesn't pay
off; the iteration round is more valuable than M=10 vs M=5 sampling
diversity).

## Final standings

| algorithm | mean LDDT | Δ% | chain wall (s) | budget? |
|---|---:|---:|---:|---:|
| baseline_naive | 0.2496 | — | 1386.7 | — |
| **+10% bar** | **0.2746** | **+10** | — | — |
| iter_R4_grow_05_10_15_25 | 0.3511 | +40.68 | 4373 | ✓ |
| iter_R4_grow_kc_kd_strict | 0.3503 | +40.34 | 5160 | ✓ (no gain) |
| sampled_uniform_M5_union | 0.3142 | +25.86 | 2458 | ✓ |
| sampled_uniform_M10_union | 0.3213 | +28.71 | 4237 | ✓ |
| iter_R3_grow_on_sampled_M10 | 0.3401 | +36.26 | 6226 | ✓ |
| **iter_R4_grow_on_sampled_M5** | **0.3564** | **+42.81** | **5613** | ✓ |
| **+50% bar** | **0.3744** | **+50** | — | — |
| gt_oracle (diagnostic) | 0.7167 | +187.14 | — | (cheating) |

**Headline: `iter_R4_grow_on_sampled_uniform_M5` — 0.3564 (+42.81%),
chain 5613 s (4.05× baseline), within 5× budget. Falls 0.018 short of
+50% bar.**

5 / 10 proteins pass +50% individually. The 5 misses still need +0.07
to +0.12 — the remaining gap is the model's contact-prediction quality
on hard proteins (oracle ceilings 0.62-0.74), which neither sampling
nor iteration can fully unlock from this checkpoint.

## Same algorithm on the 1.5B model

After rebasing onto main, `MODELS.yaml` has a `1.5B` entry pointing at
`buckets/open-athena/MarinFold/.../protein-contacts-1_5b-distance-masked-70f8f5/step-49999`.
The 1.5B has 24 hidden layers (1.5× the 1B's 16) and GQA with 8 KV
heads (vs 32 for 1B). Note the checkpoint name: **step-49999** —
likely undertrained relative to the 1B production checkpoint.

Re-ran the exact same algorithm on the same 10 train proteins with
`--model 1.5B`, output to `outputs_1.5b/`.

| run | mean LDDT | median | wall (s) |
|---|---:|---:|---:|
| 1B baseline (naive) | 0.2496 | 0.2500 | 1387 |
| 1B sampled\_uniform\_M5\_union | 0.3142 | 0.3213 | 2458 |
| **1B combined (headline)** | **0.3564** | **0.3665** | 3155 |
| 1.5B baseline (naive) | 0.2627 | 0.2577 | 1866 |
| 1.5B sampled\_uniform\_M5\_union | **0.2038** | 0.2126 | 3472 |
| **1.5B combined** | **0.2864** | **0.2665** | 4230 |

**Lift (combined vs same-model baseline):**
- 1B: +42.81%
- 1.5B: **+9.04%**

**The algorithm does NOT scale similarly to 1.5B.**

Most striking: stage A (sampled M=5 union) is *worse* than naive
baseline on 1.5B (0.2038 vs 0.2627, −22.4%). Stage B (iterative
growing-K) recovers some of the lost ground but doesn't reach
parity with even the 1.5B baseline relative to what 1B achieved.

Per-protein 1.5B-combined vs 1B-combined:

| stem | L | 1B_base | 1.5B_base | 1B_combined | 1.5B_combined | 1B lift | 1.5B lift |
|---|---:|---:|---:|---:|---:|---:|---:|
| 7y5j | 102 | 0.4485 | 0.5326 | 0.5136 | 0.5645 | +14.6% | +6.0% |
| 7ykm | 105 | 0.3317 | 0.3124 | 0.4639 | 0.4010 | +39.9% | +28.4% |
| 7ur2 | 195 | 0.2633 | 0.2577 | 0.3665 | 0.3090 | +39.2% | +19.9% |
| 7ylr | 330 | 0.2673 | 0.3071 | 0.3991 | 0.2140 | +49.3% | −30.3% |
| 7zs2 | 316 | 0.2500 | 0.2809 | 0.3778 | 0.2086 | +51.1% | −25.7% |
| 8eb9 |  95 | 0.1510 | 0.1496 | 0.3455 | 0.2665 | +128.8% | +78.1% |
| 8cba | 214 | 0.2045 | 0.2054 | 0.3028 | 0.2772 | +48.1% | +35.0% |
| 7xz3 | 325 | 0.1913 | 0.2011 | 0.2727 | 0.2146 | +42.6% | +6.7% |
| 7uk8 | 394 | 0.2010 | 0.1944 | 0.2657 | 0.1661 | +32.2% | −14.6% |
| 8baq | 208 | 0.1872 | 0.1854 | 0.2567 | 0.2425 | +37.1% | +30.8% |

**Observations:**
- 1.5B baseline (0.2627) is only +5% above 1B baseline — the larger
  model doesn't help much on the naive readout.
- 1.5B combined REGRESSES on 3/10 proteins: 7ylr (−30%), 7zs2 (−26%),
  7uk8 (−15%). All large proteins.
- 1.5B combined STILL beats its own baseline on 7/10 proteins, just
  with smaller gains than 1B managed.
- Largest-protein performance is the worst on 1.5B — possibly the
  undertrained checkpoint has not yet learned to handle long
  sequences and many seeded contacts as well as 1B has.

**Hypotheses for why the algorithm doesn't transfer:**
1. **1.5B is undertrained** (step-49999 only) — has not yet learned
   the conditional-distance distribution well enough for sampling
   constraints to help.
2. **Range-token prior is different on 1.5B** — the "uniform" fix
   was specifically tuned to undo 1B's 99% medium-bias. If 1.5B has
   a different prior over the 3 ranges, forcing uniform might
   actively hurt.
3. **Different conditional-knowledge profile** — the entropy probe
   measured on 1B showed all 3 ranges have similar position-entropy.
   1.5B might be different and need a different sampling policy.

To know which, would re-run `probe_range_entropy.py` on 1.5B and
look at the conditional distributions. Not chasing this here — the
headline answer is clear: **the 1B algorithm does not transfer to
1.5B (step-49999) without re-tuning.**

## Range-token weighting by top-position probability

Idea: pre-compute the top position-token probability after each range
token (3 forward passes per protein), use as the range-sampling
weights. Intuition: if the model has a very confident first-position
prediction for short-range, weight short higher when sampling the
range token. If all ranges are flat, fall back close to uniform.

Implementation: `measure_range_top_probs()` returns `(top_long,
top_med, top_short)`. The LP's `weighted` strategy overwrites
state-0 logits to `log(top_p)` for each range; softmax over the 3
is the per-statement range distribution.

Measured weights (1B):

| stem | L | top_p long | top_p med | top_p short | weights (L/M/S) |
|---|---:|---:|---:|---:|---|
| 8eb9 | 95  | 0.0279 | 0.0367 | 0.0520 | 0.24 / 0.31 / 0.45 |
| 7y5j | 102 | 0.0447 | 0.0599 | 0.1139 | 0.21 / 0.27 / 0.52 |
| 7uk8 | 394 | 0.0183 | 0.0238 | 0.0249 | 0.27 / 0.36 / 0.37 |

Small proteins skew toward short-range; long protein 7uk8 is close
to uniform.

Results (1B):

| algorithm | mean LDDT | Δ% |
|---|---:|---:|
| sampled\_uniform\_M5\_union | 0.3142 | +25.86 |
| sampled\_weighted\_M5\_union | 0.2902 | +16.27 |
| iter\_R4\_grow\_on\_sampled\_uniform\_M5 | 0.3564 | +42.81 |
| iter\_R4\_grow\_on\_sampled\_weighted\_M5 | 0.3493 | +39.96 |

**Weighted is worse than uniform** both alone (−0.024 LDDT) and after
iteration (−0.007 LDDT). Per-protein, weighted wins on small proteins
(7y5j +0.047, 8eb9 +0.035) and loses on most others (7zs2 −0.060 is
the biggest hit). The net trade is negative.

**Why uniform beats weighted:** uniform forces broad coverage across
the 3 ranges. Weighted concentrates sampling on the model's
already-confident ranges, leaving the others under-explored — but
the readout still needs distance info for pairs across ALL
separations. Over-fitting the sampling to the strongest range-token
signal sacrifices coverage on the others.

Sticking with `range_strategy=uniform` as the 1B headline.

## Held-out 10 proteins (1B): protein-set overfit check

Knobs (range_strategy, K schedule, min_contact_prob, growth pattern)
were tuned exclusively on the 10 train proteins. To estimate protein-
set overfit, picked 10 NEW proteins via
`random.Random(42).sample(...)` from the same FoldBench pool
(`n_residues ≤ 400`, excluding the train 10) and ran the headline
algorithm with identical knobs.

Held-out protein list (seed=42):

| stem | L |
|---|---:|
| 7t9r |  38 |
| 7y8i |  97 |
| 7zoi | 151 |
| 7wz5 | 161 |
| 8bau | 189 |
| 8gmy | 236 |
| 7xg9 | 288 |
| 7x4p | 307 |
| 7v3o | 328 |
| 7qsj | 373 |

Wall clocks:

| step | held-out wall (s) |
|---|---:|
| baseline_naive | 1226 |
| sampled\_uniform\_M5\_union | 2558 |
| iter\_R4\_grow on top | 3255 |
| **chain total** | **7039 (5.7× heldout-baseline)** |

(5.7× exceeds the train 5× budget rule. Held-out has a slightly
shorter average length than train, so the baseline is faster while
the combined algorithm scales similarly. For the generalization read
what matters is the mean-LDDT lift; the wall budget was specific
to the train set.)

Headline:

|  | mean LDDT | median | lift |
|---|---:|---:|---:|
| heldout_baseline | 0.2797 | 0.2746 | --- |
| heldout_stageA (sampled\_uniform\_M5) | 0.3189 | 0.3160 | +14.04% |
| **heldout_combined** | **0.3685** | **0.3270** | **+31.75%** |

Compared to train (+42.81%), the lift drops 11 percentage points on
held-out. **Every held-out protein gains positively:**

| stem | L | baseline | combined | lift |
|---|---:|---:|---:|---:|
| 7t9r |  38 | 0.3455 | 0.3980 | +15.2% |
| 7y8i |  97 | 0.4689 | **0.7179** | +53.1% |
| 7zoi | 151 | 0.2362 | 0.3083 | +30.5% |
| 7wz5 | 161 | 0.2222 | 0.3023 | +36.0% |
| 8bau | 189 | 0.2518 | 0.3270 | +29.9% |
| 8gmy | 236 | 0.2746 | 0.3139 | +14.3% |
| 7xg9 | 288 | 0.2980 | 0.4252 | +42.7% |
| 7x4p | 307 | 0.2319 | 0.3014 | +30.0% |
| 7v3o | 328 | 0.1594 | 0.2465 | +54.6% |
| 7qsj | 373 | 0.3080 | 0.3440 | +11.7% |

**Takeaway: the algorithm generalizes.** Real overfit cost of ~11 pp
from tuning knobs on a 10-protein set, but the lift is robustly
positive across both protein sets.

## Overfit decomposition: protein vs model

|  | drop in lift vs train | dimension changed |
|---|---:|---|
| held-out 10 (same model, different proteins) | −11 pp | protein set |
| 1.5B on train (different model, same proteins) | −34 pp | model |

**The algorithm's tuning is ~3× more model-specific than
protein-specific.** Most of the "+42.81%" headline reflects the
algorithm exploiting features of this 1B checkpoint; the protein-set
portion is a much smaller share. The +31.75% on a fresh protein set
is the more honest "this is what the algorithm achieves on this
model" number; the +42.81% should be read as "what's achievable
when the range/K knobs are let to overfit the eval set."

(Note: the "1.5B is undertrained" hypothesis was ruled out — model
authors confirmed both 1B and 1.5B saw the same number of training
steps. The 1.5B transfer failure is about the model, not training
budget.)
