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

_(next up)_
