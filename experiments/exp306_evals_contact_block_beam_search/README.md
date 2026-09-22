---
marinfold_experiment:
  issue: 306
  title: 'exp: contact-block beam search for fold discovery and eval-val precision'
  kind: evals
  branch: exp/306-contact-block-beam
---

# exp: contact-block beam search for fold discovery and eval-val precision

**Issue:** [#306](https://github.com/Open-Athena/MarinFold/issues/306) · **Kind:** `evals` · **Branch:** `exp/306-contact-block-beam`

## Question

Does short, contact-aligned beam search improve contacts-v1 decoding? Specifically, after the model decides to emit `<contact>`, search jointly over its two position tokens before committing the completed `<contact> <pos1> <pos2>` statement. Measure ordinary contact accuracy and whether this exposes both fold-specific modes in the fold-switching testbed.

## Hypothesis

Choosing each contact's two positions jointly may reject locally plausible
first positions whose likely completions are poor contacts. A truncated beam
could therefore improve contact precision. It may also concentrate the
rollout distribution and remove rare alternate-fold maps, so both ordinary
accuracy and two-mode coverage matter.

## Approach

Use the exact exp277 contacts-v1 checkpoint at step 266344. At each statement
boundary, sample one token with the exp82 settings (temperature 1, top-p 0.95,
top-k disabled). If it is `<end>`, stop. If it is `<contact>`, run vLLM beam
search for exactly two position tokens, retain complete valid contacts, and
sample one from the top beam candidates in proportion to its two-token joint
model probability. Thus every committed contact is a complete
`<contact> <pos1> <pos2>` block. A rollout gets a fresh document realization,
and the ordinary `6L+128` token cap is unchanged. The decoder sees only the
sequence and generated prefix; reference contacts are inaccessible to it.

Test beam widths 4 and 8 on the development fold-switching proteins and a
small eval-val pilot. Freeze the choice before looking at the 29 primary
fold-switching test outcomes. Compare all 97 natural eval-val proteins with
the [exp277 100-rollout baseline](../exp277_models_single_mpnn_pilot/evals/2026-09-13_rollout_v2/README.md)
using the same [exp89 metric functions](../exp89_evals_contacts_v1_model_on_eval_set/compute_metrics.py),
especially all- and long-range R-precision. Eval-test remains unread.
For fold switching, use the same exp304 25%-recall, 0.10-enrichment
contact-level criterion on every rollout and on a 16-map shortlist sealed
without references; separately report the stricter 50%-recall sensitivity.

Both comparisons report pure generation seconds, completed rollouts, number of
contacts, and rejected beam candidates. The 100-rollout comparison answers the
accuracy question; matched H100 time determines whether a gain would be worth
the extra search cost. Per-rollout maps and beam choices are retained for audit.

## Success criteria

Do not call the beam strategy an improvement unless eval-val R-precision is
at least preserved and the blind fold-switching shortlist finds more proteins
with both contact modes at a comparable H100 budget. Oracle-pool gains alone
show only that the search generated candidates, not that inference can select
them. Any new apparent alternate-fold success needs the exp304 3D validator
before it is called a recovered fold.

## Results

The width-4 smoke run produced 100/100 finished rollouts on the 75-residue
development pair `2hdma_2n54b` in 3.68 seconds of pure generation. No
contact was malformed. Its rollouts averaged 17.7 contacts versus 50.1 among
the first 100 plain iid rollouts for that pair. This was a validity check, not
an accuracy result.

The complete 20-protein development set was used to choose a beam width before
the held-out fold-switching run. On the 15 primary development proteins, both
widths found both contact modes in 1/15 oracle pools and 0/15 blind 16-map
shortlists; iid100 found 1/15 by both measures. Mean minority-mode enrichment
in the blind shortlist fell relative to iid by 0.044 for width 4 and 0.076 for
width 8. Mean paired H100 generation time was 6.7x iid for width 4 and 12.3x
for width 8. Width 4 is the frozen choice: it tied width 8 on dual-mode
coverage, preserved more minority-mode evidence, and cost roughly half as much.
These are contact-level development results, not 3D fold recoveries.

On the complete 97-protein natural eval-val set, width 4 gave mean all-range
R-precision 0.54182 versus 0.55375 for the standard exp277 iid100
rollout-and-resample baseline, a paired difference of -0.01193 with a
protein-bootstrap 95% interval [-0.01741, -0.00641]. Long-range R-precision
was 0.52932 versus 0.53802, a paired difference of -0.00870
[-0.01629, -0.00021]. All 97 targets had 100 rollouts; one rollout was capped
and excluded from voting. The beam decoder used 5,269 seconds of pure H100
generation versus 3,539 for the baseline, a 1.49x aggregate cost; the mean
per-protein time ratio was 1.37x. Thus this width does not preserve ordinary
contact accuracy. Eval-test was not read.

Held-out fold-switching results are pending.

## Conclusion

Pending the complete development, eval-val, and fold-switching comparisons.
