---
marinfold_experiment:
  issue: 308
  title: 'exp: online contact novelty penalties for fold search'
  kind: evals
  branch: exp/308-contact-novelty-search
---

# exp: online contact novelty penalties for fold search

**Issue:** [#308](https://github.com/Open-Athena/MarinFold/issues/308) · **Kind:** `evals` · **Branch:** `exp/308-contact-novelty-search`

## Question

Can online contact-count penalties make the contacts-v1 decoder explore alternate fold-specific maps more effectively than plain iid100 and the unpenalized contact-block beam search?

## Hypothesis

The first few contacts steer the rest of a rollout. Penalizing contacts that
earlier rollouts already used may create different early prefixes and increase
the chance of generating both fold-specific contact modes. A constant penalty
can also suppress contacts that both correct folds share, so a penalty that
decays to zero after the first 20 emitted contacts may preserve map quality.

## Approach

Use the exp277 contacts-v1 checkpoint at step 266344 and the same document
realization, stop sampling, token budget, and two-position beam decoder as
[exp306](../exp306_evals_contact_block_beam_search/README.md). For each valid
complete beam candidate `(i,j)`, subtract `epsilon_t * n(i,j)` from its
two-token joint log probability, softmax the adjusted beam scores, and sample.
`n(i,j)` counts earlier **completed rollouts** for that protein containing the
contact. Generate rollouts in waves of 10; each wave uses counts from finished
previous waves. No reference structure is available to the decoder.

Compare four pilot configurations on seven preselected development pairs in
`data/targets.csv`: width 16 with no penalty; width 16 with constant epsilon
0.05; width 16 with epsilon 0.05 falling linearly to zero over the first 20
emitted contacts; and width 32 with the same early-only penalty. Two short
secondary pairs already have iid dual-mode coverage, four primary pairs lack
it, and one long primary pair has it. This tests both preservation and new
discovery. Each configuration gets 100 rollouts per pair. Select one method
using development-only contact scores, then run it on the 29 primary held-out
fold-switching pairs.

Seal a reference-blind 16-map shortlist for each set before opening the two
reference contact sets. Score with exp304's 25%-recall and 0.10-enrichment
criterion, and report the 50%-recall sensitivity separately. Compare oracle
pool coverage, blind shortlist coverage, per-protein map diversity, contact
density, completed rollouts, early-contact novelty, and pure H100 time against
both iid100 and unpenalized width-4 beam search. Contact-level hits remain
distinct from validated 3D fold recovery. Eval-test is not used.

## Success criteria

The strategy is useful only if it adds held-out dual-mode proteins in the
blind shortlist or materially improves oracle coverage at a defensible time
cost, while preserving valid completed contact maps. Report gains and losses
per protein, including when the aggregate result is negative.

## Results

Pilot and held-out runs in progress.

## Conclusion

Pending complete fold-switching results.
