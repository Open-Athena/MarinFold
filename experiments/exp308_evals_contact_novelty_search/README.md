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

After inspecting partial development results for the first four settings, we added
one exploratory stronger early-only setting (width 16, epsilon 0.2, decay over
20 contacts) on the same seven pairs. The earlier penalty changed early contact
reuse but had not produced a new primary dual-mode hit on the short pairs.
This fifth setting is adaptive development tuning and must be judged on the
untouched primary held-out set.

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

The seven-pair development pilot is complete. The columns below count proteins
for which at least one of the 100 generated contact maps matches each reference
fold (oracle pool), or for which the reference-blind 16-map shortlist does so
(blind). These are **contact-level** matches, not folded 3D structures.

| Decoder | Oracle dual / 7 | Blind dual / 7 | Early contacts unseen in previous rollouts | H100 time / width 4 |
| --- | ---: | ---: | ---: | ---: |
| iid100 | 3 | 2 | — | 0.14 |
| Unpenalized width 4 | 2 | 2 | — | 1.00 |
| Width 16, no penalty | 1 | 1 | 11.9% | 3.94 |
| Width 16, constant epsilon 0.05 | 1 | 1 | 13.7% | 4.14 |
| Width 16, early epsilon 0.05 | 1 | 1 | 13.1% | 4.05 |
| Width 32, early epsilon 0.05 | 2 | 0 | 13.5% | 12.59 |
| Width 16, early epsilon 0.2 | 2 | 1 | 15.7% | 4.02 |

All five new settings produced 100 finished rollouts on every pilot protein.
None recovered both modes on any of the five primary development pairs; iid100
and width 4 each recovered one. The stronger early penalty did increase early
novelty, but its extra oracle hit was a secondary pair that iid100 had already
covered, and it did not survive the blind shortlist. The width-32 variant did
not recover the missing long primary fold despite its much higher cost.

Before opening any primary test references, we froze `b16_e0p2_d20_w10` in
`data/frozen_choice.json`. It tied width 32 for development oracle coverage,
retained one blind hit rather than zero, and used one-third as much H100 time.
The small difference in blind minority enrichment (0.008 in width 32's favor)
was not persuasive against those differences. This choice is exploratory
because epsilon 0.2 was added after inspecting partial development scores.

### Primary held-out test

The frozen setting was run on all 29 primary test pairs, with exactly 100
attempted rollouts per pair. The reference-blind 16-map shortlists were
[committed at `fa355a46`](https://github.com/Open-Athena/MarinFold/commit/fa355a46)
before opening the held-out reference contacts. Two of 2,900 rollouts were
malformed; both are ineligible for hits. The other 2,898 finished normally.

| Decoder | Oracle dual / 29 | Blind dual / 29 | Oracle dual at 50% recall / 29 | H100 inference seconds | Time / iid100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| iid100 | 6 | 2 | 2 | 270 | 1.00 |
| Unpenalized width 4 | 2 | 0 | 0 | 1,697 | 6.28 |
| Frozen width 16, early epsilon 0.2 | **1** | **0** | **0** | **7,376** | **27.28** |

The selected method found the fold-2 contact mode on 4/29 proteins, compared
with 10/29 for iid100. It found both modes on one protein,
`5jzha_5jztg`, that iid100 missed, but width 4 already found it; its fold
recalls were only about 0.28–0.29, below the stricter 0.50 threshold. The
reference-blind shortlist missed it. This pair has one sequence mismatch
between its two structures. On the 17 strictly sequence-exact test pairs, the
selected method found no dual-mode protein, while iid100 found three in the
100-map pool and one in the blind shortlist.

Relative to iid100, the selected method gained `5jzha_5jztg` in the oracle
pool and lost six: `1kcta_3t1pa`, `1qs8b_1miqb`, `2c1vb_2c1uc`,
`3g0ha_3ewsb`, `3j7wb_3j7vg`, and `4y0mj_4xwsd`. Its blind shortlist lost
`3j7wb_3j7vg` and `4y0mj_4xwsd` and gained none. The paired oracle difference
is −5/29 proteins (−17.2 percentage points; protein-bootstrap 95% interval
−34.5 to 0 points); the paired blind difference is −2/29 (−6.9 points;
interval −17.2 to 0). These intervals include zero, so the small cohort does
not establish the size of a general performance loss. It provides no evidence
of an improvement.

The penalty did alter sampling: on the pilot pairs, the fraction of the first
20 contacts that had not appeared in an earlier rollout rose from 11.9% with
zero penalty to 15.7% with the selected penalty. Mean recall of contacts
shared by both structures was 0.465 without a penalty and 0.451 with the
selected early-only penalty, so these results do not show that annealing
preserved shared-contact quality. Distinct full contact maps were generated
in all 100 pilot rollouts even without a penalty, making map uniqueness an
uninformative diversity measure here.

The raw rollout outputs and timing files are under
`s3://marin-us-east-02a/MarinFold/exp308/{pilot-v1,test-v1}/` and published
to the public Hugging Face bucket at
[`data/contact-novelty-search-exp308`](https://huggingface.co/buckets/open-athena/MarinFold/tree/main/data/contact-novelty-search-exp308).
The small per-protein tables, sealed shortlists, plot, and per-input timings
are in this experiment directory. No eval-val R-precision or 3D fold
validation was run for this negative candidate.

## Conclusion

This contact-frequency penalty modestly increased early novelty but did not
make alternate folds easier to discover at a fixed 100-rollout budget. The
frozen early-only strategy had fewer held-out dual-mode hits than iid sampling
and cost 27 times as much H100 inference. It should not replace iid100 for
fold-switching search on the evidence here. A reference-aware oracle hit is
only a contact-map match; none of these results establish recovery of a 3D
alternate fold.
