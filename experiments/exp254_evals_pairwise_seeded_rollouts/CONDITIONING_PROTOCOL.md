# Matched contact-prompt intervention, version 1

This protocol is fixed before inspecting new inference outcomes. It follows the
single-contact audit in PR #255 and directly tests whether m2-p06 uses supplied
contact statements. It stays within exp254's conditioning question.

## Inputs and conditions

Use m2-p06, `prot-exp232-cw-cv1-decontam-s02-m2-p06-aug` step 145199, on all 97
natural eval-val monomers. No eval-test or eval-denovo scoring. Two context draws
per protein; 100 independently realized rollouts per arm and draw. Sampling is
T=1, top-p=0.95, top-k disabled, with the complete 6L+128 completion-token allowance
reserved AFTER the entire prompt. Reject context overflow or unfinished rollouts.

Eight arms: no contacts (`iid`), independent-repeat no contacts (`iid_repeat`),
and true, false, or predicted contexts at 10 and floor(L/3) contacts. Context pairs
are fixed across 100 rollouts within an arm/draw, remapped into each realization,
with shuffled contact order and random pair orientation. The 10-contact set is
nested in its larger counterpart. Shared document realizations and request RNG
seeds pair the conditions; they do not guarantee identical random paths after
prompt changes. `iid_repeat` has a different request RNG seed.

True pairs are sampled from resolved GT contacts. False pairs are sampled without
replacement from resolved noncontacts, matched to the true pairs' short/medium/
long separation bins. Predicted pairs are the top vote-ranked contacts from the
archived 100 iid first-pass rollouts: the best existing deployment recipe, rather
than the weaker pairwise readout. Their selection uses sequence length and scores
only, with no GT labels or resolved-mask filtering. The original source votes and
hashes are preserved. Predictions are constant across draws; context ordering,
realizations, and continuation sampling vary. True/false sets also vary.

## Mechanistic outcomes

For each protein/draw, exclude the union of ALL provided pairs across both doses
and all three conditions. Use that same remaining resolved-pair universe and its
remaining R for every arm. Supplied contacts copied into generated continuations
are excluded too. Every planned protein/draw retains at least eight true contacts.
Report paired remaining-contact R-precision, top-R map turnover, and changes in
vote frequencies. Average the two draws within each protein before statistical bootstrapping.

Also measure the conditional distribution over a forced next contact pair on
realization zero using the
existing exact, full-vocabulary pairwise readout, separately labeled as a
mechanistic diagnostic. Its reference is the matched no-contact prompt, and its
scoring universe is the same common remainder. This forces the `<contact>` token
and does not include its probability versus stopping. It is not the production metric.

True-context improvement coupled with false-context deterioration demonstrates
use of supplied context even if predicted contexts fail. Little response to a
substantial true set warrants checking prompt serialization and positive controls
before inferring model insensitivity. Global averages do not prove absence of
local responses. Record raw completions, copied and novel contact counts,
termination, per-input timings, worker details, and exact engine versions.

## Practical outcome and decision rule

The predeclared practical arm is `pred_large`: 100 first-pass iid rollouts produce
the context, followed by 100 conditioned rollouts. Score its complete final contact
map on the full standard resolved-pair universe. Given pairs count once per
document, with 100 votes; generated repeat mentions never double-count them.

The primary comparator uses the SAME archived first-pass 100 votes plus fresh 100
iid continuation votes, totaling 200. Also report fresh iid 100 and fresh iid 200
(`iid` plus `iid_repeat`). Measure actual inference time and token counts; the
long conditioned prefixes incur extra work, so equal rollout counts alone are
not a claim of equal compute.

A useful improvement is approximately +0.03 absolute R-precision. Average draw
contrasts within protein, then use a fixed-seed 20,000-draw paired protein bootstrap
(n=97). A 95% CI wholly below +0.03 rules out that target gain for this variant and
saved run; wholly above establishes it; crossing remains unresolved. Exploratory
arm contrasts are pointwise, without multiplicity correction. Diagnostic
GT-conditioned or reduced-universe gains do not satisfy the practical criterion.

## Execution and validation

The local A5000 holds the existing checkpoint. CoreWeave also holds the pinned
export at its original S3 location; GPU jobs run in that same storage region.
Only small plan/code bundles cross regions. Run one full operational smoke before
fan-out, validating serialization, exact sample counts, and token allowance.
Use 12 independent single-H100 batch shards, interleaving length-sorted targets.
Persist complete per-protein units with hashes; retries skip only completed units.

The public run archive will include `plan.json`, `source_votes.npz`, raw model
completions, vote and probability matrices, timings, completion manifests, and
source-code/provenance records. New results will update the README and summary
PDF when every expected protein and condition has completed.

### Execution amendment (before population inference)

The checkpoint region reported zero free H100s. The successful local operational
smoke completed all 1,600 rollouts without truncation. Use a single RNO2A eight-GPU
pod instead: stage the 5.885 GB checkpoint once, then share the local files across
eight independent workers. Run the complete single-protein smoke on that engine
before the eight-worker fan-out. Disable automatic pod retries; an unplanned
second transfer could exceed the 10 GB budget. Inputs, arms, RNG, and decision
rules are unchanged. Output matrices/completions are much smaller than weights.

### Fixed-budget completion amendment (before new accuracy was examined)

The initial run found many length stops under large false contexts: for example,
73/100 on 8arl_A draw 0, versus one on 8dhj_A draw 1 and one on 8c4y_A draw 0.
The original reject-on-length-stop rule aborted those workers. Dropping those
proteins would select the cohort according to the intervention's behavior.

Retain every rollout at the original, identical 6L+128 completion budget instead.
An exact-budget `length` stop is a censored continuation, not a missing sample.
Report its incidence for every arm, score all emitted contacts with the same
fixed R, and retain empty continuations too. Unknown termination reasons and
length stops before the stated token budget remain errors. No prompts, contact
sets, sampling parameters, target membership, or practical decision rule change.
No new accuracy effects were examined before this amendment; it is nevertheless
a post-launch change to acceptance criteria, rather than fully preregistered
handling. Mechanistic claims are about behavior under this fixed inference budget,
not unlimited, voluntarily terminated continuations.

Reuse already completed units without rewriting them. Resume the remaining units
with an explicitly opted-in worker that differs only in termination acceptance.
Publish both worker sources and require an explicit hash allowlist during
verification. The model, engine versions, and input-plan hash remain identical.

Recovery does not guarantee bitwise-identical generated text, even with identical
request seeds. Before scoring accuracy, comparison with saved failed groups
found different texts and two originally capped groups that no longer capped on
rerun. Final-unit cap rates therefore describe accepted executions, not all first
attempts. Preserve the original failed groups and report their observed counts
separately; the complete first-attempt population is unavailable. As a sensitivity
check, replace each rerun false-context group's votes with its original failed
group and recompute the mechanistic contrasts. Also report the primary contrast
on proteins whose initial units were retained. This selected-subset diagnostic
does not replace the full 97-protein primary analysis.
