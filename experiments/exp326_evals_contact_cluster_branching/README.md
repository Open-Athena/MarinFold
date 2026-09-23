---
marinfold_experiment:
  issue: 326
  title: 'exp: coherent contact-cluster branching for oracle rollouts'
  kind: evals
  branch: exp/326-contact-cluster-branching
---

# exp: coherent contact-cluster branching for oracle rollouts

**Issue:** [#326](https://github.com/Open-Athena/MarinFold/issues/326) · **Kind:** `evals` · **Branch:** `exp/326-contact-cluster-branching`

## Question

Can a staged rollout policy improve oracle best-of-100 contact accuracy by
co-clustering contacts from 50 iid warm-up rollouts, then spending the other 50
rollouts on small coherent bundles from stable, under-visited clusters?

## Hypothesis

Positive residual co-occurrence should define coherent contact neighborhoods,
while negative residual association should separate them. A jointly observed
3- or 5-contact bundle may steer a continuation more strongly than the
single-contact seeds tested in #254 without incurring the error propagation of
large predicted partial maps.

## Approach

Use #277 `contacts-v1-exp277-m2-p06-full-epoch-1.5B` step 266344 and eval-val
only. Reuse #321's first 50 iid rollouts as the discovery pool and its first 100
as the control. Collapse endpoint shifts within two residues, compute binary
contact-occurrence correlations independently in the two 25-rollout halves,
define pair distance from the weaker split-half correlation, and use complete
linkage so every within-cluster pair clears a 0.2 correlation floor in both
halves. Each branch
bundle contains 3 or 5 contacts that occurred together in one real warm-up
rollout. Equal allocation across stable clusters upsamples minority states but
does not give the rarest cluster unbounded priority.

Development compares cluster-aware bundles with paired random coherent bundles
from the same source rollouts. The supplied bundle is excluded from the primary
continuation score. The complete map including the predicted seed is retained
as a secondary deployable readout. All arms use T=1, top-p=.95, no top-k, a
completion budget of `6L+128`, and fresh document realizations.

Natural development is the frozen 16-protein #321 split; the remaining 81 are
untouched until the gate below passes. The 15 fold-switch development proteins
are a secondary mechanism stress test. Eval-test remains unread.

## Success criteria

Primary: validity-gated oracle fixed-R precision, all- and long-range, for a
100-map pool consisting of 50 shared iid warm-ups and 50 seeded continuations.
Continue to the 81-protein split only if a cluster arm beats both iid100 and its
same-k random coherent-bundle control by at least 0.005 all-range on natural
development, without a material malformed or completion-rate regression.

Secondary: consensus R-precision, mean rollout precision, true-contact union
recall, union size, pairwise Jaccard, cluster coverage, timing, generated tokens,
contacts per rollout, and malformed/unfinished counts. Paired differences use a
protein bootstrap.

Raw rollout and per-target timing parquets are public at
`hf://buckets/open-athena/MarinFold/data/exp326/contact-cluster-branching-v1/`
(542 files, 19,177,484 bytes). The compact timing table and analysis results are
committed under `data/`.

## Results

The k=5 arm cleared the frozen development gate: on 16 proteins, primary
continuation-only validity-gated oracle R-precision was 0.4893, versus 0.4799
for iid100 (+0.0094) and 0.4809 for random k=5 (+0.0083). It was therefore the
only arm advanced to the untouched 81-protein split. The apparent gain was
concentrated in the eight development proteins with an eligible stable cluster
(+0.0193 versus iid); the other eight used the preregistered coherent-random
fallback.

The oracle gain did not replicate. On the 81 held-out eval-val proteins,
cluster k=5 scored 0.5296 versus 0.5331 for iid100, a paired mean difference of
-0.0035 (95% protein-bootstrap CI -0.0078 to +0.0009). It also trailed the
same-k random coherent control by -0.0023 (CI -0.0068 to +0.0019). Restricting
descriptively to the 39 proteins with an actual eligible cluster did not rescue
the effect: oracle delta versus iid was -0.0014 (CI -0.0085 to +0.0060). Across
all 97 natural proteins, oracle was -0.0014 relative to iid.

Consensus moved differently. Held-out all-range consensus rose from 0.5625 to
0.5661, +0.0036 versus iid (CI +0.0000 to +0.0072), but was -0.0008 versus the
random k=5 control. On the stable-cluster subset the consensus delta versus iid
was only +0.0009; the aggregate increase came from fallback targets. Long-range
held-out consensus fell by -0.0038 versus iid and -0.0053 versus random k=5.
Thus the small all-range consensus increase is a generic coherent-seeding or
sampling effect, not evidence that the learned clusters identify useful modes.
The cluster and random arms are identically distributed on fallback targets,
so their realized fallback-to-fallback difference is randomization noise and is
not reported as a policy contrast in the stratified summary.

The strict k=5 policy had limited coverage: an eligible split-half-stable
cluster existed for 8/16 development and 39/81 held-out proteins. k=3 covered
all 16 development proteins, but failed its oracle gate (+0.0042 versus iid,
below the required +0.005) and reduced long-range oracle accuracy.

The 15-pair fold-switch development stress test was also null. iid100, random
k=5, and cluster k=5 each reached fold 1 on 10 pairs, fold 2 on four, and both
modes on one. Mean pairwise Jaccard was 0.2771 for iid100, 0.2761 for random
k=5, and 0.2782 for cluster k=5: clustering was slightly *less* diverse by this
measure. Twelve of 15 pairs had an eligible stable cluster, so this result is
not explained by fallback coverage.

## Conclusion

Do not adopt this contact-cluster brancher for oracle best-of-100 inference. The
development signal was selection noise: it disappeared both on the full
held-out policy and on the subset where the policy actually found a stable
cluster. The experiment does show that five-contact prompt seeding can move
consensus slightly without increasing malformed generations, but a matched
random coherent bundle moves all-range consensus at least as much.

The likely issue is that pairwise contact co-occurrence from only 50 maps is a
weak proxy for a coherent structural basin. A follow-up should require a
stronger object than a contact cluster—for example, cluster whole maps using a
geometry-aware distance, select medoid partial maps, and allocate branches by
online novelty—while retaining this experiment's iid and same-prompt-size
random controls.
