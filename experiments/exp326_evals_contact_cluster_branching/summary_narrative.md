# Summary slides — exp: coherent contact-cluster branching for oracle rollouts

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide.
     Keep this current as the experiment progresses. -->

## What we're doing

Use 50 iid contacts-v1 rollouts to infer stable contact-occurrence clusters,
then allocate 50 branch rollouts equally across those clusters. Each branch is
prompted with a coherent 3- or 5-contact bundle observed together in one warm-up
map. Compare with iid100 and paired random coherent bundles.

## Why

#321 improved oracle accuracy by strengthening samples but made them more
similar. This experiment tests the complementary lever: spend part of the same
100-rollout budget on stable minority contact states. Single-contact seeding was
too weak, while large predicted prompts can amplify errors, motivating small
coherent bundles and a development gate.

## Development gate

The k=5 policy passed on 16 frozen development proteins: continuation-only
oracle fixed-R was 0.4893, +0.0094 versus iid100 and +0.0083 versus random k=5.
The k=3 gain versus iid was only +0.0042, below the +0.005 gate. We froze k=5
before opening the other 81 eval-val proteins.

## Held-out oracle: no replication

On 81 untouched proteins, cluster k=5 scored 0.5296 oracle fixed-R versus 0.5331
for iid100: delta -0.0035, 95% CI [-0.0078, +0.0009]. It also trailed random
k=5 by -0.0023. The 39-protein subset with a real stable cluster was still
flat-to-negative, so fallback dilution does not explain the failed replication.

## Consensus: small generic seeding effect

Held-out all-range consensus increased +0.0036 versus iid100, but cluster k=5
was -0.0008 versus random k=5. Long-range consensus fell -0.0038 versus iid and
-0.0053 versus random. The all-range increase came from fallback targets, not
targets on which stable clusters existed.

## Coverage is the bottleneck

The strict k=5 split-half criterion found an eligible cluster on only 8/16
development and 39/81 held-out proteins. k=3 had much better coverage but did
not clear the oracle gate. Pairwise contact co-occurrence from 50 maps appears
too noisy and local to identify reusable structural basins.

## Fold-switch stress test is also null

On 15 frozen fold-switch development pairs, iid100, random k=5, and cluster
k=5 each reached fold 1 on 10 pairs, fold 2 on four, and both modes on one.
Cluster k=5 had slightly higher pairwise Jaccard than iid100 (0.2782 vs
0.2771), despite eligible stable clusters on 12/15 pairs.

## Conclusion

Do not adopt this brancher for oracle best-of-100 inference. The development
gain was selection noise. A future attempt should cluster whole contact maps
with a geometry-aware distance and branch from medoid partial maps, while
retaining iid100 and same-prompt-size random controls.
