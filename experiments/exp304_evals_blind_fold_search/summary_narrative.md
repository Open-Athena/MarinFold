# Summary slides — exp: can blind inference search discover both folds of fold-switching proteins?

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide.
     Keep this current as the experiment progresses. -->

## Question and design

Can sequence-only inference return both observed folds of a fold-switching protein? We ran the exp277 contacts-v1 checkpoint on 67 non-capped pairs. The primary test set has 29 pairs with at least 98% identity, held apart by sequence group from 15 development pairs. Seventeen of those 29 are literally identical. Every method returned at most 16 ranked maps chosen without reference contacts or structures.

## The primary comparison is negative

At 100 shared root plus 100 arm rollouts per protein, 10-contact clustered branching improved mean minority-fold enrichment by only 0.013 over independent sampling. The paired 95% interval is -0.014 to +0.043; mean minority recall was essentially unchanged. Blind dual-contact hits were 3/29 versus 2/29. The extra hit failed a 3D check. On the 17 exact-sequence test pairs, both methods had one contact hit and the paired interval still spans zero. We cannot claim a reliable search improvement.

## What was actually discovered

For 3j7w/3j7v, both independent and branching top-16 shortlists included maps that reconstructed distinct switching-region structures. With the Fold1 sequence input, region-touch lDDT against (Fold1, Fold2) was (0.477, 0.206) and (0.356, 0.702) for the branching shortlist; the independent shortlist also favored the respective folds. True-contact controls showed comparable separation. This is an existence proof for blind inference discovery, not a branching advantage.

## Where the search is limited

Four of seven branch-10 oracle dual-contact hits were present among 200 candidates but missed by the blind 16-map selector. In a separate plain-sampling run, blind dual hits were 2, 3, and 3 at 100, 200, and 500 rollouts; the oracle pool reached 9 by 200. The first priority is better reference-free ranking of coherent minority maps. Contact-level cutoffs are operational and not calibrated as structural thresholds; only four proteins received 3D checks.

## What the 500-rollout oracle means

The oracle inspects all 500 independent maps against both reference contact maps. A dual hit needs one Fold1-like and another Fold2-like map; it does not mean the blind selector found either. Among 29 held-out primary pairs, 19 had any Fold1-like map, 13 any Fold2-like map, and 9 had both. The same counts for the 17 exact-sequence test pairs were 12, 7, and 5. Across all 67 non-capped pairs, 31 had both. The prespecified 25% contact-recall screen is permissive: raising recall to 50% reduces held-out dual coverage to 2/29. Only 3j7w/3j7v has a positive 3D validation in this experiment.
