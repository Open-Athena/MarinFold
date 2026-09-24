# Summary slides — exp: can blind inference search discover both folds of fold-switching proteins?

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide.
     Keep this current as the experiment progresses. -->

## Question and design

Can sequence-only inference return both observed folds of a fold-switching protein? We ran the exp277 contacts-v1 checkpoint on 67 non-capped pairs. The primary test set has 29 pairs with at least 98% identity, held apart by sequence group from 15 development pairs. Seventeen of those 29 are literally identical. Every method returned at most 16 ranked maps chosen without reference contacts or structures.

## How the 29-pair test set was selected

The source was 93 literature-curated pairs in the NCBI AF2 benchmark Table S1. The reference-defined funnel retained 68 premise-valid pairs, 65 with at least 98% sequence identity, 45 with at least 10 switching-region contacts unique to each fold, and 44 after excluding one context-capped pair. Sequence grouping followed by deterministic SHA-256 ordering assigned 15 pairs to development and 29 to the held-out test set. No model or Helico outcome entered selection. Of the final 29, 17 are exact-sequence pairs and 12 differ by 1–5 substitutions.

## The primary comparison is negative

At 100 shared root plus 100 arm rollouts per protein, 10-contact clustered branching improved mean minority-fold enrichment by only 0.013 over independent sampling. The paired 95% interval is -0.014 to +0.043; mean minority recall was essentially unchanged. Blind dual-contact hits were 3/29 versus 2/29. The extra hit failed a 3D check. On the 17 exact-sequence test pairs, both methods had one contact hit and the paired interval still spans zero. We cannot claim a reliable search improvement.

## What was actually discovered

For 3j7w/3j7v, both independent and branching top-16 shortlists included maps that reconstructed distinct switching-region structures. With the Fold1 sequence input, region-touch lDDT against (Fold1, Fold2) was (0.477, 0.206) and (0.356, 0.702) for the branching shortlist; the independent shortlist also favored the respective folds. True-contact controls showed comparable separation. This is an existence proof for blind inference discovery, not a branching advantage.

## Where the search is limited

Four of seven branch-10 oracle dual-contact hits were present among 200 candidates but missed by the blind 16-map selector. In a separate plain-sampling run, blind dual hits were 2, 3, and 3 at 100, 200, and 500 rollouts; the oracle pool reached 9 by 200. The first priority is better reference-free ranking of coherent minority maps. The all-rollout structural analysis below also shows that the contact-level proxy admits many candidates that do not reconstruct near the nominal fold.

## What the 500-rollout oracle means

The oracle inspects all 500 independent maps against both reference contact maps. A dual hit needs one Fold1-like and another Fold2-like map; it does not mean the blind selector found either. Among 29 held-out primary pairs, 19 had any Fold1-like map, 13 any Fold2-like map, and 9 had both. The same counts for the 17 exact-sequence test pairs were 12, 7, and 5. Across all 67 non-capped pairs, 31 had both. The prespecified 25% contact-recall screen is permissive: raising recall to 50% reduces held-out dual coverage to 2/29. The later all-rollout structural run confirms that contact-screen coverage is not a count of recovered three-dimensional folds.

## Do another 500 iid draws help?

Yes, modestly for oracle contact-pool coverage. On the same 29 primary test proteins, the count with both fold-like contact modes rises from 9 at 500 draws to 11 at 750 and stays 11 at 1,000. Neither-mode proteins fall from 6 to 5. The two new dual hits first appear at draws 509 and 587, and both are exact-sequence pairs. At the stricter 50% recall cutoff, the dual count stays 2/29. These are reference-aware screening hits in the full pool. Folding every map separately finds no new strict structural dual after draw 249.

## Every iid map was folded separately

We ran 29,000 individual iid contact maps plus 87 true-Fold1, true-Fold2, and no-contact controls through Helico: one sample, six cycles, seed 42, no MSA. All 29,087 predictions and PDBs completed. Both true-contact controls pass for 17/29 proteins. Under the exploratory strict screen—global Kabsch GDT-TS at least max(0.35, 90% of the corresponding control) plus switching-region advantage at least 0.10—the number with both modes is 0 at 100 draws, 1 at 200, and 2 from 500 through 1,000. Both dual cases are in the 17-protein exact-sequence subset.

## The structural positives are not interchangeable

3j7w/3j7v is the cleanest contact-driven example: both controls pass, the no-contact baseline matches neither, and both modes appear by draw 144. 2lel/2k0q passes the strict control-relative screen by draw 249, but its no-contact baseline is already Fold2-like; search contributes the Fold1-like mode. 1qs8/1miq is the strongest near-threshold case: its candidates reach global GDT-TS 0.855 and 0.758, but the latter is 84.7% of its unusually strong control and misses the 90% rule.

## Binary structural coverage is threshold-sensitive

Using 80% rather than 90% of the true-contact control gives 3/29 dual proteins by adding 1qs8/1miq; requiring 100% gives 1/29. Fixed absolute GDT-TS cutoffs of 0.35, 0.50, and 0.60 with the same 0.10 region margin give 8, 3, and 1. The raw scores and sensitivity grids are the result to trust. Contact and structural preference correlate (Spearman rho 0.705), but the contact screen reports 11/29 duals while the strict structural screen reports 2/29. Missing modes are inconclusive for the 12 proteins that fail at least one true-contact control.
