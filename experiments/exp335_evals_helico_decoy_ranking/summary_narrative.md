# Summary slides — exp: Helico confidence for Rosetta decoy ranking

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide.
     Keep this current as the experiment progresses. -->

## What we're doing

Can Helico's own confidence rank native and near-native protein structures above Rosetta decoys when each candidate is converted to Helico's contact-map representation, and how does it compare with AF2Rank, DeepAccNet, and the Rosetta energy function on the exact AF2Rank benchmark?

## Why

A candidate-derived contact map that is geometrically compatible with the target sequence should let contact-conditioned Helico produce a confident structure, whereas inconsistent decoy contacts should reduce Helico pTM. We therefore expect target-wise Helico pTM to correlate positively with candidate TM-score and recover high-quality candidates substantially better than Rosetta energy, but probably below AF2Rank because the contact map discards most of the template geometry available to AlphaFold. The native-versus-decoy discrimination question and the standard decoy-quality ranking question will both be reported.

## Results so far

The exact public benchmark is available: 133 targets, 180,079 decoys, 133
natives, and corrected AF2Rank outputs all join without a missing key. Our
analysis reproduces the paper's full-set AF2Rank / DeepAccNet / Rosetta rank
correlations (0.925 / 0.831 / 0.759).

## Target-balanced pilot

We ran nine targets x (24 TM-stratified decoys + native), three Helico diffusion
samples each. All 225 candidates succeeded. Primary Helico pTM reaches 0.852
mean target-wise Spearman correlation with candidate TM-score, versus 0.944 for
AF2Rank, 0.871 for DeepAccNet, and 0.820 for Rosetta on the paired pilot.

The secondary Helico composite (pTM times candidate/output TM-score) reaches
0.889 correlation and the best pilot top-1 TM-score, 0.958. It uses more than
confidence and is reported separately.

## Native versus decoy

Helico pTM ranks the exact native first on 3/9 targets and gives a mean native
rank of 3.22 among 25 candidates. AF2Rank's composite is top-1 on 5/9; the
Helico composite is top-1 on 7/9. Pure confidence is useful but does not by
itself reliably identify the exact native.

## Scale gate

The pilot used 0.544 inference H100-hours. One-candidate-at-a-time extrapolation
is about 436 H100-hours / $1,722 for all 180,212 candidates. Contact-map
deduplication will not help (224/225 pilot maps were unique), so the next step is
same-target batching to amortize trunk work before any full launch.
