# Summary slides — exp: online contact novelty penalties for fold search

<!-- Feeds plots/summary.pdf via build_summary.py. -->

## Question

Can early contact novelty penalties cause the exp277 model to sample alternate fold-specific maps more often? We compare 100-rollout contact pools and reference-blind 16-map shortlists on fold-switching proteins.

## Decoder

At each contact, beam search jointly scores two position tokens. The candidate score is joint log probability minus epsilon times the number of earlier completed rollouts containing that contact. Rollouts are processed in waves of 10. The early-only variant linearly reduces epsilon to zero over the first 20 contacts, preserving later shared contacts.

## Development pilot

Four configurations were predeclared on seven development pairs. An exploratory width-16 early-only epsilon 0.2 setting was added after inspecting partial development scores. With 100 rollouts per pair, iid100 covered both contact modes on 3/7 pairs in the oracle pool and 2/7 in the blind 16-map shortlist. The strongest novelty setting reached 2/7 and 1/7; the other width-16 variants reached 1/7 and 1/7. Width 32 reached 2/7 but zero blind hits at three times the strongest variant's H100 cost.

## Frozen test choice

Width 16 with epsilon 0.2 decaying linearly to zero over the first 20 contacts was frozen at 15:43 UTC, before reading any primary held-out references. It tied width 32 on development oracle coverage, retained one blind hit rather than zero, and cost one-third as much.

## Primary held-out result

On 29 primary held-out proteins, iid100 found both reference contact modes on 6/29 in the full 100-map pool and 2/29 in the blind 16-map shortlist. Width-4 beam found 2/29 and 0/29. The frozen novelty method found 1/29 and 0/29, with no 50%-recall dual hit. It cost 27.3 times as much H100 inference as iid100 and 4.35 times as much as width 4.

## Interpretation

The penalty increased the fraction of previously unseen early contacts in development from 11.9% to 15.7%, but this did not improve held-out fold coverage. Relative to iid100, the novelty method gained one oracle hit and lost six; the one gain was already found by width 4 and missed by the blind shortlist. These are contact-level hits, not validated 3D alternate folds. The 29-pair sample leaves uncertainty about the exact effect size, but gives no evidence that this strategy is useful at a 100-rollout budget.
