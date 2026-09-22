# Summary slides — exp: online contact novelty penalties for fold search

<!-- Feeds plots/summary.pdf via build_summary.py. -->

## Question

Can early contact novelty penalties cause the exp277 model to sample alternate fold-specific maps more often? We compare 100-rollout contact pools and reference-blind 16-map shortlists on fold-switching proteins.

## Decoder

At each contact, beam search jointly scores two position tokens. The candidate score is joint log probability minus epsilon times the number of earlier completed rollouts containing that contact. Rollouts are processed in waves of 10. The early-only variant linearly reduces epsilon to zero over the first 20 contacts, preserving later shared contacts.

## Development pilot

Four configurations were predeclared on seven development pairs. An exploratory width-16 early-only epsilon 0.2 setting was added after inspecting partial development scores. With 100 rollouts per pair, iid100 covered both contact modes on 3/7 pairs in the oracle pool and 2/7 in the blind 16-map shortlist. The strongest novelty setting reached 2/7 and 1/7; the other width-16 variants reached 1/7 and 1/7. Width 32 reached 2/7 but zero blind hits at three times the strongest variant's H100 cost.

## Frozen test choice

Width 16 with epsilon 0.2 decaying linearly to zero over the first 20 contacts was frozen at 15:43 UTC, before reading any primary held-out references. It tied width 32 on development oracle coverage, retained one blind hit rather than zero, and cost one-third as much. It will now be tested on 29 primary held-out pairs. Contact-level hits do not prove 3D fold recovery.
