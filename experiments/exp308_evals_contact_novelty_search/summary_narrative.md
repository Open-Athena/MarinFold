# Summary slides — exp: online contact novelty penalties for fold search

<!-- Feeds plots/summary.pdf via build_summary.py. -->

## Question

Can early contact novelty penalties cause the exp277 model to sample alternate fold-specific maps more often? We compare 100-rollout contact pools and reference-blind 16-map shortlists on fold-switching proteins.

## Decoder

At each contact, beam search jointly scores two position tokens. The candidate score is joint log probability minus epsilon times the number of earlier completed rollouts containing that contact. Rollouts are processed in waves of 10. The early-only variant linearly reduces epsilon to zero over the first 20 contacts, preserving later shared contacts.

## Current state

Four configurations are planned on seven development pairs: width 16 with no penalty, width 16 with constant epsilon 0.05, width 16 with an early-only epsilon 0.05, and width 32 with the same early-only schedule. One method will be frozen before scoring 29 primary held-out pairs. Results are pending.
