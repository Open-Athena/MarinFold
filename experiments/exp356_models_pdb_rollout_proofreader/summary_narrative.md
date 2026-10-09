# Summary slides — exp: bidirectional proofreading of MarinFold rollouts on experimental PDB structures

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide.
     Keep this current as the experiment progresses. -->

## What we're doing

Can an exp277-initialized bidirectional verifier identify incorrect contacts in actual MarinFold rollouts, using all supplied contacts, from one-contact prefixes through complete rollouts?

## Why

Fine-tuning a pretrained contact-language backbone on experimentally determined structures will teach contextual error detection and calibrated contact probabilities. Their mean estimates precision; a pooled readout estimates recall.

## Results so far

CoreWeave generation smoke: 48 complete experimental chain structures, eight actual exp277 rollouts each. All 384 completed; 381 passed strict parsing. No empty rollouts. Twenty-four emitted pairs were invalid contacts and remain negative examples.

Behavioral tests pass: later evidence affects earlier judgments, padding is invisible, prefixes exclude future tokens, duplicate contacts do not inflate metrics, and saved heads/tokenizers round-trip.

The one-H100 smoke resumed from a durable step-6 checkpoint, reproduced validation metrics exactly, and trained through step 12. Eight-H100 execution and larger generation batches are being checked. No trained-model quality claim yet.

## Production data and validation

Frozen: 50,000 training structures from 46,933 PDB entries, containing 34,894 distinct supplied sequences in 9,074 related groups. Validation: 1,552 structures / 287 groups. Test: 1,473 structures / 295 groups.

Every supplied residue is observed: canonical sequences or unique contiguous segments with at most 20 terminal residues omitted per end and at least 90% coverage. Internal gaps and truncated contact references are rejected. Benchmark homologs are excluded, and explicit searches check split boundaries.

Test precision, recall and calibration from one contact through full rollouts. Measure whether later evidence improves the first contact's judgment and whether proofreading improves contact selection.
