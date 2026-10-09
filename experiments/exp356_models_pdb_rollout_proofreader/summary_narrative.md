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

Eight behavioral tests pass: later evidence affects earlier judgments, padding is invisible, prefixes exclude future tokens, duplicate contacts do not inflate metrics, and saved heads/tokenizers round-trip.

Training and checkpoint recovery are being tested. No trained-model quality claim yet.

## Production data and validation

Target: 50,000 experimental chain structures for training, plus separate validation and test families. The dataset is not 50,000 independent sequence families; report both structure and family counts.

Only complete canonical sequences and complete reference contact lists are accepted. Benchmark homologs are excluded. Related structures stay in one split, followed by explicit searches across split boundaries.

Test precision, recall and calibration from one contact through full rollouts. Measure whether later evidence improves the first contact's judgment and whether proofreading improves contact selection.
