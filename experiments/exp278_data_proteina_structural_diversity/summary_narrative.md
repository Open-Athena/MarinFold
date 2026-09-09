# Summary slides — exp: diversify contacts-v1 with Proteina-generated monomers

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide.
     Keep this current as the experiment progresses. -->

## What we're doing

Can fold-conditioned Proteina generation supply one million useful, structurally diverse, sequence-paired monomer documents of length 60–500, at an acceptable cost per retained structure?

## Why

Length-stratified, fold-balanced generation followed by sequence design, refolding and structural selection will increase structural coverage relative to unconditional generation. Whether that coverage improves MarinFold must be tested with a fixed-token training comparison; generating more documents alone does not establish value.

## Results so far

Issue #278 filed and execution authorized.
One-H100 smoke test on cw-rno2a is validating the environment.
Five geometry checks pass locally.
Four 100-aa backbones generated at 1.803 seconds each (batch 4, uncompiled).
Two passed self-consistency and confidence checks and serialized to contacts-v1.
ESMFold normalized-confidence units were corrected before filtering.
Next: end-to-end smoke, then a pilot capped at 100 H100-hours.
