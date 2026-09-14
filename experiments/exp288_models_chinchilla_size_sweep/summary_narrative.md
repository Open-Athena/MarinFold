# Summary slides — Experiment: Chinchilla size sweep on cleaned contacts-v1 corpora

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide.
     Keep this current as the experiment progresses. -->

## What we're doing

For the cleaned native + MPNN-redesigned contacts-v1 corpora from #266/#274, what model size is compute/data efficient? In particular, do 0.7B, 1.5B, and 3B Qwen-style contacts-v1 models show the expected Chinchilla-style tradeoff between parameter count, token budget, validation cross-entropy, and downstream contact R-precision?

## Why

The new ~248B-token corpus is large enough that the current 1.5B recipe may be under-sized at fixed token budget. A 3B model should improve validation CE and possibly FoldBench R-precision over 1.5B if the extra MPNN/native data carries independent signal; a 0.7B model provides a cheap lower anchor for fitting a scaling trend rather than only comparing two large runs.

## Planned Chinchilla measurements

Three fixed-recipe arms: 20×1536 (~0.701B), 24×2048 (~1.471B), and 32×2560 (~3.035B). One packed epoch is ~279.3B consumed tokens, giving ~398, ~190, and ~92 tokens/parameter.

For each arm we will track validation CE vs tokens, validation CE vs GPU-hours, final token/parameter ratio, throughput, and selected exp245 eval-val contact metrics.

The goal is to collect scaling evidence for the next MarinFold recipe. A smaller arm anchors the slope; the current 1.5B scale preserves continuity; the 3B arm tests whether the cleaned/redesigned corpus makes 1.5B under-sized.

## Results so far

Training harness added. The four CoreWeave token caches were verified by `/zack/exp288-prepare-a01`, so the sweep does not require a new raw-data transfer. H100s were full, so the active target is the recent successful GB200/B200 profile on `cw-us-east-08a`: 8 nodes × 4 GB200 GPUs per trial.
