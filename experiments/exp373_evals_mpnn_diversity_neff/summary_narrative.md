# Summary slides — exp: ProteinMPNN-only prediction diversity and Neff dependence on eval-val

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide.
     Keep this current as the experiment progresses. -->

## What we're doing

Does the completed ProteinMPNN-only checkpoint produce qualitatively different predictions from the current default, particularly greater rollout diversity on eval-val and less degradation in contact accuracy at low effective MSA depth (Neff)? If it is more diverse, does that diversity capture useful alternatives or mainly additional errors?

## Why

1. **Rollout diversity.** Training without native-sequence documents produces a broader distribution of predicted contact patterns on natural proteins, potentially through reduced reliance on close native-sequence matches. This could improve discovery of correct contacts across repeated samples even when individual predictions are less accurate.
2. **Neff dependence.** The ProteinMPNN-only checkpoint has a flatter performance curve across Neff tiers on eval-val: its accuracy drops less from high-Neff to low-Neff proteins than the current default's accuracy does. This is a hypothesis about relative sensitivity to MSA depth; whether it also achieves higher absolute accuracy on low-Neff proteins is a separate question to report explicitly.

## Results so far

_(Fill in as results come in.)_
