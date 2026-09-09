## Proteina structural-diversity pilot (exp278)

Question: can synthetic monomer backbones provide diverse, usable contacts-v1 training documents at million-document scale?

Issue #278 is approved through a capped pilot and conditional scale-up milestones. Draft PR #282 contains the implementation and small result artifacts. No generated documents have entered training.

## Pipeline and current measurements

Proteina CA traces → CA ProteinMPNN sequences → ESMFold full backbone → geometry/self-consistency checks → native-only pyconfind → contacts-v1.

At 60 aa: 230/256 pass quality; 152/256 remain after evaluation decontamination and a five-per-fine-cluster cap.
At 100 aa: 205/256 pass quality; 151/256 remain after the same filters.

At 200 aa: 164/256 pass the corrected quality gate. A trans-only CA-distance bug was fixed to recognize explicitly validated cis-proline bonds; original predictions remain immutable.

Longer lengths are still running. The pilot cap is 100 H100-hours.

## Diversity is a separate result

Broad CATH conditioning produces the expected alpha/beta composition. Fine-cluster effective counts have not improved over matched unconditional controls in the short screens (~0.92× at 60 aa; 1.00× at 100 aa after decontamination).

These samples do not meet the 1.5× research target. The 100-aa control is almost all singletons at this small sample size. Broader connected-component metrics are exploratory and are not equivalent to CATH fold counts.

## Cost and numerical validation

500-aa sampling at batch 24: 10.16 seconds/backbone with reference float32 matmuls; 5.00 seconds with TF32 enabled. Both use 62.45 GB of an H100.

Matched-seed structures can change (median RMSD 0.125 Å, max 7.06 Å); the faster setting is undergoing a separate refolding check. ESMFold currently averages about 0.64 seconds at 60–100 aa.

Production cost and scale-up remain conditional on the completed length-stratified screen, acceptance yield and duplicate growth.
