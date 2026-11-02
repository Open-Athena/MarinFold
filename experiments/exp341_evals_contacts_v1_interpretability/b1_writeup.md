# exp341 B1: where exp277 computes contacts

B1 of [\#341](https://github.com/Open-Athena/MarinFold/issues/341) asks
where in exp277's 24 layers the contact prediction is computed, to inform
decision B in [`research_plan.md`](research_plan.md) (section 1.2): whether
the next size sweep should favour depth or width. Two measurements ran:
mean-ablating each layer's attention and MLP output, and decoding the
contact readout from every layer boundary with a logit lens and a tuned
lens. Both ran first on the 116 eval proteins, then on an extension of up
to 878 proteins, 516 of them 301 to 996 residues long. The pairwise probe
in the sequence section (plan section 6.1) did not run.

**Recommendation: do not make the next model shallower or move parameters
from depth into width. If a size sweep runs, run its deeper arm first,
and judge it by length bin, with most weight on proteins over 500
residues.** Both halves of the plan's "favour depth" row hold. 24 layers
are enough for short proteins and not obviously enough for long ones: on
proteins up to 150 residues the readout at boundary 19 is within 0.006 of
the model, while on proteins of 501 to 1000 residues it is 0.050 short,
and the last layer alone still adds 0.039. Neither measurement shows that
a deeper model would predict contacts better; only the sweep can show that.

## Implications for model shape and size

**Shallower is ruled out; deeper is supported for long proteins; width is
untested.**

1. **Removing late layers would cost most on long proteins.** Every MLP in
   layers 18 to 23 costs 0.047 to 0.530 R-precision when ablated (512
   proteins), and the cost grows with length. The layer 23 MLP costs 0.058
   on long-range contacts of proteins up to 150 residues and 0.176 on
   proteins of 501 to 1000; the layer 22 MLP 0.260 and 0.500. A shallower
   model at the same parameter count, or one that drops the last quarter,
   would be cut where long proteins depend on it most. (Ablation removes a
   layer from a model trained with it; a shallower model trained from
   scratch would divide the work differently. The ablation still shows the
   current model does not have spare late layers.)
2. **The readout saturates early on short proteins and not on long ones.**
   Logit-lens gap to the model, all pairs, 878 proteins:

   | Length | n | Boundary 18 | Boundary 19 | Boundary 23 |
   |---|---|---|---|---|
   | 1-150 | 162 | -0.041 | -0.006 | -0.002 |
   | 151-300 | 147 | -0.032 | -0.010 | -0.013 |
   | 301-500 | 291 | -0.045 | -0.024 | -0.022 |
   | 501-1000 | 278 | -0.066 | -0.050 | -0.039 |

   For short proteins the last five layers add almost nothing to contact
   ranking; for long ones the readout is still 0.050 short at boundary 19
   and 0.039 short at boundary 23. A model that
   ran out of depth on long proteins would look like this. So would one
   that spends its last layers sharpening the partner distribution, which
   matters more when there are 800 candidate partners than 100; a deeper
   model would then gain little. B1 cannot separate the two. The sweep
   can: a deeper arm should gain little under 150 residues and most above
   500, and if it gains nothing above 500, depth is not what limits long
   proteins.
3. **Width: no evidence either way.** B1 removes and decodes layers; it
   does not vary their width. The most lightly used parts are attention in
   layers 21 and 22 (0.010 and 0.002 when ablated), the layer 17 MLP
   (0.011) and layers 11 to 13 (under 0.02 each on the eval set). Late
   computation is MLP-heavy: in layers 21 to 23 the MLPs cost 0.111 to
   0.530 against 0.002 to 0.051 for attention. That suggests where
   capacity is idle, not that it can be removed, since single-layer
   ablation misses work two layers share.
4. **Size: B1 does not measure it.** A layer profile of one 1.5B model
   says how that model uses its depth, not whether more parameters help;
   the scaling evidence is \#262 (contacts-v1 scales worst on long
   proteins, L^0.79). The two point the same way: if more compute goes in,
   put it into depth and measure the return on long proteins.

**What the sweep needs.** At matched parameters and training tokens, the
current 24-layer shape against deeper, narrower arms, scored by length bin
(1-150, 151-300, 301-500, over 500). The eval sets lack the long proteins
to score that last bin: eval-val and eval-denovo have 116 proteins up to
761 residues, the legacy extras add almost none over 460, and the 516 long
panel structures here use AFDB-predicted contacts and have sequence
relatives in training. A held-out set of experimental structures over 500
residues would make the decisive bin measurable. The cheapest test before
any training is the same ablation and lens on an earlier exp277 checkpoint
(plan section 7.1): if the long-protein gain in the last layers is still
growing at the end of training, the gap reflects under-training, not a
depth limit.

## Decision B

The plan numbers layers 1 to 24, so its "by layer 18" is boundary 18 here
(the residual after 18 layers) and its "layers 19 to 24" are the change
from boundary 18 to boundary 24 (the model):

| Plan's condition | "Favour depth" row | "Favour width or shallower" row | Observed |
|---|---|---|---|
| Mean-ablating one late layer (19 to 24) costs | >= 0.02 R-precision | < 0.01 | Eval set: every late MLP costs 0.054 to 0.458, late attention 0.001 to 0.281. 512 proteins: MLPs 0.047 to 0.530, attention 0.002 to 0.317 |
| Lens readout from boundary 18 to 24 | still rising by >= 0.03 | within 0.01 of final by boundary 18 | Logit lens: eval set 0.042 (CI 0.037 to 0.047); 878 proteins 0.049 (0.047 to 0.051), 0.032 to 0.066 by length bin |

The plan proposed these thresholds before the run and treats them as
provisional. The ablation half does not depend on them (late MLP costs are
5 to 53 times the 0.01 boundary). The lens half does: at 0.05 instead of
0.03, the eval set and the 878-protein average would fall short, and only
proteins over 500 residues would pass. The plan names the tuned lens for
this row; this report reads the logit lens instead, because the tuned lens
decodes less than the logit lens on the eval set and fails outside its fit
lengths (section "Extension").

What the two measurements show together:

- **Contacts form in the second half of the network.** Neither lens decodes
  any contact signal through boundary 11 (all-pairs R-precision 0.004 to
  0.028, against 0.523 for the model). The logit lens then rises from 0.059
  at boundary 12 to 0.390 at 15 and 0.506 at 19. Layers 0 and 1 and the
  attention in layer 2 are still essential (ablating them costs 0.41 to
  0.51), so the first half builds
  something the contact readout needs, but not in a form the unembedding
  can read at the query.
- **The last five layers add little on short proteins and more on long
  ones.** On the eval set the logit lens at boundary 19 is 0.017 below the
  model (CI 0.012 to 0.021), then dips to 0.412 at boundary 21 and
  recovers by 24. Layers 19 to 22 are among the costliest to ablate, so
  they change the representation heavily for a small net change in contact
  ranking. On 878 proteins the boundary-19 gap grows from 0.006 (1-150
  residues) to 0.050 (501-1000), and within the panel structures alone
  from 0.021 (301-400) to 0.062 (701-1000).
- **A costly late layer shows the model uses its depth, not that it lacks
  depth.** A model with exactly enough layers and one that needs more would
  both show costly late layers; ablating layer 0 or 1 costs as much as
  layer 21, and that does not argue for more early layers. The evidence
  for depth is the length dependence above, not the size of the costs.
- **The tuned lens did not beat the logit lens.** The plan asks for this
  comparison to confirm the tuned lens adds information. On the eval set it
  decodes less than the logit lens at boundaries 13 and 14 (0.098 and
  0.155 against 0.205 and 0.324) and at 19 to 23, despite explaining 90 to
  95% of the held-out final residual's variance at boundaries 18 to 23.
  Its residual errors fall in the directions that rank contacts. On
  proteins longer than its fit set it degrades further (section
  "Extension"). The logit lens is the better reading of this model.

## What ran

| Item | Value |
|---|---|
| Model | `contacts-v1-exp277-m2-p06-full-epoch-1.5B` (step 266344), bf16 |
| Proteins | 116: eval-val (97) and eval-denovo (19) |
| Readout | production pairwise P(contact), 4 document seeds per protein |
| Ablation | clean, plus 24 layers x {attention, MLP} each mean-ablated alone |
| Lens | logit and tuned lens at all 25 layer boundaries; tuned maps fitted on 200 legacy-554 proteins |
| Compute | one NVIDIA L4 (g6.xlarge, us-east-2): ablation 4.25 GPU-hours (about 320 s per condition); lens 5.5 min fit pass plus 6 min eval pass |
| Runs | ablation `s3://marinfold-exp91-usw2/exp341/b1/20261017T142617Z-9a5b6aa0/` (commit 9a5b6aa0); lens `s3://marinfold-exp91-usw2/exp341/lens/20261017T233643Z-6d7d0cee/` (commit 6d7d0cee) |
| Extension proteins | 246 legacy-554 chains not in the eval sets (`data/b1_extra_gt.jsonl.gz`; 224 denovo_pdb, 14 cameo_hard, 8 casp_fm; up to 460 residues), and the A4 panel's removed structures over 300 residues (516, 301 to 996 residues; ablation on a seeded sample of 150) |
| Extension ablation | clean plus layers 14 to 23 x {attention, MLP}, with the eval-set means (`data/b1_sublayer_means.npz`); 10.2 GPU-hours on 8 L4s; runs `b1/20261018T003928Z-9095e0fc-extra/` and `b1/20261018T00{3957,4016,4047,4105,4138,4155,4223}Z-9095e0fc-panel-s{0..6}/` (commit 9095e0fc) |
| Extension lens | tuned maps refitted on 194 legacy-554 chains (the 200 minus 6 that share a structure with eval-denovo), scored on eval plus legacy extras (`lens/20261018T002337Z-1079813a-ext/`) and the 516 panel structures in 3 shards (`lens/20261018T00{3656,3719,3745}Z-1079813a-panel-s{0,1,2}/`); 3.3 GPU-hours |

Mean ablation replaces the sublayer's output, before it is added to the
residual stream, with one fixed vector at every position. Each vector is
the mean of that sublayer's output over all 710,640 positions the clean
readout processes (sequence-section and contact-query positions, across all
proteins and seeds). The ablated runs reuse the clean run's proteins and
document seeds, so each comparison is paired.

R-precision is scored on the seed-averaged matrix, as in production, for
all pairs (separation >= 6) and three bands: short (6-11), medium (12-23)
and long (>= 24). The clean model scores 0.522 on all pairs (0.566 short,
0.537 medium, 0.498 long). An ablation that removes all contact signal
lands near -0.52.

## Layer ablation

![B1 layer ablation](plots/b1_layer_ablation.png)

Change in all-pairs R-precision (ablated minus clean), 116 proteins, 95%
bootstrap CI over proteins:

| Layer | Attention | MLP |
|---|---|---|
| 0 | -0.511 | -0.490 |
| 1 | -0.509 | -0.507 |
| 2 | -0.405 | -0.081 |
| 3 | -0.033 | -0.029 |
| 4 | -0.092 | -0.170 |
| 5 | -0.097 | -0.140 |
| 6 | -0.113 | -0.165 |
| 7 | -0.032 | -0.198 |
| 8 | -0.016 | -0.072 |
| 9 | -0.021 | -0.073 |
| 10 | -0.047 | -0.029 |
| 11 | -0.010 | -0.019 |
| 12 | -0.012 | -0.006 |
| 13 | -0.005 | -0.016 |
| 14 | -0.059 | -0.390 |
| 15 | -0.197 | -0.089 |
| 16 | -0.027 | -0.057 |
| 17 | -0.032 | -0.014 |
| 18 | -0.253 (-0.270, -0.236) | -0.325 (-0.350, -0.300) |
| 19 | -0.281 (-0.301, -0.261) | -0.054 (-0.062, -0.046) |
| 20 | -0.278 (-0.302, -0.254) | -0.337 (-0.362, -0.311) |
| 21 | -0.010 (-0.013, -0.007) | -0.458 (-0.490, -0.427) |
| 22 | -0.001 (-0.003, +0.002) | -0.372 (-0.404, -0.338) |
| 23 | -0.049 (-0.054, -0.043) | -0.104 (-0.114, -0.093) |

Layers 0 and 1 (both sublayers) and the layer 21 MLP leave almost no
contact signal. The costly sublayers fall in three groups: layers 0 to 2,
layers 14 and 15, and layers 18 to 22, with layers 11 to 13 the cheapest
stretch between them.

**Statistics.** 47 of the 48 conditions have a CI that excludes zero; the
exception is layer 22 attention. The median CI half-width is 0.010 when
bootstrapping proteins and 0.014 when also resampling the 4 document seeds
within each protein. The second CI is centred on the mean per-seed change,
which differs from the seed-averaged change by up to 0.09 (layer 15 MLP:
-0.182 per seed, -0.089 averaged), because R-precision of an averaged
matrix is not the average of per-seed R-precisions. Effects this large do
not need repeat runs; a run with `--seed-offset 4` would tighten the
seed-level CI if small effects become the question.

**Separation bands.** The four bands give similar changes for most
sublayers. The late MLPs are the exception: the layer 22 MLP costs 0.349 on
long-range pairs and 0.217 on short-range pairs, and the layer 21 MLP 0.432
against 0.351.

**Length.** Long-range change by length tercile (39, 38 and 39 proteins):

| Sublayer | 30-173 residues | 177-290 | 297-761 |
|---|---|---|---|
| Layer 18 MLP | -0.267 | -0.324 | -0.363 |
| Layer 20 MLP | -0.278 | -0.334 | -0.383 |
| Layer 21 MLP | -0.389 | -0.420 | -0.487 |
| Layer 22 MLP | -0.251 | -0.365 | -0.431 |
| Layer 23 MLP | -0.066 | -0.083 | -0.139 |
| Layer 19 attention | -0.224 | -0.273 | -0.324 |
| Layer 20 attention | -0.199 | -0.271 | -0.338 |

Clean long-range R-precision is similar across terciles (0.529, 0.452 and
0.512), so the growth is not an artefact of a higher starting
point on long proteins. Early layers show a weaker version of the same
trend (layer 6 MLP: -0.129, -0.173, -0.231).

**Eval set.** eval-denovo (clean 0.616) loses more than eval-val (clean
0.504) in absolute terms under every costly ablation (layer 21 MLP: -0.521
against -0.446), roughly in proportion to its higher starting score.

## Lens

![B1 lens](plots/b1_lens.png)

The readout uses two next-token distributions over the protein's position
tokens: one at the `<contact>` query and one at each `<contact> <p_i>`
query. The lens takes the residual at those query positions at boundary k
(0 = embeddings, k = after layer k - 1) and decodes it:

- **Logit lens**: the model's final norm and unembedding applied directly.
- **Tuned lens**: one affine map per boundary first, fitted by ridge
  regression from the RMS-normalized boundary-k residual to the
  RMS-normalized final residual. The maps were fitted on 140,832 query
  residuals from the 200 legacy-554 proteins in `data/a2_legacy_gt.jsonl.gz`
  (4 seeds; 6 of them are chains of eval-denovo structures, which the
  extension's refit removes, with no effect on the curve), with each boundary's ridge
  strength chosen on 20% of those proteins held out.

Both decodes run in float32. At boundary 24 both are the model; their
seed-averaged R-precision matches the production readout within 0.018 on
every protein (mean 0.5229 against 0.5223), the residual difference coming
from the production readout's bf16 logits.

All-pairs R-precision decoded at each boundary from 12 on (boundaries 0 to
11 are 0.004 to 0.028 under both lenses), with the gap to the model and its
paired bootstrap CI:

| Boundary | Logit lens | Gap (CI) | Tuned lens | Gap (CI) |
|---|---|---|---|---|
| 12 | 0.059 | -0.464 (-0.495, -0.432) | 0.058 | -0.465 (-0.497, -0.432) |
| 13 | 0.205 | -0.318 (-0.339, -0.296) | 0.098 | -0.424 (-0.454, -0.394) |
| 14 | 0.324 | -0.199 (-0.212, -0.185) | 0.155 | -0.368 (-0.395, -0.341) |
| 15 | 0.390 | -0.133 (-0.142, -0.123) | 0.351 | -0.172 (-0.187, -0.157) |
| 16 | 0.405 | -0.117 (-0.128, -0.107) | 0.415 | -0.108 (-0.119, -0.097) |
| 17 | 0.445 | -0.077 (-0.085, -0.070) | 0.440 | -0.083 (-0.094, -0.072) |
| 18 | 0.481 | -0.042 (-0.047, -0.037) | 0.459 | -0.064 (-0.076, -0.053) |
| 19 | 0.506 | -0.017 (-0.021, -0.012) | 0.469 | -0.053 (-0.063, -0.044) |
| 20 | 0.470 | -0.053 (-0.060, -0.046) | 0.456 | -0.067 (-0.077, -0.057) |
| 21 | 0.412 | -0.111 (-0.126, -0.097) | 0.450 | -0.073 (-0.085, -0.062) |
| 22 | 0.454 | -0.069 (-0.081, -0.057) | 0.472 | -0.051 (-0.060, -0.043) |
| 23 | 0.508 | -0.015 (-0.020, -0.011) | 0.489 | -0.034 (-0.043, -0.026) |
| 24 | 0.523 | 0 | 0.523 | 0 |

**Length.** Long-range gap to the model by length tercile:

| Lens, boundary | 30-173 residues | 177-290 | 297-761 |
|---|---|---|---|
| Logit, 18 | -0.038 | -0.044 | -0.039 |
| Logit, 19 | +0.003 | -0.014 | -0.025 |
| Logit, 21 | -0.081 | -0.130 | -0.110 |
| Tuned, 18 | -0.013 | -0.048 | -0.113 |
| Tuned, 19 | -0.013 | -0.044 | -0.093 |

Under the logit lens, short proteins' long-range contacts are fully
decodable by boundary 19, and the longest tercile's are 0.025 short of the
model. The tuned lens shows the same order with larger gaps.

**Eval set.** eval-denovo is ahead of eval-val at every boundary (logit
lens at boundary 19: 0.603 against 0.487; model 0.615 against 0.505), with
similar gaps to the model.

**Tuned-lens fit.** Held-out R² of the tuned map is 0.16 to 0.40 over
boundaries 0 to 11, 0.57 at 12, and 0.90 to 0.95 at 18 to 23. The
logit lens's R² against the same target is negative at every boundary (-1.43
at 0, -0.18 at 23): the raw intermediate residual is far from the final one
in Euclidean terms even where it already ranks contacts well. Chosen ridge
strengths range from 1e-4 to 1 times the mean feature variance.

## Extension: 878 proteins, up to 996 residues

The eval set has 116 proteins, few of them long (the longest tercile is 297
to 761 residues), and two of the original results had CIs near a decision
boundary: the logit-lens rise of 0.042 against the 0.03 threshold, and the
length trend in the late layers. The extension adds 246 legacy-554 chains
(mostly de novo designs under 300 residues) and 516 structures over 300
residues that the A4 panel removed and exp277 never trained on. The panel's
contacts come from AFDB-predicted structures, and its proteins have
sequence relatives in training, so its absolute scores (model 0.579, all
pairs) are not comparable with eval-val's 0.505. The gaps and ablation
costs compare the model with itself on the same proteins and documents.

**Lens.** The refitted tuned maps reproduce the eval-set curve within
0.001, so dropping the 6 shared chains from the fit changed nothing.
Logit-lens gap at boundary 18 (the plan's "still rising" quantity, sign
flipped) by protein set, all pairs:

| Set | n | Model | Gap at 18 (CI) | Gap at 19 | Gap at 23 |
|---|---|---|---|---|---|
| eval-val | 97 | 0.505 | -0.041 (-0.046, -0.036) | -0.018 | -0.014 |
| eval-denovo | 19 | 0.615 | -0.047 (-0.062, -0.033) | -0.011 | -0.020 |
| legacy denovo_pdb | 224 | 0.630 | -0.034 (-0.040, -0.029) | -0.008 | -0.007 |
| panel, 301-996 | 516 | 0.579 | -0.057 (-0.059, -0.054) | -0.037 | -0.031 |
| all | 878 | 0.579 | -0.049 (-0.051, -0.047) | -0.027 | -0.022 |

The length bins in the implications section mix protein sets (most of
301-1000 is panel). Within the panel alone the gap at boundary 18 is
-0.045, -0.047, -0.062 and -0.072 for 301-400, 401-500, 501-700 and
701-1000 residues; panel proteins over 600 residues gain 0.024 more over
the last six layers than those up to 400 (CI 0.018 to 0.029), and the
Spearman correlation of length with that gain is 0.40. The model's own
score does not fall with length on the panel (0.568 at 301-400, 0.605 at
701-1000), so the larger late gain is not a lower starting point catching
up. The curve has the same shape on every set, including the dip at
boundary 21.

**Tuned lens outside its fit lengths.** The tuned maps were fitted on
chains of at most 404 residues (19 over 300). Their gap at boundary 18 is
-0.020 on proteins up to 150 residues and -0.229 on 501-1000 (logit lens
-0.041 and -0.066). The maps do not transfer to residuals of much longer
documents, so the tuned lens's length trend is mostly its own error, and
it is not used for any conclusion here.

**Ablation, layers 14 to 23.** All-pairs cost over 512 proteins (116 eval,
246 legacy extras, 150 panel), protein-bootstrap CI:

| Layer | Attention | MLP |
|---|---|---|
| 14 | -0.064 (-0.068, -0.060) | -0.428 (-0.443, -0.414) |
| 15 | -0.208 (-0.216, -0.200) | -0.099 (-0.106, -0.093) |
| 16 | -0.023 (-0.026, -0.021) | -0.061 (-0.065, -0.056) |
| 17 | -0.031 (-0.034, -0.028) | -0.011 (-0.013, -0.010) |
| 18 | -0.277 (-0.287, -0.266) | -0.358 (-0.372, -0.345) |
| 19 | -0.311 (-0.323, -0.299) | -0.047 (-0.051, -0.044) |
| 20 | -0.317 (-0.331, -0.303) | -0.388 (-0.402, -0.373) |
| 21 | -0.010 (-0.011, -0.008) | -0.530 (-0.547, -0.512) |
| 22 | -0.002 (-0.004, -0.001) | -0.433 (-0.451, -0.415) |
| 23 | -0.051 (-0.054, -0.048) | -0.111 (-0.117, -0.105) |

The profile matches the eval set's (same costly and cheap sublayers; layer
22 attention's CI now excludes zero at 0.002). Long-range cost by length:

| Sublayer | 1-150 (161) | 151-300 (147) | 301-500 (114) | 501-1000 (88) |
|---|---|---|---|---|
| Layer 18 MLP | -0.272 | -0.350 | -0.376 | -0.411 |
| Layer 20 MLP | -0.288 | -0.363 | -0.412 | -0.450 |
| Layer 21 MLP | -0.427 | -0.494 | -0.515 | -0.549 |
| Layer 22 MLP | -0.260 | -0.414 | -0.466 | -0.500 |
| Layer 23 MLP | -0.058 | -0.094 | -0.124 | -0.176 |
| Layer 23 attention | -0.034 | -0.039 | -0.050 | -0.079 |

Clean long-range R-precision is 0.548, 0.527, 0.535 and 0.561 across those
bins. Within the 150 panel structures, proteins over 600 residues lose more
than those up to 400 under the layer 23 MLP (0.071 more, CI 0.048 to
0.093), layer 23 attention (0.040, 0.028 to 0.052) and layer 21 attention
(0.016, 0.011 to 0.022); for the layer 21 and 22 MLPs the within-panel
difference (0.017 and 0.034) has a CI that spans zero with 60 against 38
proteins.

## Limits

- **The mean mixes two kinds of position.** Each replacement vector
  averages sequence-section tokens with contact-query tokens. At a late
  layer those carry different signals, so a large cost can partly come from
  pushing the model off its usual inputs, not only from removing the
  sublayer's computation. Means taken over contact-query positions only, or
  resample ablation, would separate the two; neither was run.
- **Ablation means come from the eval set.** The extension reuses the
  eval-set sublayer means, so on a long panel protein the replacement
  vector is an average over shorter documents. That can add to the cost on
  long proteins independent of what the sublayer computes; means taken per
  length bin would separate the two, and were not run.
- **Long proteins here are mostly predicted structures.** 516 of the 569
  proteins over 300 residues are AFDB predictions from the A4 panel. The
  length trends hold within that set, but no experimental set of long
  proteins was available to confirm them.
- **One model, one checkpoint.** A layer profile of the final exp277
  checkpoint cannot distinguish converged late layers from late layers that
  are still learning; the plan's optional checkpoint trajectory (section
  7.1) is the test. It also cannot show how a 32- or 48-layer model would
  divide the same work.
- **Single-layer ablation misses shared work.** If two layers compute the
  same thing, removing either costs little. The cheap layers (11 to 13,
  late attention) are candidates for spare capacity, not proof of it.
- **The lens reads only the query positions.** Contact information held
  elsewhere (at the residues' sequence-section tokens) is invisible to it;
  that is what the pairwise probe in plan section 6.1 would measure, and it
  did not run. "No decodable signal through boundary 11" means none at the
  query, not none in the network.
- **The tuned lens is a least-squares fit on 200 short proteins.** It
  explains most of the final residual's variance but decodes contacts worse
  than the logit lens, and fails on proteins longer than its fit set. A lens trained on a KL objective over position tokens, as in
  the original tuned-lens work, might do better; this one was not tried.

## Files

- [`b1_layer_ablation.py`](b1_layer_ablation.py): the driver (per condition,
  per-seed P(contact) and banded R-precision; resumable).
- [`analyze_b1.py`](analyze_b1.py): paired deltas and both bootstrap CIs by
  band, eval set and length tercile.
- [`plot_b1.py`](plot_b1.py): the ablation figure.
- [`b1_tuned_lens.py`](b1_tuned_lens.py): the lens driver (fit, then decode
  every boundary; raw per-seed P(contact); resumable).
- [`analyze_b1_lens.py`](analyze_b1_lens.py), [`plot_b1_lens.py`](plot_b1_lens.py):
  lens curves with paired gaps by band, eval set and length tercile; the lens figure.
- [`data/b1_lens_curve.csv`](data/b1_lens_curve.csv),
  [`data/b1_lens_scores.csv`](data/b1_lens_scores.csv),
  [`data/b1_lens_fit.csv`](data/b1_lens_fit.csv),
  [`data/b1_lens_timings.csv`](data/b1_lens_timings.csv): curve, per-protein
  scores, tuned-map fit table, timings (with the boundary-24 check).
- [`plots/b1_lens.png`](plots/b1_lens.png).
- [`data/b1_deltas.csv`](data/b1_deltas.csv): every condition x band x split
  with both CIs.
- [`data/b1_scores.csv`](data/b1_scores.csv): R-precision per condition,
  protein and seed (and the seed-averaged matrix).
- [`data/b1_timings.csv`](data/b1_timings.csv): per-protein timings.
- [`plots/b1_layer_ablation.png`](plots/b1_layer_ablation.png).
- Extension: [`data/b1_ext_deltas.csv`](data/b1_ext_deltas.csv),
  [`data/b1_ext_scores.csv`](data/b1_ext_scores.csv),
  [`data/b1_ext_timings.csv`](data/b1_ext_timings.csv) (ablation, with
  splits by protein set and length bins 1-150, 151-300, 301-500,
  501-1000); [`data/b1_lens_ext_curve.csv`](data/b1_lens_ext_curve.csv),
  [`data/b1_lens_ext_scores.csv`](data/b1_lens_ext_scores.csv),
  [`data/b1_lens_ext_fit.csv`](data/b1_lens_ext_fit.csv),
  [`data/b1_lens_ext_timings.csv`](data/b1_lens_ext_timings.csv) (lens);
  [`data/b1_extra_gt.jsonl.gz`](data/b1_extra_gt.jsonl.gz) and
  [`b1_select_extra.py`](b1_select_extra.py) (the legacy extras).
- Not in git, under the S3 run prefixes above: the ablation's per-seed
  P(contact) for all 49 conditions (about 3.4 GB, `out/b1/pcontact/`), and
  the lens's per-seed P(contact) for both lenses at every boundary plus the
  fitted maps (`out/lens/pcontact/`, `out/lens/lens.npz`).
