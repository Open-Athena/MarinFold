**Title:** Interpretability of contacts-v1: chain reconstruction, contact heads, and memorization vs coevolution

**Summary.** Use interpretability on the default contacts-v1 model
(`contacts-v1-exp277-m2-p06-full-epoch-1.5B`) to settle two training decisions for
the next model: (A) whether to cap how often each protein family is sampled or keep
the corpus's family redundancy, and (B) whether extra capacity should go to depth or
width, and whether the 8192-token contact budget withholds supervision on long
proteins. Full plan: `experiments/exp341_evals_contacts_v1_interpretability/research_plan.md`.

**Why not just train the ablations.** One exp277 epoch took ~73 hours on 128 H100s
(~9,300 H100-hours). This plan is budgeted at 20-25 H100-hours and picks which arm to
train, with a mechanism attached. Each decision has a rule for both outcomes, fixed
before the runs (plan section 1.2).

**Track A: memorization vs coevolution (decision A)**  
1. **A3, accuracy vs surviving relatives.** Score #225-removed AFDB structures with
   exp232 (never saw them), binned by how many relatives stayed in its training
   corpus. The saturation point is the per-cluster cap, if there is one.  
2. **A1-A2, categorical Jacobians.** Sequence-section couplings and contact
   dependence on each residue. Local, coupling-like dependence argues for keeping
   redundancy; non-local dependence argues for recall and a cap. The test is the
   partial correlation with MSA depth at fixed training homology, since #247 already
   shows raw accuracy tracks MSA depth.  
3. **A4, exposure effect.** Difference-in-differences of exp199 (saw the removed
   structures) vs exp232, per relative-count bin.  

**Track B: depth, width and contact budget (decision B)**  
1. **B1, layer profile.** Tuned-lens contact readout per layer, a bilinear pairwise
   probe in the sequence section, and mean-ablation of each layer. Still rising at
   layer 24 favours depth; flat by layer 18 favours width.  
2. **B2, budget.** Truncation rate by length from per-document metadata, and recall
   of out-of-budget contacts vs in-budget contacts of the same degree.  

**Builds on.** #247 (family abundance is the only predictor of accuracy), #213
(training-homology audit), #245 H2 and #224 (behavioural contamination and
permutation tests), #218 (the sequence-section amino-acid conditional), #262 Phase 0
(probe harness; L^0.79 enrichment scaling). Coordinate with #339, which asks Track
A's question from rollout outputs.  

**Data.** eval-val (97) and eval-denovo (19) from exp245; legacy-554 ground truth for
probe training; ~1,000 #225-removed and ~1,000 kept AFDB structures from
`afdb-24M`; no eval-test reads.  

**Cost.** An estimated 6-9 H100-hours of planned runs, budgeted at 20-25 with job
overhead and reruns: about $140-170 at AWS on-demand pricing ($6.88/H100-hour).
Storage (~60 GB peak) and egress add under $10. Derived from token counts, not
measured; see plan section 4.8.  

**Deferred.** Position-token geometry, attention-head circuits and patching, the
chain-reconstruction probe (no training decision rides on them); SAEs, TRAK, LLC,
steering.  

**Deliverables.** Experiment directory with a shared hook harness (transformers +
nnsight, `repair_rope` applied), committed CSVs and figures, and a written
recommendation for decisions A and B.
