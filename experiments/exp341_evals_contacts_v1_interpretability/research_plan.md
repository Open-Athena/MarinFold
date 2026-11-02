# Interpretability of MarinFold contacts-v1: research plan

## 1. Motivation

MarinFold's contacts-v1 model is a Qwen3 decoder with 1.47B parameters. It reads a
shuffled bag of `<pN> <AA>` statements and emits `<contact> <pX> <pY>` statements.
It scores 0.620 R-precision on the legacy 554-protein set and 0.554 on FoldBench
eval-val (exp277, `marinfold/marinfold/MODELS.yaml`). A few experiments in this repo
have looked inside the model in service of other questions (section 1.1), but none
has set out to explain how it predicts contacts: there are no layerwise probes, no
layer ablations, no Jacobians and no SAEs under `experiments/`.

This plan is scoped to two training decisions that the next model will have to make
and that behavioural evals have not settled (section 1.2):

- **A. How to weight protein families in the training data.** Cluster-balanced
  sampling (cap how often each family is seen, spend the tokens on small families)
  vs keeping the corpus's natural family redundancy.
- **B. Where the next model's capacity should go.** More depth vs more width, and
  whether the 8192-token contact budget withholds supervision on long proteins.

The source material is the survey in this directory (`compass_artifact_...md`),
which covers general LLM interpretability and the protein-model literature (Rao
attention readout, the categorical Jacobian, InterPLM, Reticular, "Two Stages of
Folding", ProGenMech). An earlier version of this plan applied four methods from it
to four descriptive questions; the parts with no training decision attached were
moved to section 10.

### 1.1 Prior work in this repo

- **#247** (`experiments/exp247_evals_protein_property_analysis/`). Family abundance
  is the only covariate block that predicts per-protein R-precision. Spearman with
  MSA depth: 0.67 for the #199 cooldown, 0.50 for #232 m2-p06, 0.41 for ESMFold2,
  0.30 for Protenix + MSA. Length, relative contact order and long-range fraction are
  |ρ| ≤ 0.04 for MarinFold. #232 m2-p06 loses 0.25 R-precision from the deepest to
  the shallowest MSA-depth quartile, more than ESMFold2 (0.18) or Protenix + MSA
  (0.08). This is the observation Track A has to explain: MSA depth measures both
  how much coevolution signal a family has and how many relatives the model trained
  on, and the two readings argue for opposite data weightings.
- **#213** (`experiments/exp213_evals_train_sequence_overlap_audit/`). 58% of the
  554 proteins have a training homolog, but accuracy barely tracks training identity
  (Spearman −0.12 pooled, against +0.53 for a sequence-KNN null). R-precision falls
  from 0.611 to 0.549 on the 231 proteins with no homolog. On the 37 proteins novel
  in both sequence and fold, MarinFold scores 0.328 against Protenix-v2
  single-sequence at 0.547. exp213's MMseqs2 search also records a training-hit
  count per eval protein.
- **#245 H2** (`experiments/exp245_evals_foldbench_held_out_monomers/`). The
  contaminated \#199 cooldown drops +0.024 more from eval-val to eval-test than the
  \#232 checkpoints do, 95% CI [−0.022, +0.073]: a behavioural contamination test on
  eval proteins that does not resolve the effect.
- **#224** (`experiments/exp224_evals_does_marinfold_fold_a_circularly_permuted/`).
  A circular permutation of DsbA (1UN2) costs 0.093 R-precision against an
  unpermuted control of identical composition. Pairs whose sequence separation the
  permutation moved lose 0.121; pairs it left alone lose 0.051.
- **#218** (`experiments/exp218_evals_proteingym_zero_shot_fitness/`). Treated the
  sequence section as an any-order amino-acid model, through the library function
  `amino_acid_conditionals`
  (`marinfold/marinfold/document_structures/contacts_v1/sequence_likelihood.py:239`).
  exp199-1.5B scores 0.296 average Spearman on ProteinGym (212 assays), against
  0.415 for ESM2-650M and 0.321 for ESM2-35M; top-1 amino-acid accuracy is 0.345 at
  context ≥ 0.8 against a 0.132 composition floor.
- **#262** (`experiments/exp262_models_nope_token_smearing/`). Its Phase 0 probes
  (`probe_attention.py`, `probe_common.py`) already load contacts-v1 checkpoints
  with hooks and build teacher-forced documents from exp245 ground truth; the harness
  here starts from `probe_common.py`. \#262 also reports that MarinFold's contact
  enrichment scales as L^0.79 against ESMFold2's L^1.15, which Track B's budget
  analysis bears on.

Related open work: **#339** asks whether accuracy is limited by memorization,
training redundancy, diversity or ranking, using rollout diagnostics on exp321's
eval-val rollouts. It shares Track A's question and answers it from the outputs;
this plan answers it from inside the model and from a controlled data split. The two
issues should link to each other.

### 1.2 Why run this instead of the training ablations directly

Each decision could be tested by training. The cost of doing so: exp277's one-epoch
run took 266,345 steps at a measured 0.858 s/step on 16 nodes (128 H100s), and just
under 73 hours wall-clock including a checkpoint-failure recovery
(`experiments/exp277_models_single_mpnn_pilot/README.md`): about 9,300 H100-hours.
A smaller-scale ablation pair is cheaper, but family weighting and depth effects are
the kind that change with scale, so a small pair would leave the question open at
1.5B. This plan's budget is 20-25 H100-hours (section 4.8), about 0.25% of one
exp277 epoch.

What the plan buys is a choice of which arm to train, with a mechanism attached.
Each decision has a rule for both outcomes, fixed before any result exists; a result
that lands between them is reported as inconclusive.

**Decision A: family weighting.**

| Result | Reading | Training change |
|---|---|---|
| exp232's accuracy on held-out structures saturates at a small number of kept relatives (A3); contact predictions on well-predicted proteins depend on many residues beyond the pair (A2, non-local); exp199's exposure advantage persists even when many relatives remain (A4) | The model recalls families and individual structures; relatives beyond the first few add little | Cap samples per cluster (the corpus rows carry Foldseek `struct_cluster_id`, exp225 README) at the saturation point from A3, and spend the freed tokens on small families, e.g. from exp91's 66.8M ESMFold2 structures |
| Accuracy keeps rising with relative count across the observed range (A3); contact dependence is concentrated on the pair (A2, local) and the Jacobian's coupling map tracks MSA depth at fixed training homology (A1) | The model learns couplings from family statistics; each extra relative adds signal | Keep family redundancy. Improve shallow-MSA proteins by adding members to their families (distillation targeted at small clusters), not by reweighting |

**Decision B: depth, width and the contact budget.**

| Result | Reading | Training change |
|---|---|---|
| The tuned-lens contact readout (B1) is still rising over layers 19-24 by ≥ 0.03 R-precision, and mean-ablating any single late layer costs ≥ 0.02 | Contact computation uses the full depth | Favour depth in the next size sweep |
| The readout is within 0.01 of its final value by layer 18, and late-layer ablations cost < 0.01 | The last quarter of the network adds little to contacts | Favour width, or a shallower model at the same parameter count |
| At matched contact degree, long proteins recover contacts that fall outside a training document's budget worse than short proteins recover the same-degree contacts that fit (B2), and truncated documents are a sizeable share of the corpus (B2 count) | The budget withholds supervision where the model is weakest | Raise context length for long documents, or a denser contact encoding |
| No matched-degree gap, or truncation is rare | The budget is not what limits long proteins | Leave the budget; the L^0.79 scaling (#262) has another cause |

The thresholds in B (0.03, 0.02, 0.01, layer 18) are proposals, to be fixed in the
issue before B1 runs. No protein-model precedent sets them: the tuned lens was
developed on text LLMs, and the "Two Stages of Folding" layer profiles come from
folding trunks (ESMFold, OpenFold, Boltz-1), not a decoder reading a contacts-v1
document. The per-layer ablations are in B1 so that the decision does not rest on
the lens alone. A layer profile at one size cannot show that a deeper model
would do better, only which arm of a size sweep to run first; B1's optional
checkpoint trajectory (section 7.1) guards against reading an under-trained late
layer as a spare one.

## 2. Why contacts-v1 is a good object for these questions

- **The output is already a contact probability.** The pairwise readout
  (`_fwd_matrix`, `contacts_v1/inference.py:284`) gives
  `log P(<p_i> | prefix, <contact>) + log P(<p_j> | prefix, <contact>, <p_i>)` for
  every pair. Mutations, layer ablations and model swaps can be scored directly on
  the contact map, with no auxiliary head.
- **The sequence section is an amino-acid language model.** #218 showed it gives
  P(AA_j | all other residues), so the categorical Jacobian of Zhang et al. (PNAS
  2024), which separates coevolution-like couplings from recall in ESM-2, applies
  without modification.
- **Graded ground truth.** pyconfind gives a continuous contact degree for every
  residue pair, while training documents carry only the thresholded, budget-truncated
  set (separation ≥ 6, degree ≥ 0.001, strongest contacts first up to 8192 tokens).
  This is what makes the budget question in B2 answerable.
- **A controlled data split exists already.** exp199 was trained on the unfiltered
  corpus; exp232 and exp277 were trained on the #225 corpus, which removed every
  training sequence at ≥ 30% identity (≥ 50% coverage) to any eval protein, and
  purged whole Foldseek clusters on the structure axis. The removed structures were
  seen by exp199 and never by exp232/exp277, and they differ in how many relatives
  survived in the decontaminated corpus.

## 3. Research questions

**Track A (decision A).**

- **A-Q1.** Does the model store coevolution-like couplings? Does the sequence
  section's Jacobian recover contacts, and does its precision follow MSA depth after
  controlling for training homology?
- **A-Q2.** Is a predicted contact computed from the two residues or from the
  surrounding motif and fold?
- **A-Q3.** For a structure the model never saw, how does accuracy grow with the
  number of its relatives that remained in training? Where does it saturate?
- **A-Q4.** How much does having seen the exact structure add, on top of its
  relatives?

**Track B (decision B).**

- **B-Q1.** At which layer does the correct contact partner become readable, and how
  much do the last layers add? Does the answer differ for long-range contacts and
  long proteins?
- **B-Q2.** How often does the contact budget truncate training documents, and does
  the model predict out-of-budget contacts worse than equally strong in-budget ones?

## 4. Shared setup

### 4.1 Models

| Role | Nickname | Why |
|---|---|---|
| Primary | `contacts-v1-exp277-m2-p06-full-epoch-1.5B` (step 266344) | Registry default; decontaminated corpus; one epoch, so each packed example was seen once |
| Contaminated arm | exp199 cooldown (`MODELS.yaml`, `-cooldown-` entry) | Saw the #225-removed structures, ~16 AFDB epochs |
| Decontaminated arm | exp232 m2-p06 (step 363000) | Same recipe family as exp277, #225 corpus |
| Null | Randomly initialized Qwen3 of the same config | Control for the Track B probes (Heap et al. 2025 found probe pipelines score similarly on random models) |

Architecture (all three trained models): 24 layers, hidden 2048, intermediate 8192,
32 query heads and 8 KV heads (head_dim 64), QK-norm, Llama3 RoPE with theta 500000,
vocab 2845, context 8192 (`experiments/exp199_optimize_contacts_v1_afdb_esm/gpu/exp199_sweep_cw.py:118-130`).

**RoPE hazard.** Configs exported by transformers 5.x load with theta 10000 under
transformers 4.x. Every harness must go through `needs_rope_repair` / `repair_rope`
(`marinfold/marinfold/inference/_config.py:55-129`); activations from an unrepaired
load come from a different model.

### 4.2 Proteins

- **eval-val** (97) and **eval-denovo** (19) from exp245
  (`hf://buckets/open-athena/MarinFold/data/contacts-v1-foldbench-monomers-exp245/`).
- **Legacy 554** (`hf://buckets/open-athena/MarinFold/data/contacts-v1-model-eval-exp89/gt_universe.jsonl`).
  Stores `(i, j, degree)` for every pair with degree > 0; the training pool for the
  Track B probes, fit on a split disjoint from eval-val.
- **#225-removed AFDB structures** and matched kept structures (Track A; section
  4.7).
- **eval-test is not used.** It has a logged read budget
  (`experiments/exp245_evals_foldbench_held_out_monomers/data/eval_test_reads.md`).

### 4.3 Covariates

- `experiments/exp213_evals_train_sequence_overlap_audit/data/eval_train_identity.csv`:
  per protein `best_identity_covered`, `stratum`, `fold_verdict`, `seq_leakage`,
  `msa_neff`, `neff_tier`, `length`, and the training-hit count.
- `experiments/exp260_evals_msa_depth_stratified/data/msa_depth.csv` (402 rows):
  `n_seqs`, `n_eff_0.8`, `n_eff_0.62`, and the frozen low-depth set.
- For the removed and kept structures: kept-relative count (section 4.7), length,
  mean pLDDT.

### 4.4 Output metric

All interventions are scored by the pairwise readout: `_prefix_and_positions`
(`inference.py:258`) builds the prefix, `_fwd_matrix` and `_pcontact_from_fwd`
(`inference.py:284,310`) give the P(contact) matrix, and R-precision is computed
over all separations and over long-range (≥ 24) separately. Pairwise reads about
0.086 lower than the rollout+resample recipe that the eval-checkpoint skill requires
for headline numbers. That is acceptable here because every comparison is internal
(intervened vs clean, or model vs model, on the same proteins and seeds); no number
from this experiment is compared with a published score.

Every readout is averaged over 4 document seeds (the ensemble salt
`f"{id}#cv1ens{k}"`, `contacts_v1/generate.py:359`). Model comparisons use the same
seeds, so each protein's shuffle and residue offset are identical across arms.

### 4.5 Tooling

- HF export loaded with PyTorch `transformers` through `marinfold`'s transformers
  backend (`TransformersBackend.from_model`, PR #361). Residual capture and mean
  ablation use plain torch forward hooks on the decoder layers, `self_attn` and
  `mlp` (`harness.py`), not nnsight: the harness needs only those three hook
  points, and plain hooks avoid a dependency whose tracing layer would sit between
  the readout and the model. Harness check 2 confirms the hooked forward is
  identical to the plain one.
- The readout functions above were private (`_`-prefixed); PR #360 promoted the
  ones the harness needs to public names.
- 1.47B in bf16 fits one 24 GB GPU for harness development and activation dumps.
  Jacobians and ablation sweeps fan out as independent 1×H100 jobs on CoreWeave,
  following the single-GPU fan-out recipe in `AGENTS.md` (explicit iris client,
  `setup_scripts=[]`, fsspec for S3, per-job exception handling, interleaved
  sharding).
- Activation dumps go to the CoreWeave bucket, co-located with the compute. CSVs and
  plots are committed.

### 4.6 Harness checks (before any result is trusted)

1. The hooked transformers forward pass reproduces `_pairwise_score_matrix` on three
   eval-val proteins within bf16 tolerance, with `repair_rope` applied, and visibly
   diverges without it.
2. The layer-ablation hook with no layer selected reproduces the clean R-precision
   exactly.
3. The Jacobian's exact path and gradient approximation agree on the top couplings
   of one short protein before either is run at scale.
4. Probes trained on the random-init model give near-chance selectivity.
5. One clean readout pass over eval-val and eval-denovo is timed on an H100, and the
   throughput assumed in 4.8 is replaced with the measured number before any sweep
   is launched.

### 4.7 Data requirements

**Existing inputs** (`hf://` paths are under `hf://buckets/open-athena/MarinFold/`):

| Data | Where | Size | Used by |
|---|---|---|---|
| exp277 HF export (primary model) | `checkpoints/contacts-v1-exp277-m2-p06-full-epoch-1.5B/hf/step-266344` | 5.9 GB (2 safetensors shards) | A1, A2, B1, B2 |
| exp199 HF export (contaminated arm) | `checkpoints/prot-exp199-cw-cv1-s02-m1-p06-aug/hf/step-145199` | 5.9 GB | A4 |
| exp232 HF export (decontaminated arm) | `s3://marin-us-east-02a/marin/protein-structure/MarinFold/exp232_sweep_cv1_decontam/evals/rollout-v2/2026-08-24/v2-01/models/exp232-decontam-train-m2-p06-step363000/hf/step-363000` (already on CoreWeave) | ~5.9 GB | A3, A4 |
| eval-val / eval-denovo ground truth | `data/contacts-v1-foldbench-monomers-exp245/gt_universe_scored.jsonl`, `eval_sets.csv` | 6.9 MB, 200 KB. 97 proteins (26,326 residues, mean L 271) and 19 proteins (2,994 residues, mean L 158) | all |
| Legacy 554 ground truth with continuous degree | `data/contacts-v1-model-eval-exp89/gt_universe.jsonl` | 8.0 MB, 554 proteins (median L 148) | B1 probe training, A1 |
| Homology and fold covariates | `experiments/exp213_evals_train_sequence_overlap_audit/data/eval_train_identity.csv` | 554 rows, in git | A1, A2 |
| MSA depth | `experiments/exp260_evals_msa_depth_stratified/data/msa_depth.csv` | 402 rows, in git | A1, A2 |
| #225 drop list | `data/decontamination/contacts_v1_eval_reference/v1/droplist_final.parquet` | 31.5 MB. 1,373,423 entries removed: 166,679 AFDB (4.04% of 4,129,682; 23-290 per shard, median 73) and 1,206,744 ESM Atlas. Each row carries `(shard, row)` into the unfiltered corpus, identity, coverage, e-value and nearest eval reference | A3, A4 |

**What has to be built:**

1. **The held-out structure panel (A3, A4).** AFDB entries only, since
   `afdb-24M` supplies Foldseek cluster IDs.
   - *As built (`panel_candidates.py`).* Sequence and ground truth come from the
     published contacts-v1 documents, not from mmCIF through pyconfind. The drop
     list's `(shard, row)` pointers locate each removed document in the unfiltered
     corpus (`data/document_structures/contacts_v1/train`, `entry_id` checked); kept
     candidates are a seeded sample of the decontaminated corpus at the same shard
     indices. A document with `truncated = False` emits every contact at degree
     ≥ 0.001 and separation ≥ 6, so its contact list is the binary ground truth
     A3 and A4 score. Two consequences: truncated documents are excluded (229 of
     3.96M kept AFDB documents, all longer than 1,000 residues; B2), and the panel
     has no continuous degree, which A3 and A4 do not use. Stage 1 reads 300 of
     2,067 shards, about 24,000 removed candidates before stratification.
   - *Originally planned.* Pull removed structures from `timodonnell/afdb-24M`
     shards (mmCIF in `cif_content`) and recompute pyconfind contacts, including the
     continuous degree.
   - **Kept-relative count.** For each removed structure, the number of sequences in
     the exp232 training corpus at ≥ 30% identity and ≥ 50% coverage, from an
     MMseqs2 search of the removed sequences against the kept corpus (the same search
     setup as exp213, `-s 7.5`). This cannot be read off `struct_cluster_id`: #225's
     structure axis purged whole clusters, so a removed structure's relatives
     surviving in training are by construction outside its cluster.
   - Sample ~1,000 removed structures stratified over log-spaced kept-relative bins
     (0, 1-3, 4-15, 16-63, 64+), plus ~1,000 kept structures matched on length,
     pLDDT and relative count as the exposure control for A4. Filling the upper bins
     may take 30-50 shards (3-5 GB).
2. **Activation dumps (B1).** The residual stream at each `<AA>` token over 25
   layers at hidden size 2048 in bf16 costs about 100 KB per residue: ~13 GB per seed
   per model over eval-val, eval-denovo and legacy 554. B1 needs exp277 and the
   random-init control (~26 GB), plus about the same again for the tuned lens's
   `<p_i>` query positions on exp277 (~13 GB). Storing every other layer halves
   both.
3. **Budget counts (B2).** The contacts-v1 generator writes per-document metadata
   (`truncated`, `contacts_passing_min_degree`, `contacts_emitted`,
   `lowest_included_contact_degree`, `seq_len`; `contacts_v1/generate.py:280-290`).
   A CPU scan of those columns over the exp277 source-document parquets gives the
   truncation rate by length. If a corpus dropped those columns, recompute them from
   the structures for a 10,000-document sample.
   *Done* (`budget_counts.py`, `data/budget_counts.csv`): both native corpora kept
   the columns. AFDB: 229 of 3,963,003 documents truncated (5.8e-05; 224 of the 523
   documents at 1,501-2,000 residues), 7.0e-05 of eligible contacts dropped. ESM
   Atlas: no document exceeds 1,000 residues and none of 65,553,178 is truncated.
   The ProteinMPNN redesign corpora (exp266) keep each structure's length and
   contact set, so they inherit these rates.
4. **Small outputs.** Only reduced L×L Jacobian maps are stored (~360 KB at L = 300,
   against ~144 MB for the full L×20×L×20 fp32 tensor).

**Placement.** Run on CoreWeave: exp232 is already there, the other two checkpoints
are ~6 GB each, and activation dumps are written to the CoreWeave bucket next to the
compute. A single model's dumps exceed 10 GB, so moving them to GCS or a workstation
is a cross-region copy that needs explicit sign-off under the AGENTS.md rule.

### 4.8 Compute and storage estimates

These are planning estimates derived from token counts. Harness check 5 replaces
the throughput assumption with a timed pass. Measured so far: 5,933 tokens/s on one
L40S (eval-val and eval-denovo, 4 seeds, transformers bf16,
`data/readout_l40s/`); the H100 figure is still outstanding.

**Basis.**

- **Forward throughput: 50,000 tokens/s per H100.** A forward pass costs about
  2 × 1.47B ≈ 2.9 GFLOP per token, so 50k tokens/s is ~150 TFLOP/s, about 15% of
  the H100's 989 TFLOP/s bf16 dense peak. For comparison, exp277's training ran at
  1.222M tokens/s over 128 H100s, ~9,500 tokens/s per GPU, or about 8.5% of peak
  counting 6 FLOPs per parameter per token (forward plus backward). Prefill-only
  inference without optimizer and communication overhead should do better than
  training, but the 15% figure is the least certain input here.
- **A backward pass costs twice a forward pass.**
- **Sequence lengths**, summed from exp245's `gt_universe_scored.jsonl` and exp213's
  CSV:

  | Set | Proteins | ΣL | ΣL² |
  |---|---|---|---|
  | eval-val | 97 | 26,326 | 9.32M |
  | eval-denovo | 19 | 2,994 | 0.54M |
  | legacy 554 | 554 | 102,554 | 24.6M |

- **One pairwise readout of a protein ≈ 4L tokens**: the ~2L-token prefix once,
  then L two-token tails against the cached prefix. One readout pass over eval-val
  plus eval-denovo is ~117k tokens, about 2.3 s of H100 time.
- **Per-job overhead of ~3 minutes**: ~2.5 minutes of pod start plus ~11 s to
  stage a 5.9 GB checkpoint in-cluster (AGENTS.md single-GPU fan-out notes).

**Planned runs.**

| Step | Work | Tokens | H100-hours |
|---|---|---|---|
| A1 gradient Jacobian | L forward+backward passes of ~2L tokens per protein (6ΣL²) on eval-val, eval-denovo and 200 legacy proteins, exp277 | 112M | 0.6 |
| A1 exact confirmation | top-L pairs × 20 substitutions on 20 proteins with L ≤ 300 (40L² each) | 32M | 0.2 |
| A2 contact Jacobian | 4 representative substitutions per residue, full readout each (16ΣL²) | 158M | 0.9 |
| A3/A4 panel readouts | ~2,000 structures × 2 models (exp199, exp232) × 4 seeds, assumed mean L 300 | 19M | 0.1 |
| A4 Jacobian subset | 100 removed + 100 kept structures under exp199 and exp232 | 216M | 1.2 |
| B1 activation dumps | exp277 and random-init, 1 seed; bound by writing ~26 GB, not compute | <1M | <0.1 |
| B1 probe fitting | rank-32 bilinear probe on ~12M pairs per layer; 25 layers × task/control × 2 models | n/a | 2-4 |
| B1 tuned lens | 24 affine 2048×2048 maps, exp277 | n/a | <1 |
| B1 layer ablation | 24 layers × 2 sites (attention, MLP) × 4 seeds = 192 readout passes | 22M | 0.1 |
| B2 budget counts | metadata scan | n/a | CPU only |
| B2 matched-degree recall | reuses the clean readouts | 0 | 0 |

The planned runs sum to **roughly 6-9 H100-hours**; probe fitting is the least
certain, since it is bounded by optimizer steps rather than forward tokens. At ~50
fan-out jobs, overhead adds ~2.5 hours. Allowing a factor of two for reruns and bugs
gives a **budget of 20-25 H100-hours**. Harness development runs on a workstation
GPU and is not counted. The MMseqs2 relative-count search is CPU work of the size
exp213 ran (554 queries then, ~1,000-2,000 now).

**Optional or avoided variants:**

| Variant | H100-hours |
|---|---|
| B1 checkpoint trajectory: tuned lens and layer ablation on 4 intermediate checkpoints (each needs a Levanter-to-HF conversion first) | +2-3 |
| A1 exact Jacobian for a whole protein (40L³ tokens), per protein at L = 300 | ~6 |
| A2 all 19 substitutions per residue instead of 4 | 4.2 per model |

**Storage.** Activation dumps (~26 GB), tuned-lens residuals (~13 GB), three
checkpoints (~18 GB) and the AFDB shards (3-5 GB): about 60 GB at peak. Dumps are
deleted once probes and lenses are fit and their CSVs committed.

**Cost** at AWS on-demand prices (us-east-1, September 2026): $6.88 per H100-hour
(`p5.4xlarge`, 1× H100), S3 Standard at $0.023/GB-month, egress at $0.09/GB. No spot
or savings-plan discounts.

| Item | Quantity | Cost |
|---|---|---|
| Planned runs | 6-9 H100-hours | $41-62 |
| Budget, with job overhead and reruns | 20-25 H100-hours | $138-172 |
| B1 checkpoint trajectory (optional) | +2-3 H100-hours | +$14-21 |
| S3 Standard storage | ~60 GB at peak | ~$1.40 per month |
| Egress of everything to a workstation | ~60 GB | ~$5.40 |

The estimated cost is **about $140-170**, or about $190 with the checkpoint
trajectory. The hour counts come from token counts, not measurement; once harness
check 5 times a real pass, the dollar figures scale with the measured throughput.

## 5. Track A: memorization vs coevolution (decision A)

**Source.** Zhang, Wayment-Steele, Brixi, … Ovchinnikov, PNAS 2024 (ESM-2's
categorical Jacobian recovers Potts-like coevolution couplings; they weigh three
hypotheses: fold lookup, motif-pair memorization, coevolution). exp213's overlap
audit and #225's decontamination split.

The order is A3 first (a behavioural dose-response that directly sets the cluster
cap), then A1-A2 (the mechanism), then A4 (the exposure effect). A3 alone is enough
to act on if its curve is clear; A1-A2 decide the case where it is ambiguous.

### 5.1 A1: sequence-section Jacobian

- With residue j's `<pN>` placed last in the prefix, the next-token distribution over
  the 20 `<AA>` tokens is P(AA_j | all other residues). Use #218's
  `amino_acid_conditionals` (`sequence_likelihood.py:239`) rather than a new
  readout.
- Mutate residue i to each of the 20 amino acids and record the change in the
  log-odds at j. This gives an [L, 20, L, 20] tensor; apply the Frobenius norm over
  the two amino-acid axes and APC to get an L × L coupling map, as in Zhang et al.
- **Cost.** Exact computation needs L² × 20 prefixes. First compute the gradient
  approximation (one backward pass per target j, gradient of the log-odds with
  respect to the one-hot input of every residue), then confirm the top couplings
  exactly on 20 proteins with L ≤ 300 and report agreement.
- **Readouts.** Long-range P@L of the coupling map against pyconfind ground truth;
  agreement with the model's own pairwise readout.
- **Expectation from #218.** The conditional is worth about an ESM2-35M model on
  ProteinGym, so weak couplings from A1 alone would not show the model lacks
  coevolution information; A2 measures dependence on the contact output directly.

### 5.2 A2: contact Jacobian

- Mutate residue k (4 representative substitutions: to Ala, Gly, Pro and a
  charge-flipped residue) and record the change in the full P(contact) matrix from
  the pairwise readout.
- **Locality score** per predicted contact (i, j): the share of |ΔP(i, j)| from
  mutations at i, j and their ±3 sequence neighbours. Coevolution-like couplings
  predict high locality; motif or fold recall predicts dependence spread over the
  motif or the chain.
- **Stratified analysis.** Regress A1 precision and A2 locality on `msa_neff`,
  `best_identity_covered`, training-hit count, `fold_verdict` and length across
  eval-val, eval-denovo and 200 legacy-554 proteins. #247 already shows accuracy
  tracks MSA depth, and MSA depth also tracks how many training relatives a protein
  has, so a raw correlation with `msa_neff` fits both hypotheses. The discriminating
  quantities are the partial correlation of A1 precision with `msa_neff` holding
  training-hit count fixed, and A2 locality at matched MSA depth.
- The 37 novel-fold, no-homolog proteins from #213 (where MarinFold trails
  Protenix-SS by 0.22) show whether the model has usable couplings where it cannot
  recall anything.

### 5.3 A3: accuracy vs surviving relatives

- Score the held-out panel (section 4.7) with exp232, which never saw the removed
  structures. Plot R-precision (all and long-range) against the kept-relative count,
  with the matched kept structures' R-precision as the ceiling at each bin.
- Fit a saturating curve (R-precision against log relative count, piecewise linear
  with one breakpoint) and report the breakpoint with a bootstrap CI over structures.
- **This is the number decision A uses.** A breakpoint at a small count (the first
  two bins) with a flat curve after it is the argument for a per-cluster cap at that
  count. A curve still rising in the 64+ bin argues for keeping redundancy.
- **Confound.** Families with many relatives may be easier for reasons other than
  training exposure (more regular folds, higher pLDDT). The kept-structure ceiling at
  each bin and the length and pLDDT matching absorb part of this; A2's locality on
  the same structures tells recall and coupling apart where the curve alone cannot.

### 5.4 A4: exposure effect (exp199 vs exp232)

- Score the same panel with exp199 (saw the removed structures) and compute the
  difference-in-differences per relative-count bin:
  (exp199 − exp232 on removed) − (exp199 − exp232 on kept).
- **Readings.** A positive DiD that stays large even in the 64+ bin means exact
  exposure adds accuracy beyond what relatives supply: the model memorizes
  individuals, and exp199's ~16 epochs are the reason. A DiD that is positive only
  where few relatives survived means relatives substitute for exposure.
- Run A1 on 100 removed and 100 kept structures under both models. If exp199's
  advantage on the removed set comes with a less local, less coupling-like Jacobian
  than exp232's on the same structures, the advantage is recall.
- **Why not reuse #245 H2.** #245 compared the same checkpoint families on eval
  proteins, which are only indirectly contaminated, and could not resolve the effect
  (+0.024, CI [−0.022, +0.073]). The removed structures test direct exposure, and
  the relative-count bins give the effect a dose. #224's permutation result (0.121
  lost on moved pairs vs 0.051 on unmoved) is the single-protein precedent for
  separation-dependent recall.
- **Confound.** exp199 and exp232 differ in training length and schedule as well as
  corpus. The kept-set term absorbs overall capability differences, not interactions
  between schedule and protein type.
- **Document-level variant.** Training used scheduled amino-acid-order augmentation
  (`shuffle_amino_acid_statements`, `exp232_sweep_cv1_decontam/training_contract.py:148`),
  which re-permutes the sequence statements but leaves every `<pN>` unchanged. If the
  document realization exp199 saw is recoverable (its offset is fixed per cached
  document), a "seen offset vs fresh offset" comparison on the removed set is a
  sharper recall test; read `epoch_data.py` and the exp232 helpers first to confirm.

**Outputs.** `jacobian_couplings/`, `jacobian_summary.csv` (precision, locality,
covariates), `relative_dose_response.csv/.png`, `contamination_did.csv`.

## 6. Track B: depth, width and the contact budget (decision B)

**Source.** "Two Stages of Folding" (arXiv 2602.06020: distance probe R² plateaus
near 0.9 in late trunk blocks across ESMFold, OpenFold and Boltz-1); tuned lens
(Belrose et al. 2023); Hewitt & Liang 2019 (control tasks and selectivity).

### 6.1 B1: layer profile of contact information

- **Tuned lens on the contact query.** For the residual at the `<p_i>` position of a
  `<contact> <p_i>` tail, fit a per-layer affine map to the final-layer residual,
  then apply the unembedding restricted to the protein's position tokens. This gives
  a partner distribution and an R-precision at every layer. Compare with the plain
  logit lens to confirm the tuned lens adds information rather than artifacts.
- **Pairwise probe in the sequence section.** A rank-32 bilinear probe
  `h_iᵀ U Vᵀ h_j` on the residuals at the `<AA>` tokens of residues i and j,
  regressing log pyconfind degree and classifying contact, per layer. Controls: the
  random-init model and shuffled pair labels. This separates information built while
  reading the sequence from information built only once the `<contact>` query
  arrives.
- **Layer ablation.** Mean-ablate each layer's attention output and MLP output
  separately (48 conditions) and record ΔR-precision. This is the causal check on
  the lens: a late layer whose lens gain is small but whose ablation is costly is
  doing work the lens misses.
- **Splits.** Report every curve by separation band (short, medium, long-range) and
  by length tercile. Depth that matters only for long-range contacts on long
  proteins points at the same regime as #262's L^0.79 scaling.
- **Optional checkpoint trajectory.** Repeat the tuned lens and layer ablation on 4
  intermediate checkpoints. If late layers' contribution is still growing at the end
  of training, a plateau in the final checkpoint means under-training, not spare
  depth, and decision B should not read it as a vote for width.

### 6.2 B2: does the contact budget withhold supervision?

- **Count.** From the per-document metadata (section 4.7), report by length bin: the
  share of documents with `truncated = True`, the share of eligible contacts dropped
  (`1 − contacts_emitted / contacts_passing_min_degree`), and
  `lowest_included_contact_degree`. Weight by training tokens as well as by document.
- **Matched-degree recall.** For each eval-val protein, rebuild the training-document
  budget for its length and mark which true contacts would have been emitted. Then
  compare the pairwise readout's recall of out-of-budget contacts on long proteins
  with its recall of in-budget contacts of the same degree on short proteins,
  binning by degree. Matching on degree is required: out-of-budget contacts are the
  weakest by construction, so an unmatched comparison would show a gap under either
  hypothesis.
- **Probe version.** Score B1's pairwise probe against the full pyconfind set and the
  in-budget set, with the same degree matching. If the sequence-section residuals
  encode out-of-budget contacts as well as matched in-budget ones, the model has the
  information and truncation is costing only the output side.

**Outputs.** `tuned_lens.csv`, `probe_pairwise.csv` (layer × R² × split),
`layer_ablation.csv`, `budget_counts.csv`, `matched_degree_recall.csv`, figures.

## 7. How the tracks connect

- A2's Jacobian can be restricted to the layers B1 finds decisive (mutation effect
  on the residual at those layers), which shows where recall or coupling is
  computed; this is a reading aid, not a separate decision.
- Both tracks use the same 4 seeds and covariates, so every result can be split by
  MSA depth, homology, fold novelty and length.

## 8. Guardrails

- Every probe reports control-task selectivity and the random-init baseline.
- Probing and lens readouts are reported as presence of information; only the layer
  ablations, the Jacobians and the model comparisons support claims about use.
- The decision thresholds in section 1.2 are fixed in the issue before the run that
  tests them.
- No eval-test reads.

## 9. Risks

- **A3's curve is confounded by family difficulty.** Mitigated by matched kept
  structures and A2's locality; if the confound dominates, decision A falls back on
  A2 and A4.
- **Thin upper bins in A3.** #225 removed relatives of eval proteins, so removed
  structures with 64+ surviving relatives may be rare. The shard count in 4.7 grows
  until each bin has ≥ 100 structures, or the top bin is merged and the plan says so.
- **Jacobian cost** grows as L²; exact confirmation is limited to L ≤ 300.
- **Layer profile at one size.** B1 ranks sweep arms; it does not predict a deeper
  model's accuracy.
- **Private readout API.** The promotion PR in 4.5 removes the dependency on
  `_`-prefixed functions.

## 10. Deferred: questions without a training decision attached

These were in the earlier version of the plan. Each is interesting about the model,
but none of their outcomes would change what the next training run does. They move
to a follow-up issue if Tracks A and B leave budget or motivation.

- **Position-token geometry and offset invariance** (Fourier/PCA of the 2000 `<pN>`
  embeddings; a shift sweep over the residue offset; seed-variance decomposition).
  The only training lever attached was a per-visit offset re-draw in augmentation,
  which is cheap enough to add without evidence if anyone wants it.
- **Attention-head analysis** (binding, neighbour and Rao contact scores for 768
  heads; grouped and single-head ablation; activation patching). #262 Phase 0 already
  identified the previous-token heads on exp199; the remaining circuit questions do
  not bear on data or capacity.
- **Chain-reconstruction probe** (decoding the residue at N ± k). Its training lever
  was emitting the sequence section in chain order, which would also break #218's
  any-order conditional.
- **SAEs and cross-layer transcoders**, **training-data attribution** (TRAK),
  **developmental interpretability** (local learning coefficient) and **steering**.
  Each needs this plan's validated harness first; TRAK in particular would sharpen
  Track A by attributing individual predictions to training structures.

## 11. Deliverables

- GitHub issue #341 (tagged `experiment`); this directory,
  `exp341_evals_contacts_v1_interpretability/`, on branch
  `exp/341-contacts-v1-interpretability`, with the survey kept as source material.
- `README.md` holding this plan and results as they arrive, with
  `marinfold_experiment` frontmatter (`issue: 341`, `kind: evals`).
- A shared harness module (model loading with `repair_rope`, hooks, readout,
  layer ablation), one driver per track step, committed CSVs and figures.
- A written recommendation for decisions A and B against the rules in section 1.2.
- A history file under `history/runs/` for any run that logs to W&B.
