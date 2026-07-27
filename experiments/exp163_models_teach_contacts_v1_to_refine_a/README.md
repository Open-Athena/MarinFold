---
marinfold_experiment:
  issue: 163
  title: "exp: teach contacts-v1 to refine a set of candidate rollouts into a better contact set"
  kind: models
  branch: main
---

# exp: teach contacts-v1 to refine a set of candidate rollouts into a better contact set

**Issue:** [#163](https://github.com/Open-Athena/MarinFold/issues/163) · **Kind:** `models` · **Branch:** `main`

## Question

Can we teach contacts-v1 to take **K candidate rollouts** (its own samples for a protein, of varying quality) and emit a contact set that beats (a) any single rollout, (b) training-free consensus voting over the same K, and (c) its own one-shot calibrated prediction — i.e. learn an **in-context aggregation / refinement operator** over a collection of candidate contact maps?

## Hypothesis

**Two measured facts frame the bet.**

*(Step 1)* The base model's per-pair calibrated matrix and a consensus vote over 16 rollouts are a **dead tie** on R-precision (0.221 vs 0.224) — voting is just a Monte-Carlo estimate of the same marginal, so consensus adds nothing the base model doesn't already output. A refiner that learns only consensus is worthless.

*(Step 2 — the decisive probe)* Yet the joint/structural signal the marginal misses is **real and large**: conditioning the base model (zero-shot) on 50% of a protein's *true* contacts lifts R-precision on the *remaining* contacts from 0.145 to **0.556** (+0.41; ΔAUC +0.10; better on 100% of proteins). Knowing part of a real contact map makes the rest highly predictable — the mechanism the refiner would exploit is present and strong, before any training.

**But that signal is precision-gated, which is exactly why training (not prompting) is required.** Conditioning the base model on a *noisy* candidate rollout (~13% precision) instead *hurts* — R drops 0.179 → 0.092 (worse on 91% of proteins) — because the base model was trained on `<begin_statements>` sections that contain only true contacts, so it trusts its context as ground truth and lets ~87% false contacts corrupt its structural prior.

So the refiner's job is precise: **learn that `<begin_candidate>` contacts are noisy hypotheses, not truth — identify the trustworthy ones (contacts recurring across the K candidate blocks are higher-precision, a signal the model can see directly) and use them to trigger the strong joint-completion the oracle probe demonstrated, climbing from ~0.22 toward the 0.556 ceiling.** GT supervision over K *separate* candidate blocks teaches exactly this discrimination; the distinct `<begin_candidate>` marker (never seen by the base model) lets it treat candidates differently from true `<begin_statements>` contacts.

Training uses a **variable candidate count K ∈ {0,…,Kmax}** (K=0 = a plain contacts-v1 doc): the model must produce GT with and without candidates — guarding the twin failure modes of *ignoring* candidates (collapse to base ~0.22) and *blindly trusting* them (the zero-shot noisy poisoning above).

## Background

### Phase 0 — feasibility probe (done; training-free, on exp98's 1,000×1,000 rollouts)

Sampled-F1 basis (`all` band, sep≥6), R-precision = F1 at top-R:

| K | consensus vote | proxy-best-of-K (nll) | mean single | oracle best-of-K | union-recall ceiling |
|---|---|---|---|---|---|
| 8  | 0.194 (AUC 0.68) | 0.137 | 0.122 | 0.208 | 0.394 |
| 16 | 0.224 (AUC 0.73) | 0.145 | 0.122 | 0.233 | 0.516 |
| 32 | 0.244 (AUC 0.78) | 0.144 | 0.122 | 0.252 | 0.638 |

- **Aggregation beats sampling:** consensus voting ≈ 97% of the (non-deployable) oracle selection and +84% over a single rollout; rises with K.
- **Confidence is a weak proxy:** within-protein Spearman(nll, F1) median −0.17; min-nll pick sits at F1 percentile 0.525 (≈ random). ⇒ no deployable quality-ordering of candidates → **train on unordered candidate sets** (the F1-sorted ramp and F1-stratified selection of the original idea are both non-deployable; kept only as oracle ceilings).
- Probe code: `scratchpad/p0_analysis.py` (reuses `exp98/rollout_metrics.py`).

### Step 1 — base calibrated-matrix control (done; 149 exp98 proteins, local GPU, exp89 scorer)

Base E8 per-pair contact-logprob matrix, same top-R metric, **paired on the same 149 proteins**:

| method | all-band R-prec | all-band AUC |
|---|---|---|
| **base calibrated matrix** | **0.221** | **0.887** |
| consensus vote (K=16) | 0.224 | 0.735 |
| single rollout | 0.125 | — |
| oracle best-of-16 | 0.241 | — |
| **union-recall ceiling** | **0.530** | — |

- long-band: base matrix R-prec **0.160** (AUC 0.870). Eval-set reference was 0.339/0.269 — the exp98 training proteins are **harder**.
- **Voting − matrix = +0.003 (vote wins 50% of proteins) = a tie.** Consensus is a noisy copy of the marginal, not a free win. Code: `scratchpad/step1_base_matrix.py`, `step1_finalize.py`.

### Step 2 — zero-shot conditional-mechanism probe (done; 50 exp98 proteins, local GPU) — **the crux result**

Does conditioning the base model on a partial contact set improve its ranking of the *remaining* pairs? Paired cond vs uncond on the identical reduced task (universe minus given set G; positives P = GT\G):

| given context G | uncond R | cond R | ΔR | uncond AUC | cond AUC | ΔAUC | cond wins |
|---|---|---|---|---|---|---|---|
| **oracle: random 50% of TRUE contacts** | 0.145 | **0.556** | **+0.41** | 0.891 | 0.993 | +0.10 | **100%** |
| **noisy: one real rollout (~13% prec, ~112 contacts)** | 0.179 | 0.092 | **−0.087** | 0.883 | 0.758 | −0.13 | 9% |

- **Mechanism exists & is powerful:** true partial context → near-perfect completion of the rest (joint structure the per-pair marginal cannot express).
- **But precision-gated:** the base model trusts its context as all-true, so raw noisy candidates poison it. This is *why* training (not prompting) is needed and precisely what it must fix.
- **Design implication:** show K *separate* `<begin_candidate>` blocks so **recurrence across blocks is a visible precision cue**; supervise on GT so the model learns to trust recurring/high-precision candidates and discount the rest. Code: `scratchpad/probe_conditional.py`, results `probe_conditional.csv`.

### Prior work this builds on / is distinct from

- **#98** — 1M rollouts (1,000 proteins × 1,000), per-band F1 precomputed, GT joined; TPU vLLM worker `gen_rollouts_worker_vllm_tpu.py`, `rollout_metrics.py`. **Reused wholesale** as the training-candidate source. Data: `hf://buckets/open-athena/MarinFold/data/contacts-v1-train-rollouts-exp98/`.
- **#102** — 200 proteins × 1,000 with **emission-order contacts + per-contact logprob** (`gen_rollouts_worker_hf.py`); join 1:1 on `(entry_id, r)` if per-contact confidence features are wanted.
- **#158 / #159 / #160** — the **single-trajectory self-correction** line (`<retract>`, backtracking corpus, self-correct training). This experiment is the **ensemble cousin**: aggregate *multiple independent* rollouts rather than correct one trajectory. Complementary; share the "re-condition the model on contacts" machinery.
- **#82** — settled rollout recipe (T=1.0/p=0.95/**k=50**) and the `_fwd_matrix` → `P(contact|prompt)` scorer; the top-k=50 causes under-generation (#142) and is the sampling we fix for any *fresh* generation. #82's rollout+resample (conditioning a rollout on its own emitted contacts) is early evidence that in-context contacts carry joint signal — Step 2 confirms it quantitatively.
- **#100** — only-correct constrained decoding (the "never wrong" corpus); ours is "given noisy candidates, produce the truth".
- **inference.py** `_fwd_matrix` / `_pcontact_from_fwd` — `[L,L]` contact-logprob matrix from teacher-forced passes; **reused to evaluate the refiner conditioned on candidate context** and for the Step-1/Step-2 probes.
- **Base checkpoint (generator + fine-tune init):** `prot-exp75-cv1-1_5b-e8-lr1e-3-wd0p2-v1-bc3084` step-35679 (E8; eval R-prec 0.339 all / 0.269 long; exp98-train R-prec 0.221/0.160). fp32 HF export in bucket `checkpoints/.../hf/step-35679` (local copy `/home/bizon/exp89_export/hf_step35679`); bf16 in `gs://marin-us-east5/checkpoints/.../hf_bf16/step-35679`. Tokenizer `timodonnell/contacts-v1-tokenizer@5d68a24a899f`.

### Format (zero vocab change)

A refinement doc = a normal contacts-v1 doc with candidate blocks prepended:

```
<contacts-v1> <begin_sequence> …seq…
  <begin_candidate> …rollout 1 contacts…      ← K candidate blocks, unordered, loss-masked
  …  (K separate blocks — recurrence across them = precision cue)
  <begin_candidate> …rollout K contacts…
<begin_statements> …TRUE contacts… <end>       ← normal contacts-v1 target, loss here
```

- `<begin_candidate>` = one **repurposed spare token** (contacts-v1 carries ~814 inherited-but-never-emitted tokens) → weight-compatible with existing checkpoints, corpus-mixable with plain contacts-v1, and semantically distinct from `<begin_statements>` (true contacts) so the model can learn candidates are noisy.
- Contacts stay 3-token `<contact> <pX> <pY>` triples; one random wrap-around position numbering per doc; re-emit each rollout's seq-index contacts under it (base shuffle + i/j flip). exp98 `pred` is already in seq-index space.
- Context budget is *not* the binding constraint: cost = `2R + 8 + K + 3·(all contacts shown)`; a mean protein with K=16 near-full candidates ≈ 3–4k tokens. Adaptive per-candidate cap only bites at R≳450.
- Optional `<think>` pause run before `<begin_statements>` (exp123 machinery, loss-masked) — the "reason about how to improve" arm.

### Metric

exp89 `compute_metrics.py`: R-precision (= per-target F1 at top-R) + AUC, in `{all, short, medium, long}` bands. True contact = pyconfind degree ≥ 0.001 ∧ |i−j| ≥ 6, exact symmetric pair match. Eval via the `eval-checkpoint` skill (554 units / 552 stems).

## Approach

**Steps 0–2 (DONE).** Feasibility probe + base calibrated-matrix control + zero-shot conditional-mechanism probe — see Background. Verdict: aggregatable joint signal exists (oracle-cond 0.556) but is precision-gated (noisy-cond poisons the base model); training must teach candidate discrimination.

**Phase 1 — build the refinement corpus (local, CPU; reuse exp98).**
- Split exp98's 1,000 proteins into refiner-train / refiner-val (e.g. 900/100). **Verify disjointness from the exp89 eval set** (leakage).
- Per doc: draw **K ~ Uniform{0,1,…,Kmax}** (Kmax≈16) **random** rollouts from the protein's pool, **unordered**, each a separate `<begin_candidate>` block; answer = true contacts after `<begin_statements>`. No binning/F1-selection — spread is free from random sampling. K=0 ⇒ a plain contacts-v1 doc.
- **Augmentation:** per-candidate subsample `uniform[1, n_pred]` (includes near-full → test-time full candidates stay in-distribution); random position numbering; re-draw K and which rollouts each epoch.
- **Optional lever (Step-2-motivated):** also try a *consensus-weighted* presentation (emphasise / tag high-recurrence candidate contacts) to approximate the oracle regime; MVP relies on the model learning recurrence from the K blocks directly.
- **Ceiling/diagnostic arms only:** oracle-F1-sorted / oracle-F1-stratified candidate selection (non-deployable) — quantify what perfect ordering/selection would buy.

**Phase 2 — train.** Fine-tune from E8. **Loss on the `<begin_statements>` (true-contacts) section only**; candidates masked (MVP). Small run (1,000 proteins) → low LR, early-stop on refiner-val; monitor candidate-conditioned R-precision, no-candidate R-precision (no one-shot regression), *and* that noisy candidates don't drag it below base. Optionally mix plain contacts-v1 docs.
- Ablations: loss on candidate blocks too; `<think>` arm; K ∈ {8,16,32}; consensus-weighted vs plain candidate presentation.

**Phase 3 — evaluate on exp89 held-out.** Generate **K base rollouts per eval protein with the same sampling regime as training** (MVP: exp98's top-k=50). Two evals:
- **Canonical (score-matrix):** condition `_fwd_matrix` on candidate context → R-precision/AUC vs **base no-candidate matrix** (control) and vs **consensus voting**, matched K. The refiner should hold the matrix's AUC (0.89) while lifting top-R.
- **Sampling (the operator):** sample the refiner's output; F1 vs best input candidate, a fresh base sample, and voting.
- Test-time-scaling curve (quality vs K).

**Phase 4 (if MVP wins) — regenerate with fixed sampling + scale protein breadth.** exp98's shape (1,000 × 1,000) over-samples rollouts and under-samples proteins; a generalizable refiner wants *many proteins × dozens of rollouts*, fixed (non-top-k) sampling on both sides.

## Success criteria

- **Primary — does it use candidates without being poisoned:** refiner **@K=Kmax vs @K=0** on exp89 held-out; candidate-conditioned R-precision (all **and** long) must **exceed** the K=0 readout (and never fall below it). Since voting ≈ the base matrix (~0.22 on exp98), beating K=0 means extracting supra-marginal signal — the whole point.
- **Primary — is it worth it:** refiner@K R-precision **>** max(base calibrated matrix, consensus voting) ≈ 0.22, climbing toward oracle best-of-K (0.24) and, ultimately, the conditional-mechanism ceiling (oracle-partial **0.556**); ideally with AUC ≥ the base matrix's 0.89.
- **Secondary:** sampled output F1 > best input candidate and > a fresh base sample; positive test-time scaling in K; consensus-weighted vs plain presentation.
- **Kill:** refiner ≈ K=0 (ignores candidates) **or** < K=0 (poisoned like zero-shot) **or** cannot beat ~0.22 (learned only consensus) → candidates add nothing learnable; ship the base matrix.

## Results

### Phase 2 — training (DONE 2026-07-27)

Refiner fine-tuned from E8 on the 10k-protein refinement corpus (18,750 docs →
6,569 packed 8192-token sequences), answer-span masked, 1-epoch cosine, batch 128,
on CoreWeave `cw-rno2a` 8×H100 at batch priority. See `SCALE_PLAN.md` §B.

| peak LR | train/loss (masked objective) | base-task eval loss | base-task **bpb** |
|---|---|---|---|
| **1e-4** | 3.985 → **2.3979** | 3.16941 | **0.39489** |
| 3e-4 | 3.833 → **2.3915** | 3.40526 | 0.42428 |

1e-4 is the better trade: 3e-4 fits the refinement objective 0.3% better but
degrades base-task retention 7.4% more. Checkpoints + HF exports at
`s3://marin-us-east-02a/MarinFold/exp163/checkpoints/<run>/hf/step-51`.

Warm-start verification uses **bpb, not per-token loss** — see `SCALE_PLAN.md` for
why #163's "step-0 val ≈ 2.7566" criterion is not well posed across harnesses.

### Phase 3 — evaluation (DONE 2026-07-27)

553 exp89 eval proteins, paired, both models under identical candidate contexts.
Harness check: base K0 = **0.3355** vs exp89's published E8 **0.3389**.

| model | K0 | K1 | K2 | K4 | K8 | K16 | consensus |
|---|---|---|---|---|---|---|---|
| base | **0.3355** | 0.1452 | 0.1024 | 0.0665 | 0.0269 | 0.0194 | 0.2023 |
| refiner | 0.1978 | 0.1555 | 0.0813 | 0.0383 | 0.0211 | 0.0165 | **0.2220** |

**The kill criterion is met** — the base model's one-shot matrix (0.3355) remains
the best deployable prediction; the refiner's best arm is 0.2220.

**But the mechanism works.** Given the same 56.7%-precision consensus block, the
refiner *gains* +0.0242 over its own K0 (wins on 63% of proteins) while the base
*loses* −0.1333 (wins on 8%) — a +0.157 swing. Candidate discrimination was learned.

**Why it still loses: catastrophic forgetting.** refiner K0 0.1978 vs base 0.3355 —
the 1-epoch full fine-tune cost 41% of the one-shot contact ability, and a +0.024
conditioning gain cannot recover a 0.138 hole. Note base-task val bpb moved only
0.9% (0.39151 → 0.39489) while the contact metric moved 41%: **the LM proxy badly
under-reports task damage.**

Also inverts the MVP's conclusion #3: raw candidate blocks hurt monotonically for
both models (even K=1, in-distribution); the precision-filtered consensus block is
the *only* context that helps.

See [WRITEUP.md](WRITEUP.md) §8 for the full analysis and next levers.

## Conclusion

_(Fill in after results are in.)_
