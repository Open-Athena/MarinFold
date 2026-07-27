# exp163 — teaching contacts-v1 to refine a set of candidate rollouts

**Issue:** [#163](https://github.com/Open-Athena/MarinFold/issues/163) · **PR:** [#164](https://github.com/Open-Athena/MarinFold/pull/164) · **Status as of 2026-07-27:** Phases 0–2 complete, Phase 3 in flight

---

## 1. The question

contacts-v1 samples badly. A single rollout from the tuned 1.5B model (E8) recovers
only ~12% of a protein's true contacts (R-precision 0.122 on the exp98 protein set).
Sample it 16 times and *some* rollout is much better — oracle best-of-16 hits 0.233,
and the union of all 16 covers 52% of the true contacts.

So the information is there, spread across samples. The question is whether the model
can be taught to **read K of its own candidate rollouts and write out a better contact
set than any of them** — an in-context aggregation operator, learned rather than
hand-coded.

The bar is not "beat a single rollout" (easy). It is:

1. beat **training-free consensus voting** over the same K, and
2. beat the model's own **one-shot calibrated prediction**.

## 2. Why this is not obvious — three training-free probes

All three ran on exp98's 1,000 proteins × 1,000 rollouts. Metric throughout is exp89's
R-precision (per-target F1 at top-R) in the `all` band, sep ≥ 6.

### Phase 0 — aggregation beats selection

| K | consensus vote | proxy best-of-K (nll) | mean single | oracle best-of-K | union-recall ceiling |
|---|---|---|---|---|---|
| 8 | 0.194 (AUC .68) | 0.137 | 0.122 | 0.208 | 0.394 |
| 16 | 0.224 (AUC .73) | 0.145 | 0.122 | 0.233 | 0.516 |
| 32 | 0.244 (AUC .78) | 0.144 | 0.122 | 0.252 | 0.638 |

Voting reaches ~97% of the (non-deployable) oracle *selection*, and rises with K.
Meanwhile `nll_per_tok` is a **useless** quality proxy: within-protein Spearman
median −0.17, and the min-nll pick sits at F1 percentile 0.525 — indistinguishable
from random. There is therefore **no deployable way to rank or filter candidates at
inference**, which kills the original "F1-sorted ramp" and "F1-stratified selection"
designs. They survive only as oracle ceilings. **Train on unordered candidate sets.**

### Step 1 — the control that reframes the whole experiment

| method | all-band R-prec | all-band AUC |
|---|---|---|
| base E8 calibrated matrix | **0.221** | **0.887** |
| consensus vote (K=16) | 0.224 | 0.735 |
| single rollout | 0.125 | — |
| oracle best-of-16 | 0.241 | — |
| union-recall ceiling | **0.530** | — |

Paired on the same 149 proteins, **voting − matrix = +0.003** (vote wins on exactly
50% of proteins). A tie. Consensus voting is just a Monte-Carlo estimate of the
model's own per-pair marginal — it adds nothing the base model doesn't already emit.

This is the result that sets the real bar: a refiner that merely learns consensus is
**worthless**. It has to extract *supra-marginal* signal.

### Step 2 — the crux: does joint structure exist?

Paired conditional-vs-unconditional on the identical reduced task (score only the
pairs outside the given set G; positives are GT \ G):

| given context G | uncond R | cond R | ΔR | ΔAUC | cond wins |
|---|---|---|---|---|---|
| **oracle: random 50% of TRUE contacts** | 0.145 | **0.556** | **+0.41** | +0.10 | **100%** |
| **noisy: one real rollout (~13% precision)** | 0.179 | 0.092 | **−0.087** | −0.13 | 9% |

Two facts, and the experiment lives in the gap between them:

* **The mechanism is real and large.** Knowing half of a true contact map makes the
  rest almost perfectly predictable (AUC → 0.99). That is exactly the joint/structural
  signal a per-pair marginal cannot express.
* **It is precision-gated.** Feed the *same* model a realistic noisy rollout and it
  gets **worse**. The base model was trained on `<begin_statements>` sections that
  contain only true contacts, so it trusts its context as ground truth and lets ~87%
  false contacts corrupt its structural prior.

**This is why training, not prompting, is required** — and it says precisely what the
training must teach: that `<begin_candidate>` contacts are *hypotheses*, not truth.

## 3. The document format

A refinement document is a normal contacts-v1 document with K candidate-rollout blocks
spliced in between the sequence header and the answer:

```
<contacts-v1> <begin_sequence> …sequence…
  <CAND> <contact> pi pj …        ← candidate block 1
  …                                  K blocks, unordered
  <CAND> <contact> pi pj …        ← candidate block K
<begin_statements> …TRUE contacts… <end>
```

Design choices that matter:

* **Zero vocab change.** `<CAND>` reuses the spare token
  `<contacts-and-distances-v1>` — the *other* format's doc-type sentinel, never emitted
  inside a contacts-v1 document, so it is collision-proof and weight-compatible with
  every existing checkpoint. E8 has never seen it inside a document, so its embedding
  is effectively fresh.
* **K separate blocks, not one merged set.** Recurrence of a contact *across* blocks is
  a precision cue the model can read directly. Merging would destroy it.
* **K ~ Uniform{0…16}, redrawn per document.** K=0 is a plain contacts-v1 document.
  The model must produce the truth both with and without candidates — this is the guard
  against the twin failure modes: *ignoring* candidates (collapse to base ~0.22) and
  *blindly trusting* them (the zero-shot poisoning above).
* **Per-candidate subsampling** `uniform[1, n_pred]`, so test-time full candidates stay
  in distribution.
* **Loss on the answer span only.** The model is never trained to reproduce the
  sequence or the candidates.

### 3.1 A complete training document, K = 0

This is the degenerate case — an ordinary contacts-v1 document. 215 tokens, 27 true
contacts. Shown in full (the `# ---` lines are annotations, not part of the document):

```
# --- sequence header (loss weight 0) ---
  <contacts-v1> <begin_sequence> <n-term> <p1725> <p1759> <GLN> <p1765> <ASP> <p1747> <SER>
  <p1739> <SER> <p1752> <VAL> <p1785> <ASP> <p1728> <ALA> <p1780> <ALA> <p1733> <SER> <p1746>
  <GLY> <p1735> <SER> <p1757> <THR> <p1773> <ALA> <p1774> <THR> <p1779> <THR> <p1744> <ILE>
  <p1741> <PHE> <p1743> <ALA> <p1756> <ALA> <c-term> <p1787> <p1726> <GLN> <p1729> <ASP> <p1762>
  <PRO> <p1731> <LEU> <p1787> <ALA> <p1753> <THR> <p1750> <SER> <p1738> <HIS> <p1763> <THR>
  <p1745> <ILE> <p1742> <ALA> <p1783> <PHE> <p1782> <ASP> <p1758> <GLY> <p1748> <SER> <p1751>
  <ALA> <p1727> <LEU> <p1768> <ALA> <p1786> <ALA> <p1760> <ARG> <p1777> <ALA> <p1734> <GLN>
  <p1732> <GLU> <p1730> <TYR> <p1784> <HIS> <p1766> <PHE> <p1749> <PRO> <p1767> <MET> <p1737>
  <SER> <p1775> <ASP> <p1754> <LEU> <p1772> <LYS> <p1776> <LYS> <p1778> <VAL> <p1781> <ASN>
  <p1771> <GLN> <p1769> <ALA> <p1764> <ARG> <p1761> <ILE> <p1740> <GLN> <p1725> <MET> <p1770>
  <ILE> <p1736> <LEU> <p1755> <TRP>
# --- answer span (loss weight 1): 27 TRUE contacts + <end> ---
  <begin_statements> <contact> <p1755> <p1770> <contact> <p1741> <p1731> <contact> <p1767>
  <p1761> <contact> <p1783> <p1727> <contact> <p1752> <p1738> <contact> <p1760> <p1754>
  <contact> <p1755> <p1766> <contact> <p1785> <p1726> <contact> <p1755> <p1727> <contact>
  <p1779> <p1771> <contact> <p1751> <p1760> <contact> <p1755> <p1783> <contact> <p1770> <p1745>
  <contact> <p1734> <p1741> <contact> <p1741> <p1770> <contact> <p1774> <p1730> <contact>
  <p1726> <p1782> <contact> <p1778> <p1727> <contact> <p1780> <p1767> <contact> <p1783> <p1762>
  <contact> <p1773> <p1745> <contact> <p1762> <p1755> <contact> <p1770> <p1727> <contact>
  <p1778> <p1770> <contact> <p1725> <p1777> <contact> <p1745> <p1766> <contact> <p1745> <p1769>
  <end>
```

Note the position tokens wrap around the 2000-slot ring (`<p1725>` … `<p1787>` then
back to `<p1725>`): each document draws a random N-terminus offset, one of the format's
nuisance symmetries. `<n-term>` / `<c-term>` mark the chain ends.

### 3.2 A complete training document, K = 5

Same format, five candidate blocks prepended. 251 tokens, 9 true contacts. Shown in
full:

```
# --- sequence header (loss weight 0) ---
  <contacts-v1> <begin_sequence> <p16> <ALA> <p25> <LEU> <p12> <ALA> <p1973> <SER> <p1990> <GLY>
  <n-term> <p1966> <p22> <GLN> <p1966> <MET> <c-term> <p27> <p1972> <LEU> <p1998> <ALA> <p2>
  <ALA> <p21> <ALA> <p7> <ARG> <p9> <ALA> <p14> <ARG> <p1991> <ALA> <p1987> <HIS> <p1975> <GLY>
  <p15> <GLU> <p3> <GLY> <p1988> <PHE> <p1994> <LEU> <p1996> <HIS> <p4> <GLU> <p1968> <SER>
  <p10> <VAL> <p1995> <PRO> <p1970> <LEU> <p6> <GLU> <p24> <ARG> <p26> <ALA> <p23> <ALA> <p1>
  <LEU> <p1999> <MET> <p18> <LYS> <p1986> <VAL> <p11> <LYS> <p27> <ALA> <p1989> <ASN> <p1980>
  <GLU> <p1997> <GLU> <p1978> <LYS> <p20> <GLY> <p1967> <LYS> <p1983> <VAL> <p1971> <ASN>
  <p1982> <ALA> <p1979> <ILE> <p1985> <VAL> <p1977> <ALA> <p17> <GLU> <p1974> <ALA> <p1976>
  <VAL> <p1969> <THR> <p1984> <GLY> <p5> <ILE> <p1992> <GLN> <p1993> <LEU> <p0> <LEU> <p1981>
  <SER> <p13> <GLN> <p8> <ALA> <p19> <ILE>
# --- candidate block 1/5  (1 contacts, loss weight 0) ---
  <contacts-and-distances-v1> <contact> <p1986> <p1992>
# --- candidate block 2/5  (6 contacts, loss weight 0) ---
  <contacts-and-distances-v1> <contact> <p1988> <p1970> <contact> <p1993> <p1976> <contact>
  <p1978> <p1989> <contact> <p5> <p1994> <contact> <p1994> <p1> <contact> <p1977> <p1988>
# --- candidate block 3/5  (9 contacts, loss weight 0) ---
  <contacts-and-distances-v1> <contact> <p18> <p24> <contact> <p7> <p1971> <contact> <p1971>
  <p11> <contact> <p1971> <p4> <contact> <p1> <p1989> <contact> <p1995> <p1979> <contact>
  <p1989> <p1999> <contact> <p1988> <p0> <contact> <p1967> <p15>
# --- candidate block 4/5  (2 contacts, loss weight 0) ---
  <contacts-and-distances-v1> <contact> <p1986> <p1995> <contact> <p1999> <p1979>
# --- candidate block 5/5  (11 contacts, loss weight 0) ---
  <contacts-and-distances-v1> <contact> <p1967> <p1999> <contact> <p1978> <p4> <contact> <p1988>
  <p1997> <contact> <p1997> <p1991> <contact> <p1999> <p1988> <contact> <p1972> <p1978>
  <contact> <p1970> <p1982> <contact> <p1978> <p1970> <contact> <p1981> <p1> <contact> <p0>
  <p1969> <contact> <p1977> <p5>
# --- answer span (loss weight 1): 9 TRUE contacts + <end> ---
  <begin_statements> <contact> <p1> <p1993> <contact> <p1980> <p1987> <contact> <p1980> <p1967>
  <contact> <p14> <p7> <contact> <p1992> <p1985> <contact> <p1978> <p1967> <contact> <p1979>
  <p1988> <contact> <p1993> <p1986> <contact> <p1978> <p1969> <end>
```

Read the structure: five candidate blocks of 1, 6, 9, 2 and 11 contacts — the
subsampling in action — followed by the 9-contact answer. The candidates are genuinely
noisy: of the 29 distinct contacts proposed across all five blocks, **none** appears in
the 9-contact answer.

Both examples above are the **shortest** documents in the corpus, chosen so they fit on
a page — and short proteins are the hardest (contacts are sparse, ~1/L prevalence), so
they are *not* representative. The typical document is 2,270 tokens with ~424 distinct
candidate contacts.

### 3.3 Recurrence really is a precision cue

The format bets that showing K *separate* blocks lets the model read recurrence as a
confidence signal. That bet is measurable directly on the corpus (1,393 sampled
documents with K ≥ 1):

| candidate contacts | precision vs the true answer |
|---|---|
| all distinct candidates (mean 424/doc) | 0.097 |
| appearing in **exactly one** block | 0.079 |
| appearing in **more than one** block (8% of them) | **0.247** |

A contact that recurs across blocks is **~3× more likely to be true** than a singleton
(24.7% vs 7.9%). The signal the design depends on is present in the raw input, before
any training — the model only has to learn to use it. This also explains the MVP result
that an explicit 44%-precision consensus block bought nothing over raw blocks: the
refiner can already recover consensus from the blocks itself.

## 4. What the loss mask actually does

The LM loss is armed on exactly `[<begin_statements>, <end>)` — trains every true
`<contact> <pX> <pY>` triple **and** the terminal `<end>` (so the model learns to
stop) — and is zero on the sequence header, every candidate block, the trailing `<eos>`
and any padding.

The mask is built from two running counts, which makes it packing-safe (many documents
share one 8192-token training sequence, and the mask must re-arm at each):

```
opened[i] = # of <begin_statements> in ids[:i+1]
closed[i] = # of <end>              in ids[:i+1]
weight[i] = 1.0 if opened[i] > closed[i] else 0.0
```

Verified on a real packed sequence — the mask arms at `<begin_statements>` (predicting
the first `<contact>`), stays armed through predicting `<end>`, and drops immediately
after:

```
  i=  453 tok=<TYR>                w=0.0  -> predicts <begin_statements>
  i=  454 tok=<begin_statements>   w=1.0  -> predicts <contact>
  i=  455 tok=<contact>            w=1.0  -> predicts <p133>
  ...
  i=  976 tok=<p145>               w=1.0  -> predicts <end>
  i=  977 tok=<end>                w=0.0  -> predicts <eos>
  i=  978 tok=<eos>                w=0.0  -> predicts <contacts-v1>   (next document)
```

**Where the mask lives changed during this work.** levanter 1.2 removed
`DatasetComponent.loss_weight_fn`, the hook exp0/exp120 used to compute masks on the
data worker. Per-token weights now only reach training *through the cache*, via
`PrebuiltLmDatasetFormat(input_ids_key=…, loss_weights_key=…)`. So the mask is
materialized offline by `tokenize_refinement_corpus.py`, which tokenizes, greedy-packs
whole documents into fixed 8192-token rows, and writes `input_ids` + `loss_weights`
side by side.

That turned out to be strictly better: the token ids are now *resolved from the live
tokenizer* rather than hard-coded, and — the bigger win — **nothing cloudpickles by
module reference any more**, so the GPU training worker never has to import experiment
code. That had been a reported blocker (the `marinfold` package pins
`transformers`/`huggingface_hub<1.0`, which clashes with the marin training stack); it
dissolved rather than needing a workaround.

## 5. The MVP (local LoRA, 900 proteins)

Before spending cluster time, the mechanism was tested with a local LoRA fine-tune on
900 exp98 proteins, evaluated on 60 held out.

**Matrix regime** (R-precision, all / long):

| model | K0 | raw K16 | consensus block (44% prec) |
|---|---|---|---|
| base E8 | 0.229 / 0.153 | **0.017 / 0.016** | 0.149 / 0.124 |
| refiner v1 (r16, 6% K=0 docs) | 0.213 / 0.129 | **0.244 / 0.169** | 0.247 / 0.163 |
| refiner v2 (r32, 21% K=0 docs) | 0.183 / 0.120 | 0.218 / 0.156 | 0.214 / 0.146 |

**Sampling regime** (40 proteins, n=2): base@K0 F1 0.144/0.104 → ref@K0 0.139/0.090 →
**ref@K16 0.197/0.131** (+0.053 all vs base sampling, wins 68%; +0.058 vs its own K0).
Best-of-16 candidates (oracle) 0.286; consensus set 0.219.

Four conclusions:

1. **Candidate-conditioning is genuinely learned.** The +0.03 all / +0.04 long gain
   over the *same model's* K=0 is stable across v1/v2 and reproduces in the generative
   regime, so it is not a matrix-calibration artifact. Base 0.017 → refiner 0.244 on
   identical input is the cleanest demonstration.
2. **The refiner's calibrated matrix (0.244) is the best deployable prediction** —
   beating base one-shot (0.229) and consensus (~0.22) — but the margin is modest.
3. **Precision-gating preprocessing is a dead lever.** A 44%-precision consensus block
   (0.247) ≈ raw K blocks (0.244): the refiner already internalizes consensus from the
   raw blocks. Keep the simple corpus.
4. **The margin is capped by overfitting at 900 proteins.** v2 fit harder and got
   *worse* held-out (K0 0.213 → 0.183). **Scale is the lever** — which motivated
   everything below.

## 6. Scale: generation

Proteins now come from the **ESM-Atlas / ESMFold2-distill** contacts-v1 corpus (exp139;
3,338 shards, 66.8M docs, one per linclust cluster, so any random shard subset is
unbiased). Rollouts are generated with **`--top-k -1`** — the #142 under-generation fix
— and `logprobs` dropped (~3.7× faster).

**10k validation batch**, 16 × 1×H100 on CoreWeave rno-2a at batch priority, ~1h:

* **225,072 rollouts over 9,375 proteins** (15/16 shards; 1 preempted — batch is
  preemptible and the worker is resume-safe, which is exactly why 9,375 ≠ 10,000)
* **`n_pred` mean 201.0 vs GT ~199** — the top-k fix confirmed (exp98's top-k=50
  rollouts averaged ~95 contacts), `frac_finished` 1.00, single-rollout `all_f1` 0.114

Refinement corpus from those real rollouts: **18,750 documents / 9,375 proteins**
(2 per protein, K ~ U{0..16}), 0 OOV, 0 over-budget. Tokenized and packed:
**44.3M tokens → 6,569 sequences** of 8192, 82.3% packing density, 25.2% of tokens
carrying loss ⇒ 52 steps/epoch at batch 128.

## 7. Scale: training (Phase 2, complete)

Warm-started from E8 via `initialize_from_hf` against the HF export already on S3;
1-epoch cosine, batch 128, seq 8192, wd 0.2, warmup 0.1, on one 8×H100 node at batch
priority. ~25 min per arm.

| peak LR | train/loss (masked objective) | base-task eval loss | base-task **bpb** |
|---|---|---|---|
| **1e-4** | 3.985 → **2.3979** | 3.16941 | **0.39489** |
| 3e-4 | 3.833 → **2.3915** | 3.40526 | 0.42428 |

**1e-4 is the better trade.** 3e-4 fits the refinement objective 0.3% better while
degrading base-task retention **7.4%** more — the same fitting-vs-retention tension the
MVP hit. Both eval curves dip at step 13 and recover, i.e. LR-warmup damage, not
divergence. Re-running the 1e-4 arm reproduced an earlier run **bit-identically**
(3.16941 / 0.39489), a free determinism check over tokenize → pack → cache → train.

### A measurement trap worth knowing about

The plan called for verifying the warm-start as "step-0 val loss ≈ 2.7566" (Eric's
logged E8 value). That check does not work, for two independent reasons:

* levanter fires eval hooks at multiples of `steps_per_eval` and has **no step-0 eval**,
  so the first recorded value already carries warmup damage;
* **per-token loss is not comparable across harnesses.** levanter's `bpb` divides by
  summed per-token-type byte lengths weighted by the loss mask, so a different
  packing/eval configuration changes the effective bytes-per-token — and hence the loss
  scale — at identical model quality:

| | loss | bpb | implied bytes/token |
|---|---|---|---|
| Eric's E8 (`eval_metrics.jsonl`, step 35679) | 2.75660 | 0.39151 | 10.16 |
| this harness (step 51) | 3.16941 | 0.39489 | 11.58 |

loss ratio 1.1497 = bpb ratio 1.0086 × bytes/token ratio 1.1399, exactly. **bpb agrees
to 0.9%** — that, not the loss, is the evidence E8 loaded (a bad load would sit near
ln(2845) = 7.95).

⚠️ This generalizes: **any MarinFold experiment quoting a cross-harness per-token loss
target is exposed to the same effect** — #137, #150 and #155 all do.

## 8. Phase 3 — evaluation (in flight)

The 10k corpus has **no held-out protein split**: all 9,375 rollout proteins went into
training. So the headline comparison must run on the **exp89 eval set** (554 units /
552 stems), which needs its own base-model rollouts.

Built and dispatched: 554 targets (sequences from the exp74/exp78 manifests, GT from
exp89's `gt_universe.jsonl` under exp89's own `degree ≥ 0.001 ∧ |i−j| ≥ 6` definition,
so the target GT is bit-identical to what the metric scores against), 24 prompts each,
8 × 1×H100 shards at batch priority.

Leakage is not a concern: the training corpus uses ESM-Atlas MD5 entry ids — a disjoint
universe from exp89's PDB-derived stems.

**Success criteria** (unchanged from the issue):

* **Does it use candidates without being poisoned?** refiner@K=16 must exceed
  refiner@K=0, and never fall below it.
* **Is it worth it?** refiner@K must beat max(base calibrated matrix, consensus voting)
  ≈ 0.22, climbing toward oracle best-of-K (0.24) and ultimately the conditional
  ceiling (0.556), ideally holding AUC ≥ 0.89.
* **Kill:** refiner ≈ K=0 (ignores candidates), or < K=0 (poisoned), or stuck at ~0.22
  (learned only consensus).

## 9. Cost of the 1M push

Measured, not estimated: the 10k batch did 225,072 rollouts in ~59 min on 16 H100s ⇒
**~14,350 rollouts/GPU-hour**. So 1M proteins × 24 rollouts = 24M rollouts ⇒
**~1,700 H100-hours** — ~4.4 days wall on 16 shards, ~26h on 64. That is ~2× the
pre-run estimate, because the top-k fix roughly doubled rollout length (which is the
point: n_pred 201 vs 95).

Two things to settle before committing:

* **Is 1M the right number?** 250k proteins is ~420 H100-hours (~26h on 16 shards) and
  plausibly buys most of the fold diversity for a quarter of the spend.
* **One known scaling wrinkle remains.** `gen_prompts_exp163.py` still writes one S3
  object per target — fine for 554, ~1M objects at full scale. Fixing it needs matched
  changes on both sides (aggregate prompt shards *and* a packed reader in the rollout
  worker), so it is deliberately not half-done.

---

## Appendix — reproduction

```bash
cd experiments/exp163_models_teach_contacts_v1_to_refine_a
set -a; source ~/.config/marin/cw-rno2a.env; set +a
WK=$(python -c "import netrc; print(netrc.netrc().authenticators('api.wandb.ai')[2])")

# corpus -> tokenized + masked
uv run python tokenize_refinement_corpus.py \
    --in  s3://marin-us-east-02a/MarinFold/exp163/val10k/refinement_corpus/'*.parquet' \
    --out s3://marin-us-east-02a/MarinFold/exp163/val10k/refinement_tokenized

# train (batch band, 8xH100)
uv run iris --cluster=cw-rno2a job run --no-wait --priority batch \
    --enable-extra-resources --cpu=2 --memory=6GB --disk=16GB --extra gpu \
    -e WANDB_API_KEY "$WK" -e EXP163_STEPS_PER_EPOCH 52 \
    -- python -m dispatch_refine_train
```

Key files: `loss_mask.py` · `tokenize_refinement_corpus.py` · `refine_ft_common.py` ·
`dispatch_refine_train.py` · `build_refinement_corpus.py` · `select_targets_eval_set.py` ·
`gen_prompts_exp163.py` · `dispatch_rollouts.py`. Narrative + operational detail in
`SCALE_PLAN.md`.
