---
marinfold_experiment:
  issue: 174
  title: 'exp: 3D coordinate-based structural eval (CA-RMSD / all-atom RMSD / LDDT / LDDT-CA / TM-score) for contacts-and-crops-v1 models'
  kind: evals
  branch: claude/github-issue-174-0d4157
---

# exp: 3D coordinate-based structural eval (CA-RMSD / all-atom RMSD / LDDT / LDDT-CA / TM-score) for contacts-and-crops-v1 models

**Issue:** [#174](https://github.com/Open-Athena/MarinFold/issues/174) · **Kind:** `evals` · **Branch:** `claude/github-issue-174-0d4157`

## Question

Every MarinFold eval so far scores *contact prediction*. contacts-and-crops-v1
([#130](https://github.com/Open-Athena/MarinFold/issues/130)) encodes real 3D
positions — Pass-1 coarse 10 Å boxes plus Pass-2 fine 0.1 Å crops — so a model
trained on it should be able to emit an actual **structure**. Can it? Scored the
way structure predictors are normally scored: CA-RMSD, all-atom RMSD, lDDT,
lDDT-CA and TM-score against the 554-protein eval set
([#74](https://github.com/Open-Athena/MarinFold/issues/74) /
[#78](https://github.com/Open-Athena/MarinFold/issues/78)).

## Approach

The issue splits into two components with different gates.

**Component 1 — inference (document → coordinates).** Plans first, no
implementation, by design. Six approaches are written up in
[`PLANS.md`](PLANS.md) with tradeoffs, compute costs and failure modes.
**Decided on #174: Plan F** — neighbour-conditioned iterative refinement, a
spatially coherent "scanning flashlight" over voxels that conditions each crop
on its already-refined neighbours and iterates until coordinates stop moving —
with sampling rather than greedy decoding and two independent temperatures
(coordinate tokens vs structural choices). Plan C is its ablation. Build order
and the still-open questions are in `PLANS.md` §8. Still unimplemented.

**Component 2 — scoring (prediction → metrics).** Built, tested and validated;
that is what the code in this directory is. It takes a directory of predicted
coordinate files and needs to know nothing about how they were produced.

### The file contract

Everything — ground truth, model predictions, baselines — is a PDB file per
protein in the contract of [`canonical_pdb.py`](canonical_pdb.py):

```
<dir>/<dataset>/<stem>.pdb      # dataset ∈ {foldbench100, denovo_pdb, cameo_hard, casp_fm}
```

single chain `A`; `resSeq` = **1-based input-sequence index**; atom names from
the 37-name heavy-atom vocabulary; unplaced atoms simply absent; the B-factor
column carrying the predictor's positional uncertainty in Å (which is what the
refined-vs-coarse split reads). The `<dataset>` directory level is load-bearing:
`7ur7_A` and `8ah9_A` appear in *both* the FoldBench-100 and exp65 manifests
with different sequences and different ground-truth files, so the record key is
`<dataset>/<stem>`, not `<stem>`.

### Pipeline

1. **[`prepare_gt_structures.py`](prepare_gt_structures.py)** — full-atom ground
   truth for all 554 records. Extracts the single protein chain, runs
   contacts-and-crops-v1's own `analyze_coordinates` (so "an atom the ground
   truth has" means exactly "an atom the format could have mentioned"), aligns
   the resolved residues to the input sequence with difflib as exp78/exp89 do,
   and renumbers to input-sequence indices. Emits the structures plus
   `gt_index.jsonl` and `gt_contacts.jsonl`.
2. **[`structure_metrics.py`](structure_metrics.py)** + **[`usalign.py`](usalign.py)**
   — the metrics. RMSD by Kabsch superposition (biotite), lDDT and lDDT-CA by
   biotite's reference implementation at the CASP convention (15 Å inclusion,
   0.5/1/2/4 Å bins, intra-residue pairs excluded), TM-score by **US-align
   `-TMscore 1`** — the sequence-*dependent* variant, since prediction and
   ground truth are the same protein and residue *i* corresponds to residue *i*.
   (TM-align's sequence-independent alignment, which is what every pip-installable
   wrapper exposes, would be free to slide the prediction along the chain.)
3. **[`score_structures.py`](score_structures.py)** — per-record CSV + aggregate
   summary, stratified by length and by dataset.
4. **[`baseline_predictions.py`](baseline_predictions.py)** +
   **[`run_baselines.py`](run_baselines.py)** — the model-free ceiling (below).

### How partial predictions are handled

contacts-and-crops-v1 documents are budget-filling and ~96 % are truncated, so a
partial prediction is the normal case, not an edge case. The harness reports two
families of metric and they must not be confused:

| family | metrics | denominator | reading |
|---|---|---|---|
| **coverage-penalized** | `lddt_all`, `lddt_ca`, `tm_score` | the ground truth | an unplaced atom costs score — **compare models on these** |
| **covered-only** | `lddt_*_covered`, `rmsd_all`, `rmsd_ca` | the covered set | "how good is the part it emitted" — meaningless without the coverage columns |

A superposition needs a common atom set, so RMSD is unavoidably covered-only: a
predictor that emits three atoms perfectly scores 0.0 Å. lDDT and TM-score take
their denominators from the reference, so missing atoms push them down exactly
as a missing loop does in CASP. A record with **no** prediction file is scored as
a total miss (zero coverage, zero lDDT, zero TM), not skipped — dropping it would
quietly inflate the mean over whatever the predictor happened to finish.

For the penalized lDDT, unpredicted atoms are given coordinates far from
everything *and from each other*, so every reference contact touching one is
scored as broken while still counting in the denominator. That is the penalized
definition, computed by biotite's own tested code path rather than a
re-derivation of it; `tests/test_structure_metrics.py` pins the arithmetic on a
4-residue chain where the contact count is countable by hand.

## Results

### The ground-truth bundle

**554/554 records, 0 failures**, 777,459 heavy atoms, minimum alignment identity
0.95, lengths 30–761 (median 161).

Building it turned up a real bug in the shared library. `analyze_coordinates`
joins the pyconfind residue list to the gemmi coordinate walk on
`(chain, author-resnum)`, and the two libraries spell a **blank** author chain id
differently — gemmi `""`, pyconfind `"_"`. Structures with no chain id (the CASP
target files here; never AFDB, which is always chain `A`) joined nothing and
produced a **document with an empty coordinate section**, silently. Fixed in both
`contacts_and_crops_v1/parse.py` and ccoord's identical copy, with
`analyze_coordinates` now raising rather than emitting a coordinate-free
document, regression tests in `marinfold/tests/`, and a note in the crops SPEC.
19 of the 554 eval proteins hit it.

### The model-free ceiling

**Read "ceiling" carefully.** The oracle-document row below is the ceiling for
**one 8192-token document**, not for the format. A single perfect document
spends its atom budget like this (measured over 246k ground-truth atoms):

| tier | share of atoms | median error |
|---|---|---|
| refined by Pass 2 | 22.6 % | 1.20 Å |
| 10 Å box only (Pass 1) | 38.4 % | 4.91 Å |
| never mentioned | 39.0 % | total miss |

Three separate budget losses, none of them about the coordinate encoding:
39 % of atoms do not fit; the boxed 38 % carry ±5 Å, which is coarser than
lDDT's 0.5–4 Å thresholds (all-boxes lDDT works out to 0.335 predicted vs 0.323
measured); and because Pass 2 re-shows a box only 10 % of the time, most
"refined" atoms are *first reads* at the schedule's σ=1 Å rather than converged
0.1 Å ones. The format's actual encoding ceiling is the ``tenths`` row —
lDDT 1.000, TM 1.000, RMSD 0.05 Å. Plan F breaks all three budget losses at
once, which is why it runs *above* the one-document ceiling on long chains.

`run_baselines.py` degrades the ground truth to each of the format's resolution
tiers and scores it with the same harness (`data/baseline_ceiling.csv`,
`plots/ceiling.png`). No model — this is what a *perfect* model would score.

| ground truth degraded to | atom cov. | lDDT | lDDT-CA | TM-score | all-atom RMSD |
|---|---|---|---|---|---|
| nothing (identity check) | 1.00 | 1.000 | 1.000 | 1.000 | 0.00 |
| 0.1 Å, all atoms | 1.00 | 1.000 | 1.000 | 1.000 | 0.05 |
| 10 Å box centers, all atoms | 1.00 | 0.323 | 0.327 | 0.511 | 4.99 |
| boxes + 50 % refined | 1.00 | 0.533 | 0.530 | 0.749 | 3.51 |
| boxes + 30 % refined | 1.00 | 0.419 | 0.418 | 0.651 | 4.16 |
| boxes + 15 % refined | 1.00 | 0.360 | 0.361 | 0.578 | 4.59 |
| **one realistic document** (65 % boxed, 25 % refined) | 0.65 | **0.167** | 0.166 | **0.406** | 4.29 |

1. **The 0.1 Å digit vocabulary costs nothing** (lDDT 1.000, TM 1.000). All the
   loss is coverage and box resolution.
2. **Pass 1 alone cannot produce a good structure.** Every atom at its *correct*
   10 Å box center still scores only lDDT 0.32 / TM 0.51.
3. **lDDT is quadratic in coverage, TM-score is linear** — an lDDT contact needs
   both of its atoms, a TM residue needs only itself. Clustering the coverage
   into whole 10 Å boxes barely helps (0.259 vs 0.250 at 50 % coverage), because
   lDDT's 15 Å inclusion radius spans several boxes.
4. **A single document tops out near lDDT 0.17 / TM 0.41.** Any inference plan
   that samples one document per protein is competing for that, and no result
   from such a plan could distinguish "the model cannot fold" from "the format
   did not get a chance". That is the central argument in [`PLANS.md`](PLANS.md).

The 65 %/25 % row uses the SPEC's own coverage table for a 150–500-residue chain
(30–70 % of atoms boxed, 12–40 % refined); our eval set's median length is 161.
Re-measuring it from real generated documents rather than the SPEC's summary is
a small, worthwhile follow-up.

### Inference: what the plans are, and what they cost

Component 1 shipped as `document_codec.py` (tokens ↔ coordinates), `sampler.py`
(two temperatures — coordinate `<xyz-DDD>` tokens vs structural choices — plus
explicit KV-cache reuse) and `plans.py`. Everything ran on CoreWeave RNO2A at
batch priority; ~57 single-H100 shards, models cast to bf16 before upload.

| plan | conditioned on | generated |
|---|---|---|
| **A** | sequence section only | contacts, Pass 1, Pass 2 — one free-running document |
| **C** | sequence + the model's own Pass 1 | one forced crop per occupied box, no neighbour context, K=1 |
| **F** | sequence + own Pass 1 + already-refined neighbours + own earlier visits | 2 spatially coherent sweeps, K=3 samples per crop |
| **E1** | sequence + **true contacts** (≤50, the format's cap) | everything downstream |
| **E2** | sequence + **true Pass-1 boxes** | Pass 2 only |
| **E3** | probe: a box shown *i* times, from ground truth | that box's crop |

A, C and F are fully de-novo: nothing is teacher-forced, and the model writes
its own contacts section before Pass 1. E1/E2/E3 are oracle diagnostics and are
never compared against a predictor.

### E3 — the model learned the refinement schedule

The format trains a box's *i*-th appearance with σ = 1/(i+1)² Å noise. Whether
the model conditions on visit count at all was the gate on Plan F's inner loop
(`data/probe_refinement.csv`, 720 trials):

| visit index | training σ (Å) | measured error (Å) |
|---|---|---|
| 0 | 1.000 | 2.11 |
| 1 | 0.250 | 1.13 |
| 2 | 0.111 | 0.44 |
| 3 | 0.063 | 0.25 |
| 4 | 0.040 | 0.16 |
| 5 | 0.028 | **0.13** |

Error falls 16× over six visits, tracking the schedule down to ~0.13 Å and then
flattening — right at the format's own 0.1 Å tenths floor. **The refinement
machinery works and is worth re-showing boxes for.**

### Model scores (554 proteins, mean; `data/scores_all.csv`, `plots/results.png`)

| run | atom cov. | refined | lDDT | lDDT-CA | TM-score | median CA-RMSD |
|---|---|---|---|---|---|---|
| *0.1 Å, all atoms* | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.05 |
| *10 Å box centres, all atoms* | 1.000 | 0.000 | 0.323 | 0.327 | 0.511 | 4.96 |
| **oracle document — 1-doc ceiling** | 0.769 | 0.319 | **0.290** | 0.300 | **0.537** | **4.16** |
| **E2** — true Pass-1 boxes | 0.772 | 0.321 | **0.278** | 0.291 | **0.522** | **4.33** |
| **F** — mix5, 2 sweeps, K=3 | **0.999** | **0.999** | **0.290** | **0.319** | 0.277 | 16.28 |
| **F** — 3way, 2 sweeps, K=3 | 0.998 | 0.998 | 0.294 | 0.323 | 0.279 | 16.48 |
| **C** — one forced sweep | 0.935 | 0.815 | 0.223 | 0.246 | 0.245 | 16.55 |
| **E1** — true contacts (≤50) | 0.754 | 0.317 | 0.161 | 0.169 | 0.222 | 12.96 |
| **A** — mix5 step-50000 | 0.758 | 0.313 | 0.141 | 0.158 | 0.193 | 16.62 |
| **A** — 3way step-20000 | 0.753 | 0.306 | 0.144 | 0.160 | 0.197 | 16.85 |

**Plan F works, and it escapes the token budget.** Coverage 0.999 and refined
fraction 0.999, against 0.31 for a single document. lDDT doubles Plan A
(0.141 → 0.290) and *equals the single-document ceiling*. Per length it goes
past that ceiling, because the ceiling collapses with chain length (fixed 8192
budget, growing atom count) while F just re-prompts:

| lDDT by length | ≤100 | 101–200 | 201–400 | >400 |
|---|---|---|---|---|
| oracle document (1 doc) | 0.519 | 0.313 | 0.158 | 0.062 |
| **F** | 0.338 | 0.297 | **0.267** | **0.164** |
| A | 0.258 | 0.153 | 0.075 | 0.018 |

F is **1.7× the one-document ceiling at 201–400 residues and 2.6× above it past
400**. That is the plan doing exactly what it was designed to do.

**But the fold is wrong, and no de-novo plan fixes it.** CA-RMSD is ~16.5 Å for
A, C and F alike (12.4 Å at ≤100 residues rising to 26 Å past 400), and TM-score
never passes 0.28. F makes a wrong fold complete and locally precise. The
decisive comparison is E2 against F: **same model, same refinement machinery,
less coverage — and TM 0.522 vs 0.277, CA-RMSD 4.33 Å vs 16.28 Å.** The only
difference is whether the coarse boxes are right.

**E1 says the contact map is not the whole story either.** Handing the model 50
true contacts cuts CA-RMSD 16.6 → 13.0 Å (−22 %) and lifts lDDT 0.141 → 0.161 —
real, and nowhere near E2's 4.33 Å. Caveat worth stating plainly: the format
caps a document at **50** contacts and samples them uniformly, so E1 could only
ever supply a small slice of a real contact map. It bounds the effect from
below; it does not exonerate contact prediction.

**The two checkpoints are indistinguishable, under both plans.** Under A,
0.141 vs 0.144 lDDT and 0.193 vs 0.197 TM. Under F, the paired per-protein
difference (3way − mix5, 554 proteins) is **lDDT +0.0044 ± 0.0027** and
**TM +0.0022 ± 0.0047** — 1.6σ and 0.5σ, i.e. nothing. The 3-way mixture restart
at step-20000 has caught up with mix5 at step-50000 and neither is better.

**Plan F had not converged.** Mean per-atom displacement between sweeps was
4.03 Å after sweep 0 and 2.02 Å after sweep 1, against a 0.1 Å stopping
threshold — it was still halving when the 2-sweep compute cap stopped it. The F
numbers are a lower bound on F.

### Interactive viewer

[`explore_predictions.ipynb`](explore_predictions.ipynb) — open in
[Colab](https://colab.research.google.com/github/Open-Athena/MarinFold/blob/claude/github-issue-174-0d4157/experiments/exp174_evals_structural_eval_contacts_and_crops_v1/explore_predictions.ipynb).
Superimposes prediction and experimental structure in 3D, coloured by the
resolution tier the document actually reached (blue = Pass-2 refined, orange =
Pass-1 box-only drawn at its box centre, grey = ground truth), with per-protein
metrics and a plan-comparison cell. Runs anonymously — no login, no token.

The committed outputs show all four plans on `denovo_pdb/7sq4_A` (L=48), which is
the whole result on one protein:

| plan | coverage | refined | lDDT | TM | CA-RMSD |
|---|---|---|---|---|---|
| A | 0.974 | 0.512 | 0.470 | 0.298 | 7.52 Å |
| F | 1.000 | 1.000 | 0.650 | 0.456 | 6.69 Å |
| E2 (oracle boxes) | 0.982 | 0.571 | 0.563 | **0.626** | **2.53 Å** |
| oracle document | 0.995 | 0.692 | 0.658 | 0.739 | 1.81 Å |

F wins on coverage and lDDT; E2 wins on the global metrics. Local precision and
global correctness come apart, and they come apart at Pass 1.

## Reproducing

```bash
cd experiments/exp174_evals_structural_eval_contacts_and_crops_v1
uv sync --extra test
bash setup_usalign.sh                       # builds _bin/USalign (needs g++ + network)
uv run python -m pytest tests -q            # 34 tests

# ground truth (needs the exp78 checkout's staged structures + pyconfind)
uv run python prepare_gt_structures.py --out-dir _scratch/gt
#   ... or pull the published bundle instead — see publish_gt_bundle.py

# the model-free ceiling
uv run python run_baselines.py --gt-dir _scratch/gt --work-dir _scratch \
    --out data/baseline_ceiling.csv
uv run python plot_ceiling.py

# scoring any predictor
uv run python score_structures.py --gt-dir _scratch/gt --pred-dir <preds> \
    --model-name <name> --out data/scores_<name>.csv
```

The ground-truth bundle is checkpoint-independent (69 MB, too big for git) and is
published to the public HF bucket by
[`publish_gt_bundle.py`](publish_gt_bundle.py) at
`buckets/open-athena/MarinFold/data/exp174-structural-eval/gt/`.

## Success criteria

1. An agreed inference approach, chosen from the Component-1 plans **after discussion**.
2. A working structure-prediction pipeline producing full-atom coordinates
   (PDB/mmCIF) per eval protein for a given checkpoint.
3. A scoring harness reporting CA-RMSD, all-atom RMSD, LDDT, LDDT-CA and
   TM-score over the 554-protein set, per-protein + aggregate, with coverage.
4. Both models scored and compared, with the length stratification and an
   honest account of coverage/truncation effects.
5. Results written up in the experiment README + a summary comment on this issue.

Status: **all five done.** Plan F agreed and implemented; the pipeline produces
full-atom coordinates per protein for a given checkpoint; the harness reports all
five metrics per-protein and in aggregate with coverage; both checkpoints are
scored and compared under both A and F, with length stratification and an
explicit coverage/truncation account; and the write-up is this README plus the
issue comment.

## Conclusion

**contacts-and-crops-v1 at 1.5B is not yet structure-capable de novo, and the
bottleneck is Pass 1 — the coarse fold — not Pass 2 refinement and not the
format's resolution.** Three independent measurements agree:

1. **The format's coordinate encoding is not the problem.** 0.1 Å quantization
   of ground truth scores a perfect 1.000 lDDT / 1.000 TM. A real generated
   document decodes to lDDT 0.290 / TM 0.537 / CA-RMSD 4.16 Å — that is the
   ceiling, and it is set by coverage and the 10 Å coarse tier, not by the digit
   vocabulary.
2. **The model's refinement is near-perfect given correct boxes.** E2 reaches
   96 % of the ceiling (lDDT 0.278 vs 0.290, TM 0.522 vs 0.537, CA-RMSD 4.33 Å
   vs 4.16 Å), and E3 shows it follows the σ=1/(i+1)² schedule down to the
   format's own 0.1 Å floor. Pass 2 works.
3. **Every de-novo plan is stuck at CA-RMSD ~16.5 Å and TM ≤0.28**, regardless
   of how much inference is spent. Plan F drives coverage and refined fraction
   to 0.999 and doubles lDDT — it beats the single-document ceiling 2.6× on long
   chains — and moves the global metrics barely at all. It produces a complete,
   locally precise, *wrong* structure.

The practical reading: **inference compute is not the lever, and neither is a
bigger `fine_reserve` in a v2 format.** The earlier ceiling analysis said the
Pass-2 refined fraction was "the whole ballgame"; that was right about the
*format's* ceiling and wrong about where this *model* sits relative to it. Plan
F already buys the refined fraction outright, and the fold does not improve.
Whatever comes next should target Pass 1 — the coarse spatial layout — with
training rather than decoding.

Two things worth carrying forward:

- **lDDT and TM-score come apart here, sharply, and reporting only one would
  mislead.** F is at the ceiling on lDDT and a third of it on TM. A local metric
  rewards a well-refined wrong fold; a global one does not. Any future
  contacts-and-crops eval should report both, with coverage.
- **E1's ≤50-contact cap is a live confound.** 50 true contacts bought a 22 %
  RMSD reduction. Whether a *full* contact map would fix the fold is untested
  and untestable in this format — which is itself an argument about the format,
  not just about the eval.
