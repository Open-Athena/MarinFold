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
implementation, by design. Five approaches are written up in
[`PLANS.md`](PLANS.md) with tradeoffs, compute costs, failure modes and a
recommendation, for discussion on the issue before anything is built.

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

### Model scores

Not yet — Component 1 is gated on the plan discussion. The two
[#137](https://github.com/Open-Athena/MarinFold/issues/137) /
[#155](https://github.com/Open-Athena/MarinFold/issues/155) checkpoints are ready
to score as soon as an inference approach is agreed (`exp137-cc1mix5-…` has HF
exports through step-50000; the 3-way restart through step-10000).

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

Status: **3 done** — the harness is built, tested and validated against a
measured ceiling. **1 open for discussion** ([`PLANS.md`](PLANS.md)). **2, 4 and
5 blocked** on that discussion.

## Conclusion

_(Fill in once the models are scored.)_

The ceiling result already stands on its own, though, and is worth carrying into
the format design as much as into this eval: **the Pass-2 refined fraction is the
entire ballgame.** Going from 15 % to 50 % of atoms refined moves the achievable
lDDT from 0.36 to 0.53 and TM-score from 0.58 to 0.75, while the 0.1 Å digit
encoding contributes no loss at all and the coarse boxes contribute nothing
beyond a 0.32 lDDT floor. Anything that raises that fraction — a bigger
`fine_reserve` in a v2 format, or an inference plan that re-prompts for more
crops — buys more than any other lever available.
