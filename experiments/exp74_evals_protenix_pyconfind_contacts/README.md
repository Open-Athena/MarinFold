---
marinfold_experiment:
  issue: 74
  title: "exp: run protenix v2 and pyconfind on updated eval set (foldbench-100 + newly curated set in exp65)"
  kind: evals
  branch: exp/74-protenix-pyconfind-contacts
---

# exp: run protenix v2 and pyconfind on updated eval set (foldbench-100 + newly curated set in exp65)

**Issue:** [#74](https://github.com/Open-Athena/MarinFold/issues/74) · **Kind:** `evals` · **Branch:** `exp/74-protenix-pyconfind-contacts`

## Question

How well does Protenix v2 do at **contact prediction** when contacts are
defined with [pyconfind](https://github.com/timodonnell/pyconfind), on an
updated eval set (FoldBench-100 from [#12](https://github.com/Open-Athena/MarinFold/issues/12)
plus the low-MSA / structurally-novel set curated in
[#65](https://github.com/Open-Athena/MarinFold/issues/65))?

## Background

We're training models that only do contact prediction now, so the eval
focus shifts from distance/structure accuracy (exp12) to **contact
accuracy**, with extra attention to **low-MSA-depth** proteins. This
experiment extends exp12 in three ways (expand the eval set; define
ground truth with pyconfind; score Protenix contact prediction in four
configs) — see the issue text reproduced under "Approach (from the
issue)" below.

This builds on:
- **exp12** — Protenix v2 on Modal (single-seq + MSA), distogram capture,
  the `best/` top-1 picker. We **reuse its harness** and its already-computed
  FoldBench-100 outputs from the HF bucket (no re-run for FoldBench).
- **exp65** — the 454-candidate low-MSA / novel-fold set (de-novo, CASP-FM,
  CAMEO-hard) with per-candidate `neff_tier` / `fold_verdict` / `seq_leakage`
  labels we stratify on.
- **contacts_v1** (`marinfold.document_structures.contacts_v1`) — the
  pyconfind side-chain-contact recipe our training documents use; the
  ground truth here is defined identically.

## Approach (from the issue)

> Extend experiment #12 in three ways: (1) add the exp65 candidates to the
> FoldBench-100 set; (2) compute ground-truth contacts with pyconfind, run
> like contacts_v1 (`native_only=True`, cd threshold 0.001) — save all
> contacts, eval only on primary separation ≥ 6; (3) score Protenix in four
> configs = {single-seq, msa} × {distogram, structure}. Distogram: rank by
> cumulative probability the representative atoms are within ~8 Å. Structure:
> run pyconfind on the predicted structure (`native_only=True`) and rank by
> contact degree. Plot contacts @ L in aggregate and split short / medium /
> long. Use Modal. Save raw results to the HF bucket; document here.

## Method (reproduction spec)

### Eval sets
- **FoldBench-100** — the 100 proteins from exp12. We reuse exp12's Protenix
  outputs (`best/{mode}/{stem}/{structure.cif,distogram.npz}`) and GT CIFs
  directly from the HF bucket `open-athena/MarinFold`
  (`data/protenix-foldbench-monomers/`); **no Protenix re-run**.
- **exp65** — all **454** unique candidates from
  [`candidate_2d_label.csv`](../exp65_evals_low_msa_depth_proteins/data/candidate_2d_label.csv)
  (396 de-novo, 32 CAMEO-hard, 26 CASP-FM; the 3 stems shared between de-novo
  and CAMEO are de-duplicated). Input sequences come from exp65's
  `candidate_sequences.csv`. Lengths 33–460 aa (median 148), so nothing is
  excluded for size.

### Protenix run (Modal)
Reuses exp12's harness ([`modal_app.py`](modal_app.py),
[`distogram_hook.py`](distogram_hook.py), [`select_best.py`](select_best.py),
[`msa_depth.py`](msa_depth.py)), pointed at the exp65 inputs
([`prepare_exp65.py`](prepare_exp65.py)). Protenix v2 on H100, **5 seeds × 8
diffusion samples**, N_cycle=10, in both `single_seq` and `msa` mode
(`--use_msa true/false`); MSAs pre-computed once via Protenix's ColabFold
backend (same pipeline as FoldBench-100). Top-1 sample per (protein, mode)
by `ranking_score`; the distogram kept is the one from that seed. The
Protenix v2 weights volume is shared with exp12.

### Ground-truth contacts (pyconfind) — [`pyconfind_contacts.py`](pyconfind_contacts.py)
Run **exactly like contacts_v1** (verified against
`contacts_v1.GenerationConfig` defaults): `native_only=True`,
`contact_distance=3.0`, `dcut=25.0`, `clash_distance=2.0`, `assembly=None`.
We **save every contact pyconfind emits** (degree > 0); a **"true" contact**
for eval is one with **degree ≥ 0.001** (`min_contact_degree`, the issue's
"cd threshold of 0.001") **and primary-sequence separation ≥ 6**
(`min_seq_separation`). The single GT chain is extracted from the
(possibly multi-chain assembly) structure before analysis.

### The four configs and the metric — [`contact_eval.py`](contact_eval.py)
Ground truth is the same for all four configs; predictors differ:
- **distogram**: per-pair score = Σ distogram mass on Protenix-v2 bins with
  center ≤ **8 Å** = P(CB-CB ≤ 8 Å) (exp12's contact probability). 8 Å is the
  default; the saved distograms make a threshold sweep cheap.
- **structure**: run pyconfind on the predicted top-1 CIF (same knobs as GT),
  score = predicted **contact degree** (0 for unpredicted pairs).

**Metric — contacts @ L**: precision among the top-L ranked pairs (L =
sequence length), also @ L/2 and L/5, reported **in aggregate** (sep ≥ 6) and
**split** short [6,11] / medium [12,23] / long [≥24]. The candidate-pair
universe is restricted to residues **resolved in the GT structure**,
identically across all four configs, so the numbers are comparable.

> **Note on the metric.** pyconfind contacts are sparser than the CB-CB ≤ 8 Å
> contacts of classic CASP precision@L (≈0.6/residue here), so precision@L is
> bounded by contact density for short proteins (a perfect predictor caps at
> `n_true/L`); L/2 and L/5 are less affected. Also, the **distogram** configs
> rank by a CB-CB-distance notion against a side-chain-contact ground truth,
> so they carry an inherent representation gap vs the **structure** configs —
> this is the eval the issue asked for, and the structure configs are the
> apples-to-apples ones.

### Index alignment
pyconfind numbers contacts over a structure's *resolved* residues; the
distogram / predicted structure are indexed by the input sequence. We align
the two by difflib (decoupled from residue numbering) and remap every contact
to input-sequence coordinates, so GT, distogram, and predicted-structure
contacts share one `[0, L)` space. A per-protein **alignment identity** is
recorded; FoldBench-100 validation showed 1.000 across the board.

### Files
| File | Role |
|---|---|
| `pyconfind_contacts.py` | pyconfind GT/prediction wrapper + chain extraction + alignment |
| `contact_eval.py` | the 4-config contacts@L scorer (→ `contact_precision.csv`, `contacts_raw.parquet`, `contact_eval_meta.csv`) |
| `prepare_exp65.py` | build Protenix inputs + eval manifest from exp65 CSVs |
| `eval_manifest.py` | build the FoldBench-100 eval manifest from the HF sync |
| `fetch_gt_structures.py` | fetch exp65 GT structures (RCSB) for scoring |
| `modal_app.py`, `distogram_hook.py`, `select_best.py`, `msa_depth.py` | Protenix-on-Modal harness (from exp12) |
| `plot.py` | contacts@L plots (by config/range; stratified by neff_tier / fold_verdict; vs Neff) |
| `cli.py` | `run` / `select-best` / `contact-eval` |

## Success criteria

A single tidy `contact_precision.csv` covering FoldBench-100 + all 454 exp65
candidates, with contacts @ L / L2 / L5 for all four configs, aggregate and
by range; plots of the above (incl. stratified by MSA-depth tier and fold
novelty); raw pyconfind contacts + Protenix outputs on the HF bucket. We
expect MSA ≫ single-seq and the structure predictor ≥ distogram predictor
(confirmed on the FoldBench-100 validation subset).

## Results

_(Fill in after the full exp65 run + scoring completes.)_

## Conclusion

_(Fill in after results are in.)_
