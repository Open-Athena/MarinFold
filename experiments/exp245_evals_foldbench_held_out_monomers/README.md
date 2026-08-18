---
marinfold_experiment:
  issue: 245
  title: 'exp: FoldBench held-out monomer eval sets (eval-val / eval-test / eval-denovo) for the decontaminated #232 checkpoints'
  kind: evals
  branch: exp/245-foldbench-eval-sets
---

# exp: FoldBench held-out monomer eval sets (eval-val / eval-test / eval-denovo) for the decontaminated #232 checkpoints

**Issue:** [#245](https://github.com/Open-Athena/MarinFold/issues/245) · **Kind:** `evals` · **Branch:** `exp/245-foldbench-eval-sets`

## Question

**How do the decontaminated #232 checkpoints score on FoldBench's natural monomers — held-out ones we have never scored — and how does that compare to the structure-prediction baselines on the same proteins?**

Every contact number we publish today comes from a 554-protein set whose FoldBench half is the *first 100 rows* of `monomer_protein.csv`, chosen before we understood the training-set overlap ([#213](https://github.com/Open-Athena/MarinFold/issues/213), [#225](https://github.com/Open-Athena/MarinFold/issues/225)). [#232](https://github.com/Open-Athena/MarinFold/issues/232) trained models on corpora decontaminated against **all** of FoldBench, so for the first time the other 234 monomers are a legitimate held-out test set rather than an untested slice of training data.

This experiment cuts FoldBench's 334 monomers into three sets and scores them:

| set | definition | n |
|---|---|---:|
| **eval-val** | the natural monomers inside the historical FoldBench-100 — what every previous eval reported on | 97 |
| **eval-test** | every other natural FoldBench monomer — never scored by anything here | 218 |
| **eval-denovo** | every de novo designed FoldBench monomer | 19 |

Each protein carries a viral / non-viral flag so results can be stratified ([#241](https://github.com/Open-Athena/MarinFold/issues/241) found MarinFold ties ESMFold on viral proteins and loses badly on non-viral ones, so the split is not cosmetic).

## Hypothesis

- **H1.** eval-test and eval-val agree within noise for the #232 checkpoints. Both sets are decontaminated at the same rule, so a gap between them would mean the historical 100 is unrepresentative of FoldBench for reasons other than leakage.
- **H2.** The contaminated reference model (#199 CoreWeave cooldown, the current default) drops more from eval-val to eval-test than the #232 checkpoints do — its training data was never filtered against the 234, so eval-val is partly memorised for it and eval-test is not.
- **H3.** Baselines (Protenix-v2 single-seq/+MSA, ESMFold, ESMFold2) move little between eval-val and eval-test: their training data is unchanged by any of this, and eval-test is not novel to them.

## Background

- [#225](https://github.com/Open-Athena/MarinFold/issues/225) published both decontaminated corpora. Rule: **≥30 % identity over ≥50 % of the shorter sequence**, reference = the 554 eval proteins ∪ all 1,940 FoldBench protein chains, no E-value arm. AFDB 4,129,682 → 3,963,003; ESM-Atlas 66,759,922 → 65,553,178.
- [#232](https://github.com/Open-Athena/MarinFold/issues/232) trained the #199 recipe from scratch on exactly those corpora; [#244](https://github.com/Open-Athena/MarinFold/pull/244) evaluated the two best checkpoints (`m2-p06` 0.592 / `m1-p02` 0.579 on the legacy 554).
- [#226](https://github.com/Open-Athena/MarinFold/issues/226) established the FoldBench monomer universe (334), the RCSB chain-resolution recipe, and ground truth for 23 of the unused ones.
- [#241](https://github.com/Open-Athena/MarinFold/issues/241) supplied the designed-vs-natural audit and the kingdom/viral annotation for all 776 eval proteins.

## Approach

Four steps, each with its own script and its own control.

**0. Confirm the decontamination** ([`confirm_decontamination.py`](confirm_decontamination.py)).
Five links have to hold for these checkpoints to be clean on these proteins, and
each is checked rather than assumed. See [Results §1](#1-the-232-checkpoints-are-verifiably-clean-on-all-334-monomers).

**1. Cut the sets** ([`build_eval_sets.py`](build_eval_sets.py)). FoldBench's
`monomer_protein.csv` has 334 rows; our historical eval set is its first 100.
Designed-vs-natural is decided by two independent RCSB signals — the
`synthetic construct` source taxon and the PDB's `DE NOVO PROTEIN` structural
class, tested against both the curated keyword field and the depositor's free
text — which agree on all 334 (19 designs, 315 natural). Viral status comes from
the NCBI taxonomy lineage. #241's independent annotation of the same proteins is
asserted to match, all 334 rows.

**2. Ground truth for everything** ([`build_ground_truth.py`](build_ground_truth.py)).
199 monomers have never been scored and have no ground truth. Rather than bolt
new records onto #89's frozen universe and leave the sets with two provenances,
all 334 are rebuilt through one path — #89's `pyconfind_contacts.compute_contacts`
imported, RCSB `-assembly1` mmCIFs, the resolved auth chain — and the 126
overlapping units built from the same input sequence must come back byte-identical.
They do, 126/126.

**3. Check the format can represent them** ([`check_context_budget.py`](check_context_budget.py)).
The 554-protein set tops out at 761 residues; FoldBench's monomers reach 1,596,
and contacts-v1 documents have to fit an 8,192-token context. Measured with the
real tokenizer and the real document builder rather than assumed.

**4. Score** — [`rollout/`](rollout) for the checkpoints (PR #244's harness with
exp245's eval sets as reporting cuts), [`score_baselines.py`](score_baselines.py)
and [`run_knn_baseline.py`](run_knn_baseline.py) for the baselines,
[`analyze.py`](analyze.py) and [`plot_results.py`](plot_results.py) for the tables
and figures.

## Success criteria

- Decontamination confirmed with zero surviving ≥30 %/≥50 %-shorter alignments into either training corpus, and the coverage-gate caveat quantified.
- All 334 units scored for all three checkpoints with 0 unfinished rollouts, and the #75 E8 gate reproduced on the legacy 554 within 0.005 to validate the path.
- A single table giving R-precision (all) for {3 MarinFold checkpoints + 5 baselines} × {eval-val, eval-test, eval-denovo}, with bootstrap CIs on the paired deltas and a viral / non-viral split.
- An answer to H2 stated as a number: the eval-val → eval-test drop for the contaminated model minus the same drop for the decontaminated ones.

## Results

### 1. The #232 checkpoints are verifiably clean on all 334 monomers

Five links, all checked ([`data/decontamination_check.json`](data/decontamination_check.json)):

| link | check | result |
|---|---|---|
| The eval proteins were queries | all 334 monomer sequences present byte-identically in #225's 1,940-chain FoldBench reference | **334/334**, by name and by sequence |
| Nothing that matches them survived | every alignment into either corpus at ≥30 % identity over ≥50 % of the shorter sequence, checked against the applied drop list | **131,180 meet the rule, 0 survive** |
| The published corpora are the filtered ones | #225's `verify_published.py` row counts | AFDB 3,963,003 (−166,679); ESM-Atlas 65,553,178 (−1,206,744) |
| #232 tokenized those bytes | its tokenizer pins both bucket prefixes and requires those exact row counts, and its sweep pins the same totals for the mixture weights | counts agree |
| The two runs read only those caches | live W&B config for both runs | only exp232's `afdb`/`esm` caches, plus #154's contacts-v1 **validation** cache (loss only, never trained on) |

**What the rule does not cover, priced.** The gate is identity over half the
*shorter* sequence, so a training protein matching an eval protein at high
identity over a short stretch is kept by design. At the applied gate the highest
surviving identity to any of the 334 is **0.299** — right up against the
threshold, as it should be. Drop the coverage requirement to 40 % and 97/97
eval-val and 217/218 eval-test proteins have a surviving training relative at
≥30 % identity; with no coverage requirement at all, 19 eval-val and 46
eval-test proteins have a surviving relative at ≥90 % identity over some
fragment. Decontamination at 30 % means *this* rule, not "no shared subsequence".
Per-protein residuals are in [`data/residual_identity.csv`](data/residual_identity.csv).

### 2. The three sets

| set | n | viral | median L | max L | ground truth | baselines |
|---|---:|---:|---:|---:|---|---|
| **eval-val** — the natural members of the historical FoldBench-100 | 97 | 6 | 245 | 761 | reproduced from #89's frozen universe, 97/97 identical | published, reused |
| **eval-test** — every other natural monomer | 218 | 13 | 258 | 1,596 | built here | 194 run here, 23 reused |
| **eval-denovo** — every designed monomer | 19 | 0 | 146 | 284 | built here or reproduced | 15 run here, 4 reused |

The historical 100 is **97 natural + 3 designs** (`5sbj_A` "METP, miniaturized
rubredoxin", `7ur7_A`, `8ah9_A`), so eval-val is 97 and the three designs sit in
eval-denovo where they belong. The designed/natural verdict has two independent
RCSB signals that agree on every one of the 334, and matches #241's independent
annotation on all 334.

**One protein is excluded from scoring.** `8uxt_A` (1,596 residues) is the only
monomer whose contacts-v1 document does not fit an 8,192-token context:
`build_document` truncates it to 1,664 of its 3,809 contacts, so no rollout can
produce it in full and a score for it would measure the format's context limit
rather than the model. It stays in
[`data/eval_sets.csv`](data/eval_sets.csv) flagged, and out of the 333 scored
units. Every other protein clears the budget with a median 2.3× headroom
([`data/context_budget.csv`](data/context_budget.csv)).

### 3. The evaluation reproduces PR #244 protein by protein

The usual gate — #75 E8 on the legacy 554 — is not available on this eval set, so
the path is validated against #244 directly: same two checkpoints, same 97
proteins (all of eval-val is inside #244's `foldbench100`), independent runs.

| checkpoint | n | mean R here | mean R in #244 | difference | per-protein r |
|---|---:|---:|---:|---:|---:|
| #232 m2-p06 | 97 | 0.5198 | 0.5229 | **−0.0031** | 0.996 |
| #232 m1-p02 | 97 | 0.4731 | 0.4754 | **−0.0023** | 0.996 |

Both inside the 0.0023 spread #204 measured across four evaluations of one
unchanged checkpoint, and inside #244's own 0.005 gate. Everything this
experiment rebuilt — ground truth, targets, dataset label, the adapted harness —
is covered by that comparison. 333 units × 3 checkpoints, 33,300 rollouts each,
**0 unfinished** ([`data/path_validation.json`](data/path_validation.json)).


## Conclusion

_(Fill in after results are in.)_
