---
marinfold_experiment:
  issue: 256
  title: 'exp: how many contacts should we hand Helico? sweep the cut past top-L, up to every pair any rollout proposed'
  kind: evals
  branch: claude/contact-probability-inference-eval-a2d2ea
---

# exp: how many contacts should we hand Helico? sweep the cut past top-L, up to every pair any rollout proposed

**Issue:** [#256](https://github.com/Open-Athena/MarinFold/issues/256) · **Kind:** `evals`

## Question

Does giving Helico more than top-L MarinFold contacts improve folding accuracy?
The existing rollout union contains many true contacts that a short vote-ranked
list omits. This experiment tests whether simply extending that list makes the
additional signal useful to the existing Helico checkpoint.

## Hypothesis

The recorded prediction was a shallow maximum around 1.5L–2L and lower lDDT by
5L and the full union. Increasing the cut changes precision, recall, and the
number of constraints together. It therefore tests a practical contact-list
policy, but cannot separately identify the effects of precision and recall.

The motivating exp254 coverage diagnostic uses **R**, the number of true
contacts; Helico's cuts use **L**, sequence length. They are different budgets.
The earlier version of this report incorrectly carried the approximately 0.52
recall at an R cut into its description of the top-L arm. All contact metrics
below now come from the actual lists passed to Helico.

## Approach

The original inference ran on 2026-08-21, using Helico exp14's target mapping
and runner. This revision audits saved results; it runs no new inference and
scores no held-out eval-test targets.

- MarinFold contacts: `marinfold-exp232-decontam-m2-p06-step145199`, from exp245's
  `fbmono-20260818-01` vote matrices. Ranking uses resolved-residue pairs with
  separation at least 6, descending vote score, and stable tie order.
- Cuts: L/5, L/2 and L reuse exp14. The new cuts are 1.5L, 2L, 3L, 5L and the
  union of positive-vote candidate pairs. New cuts stop at the positive-vote
  support, so a nominal 5L list may contain fewer than 5L pairs.
- Folding: Helico `contacts-msafree-01`, step 6000, six trunk recycles, three
  diffusion samples, no MSA, seed 42. The runner scores returned sample index
  zero, not an average over the three samples. Sampling and model settings in
  the old and new run manifests match. This is one stochastic run per arm;
  target bootstrap intervals do not measure variability across inference seeds.
- The original export checks precision at L, L/2 and L/5 against exp245's
  published values. The audit independently recomputes precision and recall
  from each delivered arm JSON, its token map and ground truth, checks the
  contact count against inference records, and uses the same target population
  for contact and structure metrics. A separate check reconstructed the ranked
  lists from the original dense matrices and found exact list equality for all
  768 eval-val arm/target pairs (eight cuts × 96 targets); this check is now
  included in `analyze.py` and the dense matrices are in the public inputs.

**Population:** eval-val has 97 targets. All eight contact-cut arms succeed on
96; `7pv5_A` has no verified MarinFold index map and is recorded as
`no_contacts`. The historical 95-target comparison also excludes `7t9r_A`
because the two Helico arms using Protenix-derived contacts have no contacts
for it. This was an
input omission in reference arms, not a failed Helico fold. We retain the
95-target table for comparison with the original report and also publish the
96-target contact-only comparison. Every missing arm/target and its recorded
status appears in `data/excluded_targets.csv`. Required arms, duplicate IDs,
missing result rows, invalid scores, and changed input hashes now fail loudly.

## Results

### Historical comparison, with contact metrics matched to the same 95 targets

| cut | pairs / L | precision | recall | lDDT | Δ vs top-L | pointwise 95% CI | better on |
|---|---:|---:|---:|---:|---:|---|---:|
| top-L/5 | 0.198 | 0.787 | 0.176 | 0.5638 | −0.0415 | [−0.0604, −0.0228] | 26% |
| top-L/2 | 0.499 | 0.664 | 0.371 | 0.5969 | −0.0084 | [−0.0184, +0.0009] | 44% |
| **top-L** | 1.000 | 0.493 | 0.548 | **0.6053** | — | — | — |
| **1.5L** | 1.499 | 0.382 | 0.636 | **0.6073** | **+0.0020** | [−0.0048, +0.0086] | 53% |
| 2L | 2.000 | 0.310 | 0.686 | 0.6044 | −0.0009 | [−0.0080, +0.0060] | 51% |
| 3L | 2.956 | 0.230 | 0.747 | 0.5999 | −0.0054 | [−0.0133, +0.0022] | 48% |
| 5L | 4.707 | 0.163 | 0.805 | 0.5918 | −0.0135 | [−0.0243, −0.0034] | 40% |
| **union** | 14.045 | 0.107 | 0.922 | **0.5808** | **−0.0245** | [−0.0346, −0.0146] | 25% |

Reference arms on those same 95 targets: Helico without contacts **0.3499**,
Helico with Protenix-v2 single-sequence contacts **0.3881**, Helico with
Protenix-v2 MSA-derived contacts **0.8342**, and Helico with oracle contacts
**0.8642**. These are all Helico folding outputs; the two Protenix names identify
where their contact lists came from, not direct Protenix structure scores.
The Helico model itself remains MSA-free in every arm. The original lDDT means and paired
intervals reproduce exactly; contact metrics change slightly because the old
report averaged them over 96 targets while averaging lDDT over 95.

### Independent coordinate rescoring

To check the measurements themselves, `score_coordinates.py` independently
reads the exported predicted PDBs and reference mmCIFs and recomputes lDDT on
CPU for **all 288 successful (target, arm) pairs** in top-L, 1.5L, and union.
Every prediction matches the original recorded atom count. This uses the
runner's pair-weighted all-atom definition: matched heavy-atom pairs with
reference distance strictly between 0.01 and 15 Å, scored at distance-error
thresholds 0.5, 1, 2, and 4 Å. Within-residue pairs and any matched cofactor
atoms are included, matching the original scorer. This is not CA-only lDDT.
The implementation uses a sparse neighbor search rather than Helico's dense
pair matrices and imports no Helico code or model.

The largest per-target absolute difference from saved lDDT is **0.0000333**,
and the mean absolute difference is **0.00000617**, consistent with exporting
coordinates rounded to 0.001 Å. On all 96 contact-available targets, coordinate
rescoring gives **1.5L − top-L = +0.0018971** and **union − top-L = −0.0243880**,
reproducing the saved-score results below. Per-target differences and paired
intervals are saved in `data/coordinate_validation.csv` and `.json`.

### Contact-only comparison, all 96 available targets

| cut | lDDT | Δ vs top-L | pointwise 95% CI |
|---|---:|---:|---|
| top-L | 0.6023 | — | — |
| 1.5L | 0.6042 | +0.0019 | [−0.0048, +0.0085] |
| 2L | 0.6016 | −0.0007 | [−0.0078, +0.0062] |
| 3L | 0.5972 | −0.0051 | [−0.0130, +0.0026] |
| 5L | 0.5889 | −0.0134 | [−0.0238, −0.0032] |
| union | 0.5779 | −0.0244 | [−0.0346, −0.0146] |

The conclusion is unchanged when the unrelated reference-arm omission is
removed. Full metrics, including L/5 and L/2, are in
`data/cut_sweep_contact_targets.csv`.

Intervals use 10,000 paired target-bootstrap draws, seed 256. They are
pointwise intervals, without correction for comparing several cuts. An
interval containing zero does not establish equivalence. In particular, the
1.5L comparison remains compatible with a small improvement as large as
approximately 0.0086 lDDT on the historical population. Its largest observed
mean does not establish a distinct optimum.

![lDDT versus contact cut](plots/contact_cut_sweep.png)

## Interpretation

**Simply increasing this vote-ranked cut has no demonstrated benefit.** The
largest observed mean is at 1.5L, but its advantage is uncertain. The much
wider 5L and union lists lower mean lDDT in these runs. Top-L is a reasonable
operating point under the tested settings; this experiment does not prove it
is optimal or that a modest improvement is impossible.

**The experiment does not show that recall is worthless or precision alone
explains folding accuracy.** Precision decreases while recall increases along
the entire sweep. In fact, lDDT rises from L/5 to L despite declining precision.
The larger-cut results show that adding these lower-ranked predictions fails
to exploit the union's extra true contacts under this conditioning scheme.
They leave open improvements to ranking, confidence weighting, training, or
selection of complementary constraints. The claim that the conditioning
channel cannot be a bottleneck was unsupported and has been removed.

**The union still preserves much of the benefit over no contacts.** On the
historical population, top-L gains 0.25540 lDDT over the no-contact reference;
the union loses 0.02453 of that gain, or **9.60%**, retaining **90.40%**. The
previous 4%/96% statements divided by total top-L lDDT instead of its gain over
the no-contact baseline. Mean per-target union precision is 10.65%; that
summarizes heterogeneous targets and is not a pooled false-to-true contact
ratio. Its precision is about 3.75-fold below the cited training floor of 0.4,
not an order of magnitude. The result demonstrates tolerance of these noisy
lists, without identifying the mechanism or a universal training limit.

## Reproduction and provenance

The public bundle is pinned in `data/artifacts.json`. From this experiment
folder, download it anonymously into a new directory and reproduce on CPU:

```bash
uv run --no-project ../exp254_evals_pairwise_seeded_rollouts/publish_to_hf.py fetch --manifest data/artifacts.json --out /tmp/exp256-inputs
uv run --locked python analyze.py --inputs /tmp/exp256-inputs --out data
uv run --locked python score_coordinates.py --inputs /tmp/exp256-inputs --out data
uv run --locked python -m unittest test_analyze -v
uv run --locked python plot_results.py --data data --out plots
uv run --locked python build_summary.py
```

The downloader verifies the archive and every extracted file by SHA-256.
To republish, use exp254's `publish_to_hf.py prepare --experiment 256` on the
`prepare_inputs.py` output, review its manifest, then use the `publish` command.

`prepare_inputs.py` extracts only eval-val records from the two original Helico
worktrees. `analyze.py --inputs <directory> --out data` consumes that portable
input directory and checks its recorded file hashes. The input directory
contains `targets.csv`, `token_map.json`, `gt_universe_scored.jsonl`,
`arms/mf_*.json`, all 12 result/error CSVs, timing CSVs, original run manifests,
and `provenance.json`. It also contains `dense/*.npz` for all 97 eval-val
targets, `gt/*.cif.gz`, and `predictions/<arm>/<target>.pdb.gz` for every
successfully scored eval-val arm/target. These are the returned sample-zero
structures originally scored, not three independent saved samples.
`score_coordinates.py --inputs <directory> --out data` reproduces the
independent coordinate audit on CPU; its default arms are top-L, 1.5L and union.
Original manifests describe the full source run before
extraction; their counts may include other eval sets. Only eval-val result
and ground-truth records are present in the extracted inputs.

Source Helico snapshots:

- New cuts: [422c75a759c411ffed2687b2f2d27f0976a8ffb6](https://github.com/Open-Athena/helico/tree/422c75a759c411ffed2687b2f2d27f0976a8ffb6),
  `experiments/exp14_foldbench_held_out_monomers/`, branch
  `claude/helico-contact-cut-sweep`.
- Original and reference arms: [4eb9b61b656569a17c61b9e424efc9c9216ffa8a](https://github.com/Open-Athena/helico/tree/4eb9b61b656569a17c61b9e424efc9c9216ffa8a),
  the same experiment directory, branch `claude/helico-marinfold-evals-155267`.

`data/provenance.json` records source paths and SHA-256 digests, populations,
bootstrap settings, and the retention calculation. Small per-target contact
metrics, structure metrics, exclusion records, and plot source CSVs are kept
in git. Plot regeneration uses `plot_results.py --data data --out plots`;
`build_summary.py` rebuilds `plots/summary.pdf` from saved plots and narrative.

## Conclusion

The existing runs support a narrow negative result: **giving this Helico
checkpoint substantially longer unweighted lists from the existing MarinFold
vote ranking does not improve mean folding accuracy, and 5L/the union reduce
it.** A small gain near 1.5L remains plausible. Neither recall as a useful signal
nor improved ways of selecting or conditioning on contacts has been ruled out.
