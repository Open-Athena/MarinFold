---
marinfold_experiment:
  issue: 335
  title: 'exp: Helico confidence for Rosetta decoy ranking'
  kind: evals
  branch: codex/helico-decoy-ranking
---

# exp: Helico confidence for Rosetta decoy ranking

**Issue:** [#335](https://github.com/Open-Athena/MarinFold/issues/335) · **Kind:** `evals` · **Branch:** `codex/helico-decoy-ranking`

## Question

Can Helico's own confidence rank native and near-native protein structures above Rosetta decoys when each candidate is converted to Helico's contact-map representation, and how does it compare with AF2Rank, DeepAccNet, and the Rosetta energy function on the exact AF2Rank benchmark?

## Hypothesis

A candidate-derived contact map that is geometrically compatible with the target sequence should let contact-conditioned Helico produce a confident structure, whereas inconsistent decoy contacts should reduce Helico pTM. We therefore expect target-wise Helico pTM to correlate positively with candidate TM-score and recover high-quality candidates substantially better than Rosetta energy, but probably below AF2Rank because the contact map discards most of the template geometry available to AlphaFold. The native-versus-decoy discrimination question and the standard decoy-quality ranking question will both be reported.

## Background

AF2Rank ([paper](https://doi.org/10.1103/PhysRevLett.129.238101), [code and data](https://github.com/jproney/AF2Rank)) evaluates 133 protein targets from the Rosetta decoy set. It masks template sidechains, replaces the template sequence with gap tokens, runs MSA-free AlphaFold model 1 with one recycle, and ranks candidates with `mean_pLDDT * pTM * TM(candidate, AF2 output)`. The paper reports mean target-wise Spearman correlation 0.925 with candidate TM-score and mean top-1 TM-score 0.933, versus 0.831 / 0.917 for DeepAccNet and 0.760 / 0.901 for Rosetta.

Data access has been verified before filing this issue:

- The public IPD archive `https://files.ipd.uw.edu/pub/decoyset/decoys.zip` is reachable and is 5,179,742,249 bytes. The local copy has SHA-256 `098a3d813b74588bfe126cb1ac5c27b3123ff70e187d636ffdb66b5dc4a57733` and is fully extracted.
- The benchmark contains 133 targets and 180,079 decoy PDBs (794--2,903 per target; median 1,000), plus native PDBs. All 180,079 `(target, decoy_id)` keys exactly match the local TM-score, RMSD, and Rosetta-score tables.
- The authors' corrected `rosetta_gapseq.csv` downloads without authentication (24,695,656 bytes; SHA-256 `ddb3b91c27561212fa9152df4a4a436b9d01990cfe801569d7adc7f925fb75c9`). It contains exactly the same 180,079 decoys plus one `native` and one `none` row for each target (180,345 rows total), including AF2Rank, DeepAccNet, and Rosetta baseline scores.

Helico exp311 established that the published contact-conditioned checkpoint exposes pTM, pLDDT, and per-sample ranking scores and can be run reproducibly on Modal H100s. This experiment pins Helico source revision `b10385d736673c81b10e70d1099962af6f2573c0` and checkpoint `contacts-msafree-01-step6000.pt` (SHA-256 `779d540e5bb45cd26bb970188da2f1937ed3079942923a1356c29d06f20fe644`) unless a later preregistered comparison is added.

## Approach

Interpret "confidence map" as the candidate-derived **contact map**, because that is Helico's trained conditioning interface.

1. **Freeze and audit the benchmark.** Add a reproducible manifest builder that validates archive/reference hashes, target membership, native coverage, and exact key equality across PDBs and the corrected AF2Rank table. Raw PDBs remain outside git; small manifests and summaries are committed.
2. **Convert candidates without silent index shifts.** Parse each candidate, align its resolved protein residues to the target/native sequence, require an unambiguous residue mapping, and derive Helico's three-state map with the pinned pyconfind geometry (`native_only=True`, 3.0 A contact distance, 25 A cutoff, 2.0 A clash distance, contact degree >=0.001, intra-chain sequence separation >=6, no assembly expansion). Preserve both PRESENT and ABSENT states for the primary full-map condition. Record every exclusion and reason. A present-only map is a preregistered sensitivity analysis because it matches the sparse deployment interface but throws away candidate non-contact information.
3. **Pilot before scaling.** Select a deterministic, target-balanced pilot spanning short/median/long targets and low/median/high decoy counts, always including each native and decoys across the TM-score range. Verify map extraction, native reconstruction, deterministic reruns, adaptive batch size, and duplicate-map rate. Benchmark actual H100 seconds/candidate and publish the full-run cost projection.
4. **Helico inference.** Use the MSA-free checkpoint, six trunk recycles, three diffusion samples, and a fixed target-derived seed shared across candidates. The primary candidate score is the maximum pTM among the three samples (identical to Helico's monomer ranking score). Also save mean pLDDT and all per-sample values. Preregistered secondary scores are mean sample pTM and an AF2Rank-analog composite `pTM * TM(candidate, Helico output)`; the latter is explicitly not a pure confidence metric. Capture per-candidate timings and worker metadata using the repository timing schema.
5. **Full evaluation if the pilot is valid and affordable.** Run all 180,079 decoys plus the native for all 133 targets, resumably and batched within target. Do not launch the full run until the pilot reports measured cost: extrapolating exp311 one-candidate inference naively gives roughly 800 H100-hours, so batching and contact-map deduplication are material design requirements.
6. **Compare on paired candidates.** Recompute AF2Rank's published composite from the corrected table and include DeepAccNet and sign-corrected Rosetta energy. Primary decoy-ranking endpoints are mean target-wise Spearman correlation with reference TM-score and mean TM-score of the top-ranked decoy, matching the paper. Native-discrimination endpoints are native rank percentile, top-1 native recovery, mean reciprocal native rank, and per-target native-vs-decoy AUROC. Also report top-1 GDT-TS/TM-score regret, bootstrap 95% target-level intervals, and candidate-level calibration plots without pooling targets for rank correlations.
7. **Robustness and leakage checks.** Report full-map versus present-only maps, pTM versus mean pLDDT, one versus three samples on the pilot, performance by target length/decoy quality, and the unconditioned Helico confidence as a per-target negative control. Keep primary conclusions on the preregistered pTM score; label all secondary analyses.

## Success criteria

- Reproducible audit accounts for all 133 targets, 180,079 decoys, 133 benchmark natives, and all corrected AF2Rank baseline rows with no unexplained key mismatch.
- Pilot has zero silent mapping failures, finite confidence for every retained candidate, deterministic scores within numerical tolerance on rerun, and a measured full-run cost/time estimate.
- On the full paired benchmark, report Helico and every baseline for all primary and native-discrimination endpoints with target-bootstrap 95% intervals and explicit coverage/exclusions.
- The hypothesis is supported if Helico pTM exceeds Rosetta's 0.760 mean target-wise Spearman correlation and 0.901 mean top-1 TM-score; AF2Rank's 0.925 / 0.933 is the stronger comparison, not an assumed pass threshold.
- Per-candidate timing rows and a run manifest pin data hashes, Helico git SHA, checkpoint hash, inference settings, software versions, and hardware.

## Results

### The exact AF2Rank benchmark is available and internally consistent

`audit_data.py` verified the 5.18 GB public archive and the authors' corrected
results by SHA-256. All 180,079 decoy PDB identifiers across 133 targets match
the TM-score, RMSD, Rosetta-energy and corrected AF2Rank tables exactly. The
corrected table also contains one native and one no-template control per target,
for 180,345 rows total. Target lengths span 53--223 residues and each target has
794--2,903 decoys (median 1,000). The committed
[`benchmark_manifest.json`](data/benchmark_manifest.json) and
[`target_summary.csv`](data/target_summary.csv) freeze that identity and
coverage.

The analysis code independently reproduces the paper's full-set numbers from
the corrected table:

| method | mean target-wise Spearman with TM-score | mean top-1 TM-score |
|---|---:|---:|
| AF2Rank composite | 0.9253 | 0.9328 |
| DeepAccNet | 0.8308 | 0.9167 |
| Rosetta energy | 0.7590 | 0.9010 |

### Full-contact-map pilot: Helico confidence is competitive but below AF2Rank

The bounded pilot selects nine targets on a 3 x 3 grid of target-length and
decoy-count percentiles. Within each target it includes the native and 24
decoys evenly spaced across the reference TM-score order, retaining both the
worst and best decoy. All 225 PDBs mapped exactly to their target sequence.
Pyconfind extraction produced 224 unique maps within target; only one pair of
decoys shared a map, so deduplication will not materially reduce the full run.

The [Modal run](https://modal.com/apps/open-athena/main/ap-a24gF08NVAYXFqRkpn6vaP)
completed 225 candidates x 3 diffusion samples with no failed or nonfinite
predictions. Results use the full three-state map (PRESENT and ABSENT), six
recycles, and the same target-derived seed reset for every candidate. Intervals
below are 95% target-bootstrap intervals over nine targets. These are pilot
estimates, not the final 133-target result.

| ranking metric | mean Spearman with TM-score | mean top-1 TM-score |
|---|---:|---:|
| **Helico pTM (primary)** | **0.8519** [0.8013, 0.8957] | 0.9244 [0.8896, 0.9567] |
| Helico mean CA pLDDT | 0.8626 [0.8157, 0.8995] | 0.9341 [0.9023, 0.9611] |
| Helico composite | 0.8895 [0.8363, 0.9349] | **0.9577** [0.9360, 0.9740] |
| AF2Rank composite | **0.9436** [0.9204, 0.9654] | 0.9515 [0.9368, 0.9637] |
| DeepAccNet | 0.8706 [0.7968, 0.9318] | 0.9512 [0.9325, 0.9692] |
| Rosetta energy | 0.8203 [0.7542, 0.8760] | 0.9557 [0.9383, 0.9716] |

The primary pure-confidence result supports the basic signal: Helico pTM tracks
candidate quality and exceeds Rosetta's rank correlation on this pilot. It does
not match AF2Rank. The secondary Helico composite, `pTM *
TM(candidate, Helico output)`, closes part of that gap and selects a slightly
better mean top-1 decoy here, but it is not a pure confidence metric and must
remain labelled separately.

![Pilot Spearman comparison](plots/pilot_spearman_comparison.png)

### Native identification is harder than ranking decoy quality

Pure Helico pTM ranks the native first for 3/9 targets, with mean native rank
3.22 of 25, mean reciprocal rank 0.532, and mean native-vs-decoy AUROC 0.907.
AF2Rank's composite ranks the native first for 5/9 (mean rank 1.44). The Helico
composite ranks it first for 7/9 (mean rank 1.22), suggesting that agreement
between the conditioned output and the candidate carries useful information
beyond confidence alone. DeepAccNet and Rosetta are omitted from native
recovery because the authors' native rows contain `-1` sentinels rather than
scores for those methods.

![Helico pTM against candidate TM-score](plots/pilot_ptm_vs_tmscore.png)

### A naive full run is too expensive without batching

The 225 predictions used 0.544 inference H100-hours, equivalent to $2.15 at the
planning rate of $3.95/H100-hour, excluding model load, startup and idle time.
Mean inference time was 8.71 seconds/candidate. Direct extrapolation to 180,079
decoys plus 133 natives is **436 H100-hours, about $1,722 of inference compute
and 54.5 hours wall time at eight continuously busy H100s**. A target-bootstrap
projection is 424--449 H100-hours; it still excludes startup and the benchmark's
longest proteins because the pilot spans 61--150 residues.

The next engineering gate is therefore same-target batching: share tokenization
and amortize trunk work across multiple contact maps while retaining all
candidate-level outputs. The full run will not launch from the one-candidate
runner. After batching is benchmarked, the full-run estimate and any remaining
cost decision will be posted to issue #335.

## Conclusion

Interim pilot conclusion: candidate-derived contact maps contain enough signal
for Helico confidence to rank structural quality, but pure pTM is not yet as
strong as AF2Rank and identifies the exact native only one-third of the time in
this small target-balanced pilot. Candidate/output agreement is a promising
secondary discriminator. Data access and provenance are fully resolved; the
remaining blocker to the 133-target evaluation is efficient batched inference,
not benchmark availability.
