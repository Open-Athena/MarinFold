---
marinfold_experiment:
  issue: 278
  title: 'exp: diversify contacts-v1 with Proteina-generated monomers'
  kind: data
  branch: exp278/proteina-pilot
---

# exp: diversify contacts-v1 with Proteina-generated monomers

**Issue:** [#278](https://github.com/Open-Athena/MarinFold/issues/278) · **Kind:** `data` · **Branch:** `exp278/proteina-pilot`

## Question

Can fold-conditioned Proteina generation supply one million useful, structurally diverse, sequence-paired monomer documents of length 60–500, at an acceptable cost per retained structure?

## Hypothesis

Length-stratified, fold-balanced generation followed by sequence design, refolding and structural selection will increase structural coverage relative to unconditional generation. Whether that coverage improves MarinFold must be tested with a fixed-token training comparison; generating more documents alone does not establish value.

## Background

- [Proteina repository](https://github.com/NVIDIA-BioNeMo/proteina), inspected at commit `a44b407daf6a5358e43cd68907f3e3f1cbc65fdc`. Its output is a Cα trace, not an amino-acid sequence or a complete backbone.
- [Published efficiency measurements, Appendix C](https://arxiv.org/html/2503.00710v1#A3): compiled, maximally batched A100-80GB inference, 400 sampling steps. The 200M model without triangle layers is the initial throughput candidate; the long-chain checkpoint uses the same architecture but needs its own benchmark.
- [Upstream designability code](https://github.com/NVIDIA-BioNeMo/proteina/blob/main/proteinfoundation/metrics/designability.py) uses Cα-only ProteinMPNN and ESMFold. Its default evaluates eight sequences per backbone and reloads ESMFold inside each per-backbone call. Production must use persistent workers and explicit sequence-attempt counts.
- MarinFold references: [contacts-v1](https://github.com/Open-Athena/MarinFold/tree/main/marinfold/marinfold/document_structures/contacts_v1), [exp78 ESMFold timings](https://github.com/Open-Athena/MarinFold/blob/main/experiments/exp78_evals_esmfold_contacts/data/timings.csv), [exp225 decontamination](https://github.com/Open-Athena/MarinFold/issues/225), [exp245 evaluation splits](https://github.com/Open-Athena/MarinFold/issues/245).

## Approach

### 1. Pilot before committing to the million

Begin with 1–2 H100s for installation, end-to-end correctness and memory tests. Pin source, checkpoint checksums, CUDA/PyTorch and container digest. Enable compilation through a supported wrapper or small explicit source change; no monkey-patching. Use fixed-length batches and keep all models warm.

Budget up to **100 H100-hours** for a roughly 10,000-candidate pilot, terminating at the resource cap if measured costs are higher. Screen configs on a small subset first; spend the remaining pilot on the strongest candidates. Include 60, 100, 200, 300, 400 and 500 aa plus boundary tests around 250–300.

Compare:

- 200M no-triangle standard checkpoint for short proteins and the long-chain checkpoint for longer ones. Test their overlap instead of assuming an arbitrary cutoff is valid.
- Default unconditional sampling versus balanced C/A-level conditioning, then feasible T-level conditioning. The long-chain release supports a restricted A/T label set; do not force every fold at every length.
- Published noise defaults and one higher-noise candidate to measure the diversity/designability tradeoff. Keep 400 sampling steps for the reference; only reduce steps after checking quality.
- A small 400M triangle-model quality control arm. Choose by accepted diverse structures per GPU-hour, not raw designability alone.
- One versus up to two ProteinMPNN/ESMFold attempts on a matched subset. Report first-attempt and best-of-two acceptance separately.

Capture per-input timings and the repository's required worker metadata at inference time, including checkpoint/seed, batch size, actual batch latency, amortized GPU time, model-load time, stage time, GPU, hostname and versions. Record p50/p95, peak memory and yield by length/fold bin. Every W&B run uses `open-athena/MarinFold` and gets a history file immediately.

### 2. Produce usable sequence–structure pairs

Preferred path:

`Proteina Cα trace → Cα-only ProteinMPNN → ESMFold → self-consistency/geometry checks → pyconfind → existing contacts-v1 serializer`

ESMFold supplies N/CA/C/O and sequence identities needed by the existing native-AA rotamer contact calculation. Retain original Cα coordinates and the accepted refolded structure, explicitly naming which geometry supplies contact labels. The first corpus uses **refolded geometry**; measure diversity after this step. Reusing the upstream per-backbone ESMFold loader would waste the throughput estimate, so write persistent stage workers and disable the stock eight-sequence designability loop.

A cheaper optional arm reconstructs a complete backbone from Cα coordinates and adds designed residue identities, with ESMFold only on an audit sample. That arm would preserve generated geometry more directly but requires reconstruction validation and has weaker sequence–structure assurance. Keep it separate; do not silently mix its labels into the refolded corpus.

Initial acceptance criteria: finite coordinates, one complete chain of the intended length, no severe clashes or chain breaks; Cα self-consistency RMSD ≤2 Å over the full chain and mean ESMFold pLDDT ≥70. These are proposed thresholds, not a claim of experimental folding or verified monomeric state. Record failures; do not label generated PDB B-factors as pLDDT. Validate the checks in every length/fold stratum.

### 3. Make diversity an explicit sampling and retention objective

- Maintain quotas over length and broad structural classes: mostly alpha, mostly beta and mixed alpha/beta. Start with equal class quotas where supported and measure realized assignments independently of requested labels.
- Within feasible length/class combinations, favor underrepresented architectures/topologies relative to the current training corpus. Keep an unconditional exploration arm. Reallocate future sampling to underfilled bins using observed acceptance, while keeping the target distribution fixed and documented.
- Cluster accepted structures using Foldseek prefiltering followed by TM alignment for candidate neighbors. Proposed redundancy definition: TM-score ≥0.8 in both normalizations and ≥80% coverage of both chains. Cap retention at five per fine structural cluster. Calibrate prefilter recall on a small exhaustive comparison; do not perform a trillion-pair all-vs-all alignment.
- Measure unique-cluster fraction, effective cluster count `exp(entropy)`, C/A/T occupancy, secondary-structure composition and nearest-training-structure TM-score distributions. Report broader fold-level comparisons separately from fine redundancy clusters. One million structures does not mean one million new folds.
- Measure both pre-refolding and post-refolding diversity and compare to an equal-size, length-matched unconditional control and an equal-size training-corpus sample. A novel sequence or low sequence identity is not evidence of structural novelty.
- Apply exp225's frozen sequence reference and rule (≥30% identity over ≥50% of the shorter sequence, including the full FoldBench reference). Add a preregistered structural near-duplicate screen against evaluation structures using the fine redundancy definition above. Record all hits and exclusions; avoid claims of complete fold-disjointness. Keep all variants of a backbone/cluster together in any internal split.

### 4. Scale in resumable shards

After the pilot, freeze the selected recipe and predicted cost. Run a **100,000-accepted-design milestone** before completing the million, because duplicate rejection can increase with corpus size.

Use root Iris jobs at batch priority, each with one GPU and independent length-bucket work. Start with 32 H100 workers on RNO2A; separate sampling, folding and CPU queues so each model remains resident and the slow stage gets the appropriate share. Balance estimated shard time, not only row count. Persist a manifest and deterministic seeds; retry only unfinished shards and preserve explicit rejection reasons.

Use the established CoreWeave S3/fsspec path, under one experiment prefix such as `s3://marin-us-east-02a/MarinFold/exp<N>-proteina/`; verify endpoint/locality before bulk I/O. Avoid routing inputs or outputs through the workstation or GCS. Regional movement above 10 GB requires the repository's explicit sign-off; no cross-region transfer is authorized by this proposal. Benchmark document generation on co-located CPU capacity and cache the pyconfind rotamer library per worker. Read the Zephyr performance skill before authoring any `map_shard` pipeline.

Store compressed, sharded structures/coordinates and documents, not one million tiny objects. At mean length 280, raw Cα float32 arrays alone are 3.36 GB/million; N/CA/C/O arrays are 13.44 GB. Full structures, text documents, rejects, search databases and manifests are larger. Initially provision **100–300 GB working storage**, then replace this allowance with pilot byte counts. Keep accepted structures, provenance and reproducible regeneration metadata; retain a stratified reject audit. Publish consolidated public artifacts under `data/exp<N>-proteina/` in the HF bucket after settling the release terms and transfer plan.

## Success criteria

- Pilot yields a reproducible measured cost model, complete timing/rejection records and no silent malformed-document drops. Production planning is conditional on ≥50% aggregate retention; lower retention triggers repricing rather than quietly exceeding the envelope.
- At the 100k milestone, length quotas hold within ±5% relative error; accepted class/fold occupancy and duplicate growth are reported. Fine-cluster retention cap is enforced without dropping desired length/class coverage silently.
- Target at least 1.5× effective fine-cluster count versus a matched unconditional sample, and broader realized class coverage. This is a proposed research target, not a forecast. Failure is a useful negative result and stops the million-scale rollout for redesign.
- Final deliverable, if milestones pass: one million distinct accepted backbones, one sequence/document each, versioned provenance, all decontamination checks, complete cost report and reproducible structural-diversity analysis. At a five-per-cluster cap this requires at least 200,000 fine clusters; assess feasibility at 100k before assuming it scales.
- Downstream utility is a separate, fixed-token continuation experiment from the same decontaminated checkpoint: natural-only control versus 5% and 15% synthetic token mixtures, with identical optimizer/compute budgets. Use eval-val for iteration and eval-denovo as a separate diagnostic; reserve and log a rare eval-test read after selection. Target a ≥0.01 absolute long-range R-precision gain with no material natural-protein regression, accompanied by paired uncertainty estimates. Repeat promising comparisons across seeds before a strong claim.

## Results

Execution started on 2026-09-09. Issue #278 is approved through the pilot and conditional scale-up milestones. The full cost model and preregistration are in [PLAN.md](PLAN.md).

Initial one-H100 smoke job: `/bizon/exp278-smoke-v4` on `cw-rno2a`. Outputs: `s3://marin-us-east-02a/MarinFold/exp278-proteina/smoke-v4/`. Four 100-residue backbones completed at 1.803 seconds/candidate with batch size 4 and no compilation. Two of four passed full-chain self-consistency RMSD ≤2 Å, pLDDT ≥70 and refolded Cα geometry checks. They yielded 451- and 502-token contacts-v1 documents (81 and 98 contacts). This is an integration smoke sample, not a production yield estimate.

The ESMFold 4.48.3 API returns pLDDT in [0, 1]; the worker now converts confidence and PDB B-factors to [0, 100] before filtering. The first smoke outputs were corrected from their saved predictions without rerunning inference, under `fold-smoke-v1-corrected/`. Original raw outputs remain available for audit. Corrected quality results are in `data/fold-smoke-quality.csv`.

The launcher uses an isolated, locked Iris environment and content-addressed S3 source bundles. Startup fixes addressed an expired Iris client, command-size limits and selecting the container Python consistently. Geometry and confidence-unit tests: 6 passed. All candidate documents remain provisional until decontamination and diversity selection complete.

## Conclusion

_(Fill in after results are in.)_

## Current pilot jobs

- `/bizon/exp278-screen-short-v1`: 768 candidates at lengths 60, 100 and 200.
- `/bizon/exp278-screen-long-v1`: 768 candidates at lengths 300, 400 and 500.
- Each length has unconditional, mostly-alpha, mostly-beta and mixed-class arms (64 candidates each).
- The compiled 100-aa probe reached about 0.80 seconds/candidate at batch 32 after its cold compilation.
- W&B: [exp278-proteina-pilot-20260909](https://wandb.ai/open-athena/MarinFold/runs/exp278-proteina-pilot-20260909).

The source image is pinned by digest, upstream Proteina is pinned by commit, and downloaded assets are cached in S3 for reuse. The two screening jobs have hard 20-hour timeouts; the overall pilot cap remains 100 H100-hours.
