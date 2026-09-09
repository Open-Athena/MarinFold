# Proposed experiment: diversify contacts-v1 with Proteina-generated monomers

Approved for execution on 2026-09-09: start with the capped pilot and proceed through the preregistered milestones if they pass.

## Kind

- Kind: data

## Question

Can fold-conditioned Proteina generation supply one million useful, structurally diverse, sequence-paired monomer documents of length 60–500, at an acceptable cost per retained structure?

## Hypothesis

Length-stratified, fold-balanced generation followed by sequence design, refolding and structural selection will increase structural coverage relative to unconditional generation. Whether that coverage improves MarinFold must be tested with a fixed-token training comparison; generating more documents alone does not establish value.

## Background

- [Proteina repository](https://github.com/NVIDIA-BioNeMo/proteina), inspected at commit `a44b407daf6a5358e43cd68907f3e3f1cbc65fdc`. Its output is a Cα trace, not an amino-acid sequence or a complete backbone.
- [Published efficiency measurements, Appendix C](https://arxiv.org/html/2503.00710v1#A3): compiled, maximally batched A100-80GB inference, 400 sampling steps. The 200M model without triangle layers is the initial throughput candidate; the long-chain checkpoint uses the same architecture but needs its own benchmark.
- [Upstream designability code](https://github.com/NVIDIA-BioNeMo/proteina/blob/main/proteinfoundation/metrics/designability.py) uses Cα-only ProteinMPNN and ESMFold. Its default evaluates eight sequences per backbone and reloads ESMFold inside each per-backbone call. Production must use persistent workers and explicit sequence-attempt counts.
- MarinFold references: [contacts-v1](https://github.com/Open-Athena/MarinFold/tree/main/marinfold/marinfold/document_structures/contacts_v1), [exp78 ESMFold timings](https://github.com/Open-Athena/MarinFold/blob/main/experiments/exp78_evals_esmfold_contacts/data/timings.csv), [exp225 decontamination](https://github.com/Open-Athena/MarinFold/issues/225), [exp245 evaluation splits](https://github.com/Open-Athena/MarinFold/issues/245).

## Available compute

Read-only inspection around 18:18–18:22 UTC on 2026-09-09. Named configs came from `/home/bizon/git/marin-freshiris/lib/iris/config/`. All three production CoreWeave controllers and the marin controller were healthy.

| Iris cluster | Configured fleet | Live schedulable GPUs | GPU requests in bound active pods | Unrequested GPUs | Recommendation |
|---|---:|---:|---:|---:|---|
| cw-rno2a | 512 H100 | 504 | 0 | 504 | First choice; x86 hosts |
| cw-us-east-02a | 256 H100 | 256 | 8 | 248 | Fallback; x86 hosts |
| cw-us-east-08a | 864 GB200 | 832 | 776 | 56 | Later benchmark; ARM hosts, 128 GPUs requested by unbound pending pods |

These are scheduling observations, not reservations or measured GPU utilization. RNO2A had zero pending Kueue workloads at the snapshot; East02 had one. Jobs can submit additional GPU work after this snapshot. Iris's RNO2A backend independently advertised 504 available H100s. CoreWeave's `0/0 workers` in `cluster status` reflects its Kubernetes execution model, not an empty GPU fleet.

The `marin` cluster had 409 healthy workers and live CPU / TPU v4, v5e, v5p and v6e pools. Ready TPU slices are not necessarily idle. Proteina's CUDA/PyTorch stack makes H100s the direct route; an XLA port would be separate engineering. `cw-us-west-04a` is configured as a small CI cluster, so exclude it from bulk production; its live capacity was not queried.

All three production CoreWeave configs describe prepaid/reserved fleets. Incremental GPU rental cost can therefore be near zero within the reservation; GPU-hours still consume shared capacity. Contract chargeback rates were not available. The usual RNO2A per-user budget is described as approximately 32 H100s with proportional full-node CPU/RAM; verify the submitting identity's budget at launch. Start with 32 GPUs, not the entire idle fleet.

## Cost model

Target means **one million retained backbones with one sequence/document each**, not one million attempts or sequence variants. Assume approximately uniform lengths over all integers 60–500 (mean 280). This gives roughly equal counts in equal-width length bins, not equal compute per bin.

Published sampling seconds per candidate on A100-80GB:

| Model | 100 aa | 200 aa | 300 aa | 400 aa | 500 aa | A100 GPU-hours / 1M raw candidates |
|---|---:|---:|---:|---:|---:|---:|
| 60M, no triangle layers | 0.29 | 0.94 | 2.01 | 3.38 | 5.26 | 583 |
| 200M, no triangle layers | 0.59 | 1.88 | 3.87 | 6.54 | 9.96 | 1,125 |
| 200M, triangle layers | 3.74 | 13.05 | 28.31 | 50.14 | 80.89 | 8,550 |
| 400M, triangle layers | 3.29 | 11.20 | 24.35 | 42.64 | 75.57 | 7,516 |

Calculation: linearly interpolate the published 100-residue grid, conservatively use the 100-aa time for lengths below 100, average across 441 lengths, multiply by 1M/3600. These are sampling-only estimates, not measured H100 performance or acceptance-adjusted costs. Compilation and batching are necessary to approach them. The 200M no-triangle mean is 4.05 seconds/candidate on A100; use a provisional H100 speedup of 1–2×, subject to the pilot. Strong guidance can add model evaluations, so baseline conditioning uses guidance weight 1.

Local exp78 H100 ESMFold measurements, restricted to lengths 60–500:

| Length | Observations | Mean seconds |
|---|---:|---:|
| 60–149 | 239 | 1.29 |
| 150–249 | 179 | 1.50 |
| 250–349 | 88 | 2.57 |
| 350–449 | 21 | 4.45 |
| 450–500 | 5 | 7.29 |

Weight these means by bin widths 90/100/100/100/51 to obtain **3.04 seconds per folding attempt**, or 844 H100-hours per million attempts. The longest bin is poorly sampled; these natural-sequence timings are a proxy for designed sequences and must be benchmarked. Using the pooled 1.5-second median would underprice this longer, more uniform corpus.

For a million accepted designs:

`GPU_hours = sum_over_bins[N_accepted_bin / yield_bin * (sampling_seconds_bin + attempts_bin * (MPNN_seconds_bin + folding_seconds_bin))] / 3600 * overhead_factor`

Include every quality, decontamination and diversity rejection in `yield_bin`. Distinguish elapsed batch latency from amortized GPU seconds per sample.

Initial planning assumptions: 50–80% overall retention; one folding attempt/candidate; 0.3–1.0 GPU seconds for ProteinMPNN as an unmeasured allowance; 20% startup, retries and I/O overhead. Together these imply roughly 2,200–5,400 H100-hours. Round the production planning envelope to **2,500–6,000 H100-hours**, pending pilot measurements. This is not an upper bound if retention collapses or more sequence attempts are needed.

| Resource scenario | Projection |
|---|---|
| 32 sustained H100s | 3.3–7.8 days compute elapsed, plus queueing/setup/CPU tail |
| 64 sustained H100s, if user budget permits | 1.6–3.9 days |
| Illustrative accounting at $2/GPU-hour | $5,000–12,000 |
| Illustrative accounting at $4/GPU-hour | $10,000–24,000 |

Dollar rates above are sensitivity assumptions, **not CoreWeave quotes or the project's contract rate**. At-scale CPU selection/contact generation gets an initial 500–2,000 core-hour allowance, explicitly unmeasured; measure it on 10k then 100k candidates. Million-scale pairwise structural alignments are excluded: use indexed prefiltering and candidate alignments. Storage and transfers are additional. Training ablations are separately budgeted.

Each extra million ESMFold attempts adds approximately 844 H100-hours before overhead. Eight attempts by default would be about 6,755 H100-hours of folding for one million raw backbones alone. Use adaptive retries only where they increase accepted structural diversity per GPU-hour.

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

## Notes

The [Proteina license](https://github.com/NVIDIA-BioNeMo/proteina/blob/main/LICENSE) restricts use of the work and derivatives to research/evaluation. Preserve provenance and license metadata; the applicability of those terms to released generated datasets and downstream models needs clarification before assigning unrestricted release terms. This proposal does not infer that output datasets automatically inherit a particular license.

No launchable code exists yet. Once the experiment issue is filed, scaffold its real issue-numbered directory and implement on a feature branch through a PR.
