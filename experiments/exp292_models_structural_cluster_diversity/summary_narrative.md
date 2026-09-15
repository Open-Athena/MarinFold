## Can cluster expansion improve useful prediction diversity?

Recover omitted sequence/structure pairs from the original AFDB and ESMFold2 clusters.
Keep each sequence with its own structure; related proteins are not automatically alternate states of one protein.
Compare existing data, matched random expansion, and structural expansion using full 1.5B models at about 500B tokens per arm.
Small jobs are for curation development. Model efficacy requires the full training budget.

## Initial AFDB curation

36 clusters; 174 current training anchors; 254 omitted candidates; 428 source structures.
2,387 pair comparisons produce five provisional whole-chain additions.
Only two remain below core TM 0.8 against every retained anchor. Other anchors explain three apparent core-diversity hits.
Inspection shows a mix of distributed backbone differences and loop/terminal disagreement.
No candidate is cleared for training; no claim of true alternative conformations.

## Recovery and scale

AFDB census: 14.06M eligible omitted members in 332,786 clusters.
At three additions per cluster this pool contributes at most 998,358 structures before further filters.
Most of the planned 10–30M expansion must come from ESMFold2 or a broader audited AFDB pool.
AWS access restored. A us-west-2 worker recovered the ESM development memberships from the original 163.14M-row TSV in 202 seconds.
29 ESM clusters, 390 omitted candidates; 419 source structures retrieved and validated, 346 pass initial quality filters. 3,408 pair comparisons and two provisional hits; job completed in 503 seconds and terminated.

## Domain inspection matters

The ESM candidate with whole-chain TM 0.680 has separate-domain TM 0.908 and 0.951 under exploratory PAE-block crops.
Much of its apparent novelty is relative domain arrangement, with interdomain uncertainty in the stored PAE pattern.
The second ESM hit mixes domain, length and loop differences. Neither is cleared for training.
PAE codes are plotted without assuming physical units. No angstrom cutoff is applied yet.

## Broader curation audit running

919 fresh ESM clusters, up to 14,284 omitted members; seed 293.
Sampled from 100 current training shards containing 1,963,845 current anchors, excluding the development shard.
One region-pinned 32-vCPU EC2 worker, with 24 alignment processes and four bounded fetch threads.
A source/training length assertion stopped the first attempt; a full anchor metadata check found three sequences with unknown residues. The corrected run excludes those clusters explicitly. The original 163M-row membership remains in us-west-2. Outputs are durable and the instance terminates automatically.
This is a curation and yield audit, not a short model-training test.

## Broader AFDB audit

1,008 fresh clusters, 4,845 retained anchors and up to 19,201 omitted members.
No cluster overlap with the initial 36-cluster development gallery.
A 32-CPU preemptible Iris worker is pinned to us-central1-a, with matching regional GCS outputs.
The audit measures within-cluster pairs and records core, coverage and sequence-quality diagnostics. Training filters remain unfrozen.

## Preliminary sequence exclusion

571 initial omitted candidates screened against 1,823 unique reference sequences from exp225.
The identity/shorter-coverage rule flags 60 candidates: 29 AFDB and 31 ESM.
None of the seven whole-chain provisional hits is flagged. An exact-reference positive control passes.
This is a small-pool screen; repeat on the final pool and frozen reference before training. E-value reporting depends on target database size.

## Larger ESM audit: scale estimate

919 sampled clusters; 12,063 quality-passing omitted members; 146,433 measured pairs.
At TM <=0.8, 49 whole-chain hits become 32 with the core diagnostic.
Population-weighted ESM projections: approximately 167K whole-chain / 90K core-checked additions at TM 0.8, or 2.20M / 760K at TM 0.9.
These are uncertain point estimates before domain review and final exclusions. The original 10-30M goal is not supported by the current strict policy. Adjust the corpus design and mixture fraction; retain the full training horizon.

## Next curation decisions

Evaluate confident-core novelty against every retained anchor and selected addition.
Inspect PAE/domain orientation where available and sequence-aware alignment coverage.
Validate the revised rules on fresh, broader samples; estimate unique-token yield and throughput.
Decontaminate every new sequence, freeze manifests and sampling, then run the full model comparisons.
