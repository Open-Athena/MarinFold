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
The original 163M-row membership remains in us-west-2. Outputs are durable and the instance terminates automatically.
This is a curation and yield audit, not a short model-training test.

## Next curation decisions

Evaluate confident-core novelty against every retained anchor and selected addition.
Inspect PAE/domain orientation where available and sequence-aware alignment coverage.
Validate the revised rules on fresh, broader samples; estimate unique-token yield and throughput.
Decontaminate every new sequence, freeze manifests and sampling, then run the full model comparisons.
