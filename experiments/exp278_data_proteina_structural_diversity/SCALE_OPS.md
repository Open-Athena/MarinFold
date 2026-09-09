# Scale run — September 9, 2026

The user authorized a run aiming for one million documents, with independent
batch jobs and preservation of every backbone and designed sequence. This
supersedes the pilot cap and production hold. Generation continues during review.

First production submission: **2026-09-09 21:32:01 UTC**. First review due:
**2026-09-10 15:32 UTC / 11:32 EDT**. Monitoring owner is Codex task
`01a08763-e824-79b2-8ecd-a8a851f4d31a`.

## Work and placement

The frozen `data/scale-20260909/manifest.json` has SHA256
`a7812d1e4173454eba2185f688cde8b60d4d8b55c97481b32acce56bdb364f80`.
It contains 4,820,608 raw candidates across every integer length 60–500 and
unconditional, alpha, beta and mixed arms. Inverse pilot retention determines raw
quotas to aim for equal accepted contributions. Estimated yield is 1,005,827
retained candidates at 12,728 pure H100-hours (15,273 with 20% overhead).
Global duplicate growth can reduce that yield. The finite manifest is bounded
below six million raw candidates; increasing it requires a new decision.

The 768 queues are balanced by predicted GPU time. Each is a root Iris job with
one H100, one replica, batch priority (`priority=3`), 100 preemption retries and
two failure retries. There is no gang or parent driver. Queue size is not GPU
allocation: the scheduler admits individual jobs as capacity becomes available.
East02 has 256 schedulable H100s shared with other users. Batch is excluded from
the interactive user-spend limit; no budget or cluster policy was changed.

East02 compute/storage are co-located. RNO2A has 512 H100s, but using East02
storage from there requires large cross-region transfers. Explicit approval was
requested and remains pending. Do not place jobs there until approval arrives.
Never manually stop health checks, other users' jobs, or the shared cluster.

## Artifacts and recovery

Root: `s3://marin-us-east-02a/MarinFold/exp278-proteina/scale-20260909/`.
Each case lives under `cases/<case-id>/`:

- `generated/batch-*.npz`: original CA coordinates and original per-input sampling
  timings in the same atomic object; `timings.csv` aggregates the case.
- `folded/sequences/*.parquet`: all designed sequences, source identities, seeds,
  model revision and original timing/worker metadata, saved before refolding.
- `folded/candidates/*.parquet`: every refolded candidate including rejects,
  original CA coordinates, sequence, complete refolded PDB, scores and reasons.
- `folded/documents-provisional/*.parquet`: quality-pass contacts-v1 documents.
  Sequence/reference decontamination and global diversity capping remain required
  before these become training data.
- `folded/timings/*.csv`, `completed-parts/*.json`: per-batch timings and commit
  markers. Retries reuse saved backbones and sequences and skip completed work.

`workers/`, `submissions/`, `manifests/`, `reports/`, `operations.json` and
`control.json` preserve run state. Raw/rejected artifacts survive final selection.

From the experiment directory, submit an unassigned range with:

```bash
uv run python scale_launch.py --manifest data/scale-20260909/manifest.json \
  --cluster cw-us-east-02a --start 0 --end 256 \
  --name-prefix exp278-scale-v1 --spacing-seconds 2 --no-wait
```

The launcher skips recorded submissions with matching manifest/cluster. Inspect
terminal failures and reconcile old jobs before replacing a worker. Never run a
worker index concurrently on two clusters. Append replacement IDs to run history.
Native preemption resumes the same immutable queue with the same job ID.

For a graceful pause, set `control.json` to `pause: true`; workers exit at batch
boundaries. For urgent cancellation, use `iris job cancel` on our root jobs.
Resume by setting `pause: false` and replacing terminal jobs after reconciliation.
`pause_at_utc` remains null during the review.

## Monitoring and first review

```bash
uv run iris --cluster cw-us-east-02a job list --prefix /bizon/exp278-scale-v1 --limit 800
uv run python scale_snapshot.py --manifest data/scale-20260909/manifest.json \
  --output data/scale-20260909/reports
```

Snapshots read metadata and small markers without downloading PDB payloads.
Counts reflect committed batch boundaries; cross-case reads are nontransactional.
If Iris logs are empty, inspect our task-container logs with kubectl.

At 18 hours report running/pending/failed jobs and retries; generated, designed,
refolded and quality counts; rejection by length/arm; allocated GPU-hours and
throughput; and storage growth. Analyze a stratified sample against frozen
sequence/structure references and compare diversity accumulation. Distinguish
sampled post-decontamination retention from exact quality retention. Include
cross-length neighbors: the six-length pilot's pruning does not apply to this
all-integer-length corpus. Requested labels alone do not establish novel folds.

[W&B](https://wandb.ai/open-athena/MarinFold/runs/exp278-proteina-scale-20260909),
[issue](https://github.com/Open-Athena/MarinFold/issues/278),
[draft PR](https://github.com/Open-Athena/MarinFold/pull/282).

## Review scheduling

The persistent workstation user timer `exp278-scale-review.timer` starts `scale_review.py` at September 10, 15:32:01 UTC. It captures exact batch counts and resource time, runs a bounded stratified retention/diversity audit, records W&B metrics, uploads reports and posts a new comment on issue #278. Its environment was tested with the canary snapshot. The workstation must be online with the user session running at the deadline; otherwise the persistent timer catches up at the next login. The workspace and local frozen reference databases must remain available. No native Codex automation was registered because its tool was unavailable.

A 16-candidate canary audit completed all sequence/structure screens and clustering; 4 survived the sample filters. That tiny integration check is not a scale-yield estimate.
