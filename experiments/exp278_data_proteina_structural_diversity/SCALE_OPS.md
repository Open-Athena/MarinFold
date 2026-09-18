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

## Preemption zombies and the September 16 recovery

Batch-priority preemption on East02 can leave a task with its attempt still
marked active: `job describe` shows `state: running` with `PodDeleted: pod was
deleted while the attempt was active` or `WorkloadEvictedDueToPreempted`, and it
never reschedules. `job list` keeps reporting `running`, so the run looks healthy
while producing nothing.

On September 16 **all 215 still-running workers were in this state**, median 9.4 h
since their last write and max 99.5 h; generation had been frozen at 87.9% of the
manifest for hours. 691 of 1,464 attempts across the run are `worker_failed`, so
the churn has been heavy throughout.

The same defect inflates the cost report. `budget_snapshot.py` charges
`(finished_at or now) - started_at`, so a zombie accrues forever: 13,185 H100-hours
reported against roughly **9,483 real** (28% phantom, growing one H100-hour per
zombie per wall-clock hour). Bound the real figure by treating each worker's last
object write as its pod death, and report closed attempts separately from open ones.

Recovery, as run on September 16:

```bash
# 1. capture the stuck set and confirm it is only ours
uv run iris --cluster cw-us-east-02a job list --prefix /bizon/exp278-scale-v1 \
  --limit 800 | awk '$2=="running"{print $1}' > zombies.txt
uv run iris --cluster cw-us-east-02a job cancel --exact --stdin --dry-run < zombies.txt

# 2. cancel, then wait until every one is terminal before resubmitting
uv run iris --cluster cw-us-east-02a job cancel --exact --stdin < zombies.txt

# 3. archive the superseded records; the launcher skips any worker that has one
mkdir -p data/scale-20260909/superseded-v1
mv data/scale-20260909/worker-<index>-submission.json data/scale-20260909/superseded-v1/

# 4. resubmit under a new prefix so replacements stay distinguishable
uv run python scale_launch.py --manifest data/scale-20260909/manifest.json \
  --cluster cw-us-east-02a --start 0 --end 768 \
  --name-prefix exp278-scale-v2 --spacing-seconds 2 --no-wait
```

Never resubmit before the old job is terminal: one worker index must never run
twice. The full range is safe in step 4 because the launcher skips the 553 workers
whose records remain. Nothing generated is lost — workers resume from durable
per-batch checkpoints and skip completed work. The replacement bundle digest
differs because the analysis module changed; no worker-side module was touched.

## Recurring snapshot timer

The one-shot `exp278-scale-review.timer` fired once on September 10 and never
rearmed, so progress went uncollected between then and September 16. A recurring
pair now covers the rest of the run:

```text
~/.config/systemd/user/exp278-scale-snapshot.service
~/.config/systemd/user/exp278-scale-snapshot.timer
```

The resource step uses `--prefix /bizon/exp278-scale-`, not a single generation's
prefix: after the September 16 resubmission the run spans `exp278-scale-v1` and
`exp278-scale-v2`, and a narrower prefix silently undercounts every replacement.

`OnCalendar=*-*-* 0/2:05:00` with `Persistent=true`, so it runs every two hours
and catches up once after the workstation is offline. Each run writes a
timestamped `snapshot-*.json` / `cases-*.csv` pair plus `latest.json` into
`data/scale-20260909/reports`, then refreshes `iris-resource-time.{csv,json}`.
The resource step is prefixed `-`, so a controller-tunnel failure cannot lose the
progress snapshot that already succeeded. One full run takes about ten minutes,
almost all of it listing the case tree; the interval keeps that under a tenth of
the duty cycle. It reads only metadata and small markers, writes nothing to S3,
and posts nothing to GitHub.

```bash
systemctl --user list-timers exp278-scale-snapshot.timer
systemctl --user start exp278-scale-snapshot.service   # snapshot right now
systemctl --user disable --now exp278-scale-snapshot.timer  # when the run ends
```

Each run also computes **staleness**: for every unfinished worker, the hours since
its newest object-store write. Iris job state cannot detect the failure above, so
recent writes are the liveness signal, and the listing is already paid for. The
snapshot gains a `staleness` block, a `staleness-*.csv` naming the idle workers,
and a `STALL DETECTED` banner past `--stale-hours` (default 3). A freshly
resubmitted or just-preempted worker on a long-length queue can exceed the
threshold legitimately until its first batch commits, so confirm with `job
describe` before replacing one: a recent attempt that is progressing is normal
churn; an old attempt stuck on `PodDeleted` is not.

Progress and service logs land in `data/scale-20260909/snapshot-service.log`. The
units hard-code this worktree as `WorkingDirectory`; moving or deleting it breaks
the timer.

## Review scheduling

The persistent workstation user timer `exp278-scale-review.timer` starts `scale_review.py` at September 10, 15:32:01 UTC. It captures exact batch counts and resource time, runs a bounded stratified retention/diversity audit, records W&B metrics, uploads reports and posts a new comment on issue #278. Its environment was tested with the canary snapshot. The workstation must be online with the user session running at the deadline; otherwise the persistent timer catches up at the next login. The workspace and local frozen reference databases must remain available. No native Codex automation was registered because its tool was unavailable.

A 16-candidate canary audit completed all sequence/structure screens and clustering; 4 survived the sample filters. That tiny integration check is not a scale-yield estimate.

## Launch verification

All 768 root jobs were submitted on East02. Each job and observed pod requests one H100 at batch priority. The launch reached 112 concurrent GPU pods; capacity subsequently varied. Iris recorded 32 task retries during startup, including `PodDeleted` worker failures. Installed Iris `task_state.py` charges worker failures to the 100-preemption retry allowance; application failures have a separate two-retry allowance. A checked example returned independently to the Kueue admission queue. No production job was terminal-failed at the startup check. All submitted IDs and content-addressed code bundles are recorded alongside the manifest.
