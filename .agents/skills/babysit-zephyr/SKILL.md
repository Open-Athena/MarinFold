---
name: babysit-zephyr
description: Start, monitor, and babysit Zephyr pipeline jobs on Iris. Use when launching a zephyr job, watching it run, or restarting after failures.
---

# Skill: Babysit Zephyr Job

Start, monitor, and keep Zephyr pipeline jobs running on Iris. If something
needs deeper investigation, escalate to **debug-zephyr-job** (not yet ported
into MarinFold).

Ported from `marin-community/marin/.agents/skills/babysit-zephyr/SKILL.md`.

Most MarinFold zephyr pipelines will live under `experiments/exp<N>_data_<name>/`
(data-generation experiments) — see [`data/README.md`](../../../data/README.md).

## Zephyr Job Structure

A zephyr pipeline job spawns child Iris jobs:

- **`*-coord`** — coordinator (1 task). Orchestrates the pipeline stages, queues tasks, tracks progress.
- **`*-workers`** — worker pool (many tasks). Workers poll the coordinator for shards to execute.

A single job may execute **multiple pipelines sequentially** (e.g. fuzzy dedup
runs connected components iteratively, each iteration is a separate pipeline).
These show as different `p<N>` values in child job names. This is normal —
don't confuse sequential pipelines with failed retries.

Failed retries show as different **hashes** with the same `p0`. Stale
coordinators from previous attempts may linger (upstream marin issue #3705).

Child job naming: `<hash>-p<pipeline>-a<attempt>-{coord,workers}`.

## Iris Config

All Iris commands below use `--config <CONFIG>`. MarinFold does not vendor iris
configs into the repo — point at whichever YAML file the user maintains for the
cluster they want to run on (see **babysit-job** for the cluster-shorthand
mapping convention).

## Dashboard

```bash
# Connect to the Iris dashboard (establishes SSH tunnel, prints URL with port)
uv run iris --config <CONFIG> cluster dashboard
```

## Starting a Job

Get the run command from the user. Typical MarinFold pattern (from a
data-gen experiment dir):

```bash
cd experiments/exp<N>_data_<name>
uv run iris --config <CONFIG> job run --region <REGION> --no-wait \
    -- python -m <script>
```

The entrypoint container defaults to 1GB memory. For long-running pipelines
that accumulate state (GCS clients, logging), increase with `--memory`:

```bash
uv run iris --config <CONFIG> job run --region <REGION> --memory 5GB --no-wait \
    -- python -m <script>
```

The command prints a job ID on success. Note this ID for monitoring.

## Run-history integration

Once the pipeline emits a W&B URL (zephyr pipelines that log to W&B), create a
history file with `marinfold history new --kind data ...`. Follow the same
pattern as in **babysit-job**'s run-history section.

## Stopping a Job

Always ask the user before stopping. Stopping kills all child jobs
(coordinators, workers).

```bash
uv run iris --config <CONFIG> job stop <JOB_ID>
```

## Monitoring

### Health Checks

Check child job states via the Iris CLI (returns per-task state and resourceUsage):

```bash
# diskMb is updated every ~60s. On K8s it is always 0 (workdir lives inside the pod).
uv run iris --config <CONFIG> rpc controller list-tasks --job-id <JOB_ID>
```

A healthy zephyr job has:
- Coordinator: RUNNING, 1 task running
- Workers: RUNNING, tasks ramping up toward target count

### Stage Progress

The coordinator logs a progress line every 5s:

```
[stage0-Map → Scatter] 347/1964 complete, 1617 in-flight, 0 queued, 1828/1891 workers alive, 63 dead
```

Fetch via the Iris CLI:

```bash
uv run iris --config <CONFIG> rpc controller get-task-logs \
  --id <COORD_JOB_ID> --max-total-lines 5000 --attempt-id -1 --tail
```

**Caveat**: With large worker pools, `pull_task` operations flood the log
buffer (upstream marin issue #3707). Filter when parsing:

```python
for entry in task_logs:
    msg = entry.get('data', '')
    if 'pull_task' in msg or 'Started operation' in msg or 'report_result' in msg or 'registered' in msg or 'tasks completed' in msg:
        continue
    print(msg)
```

### Coordinator Thread Dump

When logs are flooded, a thread dump tells you if the coordinator is alive and
working:

```bash
uv run iris --config <CONFIG> rpc controller profile-task \
  --json '{"target":"<COORD_JOB_ID>/0","durationSeconds":1,"profileType":{"threads":{}}}'
```

Key patterns:
- `actor-method_0` in `_wait_for_stage` → pipeline active, waiting for current stage to complete
- `_coordinator_loop` thread present → heartbeat/dispatch loop running
- All threads in `_worker` (thread pool idle) → pipeline exited, coordinator is a zombie

## Monitoring Lifecycle

After submitting, monitor in escalating stages:

1. **Smoke check (first 2-5 minutes)**: Confirm coordinator and workers child
   jobs appear and reach RUNNING state. Check coordinator logs for early
   errors. If it fails here — likely a code bug, config issue, or bundle
   fetch timeout.

2. **Steady-state monitoring**: Check stage progress via coordinator logs.
   Confirm two things: (a) shards are completing within the current stage,
   and (b) stages are advancing. Calibrate check-in interval to the
   pipeline — you want to see at least one stage transition between checks.
   For pipelines with many short stages, check every few minutes. For
   pipelines with few long stages, every 15-30 minutes may suffice.

3. **Failure detection**: If workers get KILLED or the coordinator goes
   zombie, the `StepRunner` may retry automatically (new child jobs with a
   different hash appear). Check the latest attempt. Stale coordinators
   from previous attempts may accumulate (upstream marin issue #3705). If
   retries keep failing, escalate to debug-zephyr-job.

**"Terminated by user" is misleading**: This diagnostic does not necessarily
mean a human killed the job. The system uses this message for various
internal termination reasons. Always check the actual logs at each level
(parent job, coordinator, workers) to determine the real cause.

## Restarting After Failure

1. Ask the user if it's okay to stop and restart.
2. Stop the job.
3. Get the run command (or reuse the previous one).
4. Submit and resume monitoring.
5. Append the new iris job ID to the run's history file:
   `marinfold history add-iris-job <run-stem> <NEW_JOB_ID>`.

## When to Escalate

Escalate to **debug-zephyr-job** (not yet ported) when:

- A stage is stuck (no shard progress for an extended period)
- Stragglers are holding up a stage (few in-flight, 0 queued, most workers idle)
- Workers are failing repeatedly with the same error

For controller issues (e.g., RPCs timing out), use the **debug-iris-controller**
skill (not yet ported).
