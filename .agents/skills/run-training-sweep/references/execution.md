# Execution

## Use Four Identities

- **Logical trial:** one opaque experiment configuration.
- **Regional run:** one logical trial in one region; owns W&B and checkpoint identity.
- **Dispatch:** one immutable Iris submission attempt.
- **Target:** one allowed region, TPU slice, and chip count; lives in the operations document.

Never parse W&B or Iris names.

## Keep Iris Operations Simple

Allowed routine job actions:

- Submit an exact unique job.
- Stop an exact job.
- Ask whether an exact dispatch is running.
- Recognize exact `unschedulable` results.
- Treat every other successfully returned job state as not-running for liveness purposes.

Iris never proves training progress. Do not routinely inspect logs, summaries, task counts,
parent-child structure, retry history, pending reasons, or cluster capacity. W&B is the truth.

An Iris timeout or service failure blocks all sweep work. Do not infer whether the request
succeeded, inspect W&B, or take another action. Schedule one later pass that checks only Iris;
keep waiting until it responds, then reconcile the affected exact dispatches before resuming.

Inspect deeper only for a recorded reason, such as the same first-attempt failure reproducing
across regions or a dispatch command failing before W&B registration. Do not derive placement
policy from unfamiliar Iris internals.

## Handle Unschedulable Targets

`unschedulable` means the requested placement is permanently unavailable, not merely
waiting for capacity.

1. Verify the dispatch requested the intended region, slice, and chip count.
2. Record `target_unschedulable`; do not retry the exact target.
3. Mark only that target `ineligible` in Operations and add a terse `Change Record`
   entry naming the grid change and its cause.
4. Reslice or relocate immediately to another eligible target.

Do not generalize one result to a region or TPU family. If many previously valid
targets become `unschedulable` together, pause target changes and investigate a
systemic problem before continuing.

## Make Every Dispatch Unique

- Assign attempt numbers in SQLite.
- Use a unique Iris job name, such as `<opaque-wandb-id>-<slice>-a<attempt>`.
- Never recover attempt numbers or metadata by parsing that name.
- Give every Iris job one immutable dispatch row.
- Stop the current dispatch before replacing it.
- Allow at most one active dispatch per regional run.

Resume comes from the regional checkpoint, not the Iris name.

## Time Stalls

Set `stall_since` to the last increase in the `run_progress` high-water mark. Before any
increase, use the first dispatch submission. Only a new high-water mark resets it.

For an unknown two-week sweep, begin near:

- Restart after 3 hours.
- Reslice after 12 hours.
- Relocate after 4 days.

Confirm these defaults during the operator interview and use them unless evidence
warrants a deliberate change. If a timeout changes, rewrite Operating Policy and add
a `Change Record` entry; do not make document maintenance part of every heartbeat.

Actions:

- **Restart:** new dispatch, same region and slice, same regional checkpoint.
- **Reslice:** new dispatch, same region, different eligible slice, same checkpoint.
- **Relocate:** new regional run, different region, starts from zero, no transferred data.

A terminal dispatch may be replaced immediately, but replacement does not reset the regional
stall timer. Do not let repeated restarts prevent reslicing or relocation.

## Classify Failures Before Retry

Observe the whole W&B fleet before acting on a `failed` run.

- If the failure is isolated, stop its dispatch if still running and immediately submit a
  unique replacement on the same region and slice from the regional checkpoint.
- If failures recur after replacement or cluster across otherwise independent trials,
  regions, or targets, pause replacements and investigate a shared cause. Inspect deeper
  Iris details only with this reason recorded.
- Resume with a concrete basis, contain or stop affected work, or wait for operator
  direction when the safe response is unclear. Do not blindly retry.

## Place Actively

No command reveals available TRC capacity. Submission is the measurement.

### Enforce Placement Diversity

> [!IMPORTANT]
> These are hard constraints. Apply them before every dispatch; placement rankings
> never override them.

- If at least two placement opportunities exist in one heartbeat and at least two
  targets are eligible, use at least two targets.
- Never make more than three consecutive dispatches to the same target while another
  eligible target exists.

Never stop progressing work, delay a dispatch, or create a replica solely for diversity.
An immediate retry after an intermittent failure may use the same target, but it counts
toward the consecutive-dispatch limit. Stall-driven reslices are the preferred exploration
opportunities.

Within these constraints, choose every placement, including a required alternative, in
this order:

1. Eligible targets with the highest `target_rate`.
2. A fresh optional fleet-utilization hint, combined with any relevant prior
   experiment throughput.
3. Remaining eligible targets across the grid, favoring untested ones.

Fall through quickly when a higher-ranked target stops placing or progressing.
Utilization never defines the grid or proves capacity. Include zero-progress
submissions as evidence and keep quiet regions visible.

- Recompute rankings every heartbeat. Never write throughput rankings or transient
  placement preferences into Operations.
- Probe with real trials, never filler jobs.
- Change placement when productive chips stay flat across repeated heartbeats.
- Reslice stalled work across untried or recently granted shapes; descending sizes is a useful
  default when large gangs do not place.
- Do not change the priority band to solve scarcity.

## Race Regions Optionally

Default to one live regional run per logical trial.

With two replicas:

- Start distinct validated regions.
- Maintain two live regional runs while the trial is incomplete and resources permit.
- Let both run until one fully completes.
- Stop every nonterminal sibling only after `run_progress >= 1` and the expected
  checkpoint is reachable.
- Record siblings as race losses.

Discourage more than two replicas without a specific operator reason.

With one replica, relocation remains allowed after its timeout. It replaces the active region
and starts from zero.

## Handle Client Revision Rejections

Apply this only when a new dispatch fails with an error like
`marin-iris client is too old (build <date>; minimum <date>)`; exact wording may vary. Do not
check the revision on every loop. `BUILD_DATE` historically lives in `iris/_build_info.py`.
Changing it is the only autonomous correction. Do not run `uv sync` or otherwise upgrade,
pull, rebase, or reinstall Iris.

On rejection:

1. Record `client_floor_failed`.
2. Stop new submissions and recovery-driven stops.
3. Set `BUILD_DATE` to satisfy the required floor.
4. Verify the runtime-reported date and prove a canary submission is accepted.
5. Resume on acceptance; otherwise stop and give the operator the dates, failure, and options.

Record in-loop `BUILD_DATE` or floor changes in Operations. Never restart or reconfigure
the shared Iris cluster.

## Stay Active

Run one heartbeat at a time as an agent decision pass. As its final action, schedule exactly
one next pass with a time-based trigger such as `CronTask`; never rely on event-based
monitoring. Every pass must refresh the full state and context before deciding. Never
delegate decisions to a shell script, daemon, or scheduled task.
