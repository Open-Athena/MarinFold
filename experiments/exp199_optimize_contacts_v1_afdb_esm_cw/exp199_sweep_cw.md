# exp199 CoreWeave Sweep Operations

> This document governs the CoreWeave exp199 training sweep managed with the
> `run-training-sweep` skill. Read it in full at the start of every heartbeat
> before inspecting code, SQLite, W&B, or Iris.

## Invariants

- **Every Iris job submission MUST pass `--priority batch`. No exception, ever.**
  It goes on the root driver; the child GPU gang inherits it. Never submit at
  `production` or `interactive`, and never omit the flag and rely on a default.
  Verify with the check in Operating Policy below.
- Do not inspect, mutate, submit, stop, or account for the sibling TRC sweep.
- Submit from this directory so Iris bundles only the CoreWeave workspace.
- A logical trial has one W&B identity and one shared S3 checkpoint root. Never
  allow two live dispatches to write that identity.
- Treat all production CoreWeave clusters as one shared storage/run domain.
  Moving a trial between clusters resumes the same run; it is not regional racing.
- Use existing S3 token caches only. Never copy, rebuild, or retokenize data in the
  experiment DAG or operating loop.
- Source `USERNAME` from `~/marin.env` and pass it as `--user "$USERNAME"` on
  every Iris command; never hard-code an Iris user.
- Use whole GPU nodes. Never target `cw-us-west-04a`.

## Sweep Definition

The entry point and opaque 20-trial catalog are `exp199_sweep_cw.py` and its
`TRIALS` mapping. Production uses CalVer `2026.08.07.2` (`s02`). Training
semantics, W&B identities, cache paths, checkpoint paths, batch fitting, and
resource requests are owned by that script.

AFDB and validation payloads were previously compared with the exp199 GCS caches
by size and MD5. The ESM cache was restored directly from Hugging Face revision
`04ff0f2cc0d7530c062b027b3c14699f65e277dc`; the successful receipt is:

```text
s3://marin-us-east-02a/marin/protein-structure/MarinFold/exp199_optimize_contacts_v1_afdb_esm/setup/esm-cache-restore/04ff0f2cc0d7530c062b027b3c14699f65e277dc/complete.json
```

The two supported GPU capacities completed scratch initialization, training,
full validation, native checkpointing, and final HF export in smoke runs. The
production 2-, 4-, 8-, and 16-node shapes use gradient accumulation 1.

## Operator Choices

- Time limit: 14 days from the first production dispatch.
- Replicas per trial: 1.
- Maximum submitted GPUs: 1,536. The retained five-trial GB200 fleet uses 288
  GPUs; optional additional ranked trials may use the remaining cap.
- Approved clusters: `cw-us-east-08a`, `cw-us-east-02a`, and `cw-rno2a`.
- Approved GPU families: GB200 and H100, using the script's 2-, 4-, 8-, and
  16-node production shapes.
- Excluded cluster: `cw-us-west-04a`.
- Priority band: `batch`.
- Retained trials: `m2-p03-base`, `m2-p06-aug`, and `m1-p02-aug`. These are the
  only trials the sweep still pursues. Every other
  trial is abandoned and must never receive another dispatch. `m1-p06-aug` is
  complete and checkpoint-verified; it is finished, not abandoned.
- Because all remaining trials are retained, there is no longer an internal
  priority ranking. Resolve contention between them on wall-clock: prefer the
  trial that is further along and more stalled.
- Operations document: this file.
- Dynamic state (repository-relative):
  `scratch/exp199_cw_s02/exp199_cw_sweep.sqlite`.

## Operating Policy

- `heartbeat_every=1h`
- Target at most 10--15 minutes of active work per heartbeat. Routine passes use
  one grouped W&B observation, exact active-dispatch Iris liveness, one decision
  snapshot, and one concise report. Inspect task details, logs, or capacity only
  when a failure or placement decision requires them.
- **Every heartbeat, confirm no exp199-cw job is running off `batch` priority.**
  The controller stores the band as an integer where `PRIORITY_BAND_BATCH = 3`
  (`PRODUCTION = 1`, `INTERACTIVE = 2`, `INHERIT = 0`). This must return `0`:

  ```bash
  .venv/bin/iris --cluster marin query "
    SELECT count(*) AS non_batch FROM job_config
    WHERE job_id LIKE '/eczech/exp199-cw-s02-%' AND priority_band != 3"
  ```

  A nonzero result means something was submitted off-band: stop that exact
  dispatch and resubmit it with `--priority batch`. Do not merely trust SQLite's
  `priority_band`, which records what was intended rather than what Iris applied.
- `reslice_after=1h`
- `restart_after=12h`
- `relocate_after=3d` is retained as a control threshold but has no distinct
  destination: every approved cluster is in the single `cw-shared` run domain.
- `pending_target_limit=1` per exact cluster/gang shape during ordinary recovery.
  The concentrated continuation may batch-admit its retained trials when the
  controller reports direct headroom for the complete admitted set.
- The concentrated continuation launched on 2026-08-08 stops the original fleet,
  retains the top 25% of trials by W&B `run_progress`, and gives each retained
  trial four times its previous node count. Additional ranked trials may be
  admitted only when they do not delay the retained five and total submitted GPUs
  remain at or below 1,536.
- Protect the priority cohort under contention. If a non-priority trial proves
  W&B progress while any priority trial is not training, stop the exact
  non-priority writer and use the released shape or its credible equivalent to
  start/recover the priority trial. Keep lower-priority trials submitted only when
  they do not prevent a priority trial from training; admit them freely during a
  capacity wave large enough to run both cohorts.
- Start with one four-node canary on each GPU family. Admit more work on a target
  after W&B proves progress; begin with matched base/augmentation trials so their
  throughput is directly comparable.
- Query W&B before Iris. Only a new `run_progress` high-water mark proves progress.
- An isolated failure is retried immediately from the shared checkpoint after the
  exact current dispatch is stopped. Repeated or correlated failures pause affected
  replacements for shared-cause investigation.
- A non-progressing dispatch may move to another eligible shape after one hour if
  the destination has pending headroom. Same-target restart is due after 12 hours.
- Preemption alone does not justify replacement; Iris owns preemption recovery.
- Completion requires `run_progress >= 1` and a reachable final checkpoint. Stop
  operating only after every trial satisfies both conditions or the time limit ends.

The authoritative target grid uses `cw-shared` as the run/storage region. `GPUs`
is stored in SQLite's generic `chips` field.

| Region | Bucket | Target | GPUs | State | Reason |
| --- | --- | --- | ---: | --- | --- |
| `cw-shared` | `marin-us-east-02a` | `cw-us-east-08a/GB200/n16` | 64 | `eligible` | concentrated continuation |
| `cw-shared` | `marin-us-east-02a` | `cw-us-east-08a/GB200/n8` | 32 | `eligible` | concentrated continuation |
| `cw-shared` | `marin-us-east-02a` | `cw-us-east-02a/H100/n16` | 128 | `eligible` | concentrated continuation |
| `cw-shared` | `marin-us-east-02a` | `cw-us-east-02a/H100/n8` | 64 | `eligible` | concentrated continuation |
| `cw-shared` | `marin-us-east-02a` | `cw-rno2a/H100/n16` | 128 | `eligible` | concentrated continuation |
| `cw-shared` | `marin-us-east-02a` | `cw-rno2a/H100/n8` | 64 | `eligible` | concentrated continuation |

## Change Record

- 2026-08-07: Iris ownership must always come from `USERNAME` in
  `~/marin.env`; the current value resolves to `eczech`.
- 2026-08-08: The operator stopped the original 20-run fleet and switched to a
  top-quartile continuation using four times each retained trial's node count.
- 2026-08-09: Heartbeats changed from 30 minutes to one hour. The operator also
  prioritized `m1-p06-aug`, `m1-p03-aug`, `m2-p03-base`, and `m1-p02-aug` by
  current validation loss and requested streamlined 10--15 minute passes.
- 2026-08-10: The operator abandoned `m1-p03-aug` as diverged, removing it from
  the priority cohort, and directed `m1-p06-aug` into the `cw-rno2a/H100/n16`
  slot it released. Measured placement behaviour: `cw-rno2a` sustains only one
  16-node gang at a time, so admitting a second one merely idles a trial.
- 2026-08-10: `m1-p06-aug` completed and was checkpoint-verified at 11:22 UTC,
  the sweep's first completion.
- 2026-08-10: The operator narrowed the sweep to four retained trials and
  abandoned the rest. Live dispatches for `m1-p01-base` and `m2-p01-aug` were
  stopped, and the released `cw-rno2a/H100/n16` slot went to `m2-p01-base`.
  Observed reliability: `cw-us-east-08a` produced correlated preemption waves
  about 3.4h apart, so prefer `cw-rno2a` for the retained trials.
- 2026-08-11: The operator abandoned `m2-p01-base` as diverged and asked for its
  capacity to be recycled. Its `cw-rno2a/H100/n16` slot went to `m1-p02-aug`,
  which had lost its child gang three times on `cw-us-east-02a`. Three trials
  remain.
