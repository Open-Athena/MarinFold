# exp232 CoreWeave Sweep Operations

> This document governs the exp232 CoreWeave GPU sweep managed with the
> `run-training-sweep` skill. Read it in full at the start of every heartbeat
> before inspecting code, SQLite, W&B, or Iris.

## Invariants

- The experiment code owns training semantics; operate only the ten trials
  declared by `experiments/exp232_sweep_cv1_decontam/exp232_sweep.py` at version
  `2026.08.14.1` (`s01`).
- Maintain at most one active writer for each trial's shared W&B ID and S3
  checkpoint root. CoreWeave target changes are reslices of that same run; never
  race clusters or GPU families.
- Every Iris root submission uses the `marin` federation controller, an exact
  `--target-cluster`, `--priority batch`, and `--user eczech`. The target cluster
  and the script's `CLUSTER` value must match.
- W&B routing is `open-athena/MarinFold`. Validate that the authenticated account
  has the required model seat before smoke; if access fails, stop and ask the
  operator rather than falling back to a personal entity or project.
- Use only the existing exp232 training caches and exp199 validation cache. Never
  copy, retokenize, or rewrite them during sweep operation.
- Production begins only after the configured smoke run succeeds and its W&B
  metrics are readable in `open-athena/MarinFold`.
- Let Iris recover ordinary preemptions. A preemption by itself is not evidence of
  a failed trial or unavailable target.
- A trial completes only when W&B `run_progress >= 1` and its expected final
  checkpoint is reachable. W&B `finished` alone is insufficient.
- Never target `cw-us-west-04a`.

## Sweep Definition

- Entry point and trial catalog:
  `experiments/exp232_sweep_cv1_decontam/exp232_sweep.py`.
- Launch and storage contract:
  `experiments/exp232_sweep_cv1_decontam/README.md`.
- Training starts from scratch. All production clusters use the script's shared
  CoreWeave S3 namespace, so checkpoint-preserving moves do not create regional
  runs or require data transfer.
- The validation-only smoke is `m1-p06-aug` on one `cw-us-east-02a` H100 node for
  ten steps. Its script-generated W&B/checkpoint identity is distinct from every
  production run.

## Operator Choices

- Time limit: 14 days from the first production dispatch.
- Regional replicas per trial: 1; represented as one shared CoreWeave run.
- Maximum active compute: 640 GPUs.
- Approved clusters: `cw-us-east-02a`, `cw-rno2a`, and `cw-us-east-08a`.
- Approved production node counts: 2, 4, 8, and 16, subject to the target grid.
- Excluded cluster: `cw-us-west-04a`.
- Priority band: `batch`.
- Operations document:
  `experiments/exp232_sweep_cv1_decontam/exp232_cw_operations.md`.
- Authoritative ledger:
  `scratch/exp232_cw_s01/exp232_cw_sweep.sqlite`.
- In the TPU-oriented persistence schema, `chips` means GPU count, `region` is the
  shared CoreWeave run domain, and `tpu_slice` stores the exact CoreWeave target.

## Operating Policy

- `heartbeat_every=30m`, `reslice_after=1h`, `restart_after=3h`, and
  `pending_target_limit=1` per exact target. A separate relocation timer is not
  applicable because every approved cluster shares the run and checkpoint
  identity; cross-cluster moves are reslices.
- At every heartbeat, read this file and relevant code, build the SQLite snapshot,
  query W&B first, persist observations, rebuild the snapshot, make one fleet
  decision, act, and report. Include the `open-athena/MarinFold` route and
  `--user eczech` submission invariant in every heartbeat report.
- Preserve any dispatch with a new W&B `run_progress` high-water mark within one
  hour. Consider any other dispatch for a different eligible node shape or cluster.
- After three hours without progress, restart or reslice from the latest shared
  checkpoint. After roughly three rapid preemptions with short runtimes, a fresh
  unique same-target submission may be used earlier because repeated Kueue gating
  can attach to the workload rather than the target.
- Observe the whole W&B fleet before classifying a failure. Retry an isolated
  failure after stopping its exact root; pause replacements and investigate when
  failures recur or correlate across independent trials or targets.
- Keep at most one active dispatch per trial and one pending dispatch per exact
  target. Admission of a planned dispatch must keep the fleet at or below 640 GPUs.
- Prefer measured target throughput and recent progress. Prior exp199 behavior is
  cold-start evidence only; it never changes eligibility or becomes permanent
  placement policy.
- Routine Iris operations are limited to submitting/stopping an exact root and
  checking whether an exact dispatch is running. W&B is the source of training
  liveness and progress.
- Every nonterminal heartbeat schedules exactly one time-based next pass for 30
  minutes later. Do not create a polling daemon or event-driven monitor.
- On full completion or the time limit, stop remaining dispatches, verify final
  checkpoints and SQLite integrity, report, and schedule no further heartbeat.

| Cluster | Bucket | GPU | Nodes | GPUs | State | Reason |
| --- | --- | --- | ---: | ---: | --- | --- |
| `cw-us-east-02a` | `marin-us-east-02a` | H100 | 2 | 16 | `eligible` | — |
| `cw-us-east-02a` | `marin-us-east-02a` | H100 | 4 | 32 | `eligible` | — |
| `cw-us-east-02a` | `marin-us-east-02a` | H100 | 8 | 64 | `eligible` | — |
| `cw-us-east-02a` | `marin-us-east-02a` | H100 | 16 | 128 | `eligible` | — |
| `cw-rno2a` | `marin-us-east-02a` | H100 | 2 | 16 | `eligible` | — |
| `cw-rno2a` | `marin-us-east-02a` | H100 | 4 | 32 | `eligible` | — |
| `cw-rno2a` | `marin-us-east-02a` | H100 | 8 | 64 | `eligible` | — |
| `cw-rno2a` | `marin-us-east-02a` | H100 | 16 | 128 | `eligible` | — |
| `cw-us-east-08a` | `marin-us-east-02a` | GB200 | 2 | 8 | `eligible` | — |
| `cw-us-east-08a` | `marin-us-east-02a` | GB200 | 4 | 16 | `eligible` | — |
| `cw-us-east-08a` | `marin-us-east-02a` | GB200 | 8 | 32 | `eligible` | — |
| `cw-us-east-08a` | `marin-us-east-02a` | GB200 | 16 | 64 | `eligible` | — |

## Change Record

None.
