# exp232 CoreWeave Sweep Operations

> This document governs the exp232 CoreWeave GPU sweep managed with the
> `run-training-sweep` skill. Read it in full at the start of every heartbeat
> before inspecting code, SQLite, W&B, or Iris.

## Invariants

- The experiment code owns training semantics. Its catalog declares ten trials at
  version `2026.08.14.2` (`s02`), but `m1-p01-aug`, `m1-p04-aug`, `m1-p03-aug`,
  `m2-p01-aug`, `m2-p04-aug`, and `m2-p03-aug` are operator-abandoned after
  divergence; operate only the remaining four and never redispatch any of them.
  Every abandoned trial belongs to point `p01`, `p03`, or `p04`; both `p02` and
  both `p06` trials remain healthy.
- Maintain at most one active writer for each trial's shared W&B ID and S3
  checkpoint root. CoreWeave target changes are reslices of that same run; never
  race clusters or GPU families.
- Every Iris root submission uses the `marin` federation controller, an exact
  `--target-cluster`, `--priority batch`, and `--user eczech`. The target cluster
  and the script's `CLUSTER` value must match.
- W&B routing is `open-athena/MarinFold`. Validate that the authenticated account
  has the required model seat before smoke; if access fails, stop and ask the
  operator rather than falling back to a personal entity or project.
- The child GPU environment retains exp199's pinned Marin, Iris, JAX, Torch,
  cuDNN, and NCCL versions. The root is a CPU driver and intentionally omits the
  `gpu` extra; `ResourceConfig.with_gpu` selects it for the training child. The
  child differs from exp199 only by using exp199's exact direct cuDNN wheel URL
  during Iris's CUDA-precedence reinstall, bypassing the failing placeholder.
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
- The training-and-validation smoke is `m1-p06-aug` on one `cw-us-east-02a`
  H100 node for ten steps followed by the full validation set. Its
  script-generated W&B/checkpoint identity is distinct from every production
  run.

## Operator Choices

- Time limit: 14 days from the first production dispatch.
- Regional replicas per trial: 1; represented as one shared CoreWeave run.
- Maximum active compute: 640 GPUs.
- Approved clusters: `cw-us-east-02a` and `cw-rno2a`.
- Unsupported cluster: `cw-us-east-08a`. Its GB200 workers are ARM, while the
  locked CUDA 13 cuDNN version publishes only an x86_64 wheel.
- Approved production node counts: 2, 4, 8, and 16, subject to the target grid.
- Excluded cluster: `cw-us-west-04a`.
- Priority band: `batch`.
- Operations document:
  `experiments/exp232_sweep_cv1_decontam/exp232_cw_operations.md`.
- Authoritative ledger:
  `scratch/exp232_cw_s02/exp232_cw_sweep.sqlite`.
- PR #233 updates are operator-directed only. Do not post sweep status or
  heartbeat updates unless the operator explicitly supplies or requests the post.
- Abandoned trials: `m1-p01-aug`, `m1-p04-aug`, `m1-p03-aug`, `m2-p01-aug`,
  `m2-p04-aug`, and `m2-p03-aug`. They are outside recovery and completion scope.
- Completed and checkpoint-verified: `m1-p06-aug` and `m1-p02-aug`. Never
  redispatch them. Two trials remain in scope: `m2-p02-aug` and `m2-p06-aug`.
- Both remaining trials are at the `n16` 128-GPU ceiling, the largest approved
  node count. No further enlargement is possible, so the sweep can draw at most
  256 GPUs regardless of visible free capacity, and GPUs freed by a completing
  trial cannot be redistributed.
- Keep deliberate free-node slack on `cw-us-east-02a`. Its 32 GPU nodes are shared
  with a production band that outranks batch, so filling all 32 forces an eviction
  of one of our gangs; that happened six times. Leaving at least 8 free nodes
  absorbed the tenant with no eviction.
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
- An explicit operator request to consume newly visible batch H100 capacity may
  reslice recently progressing dispatches to larger eligible gangs. Preserve the
  one-writer invariant, stop before replacement, and remain within the 640-GPU cap.
- After three hours without progress, restart or reslice from the latest shared
  checkpoint. After roughly three rapid preemptions with short runtimes, a fresh
  unique same-target submission may be used earlier because repeated Kueue gating
  can attach to the workload rather than the target.
- Before calling a flat or slow window a problem, check whether the run is at a
  permanent-checkpoint boundary: a multiple of `PERMANENT_CHECKPOINT_EVERY`
  (14520). A frozen `global_step` at such a boundary, alongside a current W&B
  heartbeat, means a write is in flight; confirm by listing the run's S3
  checkpoint prefix for that `step-<boundary>` directory and reading its object
  timestamps. Never stop a dispatch mid-write — that discards the checkpoint and
  buys a further ~32 minute startup.
  A HEALTHY write is fast: `m2-p06` wrote all 16.44 GiB of `step-130680` in 19
  seconds across 25 files and never visibly paused. So the duration is itself a
  diagnostic. A boundary write that blocks for many minutes — `m2-p02` needed 24
  minutes and 124 files for identical content — indicates a degraded write path
  on that gang, not a normal cost. Treat the slow case as a performance fault to
  measure, and distinguish it from the training loop: poll `global_step` over a
  minute or two, because a gang can hold full in-loop speed while losing half its
  wall clock to IO, and the throughput metric will not show it.
- Do not trust `.executor_status.lock` as proof of training. It is refreshed by
  the CPU root driver, so a fresh lock only rules out total process death (the
  `a06` wedge, whose lock went stale for 34 minutes); the training child can be
  dead underneath it. The W&B-independent proof of training is
  `checkpoints/eval_metrics.jsonl`, written from inside the training loop
  straight to S3 every `STEPS_PER_EVAL` (2114) steps. If its mtime is well past
  the next boundary's due time, training has stopped no matter what W&B or Iris
  say.
- When several trials go W&B-silent at once, suspect a W&B outage before
  suspecting simultaneous training failures, and resolve it with the
  `eval_metrics.jsonl` test rather than by stopping anything. A recovered
  heartbeat with a still-frozen `global_step` refutes the outage reading: a real
  reconnect flushes a large step jump. Stopping healthy runs during a monitoring
  outage destroys hours of work, so the independent check is worth the wait.
- Treat a W&B heartbeat older than roughly 60 seconds as a prompt to check that
  trial's exact Iris root in the same pass. A fresh progress high-water in the
  same window does not clear it, because the last logged progress can precede
  the crash. The Iris check is cheap and the alternative is losing a full
  heartbeat interval on a run that gates completion.
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
| `cw-us-east-08a` | `marin-us-east-02a` | GB200 | 2 | 8 | `invalid` | locked CUDA 13 cuDNN wheel has no ARM build |
| `cw-us-east-08a` | `marin-us-east-02a` | GB200 | 4 | 16 | `invalid` | locked CUDA 13 cuDNN wheel has no ARM build |
| `cw-us-east-08a` | `marin-us-east-02a` | GB200 | 8 | 32 | `invalid` | locked CUDA 13 cuDNN wheel has no ARM build |
| `cw-us-east-08a` | `marin-us-east-02a` | GB200 | 16 | 64 | `invalid` | locked CUDA 13 cuDNN wheel has no ARM build |

## Change Record

- 2026-08-17T23:14:12Z: `m2-p02-aug`, the sweep's final finisher, failed in
  isolation and was replaced. Iris reported `failures=1`, `preemptions=0`, task
  exit 1 after 12h59m, surfacing a nanobind crash through marin's `StepRunner`.
  That is the failure signature rather than the preemption signature, so
  preserving the gang would have been wrong; it is also a distinct mode from the
  exit-139 SIGSEGV family, whose tally stays at six. `m2-p06-aug` was healthy on
  RNO throughout, confirming the failure was not correlated. The exact root was
  stopped, root and child verified terminal, and same-target attempt `a12` was
  submitted at `n16` from the shared checkpoint; it admitted within 30 seconds.
  Taking 16 of east's 32 nodes preserves the free-node slack rule.
  Detection lag: the crash occurred at 22:41:51Z but was caught at 23:14Z. At the
  22:43Z pass the W&B heartbeat was 82 seconds old and was recorded as ordinary
  logging cadence, which was defensible because progress had just advanced, but
  it cost one heartbeat interval on the critical path. New rule below.
- 2026-08-17T18:33:11Z: `m1-p02-aug` completed, the second trial to finish. W&B
  reported `finished` with `run_progress` 1.0 at `global_step` 145199, and its
  final checkpoint verified reachable with `step-145199` holding 25 files and
  16.44 GiB, `.executor_status` SUCCESS, the `hf` export present, and all ten
  permanent checkpoints intact. Its Iris root and child both reported
  `succeeded`, so no stop was required. Its 128 RNO H100 were released but could
  not be redistributed because both remaining trials already sit at the `n16`
  ceiling. Also corrected the remaining ETAs: estimates drawn from single
  fifteen-minute progress deltas ran optimistic because an eval-free window
  overstates the rate. Rates computed over multi-hour baselines are stable near
  0.0235 to 0.0244 progress per hour, which moves `m2-p06` to roughly
  08-18 06:10Z and `m2-p02`, the final finisher, to roughly 08-18 07:40Z.
- 2026-08-17T10:10:19Z: The operator declared `m2-p03-aug` diverged and directed
  abandonment with compute rebalanced, and re-confirmed `m2-p04-aug` (already
  abandoned at 00:30Z, re-verified clean). Stopped and verified `m2-p03-aug`'s
  exact east n16 root, removed the trial from recovery and completion scope, and
  rebalanced its 128 east H100. Four in-scope trials remain. Also recorded the
  east free-node slack rule after six production-band evictions.
- 2026-08-17T00:30:18Z: The operator declared `m2-p04-aug` diverged and directed
  abandonment with its compute redistributed. Its earlier loss recovery did not
  hold and the p04 point resumed spiking. Stopped and verified its exact east n16
  root, removed the trial from recovery and completion scope, and redistributed
  its 128 east H100 to the remaining laggards. Five in-scope trials remain.
- 2026-08-16T17:48:20Z: The operator declared `m2-p01-aug` diverged and directed
  abandonment with all jobs stopped. Stopped and verified its exact active RNO n8
  root, removed the trial from recovery and completion scope, and released 64 RNO
  H100 for redistribution. Its loss was trending down at abandonment, so the
  retained checkpoints make the call reversible.
- 2026-08-16T17:46:46Z: The operator declared `m1-p03-aug` diverged and directed
  abandonment with all jobs stopped. It had already self-terminated on a rank-0
  exit 139; issued an explicit stop and verified no root, child, or pod remained.
  Removed from recovery and completion scope. Six in-scope trials remain.

- 2026-08-16T12:24:51Z: The operator declared `m1-p04-aug` diverged and
  unrecoverable. Stopped and verified its exact Iris root, removed the trial from
  recovery and completion scope, and made its east H100 allocation available for
  redistribution to healthy trials.
- 2026-08-16T01:28:41Z: The operator requested aggressive use of free H100
  capacity at batch priority. Added an explicit exception allowing
  checkpoint-preserving enlargement of recently progressing gangs, still bounded
  by one writer per trial and the 640-GPU fleet cap.
- 2026-08-15T15:48:44Z: The operator declared `m1-p01-aug` diverged and
  abandoned it. Stopped its exact active Iris root, verified no root or child job
  remained running, and removed the trial from recovery and completion scope.
- 2026-08-15T13:31:47Z: The operator reserved PR #233 updates for explicit
  prompts. Added that communication restriction to Operator Choices; autonomous
  sweep heartbeats remain in chat and will not be posted to the PR.
- 2026-08-14: Abandoned s01 by explicit operator instruction after discovering
  that seven early runs had started under transitive cuDNN/NCCL drift. Stopped
  its eight exact active roots, retained its W&B, checkpoint, and SQLite history,
  and restarted all ten trials from scratch as s02 with the exact exp199 GPU
  stack.
- 2026-08-14: Restored exp199's exact cuDNN `9.26.0.17.dev59162438` and NCCL
  `2.30.7` versions after detecting transitive lock drift. The exp232 child-job
  CUDA setup reuses that exact direct cuDNN wheel during Iris's post-sync CUDA
  precedence repair; the stock repair bypasses the lock and intermittently
  fetched NVIDIA's placeholder package with a mismatched hash.
- 2026-08-14: Removed `cw-us-east-08a` from production eligibility after its
  ARM root exposed that the then-locked CUDA 13 cuDNN package publishes only an
  x86_64 wheel. The H100 lock now pins that exact wheel URL directly, avoiding
  correlated multi-node cold-start failures in NVIDIA's placeholder package.
