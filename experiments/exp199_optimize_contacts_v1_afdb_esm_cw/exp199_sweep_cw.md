# exp199 CoreWeave Sweep Operations

> This document governs the CoreWeave exp199 training sweep managed with the
> `run-training-sweep` skill. Read it in full at the start of every heartbeat
> before inspecting code, SQLite, W&B, or Iris.

## Invariants

- Do not inspect, mutate, submit, stop, or account for the sibling TRC sweep.
- Submit from this directory so Iris bundles only the CoreWeave workspace.
- A logical trial has one W&B identity and one shared S3 checkpoint root. Never
  allow two live dispatches to write that identity.
- Treat all production CoreWeave clusters as one shared storage/run domain.
  Moving a trial between clusters resumes the same run; it is not regional racing.
- Use existing S3 token caches only. Never copy, rebuild, or retokenize data in the
  experiment DAG or operating loop.
- Use `batch` priority on the root driver and let the child GPU gang inherit it.
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
- Operations document: this file.
- Dynamic state (repository-relative):
  `scratch/exp199_cw_s02/exp199_cw_sweep.sqlite`.

## Operating Policy

- `heartbeat_every=30m`
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
