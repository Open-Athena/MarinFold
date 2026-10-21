# Execution and numerical validation

This appendix retains the chronological launch, numerical-check and recovery
record for exp368. Scientific results are in the [report](README.md).

### Execution and recovery

Jobs are independent batch H100 root jobs on `cw-us-east-02a`, submitted through
the central `marin` controller. Inputs and weights are already in its regional
S3 store. Outputs live under
`s3://marin-us-east-02a/MarinFold/exp368/`. The standalone Iris submission client
is pinned to `af9c6789e8` because the older workstation client is rejected by the
current controller. `data/dispatches.jsonl` records exact commands, source hashes,
job IDs and monitoring ownership. Per-rollout commit markers permit restart
without losing completed trajectories. A changed code or plan requires a new
output prefix. No new model training is part of this experiment.

A temporary eight-GPU RNO server staged a shared 6.014 GB model/input copy to
use available capacity. Iris preempted it four times; `--max-retries=0` only
disabled failure retries, while preemption retries defaulted to 1000. Local
state was lost, the staging counter reset, and repeated input transfers exceeded
the intended 9.5 GB campaign ceiling. The server was canceled when this was
detected at 15:20 UTC. `data/preemption_incident.json` records the evidence.
Initial recovery launches were restricted to storage-colocated `cw-us-east-02a`.
`recover_task.py` downloads and verifies each original task's frozen source
bundle and resumes directly from its durable S3 trajectories. Twelve independent
H100 recovery jobs retain the original sampling plan and sharding. The ordinary
baseline is also queued on all 1,375 structures from the 255 remaining validation
groups, excluding every pilot group. No guided winner or test policy is selected.

After a prolonged East capacity wait, a task-owned object-store bucket was
created in `RNO2A`. `prepare_regional_mirror.py` made one 6,014,167,823-byte copy
of the inputs to
`s3://marinfold-exp368-rno2a/MarinFold/exp368/inputs-v1/`. It persists multipart
progress and reserves each 32 MiB source read before transfer, with a cumulative
7 GB cap that survives preemption. A server-side copy attempt was rejected at
its first part; its empty upload was explicitly aborted before the streaming
copy. The streaming mirror completed successfully in about 70 seconds.
`regional_inputs.py` checks the full reader, head, reconstructed-generator and
target hashes against the original pilot. Regional recovery jobs use identical
H100 hardware and original frozen task bundles, with large weight reads now
colocated. New larger phases also write to the regional bucket. After East result writes began timing out at 16:55 UTC, five guided workers
exhausted their failure retries and the remaining seven were canceled. The
14,227,449-byte snapshot (4,446 objects) was copied to
`s3://marinfold-exp368-rno2a/MarinFold/exp368/results/pilot-guided/`.
Every local object matched its source ETag and every destination object was
read back and checked against SHA256. The original immutable tasks resume from
this regional copy. `data/storage_write_incident.json` and
`data/guided_results_migration.json` retain the evidence and complete file audit.

The regional ordinary-generation bootstrap now uses the vLLM container's existing
interpreter instead of installing a second CUDA/PyTorch environment solely for
staging. A separate two-protein smoke run verified identical input and generator
hashes and completed all 8/32/128 rollout waves. Scientific recovery still executes
each task's original immutable source bundle. `watch_scoring.py` is a finite local
dependency watcher: it submits the five declared scoring tasks only after their
matching generation jobs succeed, records every launch, and exits after submission.
The active experiment monitor continues to check the scoring jobs themselves.
When the regional controller stopped reporting after 17:14 UTC, the watcher
was restarted with an artifact-based dependency check: all four source/plan-pinned
completion markers and all 2,750 planned generation groups must exist before
it queues scoring against stale scheduler status. The shared cluster was not
modified. Frequency-only analysis can run directly on raw generated contacts;
its results exactly reproduce all fields for 96 previously scored pilot maps.

The East controller and eight schedulable H100s became available during the RNO
controller outage. A fresh East storage write/read check passed. The saved
guided snapshot (14.2 MB) and complete validation-generation outputs (404.6 MB)
were copied to `s3://marin-us-east-02a/MarinFold/exp368/recovery-east-v2/results/`,
with source ETags and every destination SHA256 readback verified. The queued
RNO replacement jobs were canceled and their terminal states checked before
East replacements were submitted. A new H100 fixed-length smoke passed all
model hashes, exact L and 2L counts, nested-prefix invariants and FP32 cache checks
(maximum probability error below 9e-7). The full fixed-length pilot is now queued
alongside the resumed original pilot and larger validation scoring. Large model
reads in East use the existing East checkpoints; no new cross-region model copy
was needed. Recovery manifests and cancellation audits are retained in `data/`.

### GB200 fallback checks

After the active East workers were preempted at 18:15 UTC, all replacement pods
waited for batch admission; the East controller also stopped reporting briefly.
The healthy `cw-us-east-08a` fleet admitted a small ARM/GB200 runtime check promptly.
It passed on PyTorch 2.10.0+cu130: maximum cached/full-forward probability and
recall errors were 5.96e-8, and lookahead committed only one contact. This used a
tiny random Qwen3 model and transferred zero trained-weight bytes. It validates
the runtime, not the production model. The exact result and timing are in
`data/runtime_smoke_gb200.json` and `data/runtime_smoke_gb200_timings.csv`.

The CUDA wheel is hash-pinned from the [official PyTorch CUDA 13 index](https://download.pytorch.org/whl/cu130/torch/).
A production-model check used one 6,014,167,823-byte mirror to the existing
`marin-us-east-08a` bucket, with the same durable 7 GB reservation cap and full
input hash checks as the RNO mirror. GB200 results require separate hardware
provenance and timing comparisons. The existing vLLM 0.9.2 image is amd64-only;
the pinned official 0.14.1-cu130 ARM image would constitute a separately verified
ordinary-generation backend, not a bitwise reproduction of the H100 baseline.

The production default-backend check **failed**: replaying the same eight-rollout
batches gave maximum BF16 contact-score differences of 0.0161, 0.0295 and 0.0800
on the three length-selected references, exceeding the predeclared 0.01 guard.
The separate fixed-length smoke also failed the unchanged 1e-5 FP32 cached/full
check, with error 1.2279e-5. These failures are preserved in
`data/backend_check_gb200_default*`; they do not authorize switching the production
campaign to this runtime. Explicit math and memory-efficient attention backends
with full-precision reductions are being tested without changing acceptance
thresholds. The forced math backend still failed (maximum BF16 probability
error 0.0342); the forced memory-efficient backend was unsupported for the FP32
cache path. Pinning the official ARM CUDA 12.8 wheel to match the original H100
proofreader's `torch==2.10.0+cu128` **passed** the original BF16 replay guard:
the 193- and 937-residue reference batches match exactly, and the 33-residue
batch has maximum contact-score error 0.001523. This validates the full-forward
scoring route. The separate 48-residue cache check still fails at 1.2279e-5,
so cached guidance is not yet admitted. An explicit cuBLASLt check includes that
failing prefix and keeps the original threshold.
PyTorch documents that [different attention implementations can
produce different numerical results](https://docs.pytorch.org/docs/main/generated/torch.nn.functional.scaled_dot_product_attention.html);
that is a possible explanation, not a diagnosis established by these checks.
The separate vLLM 0.14.1 ARM generation smoke completed all six ordinary pool
waves. Cost plots now require recorded, consistent GPU hardware.

At 18:56 UTC, before inspecting new GB200 pilot accuracy, the ordinary baseline
phase was expanded to the same complete 32-protein cohort on GB200: independent
pools of 8, 32, 128, 512 and 2,048 rollouts. These reuse the frozen cohort, plans
and seeds. Their outputs and timings are separate from the H100 phase. The larger
pools test whether additional ordinary sampling is competitive with expensive
guided search; no guided backend is approved by this baseline-only expansion.
After the CUDA 12.8 full-forward replay passed, the same independent ordinary
baseline was also launched on the original 1,375-structure validation extension.
`data/gb200_ordinary_phase.json` freezes the exact plan hashes, runtime, output
location and separation from H100 results before reading GB200 cohort accuracy.

### Native H100 recovery through the existing batch queue

During recovery, a read-only Kubernetes check found 320 unallocated H100s in RNO
and an active Kueue queue despite the unavailable Iris controller. Owned native
`batch/v1` jobs were submitted to the existing `cw-rno2a-lq` / `iris-cq`, using
`iris-accelerator-batch` workload priority and `iris-batch` pod priority. Kueue
admission was verified from the job events. No shared controller or scheduling
policy was changed. This follows Kueue's [documented Job admission interface](https://kueue.sigs.k8s.io/docs/tasks/manage/run_job_with_workload_priority/).

The real-model smoke completed all eight exact-length maps and passed the
original FP32 cache tolerance, with maximum error 8.94e-7. The stopped guided
snapshot was verified and repartitioned from 12 to 48 independent workers.
Only execution partitioning changed: original source bundles, plans, protein
cohort, policies, per-trajectory seeds and saved outputs are unchanged. Original
shard pins are archived with an explicit repartition ledger. Four fixed-length
workers also resumed, and the two large-pool baseline workers were submitted.
All 22 stranded East replacements were canceled; a direct check confirmed no
remaining exp368 pods there.

The native job specifications, source hashes, object UIDs and actual submission
backend are recorded in `data/native_dispatches.jsonl`. Immutable scientific
workers retain the original `runner_tag=iris` timing convention; that field is
not evidence of which controller submitted the recovery pod. Per-input GPU,
host, software and wall-time records remain measured by the unchanged workers.
Native storage credentials are held only in an owned Kubernetes Secret and are
excluded from reports and source archives. Bootstrap failures before inference
(system-pip installation, then an absent `python` alias in the vLLM image) were
fixed with an isolated installer and explicit `python3`; they produced no
scientific results.

