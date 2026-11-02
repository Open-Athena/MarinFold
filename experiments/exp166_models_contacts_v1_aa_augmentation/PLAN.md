# Plan: evaluate the exp166 AA-augmentation checkpoint

**Issue:** [#166](https://github.com/Open-Athena/MarinFold/issues/166)
**Behavioral reference:** [PR #170](https://github.com/Open-Athena/MarinFold/pull/170)
and [issue #169](https://github.com/Open-Athena/MarinFold/issues/169)

## Goal

Evaluate the final exp166 AA-augmentation checkpoint against the final exp117
control with one concise, reproducible contacts-v1 implementation. The two
checkpoints must use identical inference and output code, but they must run as
completely separate Iris jobs.

## Fixed inputs

- Candidate: `open-athena/marinfold-exp166`, exact final HF subtree at step
  35,679.
- Control: `open-athena/marinfold-exp117`, exact final HF subtree at step
  35,679; validation loss 2.703709 and prior full-set R-precision 0.5344.
- Targets: the 554-row public exp169 `eval_targets.parquet` artifact, recorded
  by URL and SHA-256 in each run manifest.
- Code: MarinFold revision `0dcb7f56b1ea03ebd38e2337d69c1fff5203b426`
  and the dependencies frozen in `uv.lock`.

## Constraints

- Submit from this MarinFold checkout with Iris installed in this directory's
  `.venv`; do not read from or package a neighboring Marin checkout.
- Use only the `marin-dev` cluster, one `v6e-4` per job. Leave region and zone
  unset so Iris can place the TPU where capacity is available.
- Let Iris provision the worker and its locked environment.
- Download and prepare exactly one checkpoint per job using one download worker.
- Use only ephemeral `/app/scratch` for downloaded and BF16 weights. Never write
  prepared checkpoints to GCS or HF, and never delete or move GCS objects.
- Persist public vote parts, completion/timing parts, the exact prompts and
  ground-truth universe, and a manifest in
  `buckets/open-athena/MarinFold/data/contacts-v1-model-eval-exp166`.
- Reimplement the useful behavior from PR #170 without importing its experiment
  code or carrying forward its staging scripts.

## Proven execution shape

The local project pins Marin's installable Iris, TPU, vLLM, and TPU-inference
packages. Iris packages this experiment directory, resolves the frozen `tpu`
extra on the worker, and runs `eval_checkpoint.py`; no remote environment
management commands are part of the evaluator.

The runner uses tensor parallelism 4 and batches all 100 rollouts for one
protein in a single `generate` call. It writes sparse vote counts first and a
timing parquet last; the latter is the durable completion marker for a
deterministic part. A retry checks each marker and evaluates only incomplete
parts. An unreadable marker is treated as incomplete.

Checkpoint preparation is intentionally small: download the exact HF subtree,
recast floating safetensors shardwise from FP32 to BF16, copy configuration and
tokenizer files unchanged, and validate the result with Transformers. No config
or tokenizer repair was needed for the published exp166 export.

## Interactive Iris RPC development

Before submitting an unattended evaluator, one provisioned `v6e-4` task was
used as a development target through Iris's supported worker RPC, specifically
`iris rpc worker exec-in-container` (the current name of the command, rather
than direct SSH or manual VM setup). Repeated calls against the same task were
used to inspect the packaged workspace and environment, resolve the frozen
`tpu` extra, exercise HF download and shardwise BF16 conversion, validate the
Transformers/vLLM imports and runtime flags, load the model, run four real
targets with all 100 rollouts, read the resulting artifacts back, and verify a
clean engine shutdown.

That interactive loop exposed the required `VLLM_TARGET_DEVICE=tpu` setting,
the conflict caused by a global `VLLM_VERSION_OVERRIDE`, and the unsafe
fork-after-OpenMP sequence. Only after the same remote task could complete the
full preparation/load/inference/write path was it encoded in `eval_checkpoint.py`
and submitted as clean smoke jobs, followed by the unattended jobs. Iris still
owned provisioning and the environment throughout; RPC only shortened the
edit-run-debug cycle on the provisioned task.

## Scientific recipe

- 100 rollouts per protein, each with a freshly resampled contacts-v1 document
- temperature 1.0, top-p 0.95, top-k disabled
- `<end>` stop token
- maximum completion length `min(8192 - prompt_tokens, 6 * L + 128)`
- contacts filtered to valid residue pairs with sequence separation at least 6
- duplicate contacts within one rollout count once
- pure vote matrix with no pairwise tie-break
- one per-input timing row for every target

## Probe record

Completed:

1. Submitted a minimal job from this checkout to `marin-dev`.
2. Provisioned and interacted with a `v6e-4` worker in `europe-west4-a` through
   Iris RPC; no direct VM setup was used.
3. Reconstructed the locked TPU environment and loaded the pinned TPU vLLM
   stack. `VLLM_TARGET_DEVICE=tpu` is required during setup; a global
   `VLLM_VERSION_OVERRIDE` must not be set because TPU inference also consumes
   it.
4. Ran the exp166 evaluator interactively over four targets with 100 rollouts
   each. Model load and inference completed, 924 nonzero vote rows and four
   complete timing rows were written, and the artifacts were readable without
   credentials. The vLLM engine shut down cleanly afterward.
5. The first clean bundled smoke proved source packaging, frozen setup,
   single-worker download, and shardwise BF16 preparation. It also exposed an
   unsafe fork-after-OpenMP sequence: vLLM's EngineCore segfaulted while loading
   weights after the parent had recast them. The runner now forces the supported
   `spawn` start method so EngineCore receives a clean OpenMP runtime.
6. The corrected clean exp166 job completed in 4m29s: four complete timing rows,
   908 nonzero vote rows, and 389/400 stop-token completions. A completely
   separate exp117-control job completed in 5m06s: four complete timing rows,
   872 nonzero vote rows, and 400/400 stop-token completions. Both jobs exited 0
   with no failures or preemptions, and both output prefixes were verified by
   anonymous HTTP reads.

The interactive output is isolated under
`.../interactive-smoke/exp166-aaaug-step-35679`; it is diagnostic and will not
be mixed with the clean smoke or full results.

## Full-run record

The first submissions at 22:52 UTC omitted Iris's `--user eczech` flag and
therefore appeared under `/exedev`. Both were stopped while still pending; no
evaluation task ran. The submitter now tests and defaults the owner explicitly.

The 23:04 UTC submissions used the correct owner and TPU but constrained the
whole job to `europe-west4-a`; they too were stopped while pending. The next
submissions used `--region europe-west4` so Iris could place supporting CPU
resources in any zone in that region.

The region-scoped submissions remained pending while the `europe-west4`
v6e-4 autoscaler entered degraded backoff. They were stopped without a task
attempt. The final submissions remove both region and zone constraints.

Submitted as two independent unattended jobs on 2026-07-31 at 23:39 UTC:

- candidate: `/eczech/marinfold-exp166-exp166-scores-anyzone`
- control: `/eczech/marinfold-exp166-exp117-control-scores-anyzone`

The final submissions use `marin-dev`, one preemptible `v6e-4` with no region
or zone constraint, batch priority, and up to three retries. Each full output
prefix includes the exact `eval_targets.parquet` and `gt_universe.jsonl`
inputs as well as the manifest, lossless sparse vote matrices, and per-target
timing/completion rows. This is the complete expensive inference state needed
to reproduce PR #170's per-protein scores, paired comparisons, and boxplots
without another TPU run.

Iris placed both regionless first attempts on `v6e-4` workers in `us-east1-d`.
Both workers reached READY without a provisioning failure, loaded their
models, and began writing 16-target result parts. The first parts were uploaded
at 23:50 UTC.

Both first workers were preempted together at about 00:00 UTC. The candidate
had 16 complete parts (256 targets) and the control had 17 (272 targets). Iris
reprovisioned replacement `v6e-4` workers in `europe-west4-a`; the retries
reported exactly 19/35 and 18/35 parts pending and skipped all durable work.
The candidate completed at 00:42:55 UTC and the control at 00:45:48 UTC. Both
Iris jobs exited successfully with zero failures and one preemption.

## Result record

Both exact checkpoints have 554 unique timing/completion rows and 35 complete
vote parts. The control produced 1,900,486 nonzero vote rows and stopped at the
requested token in all 55,400 rollouts. The candidate produced 1,684,030
nonzero vote rows and stopped in 55,373 of 55,400 rollouts; the 27 bounded
completions remain represented in the votes and timing metadata.

| Checkpoint | All R | All L | All AUC | Long R |
|---|---:|---:|---:|---:|
| exp117 control | 0.5336 | 0.4801 | 0.9324 | 0.4826 |
| exp166 AA augmentation | **0.5618** | **0.5070** | **0.9394** | **0.5133** |

The paired all-range R-precision delta is +0.02818 over 554 proteins (standard
error 0.00284; 95% CI 0.02260–0.03375). The candidate win rate is 66.6% and
the tie rate is 12.6%. The observed control value, 0.533596, differs from the
prior 0.5344 by only 0.000804 and passes the declared 0.006 tolerance.

The single-process finalizer uploaded the per-protein metric rows, complete
summary and paired tables, timings, plots and plot metadata, checksummed
derived manifest, and both 554-matrix archives to
`data/contacts-v1-model-eval-exp166/derived/` in the public HF bucket. The
matrix archives are 5,099,520 bytes (control) and 4,741,120 bytes (candidate).
All derived artifacts can be recreated without another TPU run.

The candidate-versus-control diagnostic contains all-, short-, medium-, and
long-range R-precision distributions for both checkpoints. The final
where-we-stand artifact combines two panels: five narrow
R-precision distributions first, ordered by increasing mean R-precision, with
#75/#146 sharing gray and #117/#166 sharing dark orange in both panels. The
second panel is the loss-versus-R-precision scatter for
the three 1.5B checkpoints #75, #117, and #166 plus historical exp146 3B
(loss 2.702478, R = 0.511863). The least-squares fit remains limited to the
three 1.5B checkpoints and is extrapolated to the historical Protenix-v2
single-sequence baseline (R = 0.603158), implying a crossover loss of about
2.645.

The shared footer and metadata mark #75, #146, and Protenix-v2 as historical
exp82/exp169/exp89/exp74 results and #117/#166 as generated here. The exp146
box/scatter input is a checksummed 554-row subset of exp169's public metric
table. Protenix-v2 MSA, ESMFold, ESMFold2, and PR #170's paired-delta panel are
excluded.

## Remaining sequence

1. Keep the compact tables, timings, plots, experiment code, and summary in the
   PR branch.
2. Report the result on issue #166 only when requested.

## Success criteria

- Both exact checkpoints complete independently on `marin-dev` through the same
  evaluator and frozen environment.
- Each output contains a matching manifest and all 554 durable completion rows.
- No derived checkpoint leaves job-local scratch.
- The control validates the harness before the candidate comparison is treated
  as interpretable.
- The resulting workflow is materially smaller than PR #170 and depends only on
  installable packages plus this MarinFold experiment directory.
