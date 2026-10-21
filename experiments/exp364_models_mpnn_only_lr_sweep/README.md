---
marinfold_experiment:
  issue: 364
  title: 'exp: ProteinMPNN-only training for two epochs with a learning-rate sweep'
  kind: models
  branch: main
---

# exp: ProteinMPNN-only training for two epochs with a learning-rate sweep

**Issue:** [#364](https://github.com/Open-Athena/MarinFold/issues/364) · **Kind:** `models` · **Branch:** `main`

## Question

How does the current default 1.47B contacts-v1 model train when its training data contains only ProteinMPNN redesigns, with no native-sequence training documents?

## Hypothesis

Training exclusively on inverse-folded sequences may retain structural learning while changing generalization to natural versus designed proteins. A three-point learning-rate sweep tests optimization sensitivity without changing the default architecture or other training settings.

## Background

The current default is #277's scratch-trained `contacts-v1-exp277-m2-p06-full-epoch-1.5B` (final checkpoint step 266344), as recorded in LATEST_RESULTS.md. Its #266 redesign subset contains exactly 162,574,724 documents (31,702,680 AFDB plus 130,872,044 ESM-Atlas), approximately the requested 166 million. Existing audited token caches reside on CoreWeave and contain 174,107,898,831 raw tokens. Native documents are excluded from training; the default validation set remains unchanged.

## Approach

Train three models from scratch for two complete epochs of the redesign subset, with peak learning rates 0.0005, 0.001, and 0.002. Preserve exp277's 24-layer Qwen3 (hidden 2048, MLP 8192, 32 attention / 8 KV heads, QK norm and RoPE), tokenizer, sequence length 8192, global batch 128, AdamW weight decay 0.2 and other optimizer parameters, WSD warmup 10% / cooldown 20% / minimum ratio 0.1, model/data/augmentation seeds, document packing and clipping, blocked cross-document attention, validation, and scheduled amino-acid statement-order augmentation. Scale the schedule over the requested two-epoch duration. Traverse the complete concatenated redesign corpus once per epoch with a fresh deterministic shuffle, without equal-source reweighting, oversampling, or native training documents.

Audit the packed-example counts before launch, enforce exact finite two-epoch coverage, and retain stable per-trial W&B/checkpoint identities for recovery. Use existing CoreWeave inputs and available batch-priority GPUs; TRC remains permitted if complete regional data already exists. The separately requested TPU copy has explicit approval for the pinned us-east5 input transfer described below; other cross-region transfers are not authorized. Keep W&B routing at open-athena/MarinFold. Launch separate smoke identities before production.

## Success criteria

All three trials complete two full redesign epochs with verified final native checkpoints. Preserve per-trial run history, learning curves, provenance, and final HF exports with co-located tokenizers. Report validation losses at matched exposure and any operational failures. Contact benchmark evaluation is a follow-up rather than a prerequisite for launching these requested runs.

## Results

The 2026-10-13 in-region audit (`/bizon/exp364-lr1e-3-audit-a01`) succeeded and independently reproduced the source counts in `data/epoch_corpus_counts.csv`: 23,921,189 packed examples per epoch. Two epochs expose exactly 47,842,378 examples; 373,769 updates at batch 128 leave 74 real examples and 54 padding examples in the last batch. The first epoch boundary is crossed without padding or repeating examples. The inherited 8192-token clipping policy removes one trailing token from each of 772 documents per pass; no document is dropped.

Six CPU tests pass. They compare the resolved recipe directly against exp277, enforce redesign-only training with unchanged validation, exercise complete traversal and deterministic resume across the epoch boundary through Levanter's actual loader, and check that resizing a gang does not change production identity. The default run's recorded W&B model and optimizer settings were also checked; `data/default_recipe.json` preserves that reference.

The isolated [64-H100 smoke](https://wandb.ai/open-athena/MarinFold/runs/contacts-v1-exp364-mpnn-only-2ep-lr1e-3-1.5B-smoke-a01) completed all ten updates with finite losses. Native step-9 metadata and HF weights plus `tokenizer.json` / `tokenizer_config.json` were verified in S3. Metrics are in `data/smoke.csv`; its short-run losses are only a pipeline check.

Production launched with 32 H100s per trial on `cw-us-east-02a` at batch priority (96 GPUs total). The four-sequence microbatch is below exp199/exp232's measured limit of eight sequences per H100; global batch remains 128. The 64-H100 smoke validated the complete two-epoch data path, training, validation and checkpoint/export behavior. Gang size is independent of run identity, so capacity changes resume the same trial.

| Peak LR | Production W&B | Initial Iris root |
| --- | --- | --- |
| 0.0005 | [lr5e-4](https://wandb.ai/open-athena/MarinFold/runs/contacts-v1-exp364-mpnn-only-2ep-lr5e-4-1.5B) | `/bizon/exp364-lr5e-4-train-a01` |
| 0.001 | [lr1e-3](https://wandb.ai/open-athena/MarinFold/runs/contacts-v1-exp364-mpnn-only-2ep-lr1e-3-1.5B) | `/bizon/exp364-lr1e-3-train-a01` |
| 0.002 | [lr2e-3](https://wandb.ai/open-athena/MarinFold/runs/contacts-v1-exp364-mpnn-only-2ep-lr2e-3-1.5B) | `/bizon/exp364-lr2e-3-train-a01` |

All three trials were registered as running in W&B on 2026-10-13 at 15:10 UTC. Their live saved configurations were checked against the actual default run: model fields and all optimizer values match except the requested learning rate; the exact two-source training subset, global batch 128, 373,769-step target, and augmentation duration were verified. Final results are pending.

### Validation comparison, 2026-10-21

The current curves remain above the completed native-plus-ProteinMPNN exp277 run at matched optimizer steps. All six saved W&B configurations, including the one-epoch cooldown fork, agree on the model, tokenizer, validation component and cache, context length, precision, blocked document attention, and full (uncapped) validation. Global training batch is 128 throughout. Equal steps therefore match packed token slots, although padding and corpus composition mean they do not match the number of real tokens or proteins.

| exp364 peak LR | Latest validation step | Redesign-only loss | exp277 loss at the same step | Gap (nats/token) |
| --- | ---: | ---: | ---: | ---: |
| 0.0005 | 181,804 | 3.09300 | 3.06074 | +0.03226 |
| 0.001 | 205,058 | 3.12220 | 3.05860 | +0.06360 |
| 0.002 | 101,472 | 3.15598 | 3.07419 | +0.08179 |
| 0.0005, one-epoch fork | 143,752 | 3.10221 | 3.06680 | +0.03541 |

At the common step 101,472, redesign-only losses are 3.11629, 3.13460, and 3.15598 in ascending LR order, against exp277's 3.07419. The smallest LR continues to lead this sweep; none has reached its cooldown. The one-epoch 5e-4 fork has advanced to step 145,821 in this 13:49 UTC snapshot, still before its cooldown boundary at 149,508. Its latest loss is 3.10221 versus its parent's 3.10134 at the same step 143,752. These measurements do not yet test the effect of cooldown. Only new fork validation measurements are plotted; the inherited parent history is not presented as new fork measurements.

The completed [exp277 first epoch](https://wandb.ai/open-athena/MarinFold/runs/contacts-v1-exp277-m2-p06-full-epoch-1.5B) ended at loss **2.98441**, step 266,344 (best 2.98274 at 264,250). Its [additional full-corpus epoch](https://wandb.ai/open-athena/MarinFold/runs/contacts-v1-exp277-m2-p06-full-epoch2-from213072-1.5B) ended at **2.98099**, absolute step 479,417. The latter restores the first run's pre-cooldown checkpoint at 213,072 and adds a fresh full-corpus pass; these are two completed trajectories, not independent seeds. The first run was already at about 3.059 before cooldown, so cooldown alone does not explain the current matched-step gap.

Schedule timing remains a confound: the two-epoch redesign-only schedule warms up over 37,376 updates, versus 26,634 for the first full-data run, and its amino-acid augmentation ramps over 373,769 rather than 266,345 updates. These are preliminary LM-validation results, not measurements of contact accuracy or a final causal estimate of removing native sequences.

![Validation curves against completed full-data runs](plots/validation_comparison.png)

`compare_validation.py` reproduces the figure from `data/validation_comparison.csv`; `--refresh` retrieves current W&B histories, checks every expected validation step, and verifies the saved validation configurations against the full-data run. Completed curves come from exp277's audited committed CSVs, including the continuation's deduplicated resume history. `data/validation_comparison.json` records source hashes, W&B links, observed progress, schedules, and the comparison contract.

The [public comparison artifacts](https://huggingface.co/buckets/open-athena/MarinFold/tree/data/exp364-models-mpnn-only-lr-sweep/validation/2026-10-21) include the source CSV, provenance, PNG, PDF, generating scripts, and summary slides. From the repository root, reproduce with `uv run --project experiments/exp364_models_mpnn_only_lr_sweep --frozen --with matplotlib python experiments/exp364_models_mpnn_only_lr_sweep/compare_validation.py`; append `--publish` to upload the saved snapshot with the HF CLI.

### One-epoch cooldown fork, 2026-10-19

The user requested an independent cooldown of the 5e-4 trajectory while preserving its existing two-epoch run. `cooldown_train.py` restores the full trainer state from parent checkpoint **139,053**; the next optimizer/data step is **139,054**. The checkpoint was frozen with a same-region server-side copy under `fork-sources/contacts-v1-exp364-mpnn-only-2ep-lr5e-4-1.5B/checkpoints/step-139053`. `data/cooldown_source.json` records all 105 objects (17.66 GB), source/destination paths, byte-size and boundary-byte checks, and scalar counters read back through TensorStore. The copied optimizer and learning-rate counters all equal 139,054, with LR 0.0005.

The fork retains the original first-epoch data permutation, absolute data offset, augmentation ramp over 373,769 updates, and historical 37,376-update warmup. LR stays at 0.0005 through update 149,508, then follows the default linear cooldown over the last 37,377 updates (20% of one epoch) toward 0.00005. Training stops after **186,885 total updates**, with final checkpoint **186,884**. Its final batch contains 37 real packed examples and 91 masked padding examples, so exactly 23,921,189 packed examples have been consumed across the inherited prefix and fork. This is 47,831 additional updates after the source checkpoint, not another full epoch.

The independent run identity is `contacts-v1-exp364-mpnn-only-1ep-lr5e-4-cooldown-1.5B`. The parent keeps its existing identity, two-epoch target, and schedule; no parent configuration or dispatch is changed. Ten CPU tests now pass, including actual loader resume, equality of augmented first-epoch examples, final padding masks, LR boundary checks, and independent checkpoint identities. The existing full-state continuation pattern and validated H100 microbatch profile are reused; this fork has not had a separate GPU smoke.

Submitted from commit `1e5136a2` at 2026-10-19 13:32 UTC as [Iris root `/bizon/exp364-lr5e-4-cooldown-a01`](https://iris.oa.dev/#/job/%2Fbizon%2Fexp364-lr5e-4-cooldown-a01). Its child `/bizon/exp364-lr5e-4-cooldown-a01/exp364-train-4f1ab581` uses one 8-H100 node at batch priority. After initial queue admission, [the fork registered in W&B](https://wandb.ai/open-athena/MarinFold/runs/contacts-v1-exp364-mpnn-only-1ep-lr5e-4-cooldown-1.5B) at 13:45 UTC. At 14:28 UTC it had advanced to step 139,261 with finite training loss and approximately 100,430 token slots/s; the parent continued through step 140,314. The fork's live saved source, stopping target, cooldown and augmentation settings were verified, and its run history is recorded under `history/runs/`.

### Independent v5p copy, 2026-10-21

The requested additional TPU copy is launched as `contacts-v1-exp364-mpnn-only-1ep-lr5e-4-cooldown-1.5B-tpu-us-east5`. It restores the same frozen step-139053 source as the GPU cooldown and ends at step 186884. Both the GPU cooldown and its two-epoch parent continue independently. `tpu_cooldown.py` reuses the existing scientific configuration, switches to TPU Splash attention, and supports v5p-8 or v5p-64 in us-east5-a (4 or 32 physical chips), microbatch one, global batch 128, and batch priority on both driver and child. The TPU runtime has its own lock under `tpu/`; the GPU lock is unchanged.

**Transfer completed and independently verified:** the complete redesign caches and source state were absent from the inspected TPU-region prefixes. The user approved the pinned transfer and subsequent v5p smoke and production launch; input staging was submitted as `/bizon/exp364-tpu-cooldown-stage-a01` at 15:43 UTC from commit `f238a3cf`. `data/tpu_transfer_plan.json` pins the source and destination paths, counts, byte totals, and inventory SHA-256 hashes. Required ingress is **267,730,453,188 bytes**: AFDB cache 60,773,906,131; ESM cache 189,299,410,660; full-state checkpoint 17,657,136,397. The existing exp232 validation cache in us-east5 exactly matches all 17 source paths and sizes (66,618,514 bytes), with the same 41,954 documents / 47,821,958 tokens. No validation transfer is needed. Staging finished successfully at 15:59 UTC without failures or preemptions. At 16:00 UTC, `verify_tpu_inputs.py` independently matched all 21,936 destination paths, sizes, source ETags and verified GCS MD5 values to the pinned inventories. TensorStore readback confirmed step and optimizer counters of 139,054 and LR 0.0005; `data/tpu_input_verification.json` records the evidence. The original v5p-64 smoke (`/bizon/exp364-tpu-cooldown-smoke-a01`, commit `008ddb22`) remained unallocated while the pool entered provisioning backoff; it was cancelled before registering in W&B. The smaller v5p-8 smoke was submitted as `/bizon/exp364-tpu-cooldown-smoke-a02` at 16:12:49 UTC from commit `ea6239b8` and registered in W&B at 16:15 UTC on four physical chips. The replacement smoke completed successfully at 16:32 UTC with zero failures or preemptions.

`stage_tpu_inputs.py` checks the pinned source inventories before any copy, streams to regional GCS with bounded memory, verifies each destination's MD5, supports verified-object reuse on retries, and writes a completion marker only after all inventories match. The training entry point requires that marker and checks regional cache counts and source metadata. The previous thirteen local tests pass, with an additional passing TPU placement test covering the unchanged GPU configuration, finite traversal/resume, exact optimizer and data equivalence across backends, separate run paths, and transfer inventory rejection when an object changes or disappears. The full-state TPU smoke passed all ten updates (139054–139063), with finite train losses 2.7705–2.8203 and the exact accelerated LR sequence from 5e-4 to 9.5e-5. Steady update time was 27.09 seconds on four chips. Its two-batch validation loss of 3.21906 is a pipeline check, not comparable to full validation. At 16:33 UTC, `verify_tpu_smoke.py` independently checked the ten metric rows, native final counters of 139064, both indexed HF weight shards and tokenizer files. Evidence is in `data/tpu_smoke.csv` and `data/tpu_smoke_verification.json`. Reproduce with `uv run --project experiments/exp364_models_mpnn_only_lr_sweep/tpu --frozen python -m experiments.exp364_models_mpnn_only_lr_sweep.verify_tpu_smoke --attempt 2`.

The launch commands, run from the repository root with `uv run --project scripts python experiments/exp364_models_mpnn_only_lr_sweep/launch_tpu.py`, are `stage --attempt 1` (only after transfer approval), `smoke --attempt 2 --tpu v5p-8`, then `train --attempt 1 --tpu <v5p-8|v5p-64>` after verifying the smoke checkpoint and metrics. Append `--dry-run` to inspect a source bundle without submission. Production was submitted as [Iris root `/bizon/exp364-tpu-cooldown-train-a01`](https://iris.oa.dev/#/job/%2Fbizon%2Fexp364-tpu-cooldown-train-a01) at 16:33 UTC from commit `33cc077f`, requesting v5p-64 (32 chips / eight VMs) at batch priority. The source remains the original frozen checkpoint 139053, not the accelerated smoke endpoint. The regional v5p-64 pool initially had provisioning backoff. Its child `/bizon/exp364-tpu-cooldown-train-a01/exp364-tpu-cooldown-5e68abef` registered in W&B at 16:51:58 UTC and is training. At 17:50 UTC it had reached step 139702, with full validation loss 3.10391 and approximately 244,184 packed token slots/s (4.29 seconds/update). That rate implies about 56 compute hours remaining, excluding validation, checkpointing and interruptions. The production history is recorded under `history/runs/`. All TPU outputs remain under `gs://marin-us-east5/protein-structure/MarinFold/exp364_models_mpnn_only_lr_sweep/`; the TPU run cannot write the GPU paths.

### Contact evaluation, 2026-10-21

Score the latest complete checkpoints of the three learning-rate trials and the GPU one-epoch fork on **eval-val (97 natural proteins) and eval-denovo (19 designs)**. Eval-test is excluded. All-range and long-range R-precision are reported separately for the two sets. No complete new scores are available yet.

The user approved local CoreWeave scoring and subsequently approved extending the token limit for capped samples. The refreshed `latest-02` snapshot pins steps **184314 (5e-4), 207865 (1e-3), 105722 (2e-3), and 146264 (one-epoch 5e-4 fork)**. These are unequal training exposures; the fork remains before cooldown at 149508. The TPU copy is not included. `data/contact_eval/plan.json` records source object sizes/ETags, model identities, evaluation inputs, approvals and sampling policy.

CoreWeave storage writes recovered after the earlier account suspension. All four native sources are now frozen in the same CoreWeave bucket under `s3://marin-us-east-02a/MarinFold/exp364_models_mpnn_only_lr_sweep/evals/rollout-v2/2026-10-21/latest-02/sources/`. An independent inventory matched every file and byte count and read back the expected steps; `freeze_verification_latest_02.json` records 124 files totaling 70.63 GB. Four batch CPU jobs `/bizon/exp364-contact-export-<trial>-a05` were accepted from commit `fd1e25f3` and are exporting the model weights with their tokenizers to reusable regional HF-format directories. GPU retries read these exports, avoiding repeated native-state staging and conversion. Model bytes stay on CoreWeave; only small evaluation artifacts are published publicly.

The extended worker is a narrowly changed copy of the validated exp277 rollout worker (original SHA-256 `dd2f76dd5d34d1549e0be1197d9053d6e3aa0ef909f062ff5355d65f31cd571c`). Initial sampling, prompt realizations, target order, vote parsing and exp89 metrics are unchanged: 100 rollouts, temperature 1, top-p 0.95, top-k disabled, seed 0 and initial budget 6L+128. Only length-capped samples continue. Their original prompt and generated token IDs are retained exactly, and the suffix is sampled within the remaining 8192-token context using a deterministic SHA-256-derived independent seed. Completed samples are untouched. A sample that exhausts the model context remains incomplete and cannot enter a complete score.

Per-part extension records retain the original budget and seed, preserved-prefix digest, continuation seed, added budget/tokens, final length and stop reason. Timing records include extension counts and generated tokens; inference timing includes the continuation. The result manifest records both worker revisions and the policy. Reporting independently checks all 116 targets, twenty metric rows per protein, checkpoint labels, timing coverage, zero unfinished samples and consistent extension counts before calculating macro means and protein-bootstrap intervals. Viral/nonviral and frozen low-MSA subgroup tables remain separate. The natural low-MSA targets are outside the requested sets; the evaluator does not read eval-test to fill them. Thirty-five CPU checks pass, including actual local-filesystem reconstruction, held-out exclusion, incorrect-result rejection, prefix preservation, completed-sample preservation and context exhaustion.

The previous `latest-01` attempt is preserved in `plan_latest_01.json` and the public artifact prefix. Three two-epoch smokes passed; the fork capped 2/100 samples on the 30-residue design `5sbj_A`. Repeated preemptions delayed full scoring, and temporary source objects disappeared before later retries could read them. No full-evaluation protein group completed. The three terminal failed jobs are preserved; the remaining outdated 1e-3 job was cancelled before refreshing the snapshot. The new frozen sources prevent that retention failure. `data/contact_eval/launches.json` retains every accepted attempt.

Launch from the repository root with `uv run --project experiments/exp364_models_mpnn_only_lr_sweep/eval_client --frozen python -m experiments.exp364_models_mpnn_only_lr_sweep.launch_eval_direct --submit --attempt <N> --export-only` for CPU exports; omit `--export-only` after all export manifests verify to launch the four one-H100 evaluations. Omit `--submit` for a dry run. The isolated client lock is compatible with the October 21 Iris controller; neither training lock is changed. Do not submit duplicate writers. `eval_prepare --snapshot <dated-unique-name>` creates a new inventory and must not overwrite a running snapshot.

Fetch completed results with `uv run --project experiments/exp364_models_mpnn_only_lr_sweep --frozen python -m experiments.exp364_models_mpnn_only_lr_sweep.eval_report --fetch`. [Public artifacts](https://huggingface.co/buckets/open-athena/MarinFold/tree/data/exp364-models-mpnn-only-lr-sweep/contact-eval/2026-10-21/latest-02) retain score parts, per-protein timings, extension diagnostics, source bundles and provenance. The original failed smoke diagnostics remain at `data/contact_eval/fork_smoke_failure.json`.

Completed full-data reference measurements are in exp277's `data/eval_rollout_v2_epochs/epoch_comparison.csv`. First-epoch R-precision is 0.55510 / 0.53706 on eval-val and 0.69640 / 0.67497 on eval-denovo (all / long). The completed continuation scores 0.55750 / 0.53935 and 0.68600 / 0.64095 respectively. The continuation had eight capped samples on `5sbj_A`; a comparison with extended complete samples must disclose this method difference. The existing native-corpus sequence-KNN null is not matched to redesign-only training, so these results cannot support a redesign-corpus homology claim.

## Reproduction

`launch.py audit` verifies the regional token caches and recomputes packed-example counts. `launch.py train-smoke --trial lr1e-3 --nodes 8 --attempt 1` launches the isolated ten-update gate. `launch.py train --trial <lr5e-4|lr1e-3|lr2e-3> --nodes 4 --attempt 1` launches one production trial. Run launch commands with `uv run --project scripts python experiments/exp364_models_mpnn_only_lr_sweep/launch.py` from the repository root. The launcher uses the experiment's committed lock and source bundle on workers.

`observe.py` reads the experiment's W&B group and optionally persists observations in the sweep SQLite database; it makes no placement or recovery decisions. `utilization.py` reads current Iris v3 fleet availability (free/held accounting is unchanged from v2; v3 changes priority-band ordering). Launch and recovery decisions remain agent-owned.

`launch.py cooldown --trial lr5e-4 --nodes 1 --attempt 1` submits the independent one-epoch fork. Its final native checkpoint is `runs/contacts-v1-exp364-mpnn-only-1ep-lr5e-4-cooldown-1.5B/checkpoints/step-186884` beneath the same experiment prefix. Optional `cooldown-smoke` uses an isolated identity and ten additional updates with an accelerated ten-update decay; it does not consume the production identity.

Each trial writes to `s3://marin-us-east-02a/MarinFold/exp364_models_mpnn_only_lr_sweep/runs/<run-id>/`, with native final checkpoint `checkpoints/step-373768` and the default trainer's HF export. Preserve the tokenizer with published exports. The run IDs are generated by `config.py` and stay fixed across attempts and gang sizes.

## Conclusion

Training is in progress. As of 2026-10-21, all three redesign-only trials have higher LM validation loss than the full-data run at matched optimizer steps; 5e-4 leads this sweep. The one-epoch fork now has validation measurements but has not reached its cooldown boundary. Repeated batch-priority preemptions have slowed wall-clock progress; checkpoint recovery is working. Final checkpoints and contact evaluations are still needed, and differing schedule durations limit attribution to the training subset alone.
