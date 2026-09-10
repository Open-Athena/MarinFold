---
marinfold_experiment:
  issue: 279
  title: 'exp: exact soft contact targets with the best decontaminated training recipe'
  kind: models
  branch: exp/279-exact-soft-targets
---

# Exact soft contact targets

Implementation for [#279](https://github.com/Open-Athena/MarinFold/issues/279).
The question is whether exact generator-conditional contact targets improve
contact prediction or learning efficiency over ordinary next-token CE.

## Method

A complete contacts-v1 document supplies its selected, randomly numbered contact
set. Filtering and strongest-contact truncation have already happened in the
existing generator. No new corpus or document format is introduced.

For the remaining undirected contacts E, the first endpoint distribution is
q(a) = degree_E(a)/(2|E|). Once endpoint a is observed, the second endpoint is
uniform over a's remaining neighbors. Consume the observed edge after its second
endpoint. Contact degrees affect selection, not probability weights. Sequence
statements, framing, contact markers and end tokens retain ordinary CE.

The objective is categorical cross-entropy, -sum(q * log p). Future ground truth
enters only this objective. Transformer inputs, causal/segment masks, numbering,
RoPE positions and sampled prefixes are unchanged.

## Implementation

- `targets.py` validates complete serialized documents and stores ordered edges
  plus per-prediction suffix ranges. Targets are built from full source rows
  before applying the stock packer's slices. This preserves future contacts even
  beyond a packed fragment and excludes already consumed contacts.
- `data.py` uses the stock packer, component shuffle and mixture. It reuses
  exp232's deterministic sequence-statement augmentation. Both arms receive
  identical examples and metadata; the CE arm ignores the contact targets.
- `model.py` subclasses Qwen3 only to change training loss. Parameters, forward
  pass and HF serialization remain ordinary Qwen3. FP32 loss runs in 128-position
  blocks, with rematerialization during backpropagation. Dense targets are never
  stored in the dataset. Ordinary validation examples use stock CE in both arms.
- `train.py` calls the stock Levanter entry point, with matched CE/soft switches,
  native checkpoints, full-state resumes, W&B history registration and HF exports
  including tokenizers. It initializes distributed JAX before backend-dependent
  preflight checks. `checkpoints.py` permits only the five new SkipStep buffers
  at the recovery transition; every other state array must match the manifest.
- `inputs.py` records cache ledgers, counts, tokenizer revision, git SHA, runtime
  source hash, dependency lock hash and relevant installed package versions.
  Changed inputs/code and cross-arm or changed-microbatch resumes fail.
- `diagnostics.py` measures ordinary CE, exact soft CE, target entropy and KL on
  identical unaugmented validation packs, saving scored-token sums and timings.
  This is an explicit checkpoint diagnostic, separate from the stock ordinary-CE
  training callback.

The initial metadata implementation reads complete cache rows in addition to the
stock packed read. That extra I/O and the loss overhead need a production pilot.
The only pinned private data API used is the packer's document-index ranges;
every read checks that these reconstruct the exact stock token input. Checkpoint
preflight also uses the pinned serializer's path encoding.

## Reference recipe

The reference is exp232 m2/p06, `contacts-v1-exp232-m2-p06-train-1.5B`,
checkpoint **step-363000**, from the
[decontaminated reference run](https://wandb.ai/open-athena/MarinFold/runs/prot-exp232-trc-cv1-decontam-train-s01-m2-p06-srcpeak-augcont-lr005-us-east1).
The source recipe was inspected at
[`7a18dd5`](https://github.com/Open-Athena/MarinFold/tree/7a18dd5158e71b0db8d782be77cc4b9a6d59c8df/experiments/exp232_sweep_cv1_decontam).

Both fresh arms use Qwen3 (2048 hidden, 8192 intermediate, 24 layers, 32 query /
8 KV heads, QK normalization, Llama3 RoPE), context 8192 and global batch 128.
Adam uses LR peak 1e-3, weight decay 0.2, betas (0.9, 0.95), epsilon 1e-8 and
norm clipping 1. Parameters are FP32 and compute is BF16. The tokenizer is pinned
to `eczech/contacts-v1-tokenizer-5d68a24a899f@80f4e411b957641e8c350b07bbe1e2c832697518`.
Marin/Levanter/Haliax use release `0.2.99.dev33617068238`, with a committed lock.

All training uses the same decontaminated native AFDB and ESM-Atlas caches from
exp232/#225. The m2 mixture weights are proportional to 4,432,940,838 AFDB tokens
and 70,042,923,165 ESM tokens; expected document counts are 3,963,003 and
65,553,178. The unchanged validation source is contacts-v1 AFDB val.

| Phase | Trainer updates (inclusive) | Data seed | SkipStep |
| --- | --- | --- | --- |
| base | 0–217800 | 0 | off |
| recovery | 217801–333960 | 0 | on |
| final | 333961–363000 | 232 | on |

The LR warms from zero to 1e-3 over 14,520 updates, stays at peak through update
333961, declines linearly to 5e-5 at 344850, then stays there through 363000.
This reproduces the initial sweep's retained prefix and its continuations; the
abandoned original cooldown and failed/replayed source attempts are not extra
training. The original continuation boundary at 116161 does not change LR,
data or augmentation, so it requires no restart here. Sequence augmentation
ramps on the original 145,200-step schedule and remains at 100% thereafter.

Optax's outer schedule count advances on rejected SkipStep updates; the inner
Adam moments/count do not. An actual rejection test checks this distinction.
The final checkpoint contains 363,001 attempted updates, not 363,000.

Levanter averages normalized microbatch losses during accumulation. With varying
padding, changing microbatch size changes weighting. Keep device count, mesh and
microbatch settings identical between arms; the native resume identity records
these counts. A second paired initialization seed is available through `--seed`.

## Running

Run the modules from the repository root. CPU tests need the `test` extra;
CoreWeave training uses `gpu`. The workstation accelerator checks use the
separate `local-gpu` (CUDA 12) extra. Use a fresh venv when switching CUDA families:
their NVIDIA wheels can share installation paths.

```bash
uv sync --project experiments/exp279_models_exact_soft_contact_targets --extra test
JAX_PLATFORMS=cpu uv run --no-sync --project experiments/exp279_models_exact_soft_contact_targets pytest experiments/exp279_models_exact_soft_contact_targets/tests -m 'not accelerator'
```

Freeze the existing **tokenized split cache** paths in the compute region (each
contains `shard_ledger.json`). This reads metadata, not the full corpus:

```bash
uv run --project experiments/exp279_models_exact_soft_contact_targets python -m experiments.exp279_models_exact_soft_contact_targets.inputs --afdb AFDB_TRAIN_CACHE --esm ESM_TRAIN_CACHE --val AFDB_VAL_CACHE --output inputs.json
uv run --project experiments/exp279_models_exact_soft_contact_targets python -m experiments.exp279_models_exact_soft_contact_targets.train --manifest inputs.json --arm soft --phase base --run-name exp279-soft-s0 --output REGION_LOCAL_PREFIX
```

The second command previews the resolved configuration. Add `--run` on an
allocated training worker. Launch the CE arm with `--arm ce` and its own
`exp279-ce-s0` name, using the same manifest, hardware and microbatch setting.
Continue with `--phase recovery --resume .../step-217800`, then
`--phase final --resume .../step-333960`. Resume preemptions from an explicit
checkpoint within the current phase. Resume identity excludes phase, allowing
these transitions, and includes the data, code, arm, seed and batch settings.
Keep the generated W&B history files in git and append allocated Iris job IDs
with the repository's history command.

HF exports live at `checkpoints/RUN_NAME/hf/step-N`; native state lives at
`checkpoints/RUN_NAME/step-N`. No existing experiment runs are modified.

```bash
uv run --project experiments/exp279_models_exact_soft_contact_targets python -m experiments.exp279_models_exact_soft_contact_targets.diagnostics --manifest inputs.json --checkpoint REGION_LOCAL_PREFIX/checkpoints/RUN_NAME/step-N --max-examples 128 --output diagnostic.json
```

For synthetic end-to-end verification, use the local-gpu environment and
`smoke --arm soft --output /tmp/exp279-soft` (and `--arm ce`). Each performs
three actual training steps, native saves, stock validation and HF export.
Then run `inference_smoke.py --arm soft --smoke-dir /tmp/exp279-soft --timings ...`
using `uv run --project marinfold --extra transformers python` so inference uses
MarinFold's ordinary transformers<5 environment. The full loss-size test is
`pytest .../tests/test_accelerator.py` with `JAX_PLATFORMS=cuda,cpu` and
`XLA_PYTHON_CLIENT_PREALLOCATE=false`.

## Validation and results

Implementation validation includes independent enumeration of every permutation
and orientation for six tiny graphs; all fragment cuts; real generator filtering,
truncation and numbering; real TensorStore packing; shuffled mixture and
augmentation parity; FP32/BF16 losses and both activation/head gradients; stock CE
parity; normalization, entropy, masks and corrupted inputs; actual trainer
accumulation; full-state resume; and strict SkipStep state migration.

The accelerator loss test uses **8192 positions × 2048 hidden × 2845 vocabulary**
and 1500 contacts, compared with an independent dense BF16-input oracle.
Synthetic three-step runs of **both arms** exercised the stock entry point,
validation and exports on the RTX A5000. Their exported BF16 weights also pass
the normal MarinFold prefix-cache inference check against a causal FP32 Levanter
reference. Small inference timing records are in `data/`.

**No scientific accuracy result yet.** A full 24-layer, 1.47B, 32-H100 pilot
completed 32 updates with JAX_FLASH/CUDA 13 and the production decontaminated
mixture. It measured roughly 3.5 seconds/update, 300k nominal tokens/second and
13% reported MFU. Native step-31 state and HF weights/tokenizer were saved.
These are startup/throughput checks, not evidence of an accuracy improvement.

The first pilot exposed a validation configuration bug: Levanter ignores
`split="validation"` for flat caches. That run therefore has no validation metrics.
Earlier synthetic smoke claims of validation coverage were also incorrect.
The corrected configuration uses the validation cache's parent and requires a
nonempty validation set. A regression test reads actual held-out cache examples
through `tagged_eval_sets`, excludes both training sources, and reproduces the
previous silent omission as a hard failure. Both local GPU smokes were rerun
with actual stock validation, native/HF export and ordinary inference checks.

Ledger hashes pin metadata, not every payload byte; this implementation did not
independently re-audit decontamination or read all live cache payloads. Reported
training throughput counts nominal tokens, including padding. Diagnostic CSVs
retain scored-token denominators and real token counts. Multi-host TPU execution
and scientific accuracy comparisons remain unverified.

## Scientific readout

Use natural-protein **exp245 eval-val R-precision**, with the established exp82
rollout/resample settings, as the primary metric. Compare identical intermediate
and final exposures, equal inference budgets and paired protein-level uncertainty.
Evaluate the fresh CE reproduction before attributing any difference to soft
targets. Keep ordinary document CE comparable; report soft CE with entropy and KL,
not as if its raw magnitude were comparable to one-hot training CE.

Report accuracy versus both exposure and accelerator time, including metadata
and loss overhead. Keep per-protein predictions, scores, timings and precise
checkpoint identities. Repeat a promising result with another paired seed;
reserve eval-test for final confirmation. A null or negative result is valid.

## Production launch (2026-09-09)

The current default registry still selects exp232 m2/p06 step-363000. GPU training
uses `cw-us-east-02a`, four complete H100 nodes (32 GPUs), batch priority, global
batch 128 and per-device batch 1. Existing S3 copies of the same decontaminated
AFDB/ESM caches and AFDB validation cache avoid any bulk input transfer. Frozen
ledgers match the reference counts for all three sources.

The first completed pilot was `/bizon/exp279-soft-pilot-cw-h100x32-a03`, W&B
[exp279-soft-s0-cw-h100x32-b01](https://wandb.ai/open-athena/MarinFold/runs/exp279-soft-s0-cw-h100x32-b01),
source `b80b42d4`. It completed 32 updates, but validation was absent (see above),
so it will not be resumed as the production experiment. Checkpoints are under
`s3://marin-us-east-02a/MarinFold/exp279/checkpoints/exp279-soft-s0-cw-h100x32-b01/`.
Earlier TPU requests were cancelled while pending, and two GPU startup attempts
failed before training (venv selection, then pinned tokenizer resolution).

The corrected run is
[exp279-soft-s0-cw-h100x32-b02](https://wandb.ai/open-athena/MarinFold/runs/exp279-soft-s0-cw-h100x32-b02),
source `b0dd33eca8836c1dddbe6588e4befc3b2dd5c67d`, pilot job
`/bizon/exp279-soft-pilot-cw-h100x32-a04`. Its frozen manifest is
`data/soft_pilot_cw_launch.json`; native checkpoints and HF exports live under
`s3://marin-us-east-02a/MarinFold/exp279/checkpoints/exp279-soft-s0-cw-h100x32-b02/`.
It completed 32 updates with training loss 6.3659, ordinary validation CE 6.3678,
and 3.464 seconds/update. Native step-31 state and HF model/tokenizer were
verified; `data/soft_pilot_cw_result.json` records the result. The full production
driver resumes this state. These early losses are operational checks only.

Production driver: `/bizon/exp279-soft-production-cw-h100x32-a01`.
Its base-phase child is
`/bizon/exp279-soft-production-cw-h100x32-a01/exp279-soft-production-cw-h100x32-a01-base`.
The workers restored the native checkpoint and completed update 32 with finite
loss 6.3391 at 21:59:58 UTC. The submit/verification record is
`data/soft_production_cw_launch.json`; both stages share the exact frozen runtime
source and inputs. At the measured rate the full schedule is about 15 days,
plus validation, startup and preemptions. Only the soft arm is launched here;
a fresh matched CE run and accuracy evaluations remain future work.

The corrected run starts fresh with a new identity. `launch_gpu.py` submits a
short pilot with `--pilot-updates 32`; after validation, omitting that argument
launches the production driver. `--resume-record` preserves the pilot's frozen
manifest even after history-only commits. Runtime source, dependency, cache or
placement changes are rejected. The CPU driver waits for each prescribed phase
job and advances only after a complete checkpoint. Preempted workers choose the
latest committed native checkpoint on rank zero and broadcast that choice to all
ranks, restoring full state and automatically selecting the correct phase.

The CPU suite has 53 passing tests. Bundle tests verify identical source identity
without `.git` and detect changed worker files. Tests cover actual pinned
tokenizer loading, inherited Iris venv selection, GPU/TPU resource requests,
phase boundaries and unchanged production LR during the pilot.

## Learning-rate continuations (2026-09-10)

The approved follow-up tests whether the soft objective benefits from a higher
learning rate. Three full-state continuations use constant LR **0.001, 0.0015,
and 0.002**, starting from the same permanent production checkpoint
`exp279-soft-s0-cw-h100x32-b02/step-14520`. Each performs exactly **5,000 additional
updates** (5.243B nominal tokens), ending at native checkpoint `step-19520`.
The original production run continues independently.

`lr_trial.py` contains the catalog and continuation entry point. Model weights,
Adam moments and counters, RNG, next data batch, cache manifests, dependency
versions, augmentation, batch 128, context 8192, and microbatch 32 are preserved.
The continuation scheduler replaces only the LR and retains the optimizer's
native state structure. Every original runtime source file must still match the
parent's frozen code hash; new continuation code is additive. Lineage records
pin the parent checkpoint metadata and reject changed inputs or trial identity.

All branches use four eight-H100 nodes at batch priority, submitted through the
`marin` federation as user `bizon`. `launch_lr.py` defaults to `cw-us-east-02a`;
`--cluster cw-rno2a` is an alternative only after authorization for the cross-region
storage access. Neither placement copies or modifies the original caches.
A two-update pilot (`--stop-after 14523`) exercises the actual full-model restore,
validation, diagnostics and export. Its trial resumes with the same run name and
`--resume-record`, without changing its 5,000-update LR schedule.

Ordinary full validation CE runs every 1,000 absolute steps and at the end.
Separately, `lr_metrics.py` evaluates ordinary one-hot CE only at contact endpoint
predictions, on the same fixed 128 packed validation examples, every 1,000
continuation updates and at the end. This preserves the existing `eval/loss`;
endpoint CE is `contact_eval/loss`. The diagnostic logs pre-clipping global
gradient norm, whether the existing clip-at-one rule applies, its scaling factor,
and actual optimizer update norm every step. These norms do not themselves
estimate gradient variance.

Compare the branches at matched additional updates, primarily by final ordinary
validation CE and endpoint CE, checking the preceding evaluation points for a
consistent effect and monitoring instability/clipping. A positive result selects
a soft-loss LR candidate; it does not establish that soft targets outperform
one-hot training. That claim still needs a matched one-hot control (including
appropriate LR tuning) and the contact-accuracy evaluation described above.
These short forks test optimization from an already-trained state, not which LR
is best from initialization.

The stock `run_progress` includes the pre-fork updates. Use
`lr_trial/completed_updates` and `lr_trial/progress` for the bounded continuation's
progress and ETA. The sweep ledger normalizes its progress field from the latter.

The full CPU suite passes 60 tests; the new GPU integration smoke also passes.
Validation includes native checkpoint round-trips through actual Levanter/Adam,
identical restored states and next batches, correct first-update LR and parameter
scaling across all three branches, independently derived endpoint masks and CE,
resume identity guards, shell/resource checks, and a local GPU run through the
stock training entry point. The production parent's full model/optimizer/RNG
array manifest passed the strict restore preflight. The distributed full-model
pilot remains required before the sweep is treated as operationally validated.

The user approved Reno cross-region storage access. The two-update distributed
pilot `/bizon/exp279-lr100-rno-pilot-a01` completed successfully on 2026-09-10,
using frozen runtime `5e11b8a6847f79b00a572f5c9777a9394fd69ee9`. It restored the
parent at update 14521, applied LR 0.001 immediately, and logged ordinary CE
**3.385615** and endpoint CE **4.433615** at step 14522. Its native full-state
manifest, own-run resume position (14523), and HF weights/tokenizer were verified.
`data/lr_pilot_result.json` preserves the evidence. These two updates belong to
[the control trial](https://wandb.ai/open-athena/MarinFold/runs/exp279-soft-lr100-step14520-a01)
and are included in its fixed 5,000-update budget. This is operational validation;
no LR comparison result is available yet.

All three full continuations are running on Reno, each on 32 H100s at batch
priority, with identical source/input manifests. Their immutable dispatch record
is `data/lr_sweep_launch.json`:

| LR | W&B run | Iris root |
| --- | --- | --- |
| 0.001 | [exp279-soft-lr100-step14520-a01](https://wandb.ai/open-athena/MarinFold/runs/exp279-soft-lr100-step14520-a01) | `/bizon/exp279-lr100-rno-full-a02` |
| 0.0015 | [exp279-soft-lr150-step14520-a01](https://wandb.ai/open-athena/MarinFold/runs/exp279-soft-lr150-step14520-a01) | `/bizon/exp279-lr150-rno-full-a01` |
| 0.002 | [exp279-soft-lr200-step14520-a01](https://wandb.ai/open-athena/MarinFold/runs/exp279-soft-lr200-step14520-a01) | `/bizon/exp279-lr200-rno-full-a01` |

The first production update independently checks the matched starting state:
all three log loss 3.2835784 and gradient norm approximately 0.1049417; their
update norms are 6.772814, 10.159221 and 13.545628 (1x, 1.5x and 2x).
The applied LRs also match. `data/lr_first_update.csv` contains the W&B evidence.
Steady training is approximately 3.5 seconds/update, implying roughly five hours
of compute per branch plus evaluation/checkpoint overhead.

Both higher rates show early loss spikes. Over the same first 23 updates,
the peak training losses are 3.3050 (1x), 5.1286 (1.5x), and 9.1566 (2x);
clipping fires on 0, 5, and 15 updates respectively. `data/lr_first_23_updates.csv` records
all three branches at the same first 23 updates. These training diagnostics do
not establish a validation improvement or a winner. Both higher-rate branches
remain within the approved fixed budget; their first scheduled validation is
still pending at this launch check. The original production run is unchanged.
