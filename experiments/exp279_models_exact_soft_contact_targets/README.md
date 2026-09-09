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

**No scientific training result yet.** These checks do not establish an accuracy
improvement or reproduce the full 1.47B reference run. The full 24-layer model,
production corpora and multi-host accelerator execution have not been run here.
Two virtual CPU devices test sharding/accumulation/resume; that is not a
multi-host GPU/TPU validation. The local smoke uses vanilla attention and CUDA 12,
not the production JAX_FLASH/CUDA 13 or TPU stack. CUDA/cuDNN driver warnings on
the workstation did not prevent the tested CUDA matmul/training paths.

Production launch still needs region-local manifests and a short full-model
throughput/memory pilot. Ledger hashes pin metadata, not every payload byte;
this implementation did not independently re-audit decontamination or read all
live cache payloads. The launcher logs nominal exposure and standard trainer
throughput; audit actual scored-token/document/padding telemetry in that pilot.
Diagnostic CSVs already retain scored-token denominators and real token counts.

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

The current default registry still selects exp232 m2/p06 step-363000. The active
full-model pilot uses the existing us-east5 caches, 32 preemptible v6e chips
(eight hosts), batch priority, global batch 128 and per-device batch 1. No bulk
input transfer is needed. The cache ledgers match all three reference counts.

The pilot job is `/bizon/exp279-soft-pilot-use5-v6e32-a01`, with W&B identity
`exp279-soft-s0-use5-v6e32`. It was submitted from commit `8cb5d97b` for 32
updates; allocation and actual training remain to be verified. Its frozen source,
lock, package versions and input ledgers are in `data/soft_pilot_use5_launch.json`.
The initial us-east1 pilot was cancelled while still pending because the pool
was blocked by autoscaler tier backoff; it performed no training. The us-east5
request uses the already-existing regional copies, without a bulk transfer.

```bash
uv run --no-sync --project experiments/exp279_models_exact_soft_contact_targets python -m experiments.exp279_models_exact_soft_contact_targets.launch --arm soft --run-name exp279-soft-s0-use5-v6e32 --job-name exp279-soft-pilot-use5-v6e32-a01 --region us-east5 --tpu v6e-32 --pilot-updates 32 --record scratch/exp279/soft-pilot-use5.json
```

After the pilot is verified, omit `--pilot-updates`, use a new driver job name,
and pass `--resume-record scratch/exp279/soft-pilot-use5.json`. This preserves the
pilot's frozen manifest even after history-only commits. Any actual runtime
source, dependency, cache or placement change is rejected. The CPU driver waits
for each prescribed phase job and advances only after a complete checkpoint.
Preempted workers choose the latest committed native checkpoint on rank zero and
broadcast that choice to all ranks. They restore full state and automatically
select the correct phase. A restart before the first checkpoint may reinitialize
only when the persisted experiment identity matches exactly.

The launch additions passed eight focused tests and the full CPU suite now has
48 passing tests. Bundle tests verify identical source identity without `.git`
and detect a changed worker file. Phase-boundary tests include final completion;
pilot configuration tests confirm the production LR schedule is unchanged.
