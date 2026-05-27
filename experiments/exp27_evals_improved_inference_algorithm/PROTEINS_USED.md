# Proteins used (and over-tuned on) in exp27

This file lists every FoldBench monomer that was looked at — and in
particular **tuned on** — in `exp27_evals_improved_inference_algorithm`.
Future experiments that want a clean held-out set of MarinFold-1B
inference-time evaluations should **avoid these PDB IDs**: every knob
of the headline algorithm (`iter_R4_grow_on_sampled_uniform_M5`) was
selected by looking at LDDT on these proteins, so any later run on the
same set is no longer an honest generalization signal.

Concretely, "knob-tuning" here means: range_strategy choice (uniform
vs model vs round_robin vs weighted), K schedule (fixed K=L vs growing
K), `min_contact_prob` (0.1 vs 0.3 vs 0.5), number of rounds (R=2, 3,
4, 5), distance-commit on/off, sharpening T sweeps, and the choice
between sampled / iterative / combined pipelines. All of these knob
decisions were made by reading per-protein LDDT on the proteins below.

## Train set — `data/train_proteins.csv`

Picked with `random.Random(27).sample(...)` from the FoldBench monomer
manifest filtered to `n_residues ≤ 400`. **All knob tuning happened
here.** Held-out only relative to its own evaluation; for downstream
generalization claims, treat as fully training data.

| PDB | chain | length |
|---|---|---:|
| 7uk8 | A | 394 |
| 7y5j | A | 102 |
| 7ykm | A | 105 |
| 7ur2 | A | 195 |
| 8baq | A | 208 |
| 8cba | A | 214 |
| 7zs2 | A | 316 |
| 7xz3 | A | 325 |
| 7ylr | A | 330 |
| 8eb9 | A |  95 |

## Held-out set — `data/heldout_proteins.csv`

Picked with `random.Random(42).sample(...)` from the same pool,
excluding the train set. Used **once** at the end of the experiment
for a generalization check on the FROZEN headline algorithm — no knob
changes were made after seeing these scores. Still, every PDB ID below
has had its mean LDDT under the headline algorithm published in this
experiment dir; treat as "evaluation set, do not re-use as a clean
held-out signal for an algorithm you intend to compare with this one."

| PDB | chain | length |
|---|---|---:|
| 7t9r | A |  38 |
| 7y8i | A |  97 |
| 7zoi | A | 151 |
| 7wz5 | A | 161 |
| 8bau | A | 189 |
| 8gmy | A | 236 |
| 7xg9 | A | 288 |
| 7x4p | A | 307 |
| 7v3o | A | 328 |
| 7qsj | A | 373 |

## Untouched FoldBench monomers (safe for future use)

The remaining 80 of the 100 FoldBench monomers were never looked at
in exp27 — not scored, not iterated on, not even loaded. For a fresh
generalization signal, sample from `protenix_data/.../manifest.csv`
**excluding the 20 PDB IDs listed above**.

(The 14 monomers with `n_residues > 400` were excluded from the
sampling pool entirely, so they're also untouched here — but they
were excluded for compute-time reasons, not freshness reasons.)
