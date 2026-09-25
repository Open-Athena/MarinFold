#!/usr/bin/env python
# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""7 · make — Top7 rollouts kept in **emission order**.

Same protein, same recipe and same 100 rollouts as
``1_make_top7_heatmap_data.ipynb``. The one difference is what is kept: #1 stores
the vote matrix, which is the rollouts summed and their order thrown away, and
this stores every ``<contact> <pX> <pY>`` statement as the model wrote it, in the
order it wrote it, so :mod:`7_plot_rollout_animation` can replay a rollout
statement by statement.

That order is not recoverable after the fact — it is why #98's published rollouts
could not answer #102 and #102 had to regenerate them (issue
`#102 <https://github.com/Open-Athena/MarinFold/issues/102>`_). Anything that
wants to *watch* a rollout has to capture it here.

The drawing itself is :func:`figlib.draw_rollout_votes` — #82's settled rollout +
resample, shared with
:mod:`8_make_contact_ranking_data`. This script is the Top7-specific part: which
protein, and scoring the result against #89's ground truth.

**Needs a GPU.** Roughly two minutes on an RTX A5000 for Top7's 92 residues.

    .venv/bin/python 7_make_rollout_animation_data.py
"""

import argparse
import json
import time

import numpy as np
import torch

import figlib

# --- parameters ---------------------------------------------------------------------------------
DATASET = "7_rollout_animation"
PROTEIN = "denovo_pdb__1qys_A"   # dataset__stem in the legacy 554
MODEL = "contacts-v1-exp277-m2-p06-full-epoch-1.5B"   # the current default (MODELS.yaml)

N_ROLLOUTS = 100                 # exp82's settled count
TEMPERATURE = 1.0
TOP_P = 0.95
TOP_K = -1                       # disabled, as in the published harness
SEED = 0                         # the animation has to replay the same rollout every time
BACKEND = "transformers"         # seeded and reproducible; vLLM's sampler is not (see figlib)
DTYPE = "auto"                   # auto -> bfloat16; float16 overflows these weights
BATCH_SIZE = 64


def main() -> None:
    """Draw the rollouts and write the dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-rollouts", type=int, default=N_ROLLOUTS)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--protein", default=PROTEIN)
    parser.add_argument("--backend", default=BACKEND, choices=("transformers", "vllm"))
    arguments = parser.parse_args()

    parameters = dict(protein=arguments.protein, model=arguments.model,
                      n_rollouts=arguments.n_rollouts, temperature=TEMPERATURE, top_p=TOP_P,
                      top_k=TOP_K, seed=SEED, backend=arguments.backend, dtype=DTYPE,
                      batch_size=BATCH_SIZE)
    print(json.dumps(parameters, indent=2))

    inputs = figlib.Inputs()
    targets, ground_truth = figlib.load_legacy_universe(inputs)
    dataset_name, stem = arguments.protein.split("__", 1)
    target = targets[(targets.dataset == dataset_name) & (targets.stem == stem)].iloc[0]
    truth = ground_truth[(dataset_name, stem)]
    print(f"{arguments.protein}  L={target.L}  {len(truth['contacts'])} ground-truth contacts  "
          f"{len(truth['resolved'])} resolved residues")

    dtype = DTYPE if DTYPE != "auto" else ("bfloat16" if torch.cuda.is_available() else "float32")
    model_identity = figlib.model_identity(arguments.model)
    print(f"backend {arguments.backend} · dtype {dtype} · "
          f"rope_theta {model_identity['rope_theta']}")

    started = time.time()
    draw = figlib.draw_rollout_votes(
        target.input_seq, entry_id=stem, model=arguments.model,
        n_rollouts=arguments.n_rollouts, temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K,
        seed=SEED, backend=arguments.backend, dtype=dtype, batch_size=BATCH_SIZE)
    print(f"{arguments.n_rollouts} rollouts in {time.time() - started:.1f}s · "
          f"max_new_tokens {draw.max_new_tokens}")

    metrics = figlib.score_metrics(draw.score, truth)
    headline = metrics[(metrics.range == "all") & (metrics.cut == "R")].iloc[0]
    print(f"R-precision {headline.value:.4f} over {int(headline.n_true)} true contacts "
          f"({int(headline.n_candidate)} candidate pairs)")
    print(f"{len(draw.statements)} statements over {arguments.n_rollouts} rollouts · "
          f"{int(draw.per_rollout.truncated.sum())} truncated · "
          f"{int((draw.statements.seq_i < 0).sum())} off-protein · "
          f"{int(draw.votes.max())} votes on the strongest pair")

    figlib.write_dataset(
        DATASET,
        notebook="7_make_rollout_animation_data.py",
        parameters=parameters,
        inputs=inputs,
        files={
            # The rollouts as text, exactly as decoded — the receipt every other file derives
            # from, one completion per line.
            "completions.txt": "\n".join(draw.completions).encode(),
            # The parse: one row per statement, in emission order. This is what the animation
            # replays.
            "statements.csv": lambda path: draw.statements.to_csv(path, index=False),
            # Per rollout: `n_term_index` (what maps a `<pN>` in completions.txt back to a
            # residue) and whether the rollout ran out of tokens instead of writing `<end>`.
            "rollouts.csv": lambda path: draw.per_rollout.to_csv(path, index=False),
            "votes.npy": lambda path: np.save(path, draw.votes.astype(np.int16)),
            "score.npy": lambda path: np.save(path, draw.score.astype(np.float32)),
            "ground_truth.json": json.dumps({
                "dataset": dataset_name, "stem": stem, "L": int(truth["L"]),
                "resolved": [int(i) for i in truth["resolved"]],
                "contacts": [[int(i), int(j), float(d)] for i, j, d in truth["contacts"]],
                "sequence": target.input_seq,
            }, indent=2).encode(),
            "metrics.csv": lambda path: metrics.to_csv(path, index=False),
        },
        extra={
            "model": model_identity,
            "recipe": {"method": "rollout+resample+tiebreak", "backend": arguments.backend,
                       "dtype": dtype, "n_rollouts": arguments.n_rollouts,
                       "temperature": TEMPERATURE, "top_p": TOP_P, "top_k": TOP_K, "seed": SEED,
                       "max_new_tokens": int(draw.max_new_tokens),
                       "min_seq_separation": figlib.MIN_SEPARATION},
            "protein": {"dataset": dataset_name, "stem": stem, "L": draw.seq_len,
                        "sequence_sha256": figlib.digest(target.input_seq.encode()),
                        "n_true_contacts": int(headline.n_true),
                        "n_candidate_pairs": int(headline.n_candidate)},
            "rollouts": {
                "n_statements": int(len(draw.statements)),
                "n_off_protein": int((draw.statements.seq_i < 0).sum()),
                "n_truncated": int(draw.per_rollout.truncated.sum()),
                "statements_per_rollout_median": float(draw.per_rollout.n_statements.median()),
                "max_votes": int(draw.votes.max()),
                "note": "statements.csv is in emission order — the whole point of this dataset. "
                        "seq_i/seq_j are -1 for a position token outside the protein.",
            },
            "result": {"r_precision_all": float(headline.value),
                       "auc_all": float(metrics[(metrics.range == "all")
                                                & (metrics.cut == "AUC")].value.iloc[0])},
        })


if __name__ == "__main__":
    main()
