#!/usr/bin/env python
# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""8a · make — MarinFold's ranked contacts for one FoldBench monomer.

Step one of three for the contact-titration animation. Draws 100 rollouts for
``8ubs_A`` and writes the vote matrix, so
:mod:`8_make_titration_data` can feed Helico the top *k* contacts for every *k*
and :mod:`8_plot_contact_titration` can draw the map filling up.

Only this step needs MarinFold; the Helico step runs in a different environment
entirely and reads what this writes.

**Which protein and why.** ``8ubs_A`` is a 150-residue natural monomer in
[#245](https://github.com/Open-Athena/MarinFold/issues/245)'s ``eval-test``, and
it is the case where contact conditioning is most visible: Helico folds it to
0.225 lDDT with no contacts and 0.874 with MarinFold's top-L, against an oracle
ceiling of 0.868 — the contacts are worth the entire distance. MarinFold's
contact R-precision on it is 0.821 against Protenix-v2 single-sequence's 0.160.
``eval-test`` is the rarely-read held-out set, but every number above is already
published (#245's per-protein table, helico exp14's per-target scores), so
animating it reads nothing new.

**The prompt is #245's, 151 residues.** The deposited chain is 150 — one
N-terminal serine is unresolved — so the two indexings differ by one.
:mod:`8_make_titration_data` re-seats the ranking onto Helico's residues and
asserts that offset against the sequences themselves rather than assuming it.

**Needs a GPU.** About a minute on an RTX A5000.

    .venv/bin/python 8_make_contact_ranking_data.py
"""

import argparse
import json
import time

import numpy as np
import torch

import figlib

# --- parameters ---------------------------------------------------------------------------------
DATASET = "8_contact_ranking"
PROTEIN = "8ubs_A"               # stem in #245's FoldBench monomer universe
MODEL = "contacts-v1-exp277-m2-p06-full-epoch-1.5B"   # the current default (MODELS.yaml)

N_ROLLOUTS = 100                 # exp82's settled count
TEMPERATURE = 1.0
TOP_P = 0.95
TOP_K = -1                       # disabled, as in the published harness
SEED = 0                         # the titration has to condition on the same contacts every time
BACKEND = "transformers"         # seeded and reproducible; vLLM's sampler is not (see figlib)
DTYPE = "auto"                   # auto -> bfloat16; float16 overflows these weights
BATCH_SIZE = 64


def main() -> None:
    """Draw the rollouts and write the ranking dataset."""
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
    targets = figlib.load_foldbench_universe(inputs)
    target = targets[targets.stem == arguments.protein].iloc[0]
    print(f"{arguments.protein}  L={target.L}  {target.eval_set}  "
          f"designed={target.designed}  viral={target.is_viral}")

    dtype = DTYPE if DTYPE != "auto" else ("bfloat16" if torch.cuda.is_available() else "float32")
    model_identity = figlib.model_identity(arguments.model)
    print(f"backend {arguments.backend} · dtype {dtype} · "
          f"rope_theta {model_identity['rope_theta']}")

    started = time.time()
    draw = figlib.draw_rollout_votes(
        target.input_seq, entry_id=arguments.protein, model=arguments.model,
        n_rollouts=arguments.n_rollouts, temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K,
        seed=SEED, backend=arguments.backend, dtype=dtype, batch_size=BATCH_SIZE)
    print(f"{arguments.n_rollouts} rollouts in {time.time() - started:.1f}s · "
          f"max_new_tokens {draw.max_new_tokens}")

    # The ranking itself is deliberately NOT cut here. Which pairs are candidates depends on which
    # residues the deposited structure resolves, and that is Helico's side of the pipeline; this
    # writes the dense matrices and lets the next step rank what it can actually fold.
    upper = np.triu_indices(draw.seq_len, k=figlib.MIN_SEPARATION)
    ranked_votes = np.sort(draw.votes[upper])[::-1]
    print(f"{len(draw.statements)} statements · {int(draw.per_rollout.truncated.sum())} truncated "
          f"· {int((draw.statements.seq_i < 0).sum())} off-protein")
    print(f"votes on the top 1 / 10 / 150 / 300 candidate pairs: {ranked_votes[0]} / "
          f"{ranked_votes[9]} / {ranked_votes[149]} / {ranked_votes[299]}")

    figlib.write_dataset(
        DATASET,
        notebook="8_make_contact_ranking_data.py",
        parameters=parameters,
        inputs=inputs,
        files={
            "completions.txt": "\n".join(draw.completions).encode(),
            "statements.csv": lambda path: draw.statements.to_csv(path, index=False),
            "rollouts.csv": lambda path: draw.per_rollout.to_csv(path, index=False),
            "votes.npy": lambda path: np.save(path, draw.votes.astype(np.int16)),
            "score.npy": lambda path: np.save(path, draw.score.astype(np.float32)),
            "target.json": json.dumps({
                "dataset": target.dataset, "stem": target.stem, "L": int(target.L),
                "eval_set": target.eval_set, "designed": int(target.designed),
                "is_viral": int(target.is_viral), "sequence": target.input_seq,
            }, indent=2).encode(),
        },
        extra={
            "model": model_identity,
            "recipe": {"method": "rollout+resample+tiebreak", "backend": arguments.backend,
                       "dtype": dtype, "n_rollouts": arguments.n_rollouts,
                       "temperature": TEMPERATURE, "top_p": TOP_P, "top_k": TOP_K, "seed": SEED,
                       "max_new_tokens": int(draw.max_new_tokens),
                       "min_seq_separation": figlib.MIN_SEPARATION},
            "protein": {"stem": target.stem, "L": draw.seq_len,
                        "sequence_sha256": figlib.digest(target.input_seq.encode())},
            "rollouts": {
                "n_statements": int(len(draw.statements)),
                "n_off_protein": int((draw.statements.seq_i < 0).sum()),
                "n_truncated": int(draw.per_rollout.truncated.sum()),
                "statements_per_rollout_median": float(draw.per_rollout.n_statements.median()),
                "max_votes": int(draw.votes.max()),
                "note": "votes.npy / score.npy are in the #245 PROMPT's 0-based indexing "
                        "(L=151 for 8ubs_A), not the deposited chain's. "
                        "8_make_titration_data.py re-seats them.",
            },
        })


if __name__ == "__main__":
    main()
