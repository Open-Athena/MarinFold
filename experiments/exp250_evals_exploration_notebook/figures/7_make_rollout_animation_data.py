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

The recipe is #82's settled rollout + resample, reproduced from the public
pieces of ``marinfold.document_structures.contacts_v1`` rather than run through
``predict``: ``predict`` returns the summed matrix only. Every step below is the
same one ``_rollout_score_matrix`` takes, and the vote matrix this writes is the
matrix ``predict`` would have returned for these rollouts.

**Needs a GPU.** Roughly two minutes on an RTX A5000 for Top7's 92 residues.

    .venv/bin/python 7_make_rollout_animation_data.py
"""

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

import figlib
from marinfold import load_backend
from marinfold.document_structures.contacts_v1 import (
    CONTEXT_LENGTH,
    NUM_POSITION_INDICES,
    GenerationConfig,
    build_document,
    iter_structure_statements,
    residues_from_sequence,
    sample_contacts,
)
from marinfold.document_structures.contacts_v1.vocab import (
    BEGIN_STRUCTURE_TOKEN,
    CONTACT_TOKEN,
    position_token,
)

# --- parameters ---------------------------------------------------------------------------------
DATASET = "7_rollout_animation"
PROTEIN = "denovo_pdb__1qys_A"   # dataset__stem in the legacy 554
MODEL = "contacts-v1-exp277-m2-p06-full-epoch-1.5B"   # the current default (MODELS.yaml)

N_ROLLOUTS = 100                 # exp82's settled count
TEMPERATURE = 1.0
TOP_P = 0.95
TOP_K = -1                       # disabled, as in the published harness
SEED = 0                         # the animation has to replay the same rollout every time
BACKEND = "transformers"         # see RECIPE NOTE below — not "auto"
DTYPE = "auto"                   # auto -> bfloat16; float16 overflows these weights
BATCH_SIZE = 64

# RECIPE NOTE — transformers, not vLLM. Notebook 1 prefers vLLM where it is usable, because it
# only needs the vote matrix and vLLM is faster. This dataset is replayed frame by frame, so the
# rollout it draws has to be the same one next time it is regenerated; the transformers backend
# seeds `torch` and is reproducible, vLLM's sampler is not seedable per request. Top7 is 92
# residues, so the speed the seed costs is about a minute.

# Probability floor for the pairwise log-score. `inference.py`'s `_PROB_FLOOR`, which is private.
PROB_FLOOR = 1e-12


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

    backend = load_backend(arguments.backend, model=arguments.model, dtype=dtype,
                           tail_batch_size=BATCH_SIZE)
    tokenizer = backend.tokenizer

    # One *fresh document realization* per rollout: contacts-v1 randomizes both the N-terminal
    # position index and the statement order, and #82's recipe resamples that per rollout rather
    # than drawing 100 samples from one prompt. `n_term_index` is what maps a `<pN>` token in the
    # completion back to a residue, and it differs per rollout, so it is stored per rollout too.
    residues = residues_from_sequence(target.input_seq)
    prefixes: list[list[int]] = []
    n_term_indices: list[int] = []
    seq_len = 0
    for rollout in range(arguments.n_rollouts):
        built = build_document(f"{stem}:r{rollout}", residues, [], config=GenerationConfig())
        if built is None:
            raise SystemExit(f"{stem} cannot be serialized as a contacts-v1 document")
        document = built.document
        cut = document.index(BEGIN_STRUCTURE_TOKEN) + len(BEGIN_STRUCTURE_TOKEN)
        prefixes.append(list(tokenizer.encode(document[:cut], add_special_tokens=False)))
        n_term_indices.append(built.n_term_index)
        seq_len = built.seq_len
    if seq_len != int(target.L):
        raise SystemExit(f"document length {seq_len} != input sequence length {target.L}")

    # #82's budget: 4L + 64 tokens, sized from L only — never from the ground-truth contact
    # count, which would make the rollout oracle-dependent.
    max_new_tokens = min(CONTEXT_LENGTH - len(prefixes[0]), 4 * seq_len + 64)
    print(f"{len(prefixes)} prefixes of {len(prefixes[0])} tokens · "
          f"max_new_tokens {max_new_tokens}")

    started = time.time()
    completions = sample_contacts(backend, prefixes, max_new_tokens=max_new_tokens,
                                  temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K, seed=SEED,
                                  batch_size=BATCH_SIZE)
    print(f"{len(completions)} rollouts in {time.time() - started:.1f}s")

    # --- the parse: every statement, in the order it was emitted --------------------------------
    # `seq_i` / `seq_j` are the sorted sequence indices the statement's two position tokens map
    # to, or -1 for a token outside this protein's slice of the 2000-position ring (the model can
    # write any `<pN>`). Nothing is filtered here: the near-diagonal band, the repeats and the
    # unmapped statements are all part of what a rollout actually emits, and the plot decides
    # what to do with them.
    rows = []
    texts = []
    for rollout, (token_ids, n_term_index) in enumerate(
            zip(completions, n_term_indices, strict=True)):
        text = tokenizer.decode(token_ids, skip_special_tokens=False)
        texts.append(text)
        for order, (kind, pos_a, pos_b) in enumerate(iter_structure_statements(text)):
            mapped = []
            for position in (pos_a, pos_b):
                index = (position - n_term_index) % NUM_POSITION_INDICES
                mapped.append(int(index) if index < seq_len else -1)
            low, high = (min(mapped), max(mapped)) if -1 not in mapped else (-1, -1)
            rows.append(dict(rollout=rollout, order=order, kind=kind, pos_a=pos_a, pos_b=pos_b,
                             seq_i=low, seq_j=high))
    statements = pd.DataFrame(rows)

    per_rollout = pd.DataFrame([
        dict(rollout=rollout, n_term_index=n_term_index, n_tokens=len(token_ids),
             truncated=len(token_ids) >= max_new_tokens,
             n_statements=int((statements.rollout == rollout).sum()))
        for rollout, (token_ids, n_term_index) in enumerate(
            zip(completions, n_term_indices, strict=True))])

    # --- the vote matrix: what `predict` would have returned for these rollouts ------------------
    # One vote per *distinct* pair per rollout, band-filtered — the same accumulation
    # `_rollout_score_matrix` does, run over the statements above instead of re-reading the text.
    # Today's models never emit `<retract>`, so the live set is the emitted set; a retracting
    # model would need the fold in `read.py` here instead, and the assert below is what would
    # notice.
    if (statements.kind != "contact").any():
        raise SystemExit("a <retract> statement was emitted: the vote accumulation below assumes "
                         "the emitted set is the live set — fold read.py's edit list instead")
    votes = np.zeros((seq_len, seq_len), dtype=np.int64)
    scorable_statements = statements[(statements.seq_i >= 0)
                                     & (statements.seq_j - statements.seq_i
                                        >= figlib.MIN_SEPARATION)]
    for _, pairs in scorable_statements.groupby("rollout"):
        for i, j in pairs[["seq_i", "seq_j"]].drop_duplicates().itertuples(index=False):
            votes[i, j] += 1
            votes[j, i] += 1
    if votes.max() > arguments.n_rollouts:
        raise SystemExit(f"{votes.max()} votes from {arguments.n_rollouts} rollouts")

    # --- #82's tie-break ------------------------------------------------------------------------
    # The vote matrix leaves most pairs on zero, so the metric ranks `votes + a pairwise log-prob
    # term scaled into [0, 0.5)`: integer vote gaps are never crossed, so this only orders pairs
    # *tied* on votes. The three steps are `_fwd_matrix`, `_sym_from_fwd` and `_tiebreak` in
    # `inference.py`, which are private; they are short and stated in full rather than reached
    # into, and `backend.next_token_probs` — the only model call — is public.
    prefix_ids = prefixes[0]
    seq_positions = [(n_term_indices[0] + k) % NUM_POSITION_INDICES for k in range(seq_len)]
    contact_id = tokenizer.convert_tokens_to_ids(CONTACT_TOKEN)
    position_ids = [tokenizer.convert_tokens_to_ids(position_token(p)) for p in seq_positions]
    first = backend.next_token_probs(prefix_ids, [[contact_id]], position_ids)      # (1, L)
    second = backend.next_token_probs(prefix_ids, [[contact_id, pid] for pid in position_ids],
                                      position_ids)                                # (L, L)
    forward = (np.log(np.clip(np.asarray(first[0], dtype=np.float64), PROB_FLOOR, None))[:, None]
               + np.log(np.clip(np.asarray(second, dtype=np.float64), PROB_FLOOR, None)))
    pairwise = 0.5 * (forward + forward.T)
    upper = np.triu_indices(seq_len, k=1)
    low, high = float(pairwise[upper].min()), float(pairwise[upper].max())
    score = votes + (pairwise - low) / (high - low + 1e-9) * 0.5

    metrics = figlib.score_metrics(score, truth)
    headline = metrics[(metrics.range == "all") & (metrics.cut == "R")].iloc[0]
    print(f"R-precision {headline.value:.4f} over {int(headline.n_true)} true contacts "
          f"({int(headline.n_candidate)} candidate pairs)")
    print(f"{len(statements)} statements over {arguments.n_rollouts} rollouts · "
          f"{int(per_rollout.truncated.sum())} truncated · "
          f"{int((statements.seq_i < 0).sum())} off-protein · "
          f"{int(votes.max())} votes on the strongest pair")

    figlib.write_dataset(
        DATASET,
        notebook="7_make_rollout_animation_data.py",
        parameters=parameters,
        inputs=inputs,
        files={
            # The rollouts as text, exactly as decoded — the receipt every other file derives
            # from, one completion per line.
            "completions.txt": "\n".join(texts).encode(),
            # The parse: one row per statement, in emission order. This is what the animation
            # replays.
            "statements.csv": lambda path: statements.to_csv(path, index=False),
            # Per rollout: `n_term_index` (what maps a `<pN>` in completions.txt back to a
            # residue) and whether the rollout ran out of tokens instead of writing `<end>`.
            "rollouts.csv": lambda path: per_rollout.to_csv(path, index=False),
            "votes.npy": lambda path: np.save(path, votes.astype(np.int16)),
            "score.npy": lambda path: np.save(path, score.astype(np.float32)),
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
                       "max_new_tokens": int(max_new_tokens),
                       "min_seq_separation": figlib.MIN_SEPARATION},
            "protein": {"dataset": dataset_name, "stem": stem, "L": seq_len,
                        "sequence_sha256": figlib.digest(target.input_seq.encode()),
                        "n_true_contacts": int(headline.n_true),
                        "n_candidate_pairs": int(headline.n_candidate)},
            "rollouts": {
                "n_statements": int(len(statements)),
                "n_off_protein": int((statements.seq_i < 0).sum()),
                "n_truncated": int(per_rollout.truncated.sum()),
                "statements_per_rollout_median": float(per_rollout.n_statements.median()),
                "max_votes": int(votes.max()),
                "note": "statements.csv is in emission order — the whole point of this dataset. "
                        "seq_i/seq_j are -1 for a position token outside the protein.",
            },
            "result": {"r_precision_all": float(headline.value),
                       "auc_all": float(metrics[(metrics.range == "all")
                                                & (metrics.cut == "AUC")].value.iloc[0])},
        })


if __name__ == "__main__":
    main()
