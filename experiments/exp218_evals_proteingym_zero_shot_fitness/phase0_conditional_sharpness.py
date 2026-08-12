# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Phase 0 — is contacts-v1's amino-acid conditional sharp at all?

The go/no-go for the rest of issue #218. Before asking whether the conditional
*ranks mutations*, ask whether it carries information about amino acids at all,
and how that scales with how much of the protein the model has already seen.

Three arms, all on ProteinGym's own target sequences (the population Phase 1
scores, so this measures the conditional where it will actually be used):

- **model** — ``P(residue i | a random subset of the others)``, read off the
  contacts-v1 any-order conditional.
- **composition floor** — always guess the protein's own modal amino acid. A
  model that has learned nothing beyond "this protein is leucine-rich" scores
  here, so this is the bar the model must clear to be doing anything.
- **scrambled** — the identical document built from a random permutation of the
  same protein's amino acids. Composition is preserved exactly; real sequence
  structure is destroyed. The gap between *model* and *scrambled* is the part
  of the model's accuracy that comes from protein structure rather than
  composition — the part a fitness predictor needs.

Gate: if the model arm does not clearly beat both controls at high context,
the mechanism this experiment rests on is not there and Phase 1 is pointless.

Usage::

    uv run python phase0_conditional_sharpness.py --num-proteins 24 --orderings 16
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

import proteingym
from marinfold.document_structures.contacts_v1 import sequence_likelihood as sl
from marinfold.document_structures.contacts_v1.parse import residues_from_sequence
from marinfold.inference.core import load_backend

HERE = Path(__file__).resolve().parent
# Context-fraction buckets. The lowest is "saw almost nothing", the highest is
# the masked-marginals regime Phase 1 actually uses.
BUCKETS = ((0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01))


def select_proteins(count: int, max_length: int, seed: int) -> pd.DataFrame:
    """One target sequence per UniProt id, spread across the length range.

    Deduplicating by UniProt id keeps a heavily-assayed protein from dominating;
    stratifying by length keeps the answer from being about short domains only.
    """
    frame = proteingym.reference()
    frame = frame[frame.scorable & (frame.seq_len <= max_length)]
    frame = frame.drop_duplicates(subset="UniProt_ID")
    frame = frame[~frame.target_seq.str.contains(f"[^{sl.AA_ALPHABET}]", regex=True)]
    frame = frame.sort_values("seq_len").reset_index(drop=True)
    if len(frame) <= count:
        return frame
    picks = np.linspace(0, len(frame) - 1, count).round().astype(int)
    return frame.iloc[np.unique(picks)].reset_index(drop=True)


def score_one(backend, sequence: str, entry_id: str, orderings: int, batch: int):
    """Per-slot correctness and NLL for one sequence, flattened over orderings."""
    conditionals = sl.amino_acid_conditionals(
        backend,
        residues_from_sequence(sequence),
        entry_id=entry_id,
        num_orderings=orderings,
        batch_size=batch,
    )
    truth = np.array([sl.AA_ALPHABET.index(c) for c in sequence])
    truth_grid = np.broadcast_to(truth, conditionals.context_sizes.shape)
    logprobs = conditionals.logprobs
    correct = logprobs.argmax(axis=-1) == truth_grid
    nll = -np.take_along_axis(logprobs, truth_grid[:, :, None], axis=2)[:, :, 0]
    return {
        "context_fraction": conditionals.context_fractions().ravel(),
        "correct": correct.ravel(),
        "nll": nll.ravel(),
        "target_mass": conditionals.target_mass.ravel(),
    }


def composition_floor(sequence: str) -> float:
    """Accuracy of always guessing this protein's most common amino acid."""
    counts = pd.Series(list(sequence)).value_counts()
    return float(counts.iloc[0] / len(sequence))


def bucketize(rows: dict) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    out = []
    for low, high in BUCKETS:
        mask = (frame.context_fraction >= low) & (frame.context_fraction < high)
        if not mask.any():
            continue
        chunk = frame[mask]
        out.append(
            {
                "context_low": low,
                "context_high": min(high, 1.0),
                "n_slots": int(mask.sum()),
                "top1": float(chunk.correct.mean()),
                "nll": float(chunk.nll.mean()),
                "perplexity": float(np.exp(chunk.nll.mean())),
                "target_mass": float(chunk.target_mass.mean()),
            }
        )
    return pd.DataFrame(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-proteins", type=int, default=24)
    parser.add_argument("--orderings", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--model", default=None, help="MODELS.yaml nickname")
    parser.add_argument("--backend", default="transformers")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=HERE / "data")
    args = parser.parse_args()

    proteins = select_proteins(args.num_proteins, args.max_length, args.seed)
    print(
        f"Phase 0 on {len(proteins)} proteins "
        f"(lengths {proteins.seq_len.min()}–{proteins.seq_len.max()}), "
        f"K={args.orderings}"
    )

    backend = load_backend(args.backend, model=args.model)
    rng = random.Random(args.seed)
    arms: dict[str, list[dict]] = {"model": [], "scrambled": []}
    floors, per_protein = [], []

    for _, row in proteins.iterrows():
        real = row.target_seq
        scrambled = "".join(rng.sample(real, len(real)))
        real_rows = score_one(
            backend, real, row.UniProt_ID, args.orderings, args.batch_size
        )
        scrambled_rows = score_one(
            backend,
            scrambled,
            f"{row.UniProt_ID}-scrambled",
            args.orderings,
            args.batch_size,
        )
        arms["model"].append(real_rows)
        arms["scrambled"].append(scrambled_rows)
        floors.append(composition_floor(real))

        high = real_rows["context_fraction"] >= 0.8
        per_protein.append(
            {
                "uniprot_id": row.UniProt_ID,
                "dms_id": row.DMS_id,
                "seq_len": int(row.seq_len),
                "top1_high_context": float(real_rows["correct"][high].mean()),
                "nll_high_context": float(real_rows["nll"][high].mean()),
                "composition_floor": composition_floor(real),
            }
        )
        print(
            f"  {row.UniProt_ID:>22s} L={row.seq_len:4d} "
            f"top1@0.8+={per_protein[-1]['top1_high_context']:.3f} "
            f"floor={floors[-1]:.3f}"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tables = {}
    for arm, chunks in arms.items():
        merged = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
        table = bucketize(merged)
        table.insert(0, "arm", arm)
        tables[arm] = table
    curve = pd.concat(tables.values(), ignore_index=True)
    curve.to_csv(args.out_dir / "phase0_context_curve.csv", index=False)
    pd.DataFrame(per_protein).to_csv(
        args.out_dir / "phase0_per_protein.csv", index=False
    )

    floor = float(np.mean(floors))
    top_bucket = tables["model"].iloc[-1]
    scrambled_top = tables["scrambled"].iloc[-1]
    verdict = {
        "n_proteins": int(len(proteins)),
        "orderings": args.orderings,
        "composition_floor_top1": floor,
        "model_top1_high_context": float(top_bucket.top1),
        "scrambled_top1_high_context": float(scrambled_top.top1),
        "model_perplexity_high_context": float(top_bucket.perplexity),
        "scrambled_perplexity_high_context": float(scrambled_top.perplexity),
        "passes_gate": bool(
            top_bucket.top1 > floor + 0.05 and top_bucket.top1 > scrambled_top.top1 + 0.05
        ),
    }
    (args.out_dir / "phase0_verdict.json").write_text(json.dumps(verdict, indent=2))

    print("\n" + curve.to_string(index=False))
    print(f"\ncomposition floor (top-1): {floor:.3f}")
    print(f"model    @ context>=0.8: top1 {top_bucket.top1:.3f}  ppl {top_bucket.perplexity:.2f}")
    print(f"scrambled@ context>=0.8: top1 {scrambled_top.top1:.3f}  ppl {scrambled_top.perplexity:.2f}")
    print(f"\nGATE: {'PASS' if verdict['passes_gate'] else 'FAIL'}")


if __name__ == "__main__":
    main()
