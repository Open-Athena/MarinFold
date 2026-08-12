# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Turn cached amino-acid conditionals into per-assay Spearman.

Pure CPU, no model. Reads the ``.npz`` files :mod:`cache_conditionals` wrote
and applies the masked-marginals rule:

    score(variant) = Σ_{mutated sites}  log P(mutant aa | context)
                                      − log P(wild-type aa | context)

with the wild-type sequence as the context throughout (ESM-1v's convention,
and what ProteinGym's own ESM entries use).

Two knobs, both swept rather than assumed:

- ``orderings`` — how many document permutations to ensemble. Ratios are taken
  *within* an ordering and averaged after, so whatever offset an ordering
  contributes to both terms cancels before the average rather than after.
- ``min_context_fraction`` — how much of the rest of the protein a slot must
  have been conditioned on to count. 0 uses every slot (mean context ≈ half the
  protein); 0.9 keeps only near-full-context conditionals, at the cost of
  fewer samples per residue.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import proteingym
from marinfold.document_structures.contacts_v1 import sequence_likelihood as sl

HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ScoringRule:
    """One (K, context threshold) point of the sweep."""

    orderings: int
    min_context_fraction: float

    @property
    def label(self) -> str:
        return f"K{self.orderings}_ctx{self.min_context_fraction:g}"


def load_conditionals(path: Path, orderings: int) -> sl.AAConditionals:
    """Read a cached tensor, truncated to the first ``orderings`` permutations.

    Truncation (rather than resampling) is what makes the K-sweep a nested
    comparison: K=4 uses the same first four orderings K=200 starts with, so
    the curve isolates ensemble size and not which shuffles were drawn.
    """
    with np.load(path) as data:
        available = int(data["logprobs"].shape[0])
        if available < orderings:
            raise ValueError(
                f"{path.name} holds {available} orderings; {orderings} requested."
            )
        return sl.AAConditionals(
            entry_id=path.stem,
            seq_len=int(data["logprobs"].shape[1]),
            logprobs=data["logprobs"][:orderings],
            context_sizes=data["context_sizes"][:orderings],
            target_mass=data["target_mass"][:orderings],
        )


def score_assay(
    conditionals: sl.AAConditionals,
    assay: proteingym.Assay,
    rule: ScoringRule,
) -> tuple[np.ndarray, dict]:
    """Masked-marginal score per variant, plus what had to be dropped.

    A variant is dropped when the reference file's wild-type letter disagrees
    with ``target_seq`` at that site, when a site falls outside the sequence,
    or when the log-ratio there is undefined (non-canonical wild type, or no
    ordering clearing the context threshold). Dropping is counted and reported
    — a silent drop would inflate Spearman by removing exactly the variants the
    model has nothing to say about.
    """
    ratios = sl.substitution_log_ratios(
        conditionals,
        assay.target_seq,
        min_context_fraction=rule.min_context_fraction,
    )
    parsed = proteingym.parse_mutants(assay.variants)
    scores = np.full(len(parsed), np.nan)
    reasons = {"site_out_of_range": 0, "wt_mismatch": 0, "undefined_ratio": 0}

    for index, sites in enumerate(parsed):
        total = 0.0
        for wt_aa, position, mut_aa in sites:
            if not 0 <= position < conditionals.seq_len:
                reasons["site_out_of_range"] += 1
                total = np.nan
                break
            if assay.target_seq[position] != wt_aa:
                reasons["wt_mismatch"] += 1
                total = np.nan
                break
            value = ratios[position, sl.AA_ALPHABET.index(mut_aa)]
            if not np.isfinite(value):
                reasons["undefined_ratio"] += 1
                total = np.nan
                break
            total += float(value)
        scores[index] = total

    return scores, reasons


def score_all(
    rules: list[ScoringRule],
    conditionals_dir: Path,
    data_root: Path | None = None,
) -> pd.DataFrame:
    """Per-assay Spearman for every rule. One row per (assay, rule)."""
    reference = proteingym.reference(data_root)
    scorable = reference[reference.scorable]
    rows = []
    for _, meta in scorable.iterrows():
        path = conditionals_dir / f"{meta.DMS_id}.npz"
        if not path.exists():
            continue
        assay = proteingym.load_assay(meta, data_root)
        for rule in rules:
            conditionals = load_conditionals(path, rule.orderings)
            scores, reasons = score_assay(conditionals, assay, rule)
            keep = np.isfinite(scores)
            if keep.sum() < 2:
                raise ValueError(
                    f"{meta.DMS_id} under {rule.label}: only {int(keep.sum())} "
                    f"scorable variants ({reasons})."
                )
            rows.append(
                {
                    "DMS_id": meta.DMS_id,
                    "orderings": rule.orderings,
                    "min_context_fraction": rule.min_context_fraction,
                    "spearman": proteingym.assay_spearman(
                        scores[keep], assay.variants.DMS_score.values[keep]
                    ),
                    "n_variants": int(keep.sum()),
                    "n_dropped": int((~keep).sum()),
                    **reasons,
                }
            )
        print(f"  {meta.DMS_id:<45s} " + " ".join(
            f"{r['orderings']}/{r['min_context_fraction']:g}:{r['spearman']:+.3f}"
            for r in rows[-len(rules):]
        ))
    return pd.DataFrame(rows)


def mutational_depth(assay: proteingym.Assay) -> np.ndarray:
    """Number of substitutions per variant — the leaderboard's depth axis."""
    return np.array([len(s) for s in proteingym.parse_mutants(assay.variants)])
