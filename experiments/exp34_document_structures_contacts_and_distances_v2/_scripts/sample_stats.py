# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Compute distributional stats for v2 documents on a representative sample.

Generates a v2 document per input structure, then counts what each
doc *actually contains* — sequence length, per-mode contacts shown,
distance statements shown, think-token cost, total token use — and
also tallies what was *available* (eligible) in each protein so we
can report "fraction of contacts captured" alongside the raw counts.

Default sample is the 100 FoldBench monomer ground-truth CIFs that
exp20 / exp26 used. They're real PDB single-chain proteins covering
a wide length range. Two important caveats for interpretation:

1. PDB B-factors are not pLDDT. v1/v2's ``residue_plddt_min`` uses
   B-factor as a confidence proxy; the default 70.0 is calibrated
   for AFDB (0..100 pLDDT) and would filter virtually every PDB
   residue out. We disable that filter for the sample run
   (``--residue-plddt-min 0.0``) so the resulting stats reflect the
   *document-structure* mechanics, not the unrelated pLDDT-cutoff
   interaction with B-factor units. AFDB runs of the same generator
   filter ~50-60% of residues at the default cutoff — that's a
   separate story.
2. FoldBench monomers are PDB experimental structures, not AFDB
   predictions, so the absolute length distribution and the pool of
   eligible contacts skew differently from the AFDB-24M training
   distribution. The shape of the distributions reported here is
   what we care about; absolute numbers will shift on AFDB.

Writes ``data/sample_stats.csv`` (one row per protein, committed)
with the columns documented in :data:`_COLUMNS` below.
"""

import argparse
import csv
import dataclasses
import hashlib
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP_ROOT = HERE.parent

# Run as a top-level script from the experiment dir so the bare
# ``import generate``/``import vocab`` in those modules resolves.
sys.path.insert(0, str(EXP_ROOT))

import generate
from parse import (
    ParsedStructure,
    cb_or_ca_position,
    euclidean,
    iter_parsed_structures,
)
from vocab import CONTEXT_LENGTH, NAME, THINK_TOKEN


# gemmi assigns block-name as ``structure.name``. For CIFs whose
# block name is the generic placeholder "XXXX" (protenix /
# foldbench-monomer convention), prefer the file stem so each
# protein gets a distinct seed.
_PLACEHOLDER_NAMES = frozenset({"", "XXXX", "UNNAMED"})


def _disambiguate_entry_id(structure: ParsedStructure) -> ParsedStructure:
    """Replace a placeholder entry_id with the source path's stem.

    The default v1/v2 parser uses ``structure.name or path.stem``,
    which is fine for AFDB CIFs (real ``AF-...`` block names) but
    collapses every protenix-foldbench monomer onto the same RNG
    seed because they all share ``_data_XXXX``. Override here so the
    sample run actually exercises a hundred distinct seeds.
    """
    if structure.entry_id in _PLACEHOLDER_NAMES:
        return dataclasses.replace(
            structure, entry_id=structure.source_path.stem,
        )
    return structure


_DEFAULT_INPUT = (
    EXP_ROOT.parent
    / "exp20_evals_marinfold_1b_foldbench"
    / "protenix_data" / "data" / "protenix-foldbench-monomers" / "gt"
)


_COLUMNS = [
    "entry_id",
    "n_residues",
    "global_plddt",
    "n_eligible_residues",
    # Eligible contacts: pairs in the eligible-residue pool with
    # CB-CB <= 8 Å and sequence-sep >= 6. This is *what the generator
    # is sampling from* once pLDDT filtering is done.
    "n_long_eligible",
    "n_medium_eligible",
    "n_short_eligible",
    # Total contacts in the protein at the same CB-CB threshold but
    # ignoring pLDDT (i.e. the universe of possible long-range
    # contacts at this sequence separation and distance cutoff).
    "n_long_total",
    "n_medium_total",
    "n_short_total",
    # Counts actually serialized into the v2 document.
    "n_long_shown",
    "n_medium_shown",
    "n_short_shown",
    "n_distance_shown",
    "n_think_total",
    "n_think_initial_k1",
    "n_think_additional_runs",
    "n_total_tokens",
]


def _count_eligible_contacts(
    structure: ParsedStructure, *, plddt_min: float, cfg: generate.GenerationConfig,
) -> tuple[int, int, int, int]:
    """Return (n_eligible_residues, n_long, n_medium, n_short) at given pLDDT."""
    cb_positions: dict[int, tuple[float, float, float]] = {}
    for r in structure.residues:
        if r.plddt < plddt_min:
            continue
        pos = cb_or_ca_position(r)
        if pos is not None:
            cb_positions[r.index] = pos
    n_long = n_med = n_short = 0
    idxs = sorted(cb_positions)
    for ii in range(len(idxs)):
        for jj in range(ii + 1, len(idxs)):
            i, j = idxs[ii], idxs[jj]
            sep = j - i
            if sep < cfg.short_range_sep:
                continue
            if euclidean(cb_positions[i], cb_positions[j]) > cfg.contact_cutoff_angstrom:
                continue
            if sep >= cfg.long_range_sep:
                n_long += 1
            elif sep >= cfg.medium_range_sep:
                n_med += 1
            else:
                n_short += 1
    return len(cb_positions), n_long, n_med, n_short


def _resample_think_overhead(
    structure: ParsedStructure, cfg: generate.GenerationConfig,
) -> tuple[int, int]:
    """Replay the first RNG draws to recover (k1, n_additional_runs) for a doc.

    Useful because ``_generate_one`` doesn't expose those numbers
    directly; rather than refactor we just redo the seeding (cheap)
    using the same deterministic seed-from-entry_id rule. Returns
    raw k1 (0 if the gate missed) and the number of additional runs.
    """
    seed = int(hashlib.sha1(structure.entry_id.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)
    k1, additional = generate._sample_think_overhead(rng, cfg)
    return k1, len(additional)


def _count_doc_tokens(doc: str) -> dict[str, int]:
    """Token-level inventory of a v2 document."""
    parts = doc.split()
    counts = {
        "n_long_shown": parts.count("<long-range-contact>"),
        "n_medium_shown": parts.count("<medium-range-contact>"),
        "n_short_shown": parts.count("<short-range-contact>"),
        "n_distance_shown": parts.count("<distance>"),
        "n_think_total": parts.count(THINK_TOKEN),
        "n_total_tokens": len(parts),
    }
    return counts


def _row(
    structure: ParsedStructure, *,
    cfg: generate.GenerationConfig, context_length: int,
) -> dict[str, object] | None:
    doc = generate._generate_one(
        structure, context_length=context_length, cfg=cfg,
    )
    if doc is None:
        return None
    n_elig_res, n_long_e, n_med_e, n_short_e = _count_eligible_contacts(
        structure, plddt_min=cfg.residue_plddt_min, cfg=cfg,
    )
    # No-pLDDT-filter universe: same calc with min = -inf.
    _, n_long_t, n_med_t, n_short_t = _count_eligible_contacts(
        structure, plddt_min=float("-inf"), cfg=cfg,
    )
    k1, n_additional = _resample_think_overhead(structure, cfg)
    counts = _count_doc_tokens(doc)
    row: dict[str, object] = {
        "entry_id": structure.entry_id,
        "n_residues": len(structure.residues),
        "global_plddt": structure.global_plddt,
        "n_eligible_residues": n_elig_res,
        "n_long_eligible": n_long_e,
        "n_medium_eligible": n_med_e,
        "n_short_eligible": n_short_e,
        "n_long_total": n_long_t,
        "n_medium_total": n_med_t,
        "n_short_total": n_short_t,
        "n_think_initial_k1": k1,
        "n_think_additional_runs": n_additional,
    }
    row.update(counts)
    return row


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--input", type=Path, default=_DEFAULT_INPUT,
        help=(
            "Directory of PDB / mmCIF (.gz) inputs. Defaults to the "
            "FoldBench monomer GT set from exp20."
        ),
    )
    ap.add_argument(
        "--out", type=Path,
        default=EXP_ROOT / "data" / "sample_stats.csv",
        help="Output CSV path.",
    )
    ap.add_argument(
        "--context-length", type=int, default=CONTEXT_LENGTH,
        help="Token budget per document (matches CLI default).",
    )
    ap.add_argument(
        "--residue-plddt-min", type=float, default=0.0,
        help=(
            "pLDDT (== mean B-factor for PDB inputs) floor for the "
            "contact-eligibility pool. Default 0.0 disables the "
            "filter on PDB inputs whose B-factors aren't pLDDT; pass "
            "70.0 to mirror the AFDB-training default."
        ),
    )
    ap.add_argument(
        "--num-docs", type=int, default=None,
        help="Cap on docs to process; default = all inputs.",
    )
    args = ap.parse_args(argv)

    cfg = generate.GenerationConfig(residue_plddt_min=args.residue_plddt_min)

    rows: list[dict[str, object]] = []
    skipped = 0
    for structure in iter_parsed_structures(args.input):
        structure = _disambiguate_entry_id(structure)
        row = _row(structure, cfg=cfg, context_length=args.context_length)
        if row is None:
            skipped += 1
            continue
        rows.append(row)
        if args.num_docs is not None and len(rows) >= args.num_docs:
            break

    if not rows:
        raise SystemExit(f"No documents generated from {args.input}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_COLUMNS)
        w.writeheader()
        for r in rows:
            # Round float columns for csv readability.
            r = dict(r)
            r["global_plddt"] = (
                f"{r['global_plddt']:.3f}"
                if isinstance(r["global_plddt"], float) and math.isfinite(r["global_plddt"])
                else r["global_plddt"]
            )
            w.writerow(r)

    print(f"[{NAME}] wrote {args.out} ({len(rows)} rows, {skipped} skipped)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
