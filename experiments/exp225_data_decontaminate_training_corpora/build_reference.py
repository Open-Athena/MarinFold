# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage 1 — pin the decontamination reference: 554 eval sequences + structures.

This is the durable deliverable. Today every corpus that has *any* eval filter
derived its own: #91's ESM-Atlas funnel pointed ``--eval-ref`` at exp65's
``candidate_sequences.csv``, which is 454 proteins — it never saw FoldBench-100,
the most contaminated slice we have (#41: 99/100 FoldBench monomers fall in a
trained fold). One pinned reference, versioned, is what stops the next corpus
repeating that.

The reference has two halves, because contamination has two axes:

* **Sequences** — ``data/reference/eval_queries.fasta``, 554 records headed
  ``{dataset}__{stem}``. Small, committed, and byte-checked against exp213's
  copy of the same file so the two experiments provably search the same
  queries. (Keyed on *both* dataset and stem: a few stems recur across
  FoldBench and exp65 with different sequences.)
* **Structures** — the 554 ground-truth mmCIFs are 228 MB, too big for git, so
  what is committed is ``data/reference/eval_structures.csv``: one row per
  protein with its sha256, source path, evaluated chain and residue count. The
  structures themselves are staged to ``--structures-out`` and published to the
  bucket by ``publish_reference.py``; the manifest is what makes a staged copy
  verifiable.

Both halves come from the **exp78 eval manifests**, which are the definition of
the 554-protein benchmark that #89 scores and #180 tracks. Nothing here
re-derives the eval set; it only freezes it.

    uv run python build_reference.py --structures-out /data/exp225_decontam/eval_structures
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import gemmi

from decontam_lib import N_EVAL_PROTEINS, REFERENCE_VERSION

HERE = Path(__file__).resolve().parent

#: The 554-protein benchmark, as exp78 froze it. Each manifest's ``gt_cif`` is
#: relative to its own base, which is why the two are carried as pairs.
EXP78_ROOT = Path("/home/bizon/git/MarinFold-exp78/experiments")
MANIFEST_SOURCES: tuple[tuple[Path, Path], ...] = (
    (
        EXP78_ROOT / "exp78_evals_esmfold_contacts/data/eval_manifest_foldbench.csv",
        EXP78_ROOT / "exp78_evals_esmfold_contacts/_scratch/gt_foldbench",
    ),
    (
        EXP78_ROOT / "exp78_evals_esmfold_contacts/data/eval_manifest_exp65.csv",
        EXP78_ROOT / "exp65_evals_low_msa_depth_proteins",
    ),
)

#: exp213 committed the same 554 queries. Building ours independently and then
#: diffing is a stronger guarantee than copying the file would be.
EXP213_QUERIES = (
    HERE.parent / "exp213_evals_train_sequence_overlap_audit" / "data" / "eval_queries.fasta"
)


@dataclass(frozen=True)
class EvalProtein:
    """One member of the pinned reference."""

    dataset: str
    stem: str
    sequence: str
    gt_cif: Path
    gt_chain: str

    @property
    def key(self) -> str:
        return f"{self.dataset}__{self.stem}"


def read_manifests(sources: tuple[tuple[Path, Path], ...]) -> list[EvalProtein]:
    """Load every eval protein, resolving each ``gt_cif`` against its own base."""
    proteins: list[EvalProtein] = []
    seen: set[str] = set()
    for manifest, base in sources:
        if not manifest.exists():
            raise SystemExit(f"missing eval manifest {manifest}")
        with manifest.open() as fh:
            for row in csv.DictReader(fh):
                protein = EvalProtein(
                    dataset=row["dataset"],
                    stem=row["stem"],
                    sequence=row["input_seq"].strip().upper(),
                    gt_cif=(base / row["gt_cif"]).resolve(),
                    gt_chain=row["gt_chain"],
                )
                if protein.key in seen:
                    raise ValueError(f"duplicate eval key {protein.key} in {manifest}")
                seen.add(protein.key)
                if not protein.gt_cif.exists():
                    raise SystemExit(f"{protein.key}: missing structure {protein.gt_cif}")
                proteins.append(protein)
    return proteins


def write_query_fasta(proteins: list[EvalProtein], out: Path) -> None:
    """The sequence half. Header grammar is exp213's, so the tables join."""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(f">{p.key}\n{p.sequence}\n" for p in proteins))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def select_chain(structure: gemmi.Structure, gt_chain: str) -> tuple[gemmi.Chain, str]:
    """The evaluated polymer chain, and how it was resolved.

    Two cases, both exhaustively checked across the 554 (533 / 21 respectively):

    * **one polymer chain** — take it, whatever it is called. 10 of the 533
      single-chain files carry a chain name that differs from the manifest's
      ``gt_chain`` (the manifest names the *entry's* chain; the staged file may
      use the label rather than the auth asym id — ``5sbj_A`` is chain ``C``
      inside its own file). With one candidate there is no ambiguity to
      resolve, so the name is not consulted.
    * **several polymer chains** — select by ``gt_chain``. These are the 21
      CAMEO/CASP entries staged as whole PDB entries; only one chain of each is
      evaluated, and matching training structures against its neighbours would
      over-filter for contamination we do not measure.

    Raising rather than falling back is deliberate: a mis-selected chain
    silently changes what gets purged from the training corpus.
    """
    polymers = [ch for ch in structure[0] if len(ch.get_polymer()) > 0]
    if not polymers:
        raise ValueError("no polymer chain")
    if len(polymers) == 1:
        return polymers[0], "single"
    for chain in polymers:
        if chain.name == gt_chain:
            return chain, "by_name"
    raise ValueError(
        f"gt_chain {gt_chain!r} not among polymer chains {[c.name for c in polymers]}"
    )


def stage_structure(protein: EvalProtein, out_dir: Path) -> tuple[Path, str, int]:
    """Write the evaluated chain alone to ``{key}.cif``; return path, mode, length.

    Two things are happening here and both matter for the drop list:

    * **The name is the reference key.** Foldseek keys a query on its filename,
      so a flat ``{dataset}__{stem}.cif`` is what lets a hit be attributed back
      to an eval protein — and keying on dataset *and* stem is what stops the
      few stems that recur across FoldBench and exp65 from colliding.
    * **Only the evaluated chain is kept.** A single-chain query also means
      Foldseek emits exactly one DB entry per file, so the query name needs no
      ``_<chain>`` suffix stripping (exp41 had to guess at that, and our keys
      already end in ``_<chain>`` often enough to make the guess wrong).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    structure = gemmi.read_structure(str(protein.gt_cif))
    structure.setup_entities()
    chain, mode = select_chain(structure, protein.gt_chain)
    n_residues = len(chain.get_polymer())

    single = gemmi.Structure()
    single.name = protein.key
    single.cell = structure.cell
    single.spacegroup_hm = structure.spacegroup_hm
    model = gemmi.Model("1")
    model.add_chain(chain)
    single.add_model(model)
    single.setup_entities()

    dst = out_dir / f"{protein.key}.cif"
    single.make_mmcif_document().write_file(str(dst))
    return dst, mode, n_residues


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--structures-out",
        type=Path,
        default=Path("/data/exp225_decontam/eval_structures"),
        help="where the 554 GT mmCIFs are staged (too big for git)",
    )
    ap.add_argument("--fasta-out", type=Path, default=HERE / "data/reference/eval_queries.fasta")
    ap.add_argument(
        "--manifest-out", type=Path, default=HERE / "data/reference/eval_structures.csv"
    )
    ap.add_argument(
        "--provenance-out", type=Path, default=HERE / "data/reference/reference.provenance.json"
    )
    ap.add_argument(
        "--exp213-queries",
        type=Path,
        default=EXP213_QUERIES,
        help="exp213's committed eval_queries.fasta, to prove both experiments search the "
        "same 554 queries. exp213 is still on a branch (PR #216), so extract it with "
        "`git show <ref>:experiments/exp213_.../data/eval_queries.fasta > <path>` and pass "
        "the path. Skipped with a warning when absent.",
    )
    args = ap.parse_args()

    proteins = read_manifests(MANIFEST_SOURCES)
    if len(proteins) != N_EVAL_PROTEINS:
        raise SystemExit(f"expected {N_EVAL_PROTEINS} eval proteins, got {len(proteins)}")
    print(f"[reference] {len(proteins)} eval proteins", flush=True)

    write_query_fasta(proteins, args.fasta_out)
    print(f"[sequences] -> {args.fasta_out}", flush=True)

    cross_checked = args.exp213_queries.exists()
    if cross_checked:
        ours = args.fasta_out.read_text()
        theirs = args.exp213_queries.read_text()
        if ours != theirs:
            raise SystemExit(
                f"the reference queries disagree with {args.exp213_queries} — the two "
                "experiments would be measuring different eval sets; reconcile before "
                "going further"
            )
        print(f"[sequences] byte-identical to {args.exp213_queries}", flush=True)
    else:
        print(
            f"[sequences] {args.exp213_queries} not present; skipping the cross-check "
            "(see the README for how to extract it from exp213's branch)",
            flush=True,
        )

    rows = []
    for i, protein in enumerate(proteins, 1):
        try:
            staged, mode, n_residues = stage_structure(protein, args.structures_out)
        except ValueError as exc:
            raise SystemExit(f"{protein.key}: {exc}") from exc
        rows.append(
            {
                "dataset": protein.dataset,
                "stem": protein.stem,
                "key": protein.key,
                "gt_chain": protein.gt_chain,
                "chain_resolution": mode,
                "seq_len": len(protein.sequence),
                "n_resolved_residues": n_residues,
                "sha256": sha256(staged),
                "staged_name": staged.name,
                "source_path": str(protein.gt_cif),
            }
        )
        if i % 100 == 0 or i == len(proteins):
            print(f"[structures] {i}/{len(proteins)}", flush=True)

    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest_out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[structures] -> {args.manifest_out}", flush=True)

    by_dataset = dict(Counter(row["dataset"] for row in rows))
    by_resolution = dict(Counter(row["chain_resolution"] for row in rows))
    args.provenance_out.write_text(
        json.dumps(
            {
                "reference_version": REFERENCE_VERSION,
                "n_proteins": len(proteins),
                "by_dataset": by_dataset,
                "chain_resolution": by_resolution,
                "manifests": [str(m) for m, _ in MANIFEST_SOURCES],
                "structures_staged_to": str(args.structures_out),
                "queries_cross_checked_against": str(args.exp213_queries) if cross_checked else None,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"[provenance] {by_dataset} -> {args.provenance_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
