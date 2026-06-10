# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Extract one query sequence per candidate structure -> FASTA + CSV.

Both sequence-axis measures need the candidate *sequences*, not the
structures: MSA depth (feed each to ColabFold MMseqs2 -> a3m -> Neff) and the
training-set sequence-leakage check (MMseqs2 vs afdb-24M). This reads the
three fetch manifests, pulls the query chain's sequence from each downloaded
structure with gemmi, and writes:

- ``data/candidate_sequences.csv``  (source, stem, pdb_id, chain, length, sequence)
- ``data/candidates.fasta``         (all candidates, header = stem)
- ``data/seqs/<source>.fasta``      (one FASTA per source)

Sequence source: gemmi's polymer one-letter sequence for the manifest's named
chain (or the longest polymer chain when the manifest chain is blank, e.g. the
single-chain CASP domain PDBs). This is the observed construct sequence, which
is what the CAMEO/CASP/de-novo query sequences are.
"""

import argparse
import csv
from pathlib import Path

import gemmi

HERE = Path(__file__).resolve().parent
MANIFESTS = {
    "denovo_pdb": HERE / "data" / "denovo_pdb_manifest.csv",
    "casp_fm": HERE / "data" / "casp_fm_manifest.csv",
    "cameo_hard": HERE / "data" / "cameo_hard_manifest.csv",
}


def sequence_for(structure_path: Path, chain_id: str) -> str:
    """Return the one-letter polymer sequence for ``chain_id`` (or longest chain).

    ``setup_entities()`` first so raw blank-chain PDBs are recognised as
    polymers. Falls back to the longest polymer chain when the requested chain
    id isn't found (manifest ``chain`` is blank for single-chain CASP domains).
    """
    st = gemmi.read_structure(str(structure_path))
    if not st:
        return ""
    st.setup_entities()
    model = st[0]
    chain = None
    if chain_id:
        chain = next((c for c in model if c.name == chain_id), None)
    if chain is None:
        # Longest polymer chain.
        chain = max(model, key=lambda c: len(c.get_polymer()), default=None)
    if chain is None:
        return ""
    return gemmi.one_letter_code(chain.get_polymer().extract_sequence()).upper()


def run(out_csv: Path, fasta_dir: Path, combined_fasta: Path) -> list[dict]:
    rows: list[dict] = []
    fasta_dir.mkdir(parents=True, exist_ok=True)
    for source, manifest in MANIFESTS.items():
        if not manifest.exists():
            print(f"  (skipping {source}: {manifest.name} not found)")
            continue
        n_src = 0
        with (fasta_dir / f"{source}.fasta").open("w") as fa:
            for r in csv.DictReader(manifest.open()):
                if not r["local_path"]:
                    continue  # no structure on disk (e.g. unresolved CASP FM)
                path = HERE / r["local_path"]
                seq = sequence_for(path, r["chain"])
                if not seq:
                    print(f"  WARNING: no sequence extracted for {r['stem']} ({path.name})")
                    continue
                rows.append(
                    {
                        "source": source,
                        "stem": r["stem"],
                        "pdb_id": r["pdb_id"],
                        "chain": r["chain"],
                        "length": len(seq),
                        "sequence": seq,
                    }
                )
                fa.write(f">{r['stem']}\n{seq}\n")
                n_src += 1
        print(f"  {source}: {n_src} sequences")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["source", "stem", "pdb_id", "chain", "length", "sequence"])
        w.writeheader()
        w.writerows(rows)
    with combined_fasta.open("w") as fa:
        for r in rows:
            fa.write(f">{r['stem']}\n{r['sequence']}\n")
    print(f"Wrote {len(rows)} sequences -> {out_csv} and {combined_fasta}")
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-csv", type=Path, default=HERE / "data" / "candidate_sequences.csv")
    p.add_argument("--fasta-dir", type=Path, default=HERE / "data" / "seqs")
    p.add_argument("--combined-fasta", type=Path, default=HERE / "data" / "candidates.fasta")
    args = p.parse_args()
    run(args.out_csv, args.fasta_dir, args.combined_fasta)


if __name__ == "__main__":
    main()
