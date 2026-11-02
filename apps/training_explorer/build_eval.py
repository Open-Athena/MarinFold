"""Build the complete eval-val and eval-denovo catalog for the explorer."""

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "experiments/exp245_evals_foldbench_held_out_monomers/data/eval_sets.csv"
OUTPUT = Path(__file__).resolve().parent / "data/eval.json"
SETS = ("eval-val", "eval-denovo")


def main() -> None:
    """Preserve every requested eval protein and its source metadata."""
    with INPUT.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["eval_set"] in SETS]
    proteins = [
        {
            "id": f"{row['eval_set']}:{row['stem']}",
            "label": row["stem"],
            "subset": row["eval_set"],
            "source": "PDB experiment",
            "sequence": row["sequence"],
            "length": int(row["seq_len"]),
            "title": row["title"],
            "organism": row["source_organisms"],
            "pdbId": row["pdb_id"],
            "chainId": row["chain_id"],
            "structureUrl": f"https://files.rcsb.org/download/{row['pdb_id'].upper()}.cif",
            "structureFormat": "mmcif",
            "pdbFallbackUrl": f"https://files.rcsb.org/download/{row['pdb_id'].upper()}.pdb",
            "viewerChain": row["auth_asym_ids"].split(";")[0],
            "neighbors": [],
        }
        for row in rows
    ]
    assert len(proteins) == 116, len(proteins)
    assert sum(p["subset"] == "eval-val" for p in proteins) == 97
    assert sum(p["subset"] == "eval-denovo" for p in proteins) == 19
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(
            {
                "title": "Evaluation proteins",
                "description": "All 97 eval-val and 19 eval-denovo proteins from exp245.",
                "population": 116,
                "provenance": "exp245/data/eval_sets.csv",
                "neighborCorpus": "latest",
                "neighborsComplete": False,
                "proteins": proteins,
            },
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
