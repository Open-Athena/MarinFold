"""Build a conservative sequence reference for Helico's date-eligible PDB pool.

This is an exposure screen, not a record of the crops actually sampled by the
6000-step fine-tune, nor a complete reconstruction of inherited Protenix data.
All protein chains of eligible entries are retained to avoid false clean claims.
"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path


def main() -> None:
    """Select protein SEQRES records by the processed manifest's training date rule."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("/data/tim/helico-data"))
    ap.add_argument("--out", type=Path, default=Path("/data/exp350"))
    args = ap.parse_args()
    manifest_path = args.root / "processed/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    selected = {
        p.lower()
        for p, m in manifest.items()
        if not m.get("release_date") or m["release_date"] < "2021-09-30"
    }
    dates = {p.lower(): m.get("release_date", "") for p, m in manifest.items()}
    del manifest
    source = args.root / "raw/pdb_seqres.txt.gz"
    out = args.out / "helico_training_pool.fasta"
    n_records = 0
    entries = set()
    with gzip.open(source, "rt") as fh, out.open("w") as dest:
        keep = False
        for line in fh:
            if line.startswith(">"):
                name = line[1:].split()[0]
                pdb = name.split("_")[0].lower()
                keep = pdb in selected and "mol:protein" in line
                if keep:
                    dest.write(f">helico|{name}|{dates[pdb]}\n")
                    n_records += 1
                    entries.add(pdb)
            elif keep:
                dest.write(line)
    provenance = {
        "manifest": str(manifest_path),
        "sequence_source": str(source),
        "eligible_entries": len(selected),
        "protein_entries_written": len(entries),
        "protein_chains": n_records,
        "cutoff": "2021-09-30 (exclusive; missing date included)",
        "scope": "all protein chains of date-eligible preprocessed entries; conservative fine-tuning pool, not actual sampled crops or complete Protenix pretraining",
        "source_sha256": hashlib.file_digest(source.open("rb"), "sha256").hexdigest(),
        "fasta_sha256": hashlib.file_digest(out.open("rb"), "sha256").hexdigest(),
    }
    (args.out / "helico_reference.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print(json.dumps(provenance, indent=2), flush=True)


if __name__ == "__main__":
    main()
