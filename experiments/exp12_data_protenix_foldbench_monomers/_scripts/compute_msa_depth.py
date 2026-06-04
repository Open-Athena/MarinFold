"""Compute per-protein MSA depth (n_seqs + N_eff) and write data/msa_depth.csv.

Reads inputs/manifest.csv, dispatches ``compute_msa_depth`` across all
rows via the Modal app (reading each protein's ``non_pairing.a3m`` off
the protenix-foldbench-msa Volume), and writes one row per protein.

MSA depth is a per-protein property (independent of mode / seed /
sample), so it lives in its own CSV and is joined to data/scores.csv at
plot time via the (pdb_id, chain_id) key.

Invoke from the experiment dir::

    .venv/bin/python _scripts/compute_msa_depth.py
    .venv/bin/python _scripts/compute_msa_depth.py --stems-file /tmp/some_stems.txt
"""

import argparse
import csv
import sys
from pathlib import Path

# Make the experiment dir importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import modal_app
import msa_depth as md


import tempfile

# HF bucket where the 100-protein manifest is mirrored (see
# _scripts/upload_to_hf.py — it uploads via the HF *buckets* API, not a
# dataset repo). Used as a fallback when there's no local
# inputs/manifest.csv (a prepare-inputs artifact that isn't committed).
HF_BUCKET_ID = "open-athena/MarinFold"
HF_MANIFEST_PATH = "data/protenix-foldbench-monomers/manifest.csv"


def load_manifest(inputs_dir: Path) -> list[dict]:
    """Read manifest.csv from ``inputs_dir`` if present, else from the HF bucket."""
    local = inputs_dir / "manifest.csv"
    if local.exists():
        print(f"using local manifest: {local}")
        return list(csv.DictReader(local.open()))
    from huggingface_hub import HfApi
    dest = Path(tempfile.mkdtemp(prefix="exp12_manifest_")) / "manifest.csv"
    print(f"no {local}; downloading manifest from HF bucket {HF_BUCKET_ID}:{HF_MANIFEST_PATH}")
    HfApi().download_bucket_files(
        HF_BUCKET_ID, files=[(HF_MANIFEST_PATH, dest)], raise_on_missing_files=True,
    )
    return list(csv.DictReader(dest.open()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inputs", type=Path, default=Path("inputs"),
                        help="Dir produced by prepare-inputs (contains manifest.csv).")
    parser.add_argument("--out", type=Path, default=Path("data/msa_depth.csv"),
                        help="Output CSV path.")
    parser.add_argument("--stems-file", type=Path, default=None,
                        help="Optional file with one stem per line; restrict to those.")
    args = parser.parse_args()

    manifest = load_manifest(args.inputs)
    by_stem = {row["stem"]: row for row in manifest}
    stems = list(by_stem)
    if args.stems_file is not None:
        wanted = {s.strip() for s in args.stems_file.read_text().splitlines() if s.strip()}
        stems = [s for s in stems if s in wanted]
        print(f"filtering to {len(stems)} stems from {args.stems_file}")

    print(f"computing MSA depth for {len(stems)} proteins...")
    with modal_app.app.run():
        results = list(modal_app.compute_msa_depth.starmap([(s,) for s in stems]))

    eff_cols = [f"n_eff_{t}" for t in md.DEFAULT_THRESHOLDS]
    fieldnames = ["pdb_id", "chain_id", "stem", "n_residues", "query_len",
                  "n_seqs", *eff_cols]

    missing = [r["stem"] for r in results if not r.get("found")]
    if missing:
        print(f"WARN: {len(missing)} proteins have no non_pairing.a3m: {missing}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in sorted(results, key=lambda x: x["stem"]):
            row = by_stem[r["stem"]]
            found = r.get("found", False)
            # Leave depth columns blank (not 0 / nan) when the MSA wasn't
            # found, so "no a3m on this volume" is distinguishable from a
            # genuinely shallow MSA. A later backfill can fill these in.
            writer.writerow({
                "pdb_id": row["pdb_id"],
                "chain_id": row["chain_id"],
                "stem": r["stem"],
                "n_residues": row["n_residues"],
                "query_len": r["query_len"] if found else "",
                "n_seqs": r["n_seqs"] if found else "",
                **{c: (r[c] if found else "") for c in eff_cols},
            })
    print(f"Wrote {args.out} with {len(results)} rows.")


if __name__ == "__main__":
    main()
