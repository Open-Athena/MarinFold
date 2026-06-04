"""Backfill MSAs for proteins whose a3m isn't on the active workspace's volume.

exp12 ran in two halves on two Modal workspaces (``timodonnell`` then
``open-athena``); the MSAs were never consolidated, so the first ~48
proteins' ``non_pairing.a3m`` files live only on the ``timodonnell``
volume. Their MSAs are deterministic (ColabFold MMseqs2), so we just
regenerate them on whatever workspace is active.

For each requested stem this script:
  1. downloads the GT biological-assembly CIF from the HF bucket,
  2. extracts the canonical one-letter sequence (same path
     prepare-inputs used originally, so the sequence — and therefore the
     MSA — is identical), and
  3. dispatches ``precompute_msa`` (idempotent; skips stems already on
     the volume).

By default it targets the stems with no depth yet in
``data/msa_depth.csv``; pass ``--stems-file`` to override.

Run on the open-athena workspace (the active profile)::

    .venv/bin/python _scripts/backfill_msa.py

Then recompute depth for all 100 and replot::

    .venv/bin/python _scripts/compute_msa_depth.py --inputs inputs --out data/msa_depth.csv
    .venv/bin/python cli.py plot --scores data/scores.csv --out plots/ --msa-depth data/msa_depth.csv
"""

import argparse
import csv
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import modal_app
import prepare_inputs
from compute_msa_depth import HF_BUCKET_ID, load_manifest

HF_GT_PREFIX = "data/protenix-foldbench-monomers/gt"


def missing_stems_from_depth(depth_csv: Path) -> list[str]:
    """Stems in ``depth_csv`` whose n_seqs cell is blank (no MSA computed)."""
    out: list[str] = []
    with depth_csv.open() as f:
        for row in csv.DictReader(f):
            if not (row.get("n_seqs") or "").strip():
                out.append(row["stem"])
    return out


def fetch_sequences(stems: list[str], by_stem: dict[str, dict]) -> list[tuple[str, str]]:
    """Download each stem's GT CIF from the HF bucket and extract its sequence."""
    from huggingface_hub import HfApi

    api = HfApi()
    tmp = Path(tempfile.mkdtemp(prefix="exp12_gt_"))
    files = [(f"{HF_GT_PREFIX}/{stem}.cif", tmp / f"{stem}.cif") for stem in stems]
    print(f"downloading {len(files)} GT CIFs from HF bucket {HF_BUCKET_ID} ...")
    api.download_bucket_files(HF_BUCKET_ID, files=files, raise_on_missing_files=True)

    out: list[tuple[str, str]] = []
    for stem in stems:
        chain_id = by_stem[stem]["chain_id"]
        seq = prepare_inputs.extract_canonical_sequence(tmp / f"{stem}.cif", chain_id)
        out.append((stem, seq))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inputs", type=Path, default=Path("inputs"),
                        help="Dir with manifest.csv (falls back to HF bucket if absent).")
    parser.add_argument("--depth-csv", type=Path, default=Path("data/msa_depth.csv"),
                        help="Used to pick stems with no depth yet (default target).")
    parser.add_argument("--stems-file", type=Path, default=None,
                        help="Override: file with one stem per line to backfill.")
    args = parser.parse_args()

    manifest = load_manifest(args.inputs)
    by_stem = {row["stem"]: row for row in manifest}

    if args.stems_file is not None:
        stems = [s.strip() for s in args.stems_file.read_text().splitlines() if s.strip()]
    else:
        stems = missing_stems_from_depth(args.depth_csv)
    stems = [s for s in stems if s in by_stem]
    print(f"backfilling MSAs for {len(stems)} stems")

    seq_args = fetch_sequences(stems, by_stem)
    print(f"dispatching precompute_msa for {len(seq_args)} proteins (open-athena)...")
    with modal_app.app.run():
        results = list(modal_app.precompute_msa.starmap(seq_args))

    n_ran = sum(1 for r in results if not r.get("skipped") and not r.get("failed"))
    n_skipped = sum(1 for r in results if r.get("skipped"))
    n_failed = sum(1 for r in results if r.get("failed"))
    print(f"done: {n_ran} ran, {n_skipped} skipped, {n_failed} failed")
    for r in results:
        if r.get("failed"):
            print(f"  FAILED {r['stem']}: {r.get('error')}")


if __name__ == "__main__":
    main()
