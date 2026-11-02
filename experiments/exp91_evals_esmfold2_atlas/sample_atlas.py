# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Draw a small, reproducible, streamed sample of the ESM Atlas folds.

The full Atlas is multi-TB and lives in AWS (us-west-2) while our compute is
on GCS/marin, so we never bulk-copy it (a cross-region copy >10 GB needs human
sign-off). Instead we pull:

- a **metadata** sample (``protein_hash, sequence, mean_plddt, ptm``) of
  ``--n-meta`` rows → ``data/atlas_sample.parquet`` — enough for the length /
  pLDDT / pTM distributions, and
- a smaller **structure** sub-sample of ``--n-struct`` rows whose
  ``structure_blob`` we decode (brotli + msgpack → atom37) and write as PDB
  files under ``--cif-dir`` for Foldseek novelty queries, plus a
  ``data/atlas_struct_sample.csv`` with per-row geometry (n_res, n_atoms,
  is_monomer, mean_plddt, ptm).

Sampling is reproducible and cheap: ``protein_hash`` is an MD5 of the sequence,
so the dataset's row order is uniform-random w.r.t. biology (length, pLDDT,
source). We therefore read a handful of **scattered contiguous blocks** (seeded
random offsets, ``scanner(offset, limit)``) rather than random single-row
``take``s — same statistical sample, ~20× less latency cross-region. Bytes
pulled are logged so we can show the run stayed far under any transfer budget.

Usage::

    uv run python sample_atlas.py --n-meta 100000 --n-struct 2000
    uv run python sample_atlas.py --n-meta 1000 --n-struct 50   # smoke
"""

import argparse
import time
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import atlas_io

HERE = Path(__file__).resolve().parent
META_COLS = ["protein_hash", "sequence", "mean_plddt", "ptm"]


def _block_offsets(n_total: int, n_rows: int, block: int, seed: int) -> list[int]:
    """Seeded random start offsets for contiguous blocks covering ~n_rows."""
    n_blocks = max(1, -(-n_rows // block))  # ceil
    rng = np.random.default_rng(seed)
    hi = max(1, n_total - block)
    return sorted(int(o) for o in rng.integers(0, hi, size=n_blocks))


def scan_blocks(ds, columns: list[str], n_rows: int, block: int, seed: int):
    """Read scattered contiguous blocks and concat to one table."""
    offsets = _block_offsets(atlas_io.N_FOLDS_1B, n_rows, block, seed)
    tables = []
    for off in offsets:
        tables.append(ds.scanner(columns=columns, offset=off, limit=block).to_table())
    return pa.concat_tables(tables)


def sample_metadata(ds, n_meta: int, seed: int, block: int) -> pd.DataFrame:
    print(f"[meta] scanning ~{n_meta:,} rows in blocks of {block:,} "
          f"(cols={META_COLS}) …", flush=True)
    t0 = time.time()
    df = scan_blocks(ds, META_COLS, n_meta, block, seed).to_pandas()
    df["seq_len"] = df["sequence"].str.len()
    print(f"[meta] done in {time.time() - t0:.1f}s, {len(df):,} rows", flush=True)
    return df


def sample_structures(ds, n_struct: int, seed: int, block: int,
                      cif_dir: Path) -> pd.DataFrame:
    """Decode ~n_struct structure_blobs from scattered blocks."""
    cif_dir.mkdir(parents=True, exist_ok=True)
    cols = ["protein_hash", "sequence", "mean_plddt", "ptm", "structure_blob"]
    print(f"[struct] scanning ~{n_struct:,} structure_blobs …", flush=True)
    t0 = time.time()
    # Different seed → blocks disjoint from the metadata offsets; smaller blocks
    # because structure_blob is ~10–300 KB/row.
    tbl = scan_blocks(ds, cols, n_struct, block, seed + 1)
    d = tbl.to_pydict()
    rows = []
    blob_bytes = 0
    for i in range(len(d["protein_hash"])):
        blob = d["structure_blob"][i]
        blob_bytes += len(blob)
        struct = atlas_io.decode_structure_blob(blob)
        mask = np.asarray(struct["atom37_mask"], dtype=bool)
        n_res = mask.shape[0]
        n_atoms = int(mask.sum())
        ph = d["protein_hash"][i]
        cif_path = cif_dir / f"{ph}.cif"
        atlas_io.write_structure_cif(struct, cif_path, name=ph)
        # Validate it round-trips as a polymer (the check Foldseek's parser makes).
        st = gemmi.read_structure(str(cif_path))
        assert st[0][0].get_polymer().length() == n_res, f"polymer parse {ph}"
        rows.append({
            "protein_hash": ph,
            "seq_len": len(d["sequence"][i]),
            "n_res": n_res,
            "n_present_atoms": n_atoms,
            "n_chains": len(struct.get("chain_boundaries", [])),
            "is_monomer": atlas_io.is_monomer(struct),
            "mean_plddt": d["mean_plddt"][i],
            "ptm": d["ptm"][i],
            "blob_bytes": len(blob),
        })
    df = pd.DataFrame(rows)
    print(f"[struct] decoded {len(df):,} structures in {time.time() - t0:.1f}s; "
          f"pulled {blob_bytes / 1e6:.1f} MB of structure_blob; "
          f"all monomer={bool(df['is_monomer'].all())}", flush=True)
    return df


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-meta", type=int, default=100_000)
    ap.add_argument("--n-struct", type=int, default=2_000)
    ap.add_argument("--seed", type=int, default=91)
    ap.add_argument("--meta-block", type=int, default=5_000,
                    help="contiguous rows per metadata block scan")
    ap.add_argument("--struct-block", type=int, default=200,
                    help="contiguous rows per structure block scan")
    ap.add_argument("--dataset", default=atlas_io.FOLDS_1B_URI)
    ap.add_argument("--out-dir", type=Path, default=HERE / "data")
    ap.add_argument("--cif-dir", type=Path, default=HERE / "data" / "struct_sample")
    args = ap.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ds = atlas_io.open_folds(args.dataset)

    meta_df = sample_metadata(ds, args.n_meta, args.seed, args.meta_block)
    meta_path = args.out_dir / "atlas_sample.parquet"
    pq.write_table(pa.Table.from_pandas(meta_df, preserve_index=False), meta_path)
    print(f"[meta] wrote {meta_path} ({meta_path.stat().st_size / 1e6:.1f} MB)")

    struct_df = sample_structures(ds, args.n_struct, args.seed, args.struct_block,
                                  args.cif_dir)
    struct_path = args.out_dir / "atlas_struct_sample.csv"
    struct_df.to_csv(struct_path, index=False)
    print(f"[struct] wrote {struct_path} and {len(struct_df)} PDBs to {args.cif_dir}")


if __name__ == "__main__":
    main()
