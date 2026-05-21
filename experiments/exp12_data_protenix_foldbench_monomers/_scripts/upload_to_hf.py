"""Upload exp12 outputs to the open-athena/MarinFold HF bucket.

What goes up (under data/protenix-foldbench-monomers/ in the bucket):
  - scores CSVs (data/scores.csv, scores_all_samples.csv, scores_summary.csv)
  - manifest.csv (the 100-protein index from inputs/)
  - gt/*.cif (100 GT biological-assembly mmCIFs from inputs/gt/)
  - best/{mode}/{stem}/{structure.cif, confidence.json, distogram.npz,
    provenance.json} — the curated top-1 picks (200 entries × 4 files)
  - msa/{stem}/0/{...} — pre-computed ColabFold MSAs (synced from the
    protenix-foldbench-msa Modal Volume on the open-athena workspace)
  - README.md — quick orientation pointing back to the experiment + issue

Bucket: huggingface.co/buckets/open-athena/MarinFold
Path:   data/protenix-foldbench-monomers/

Idempotency: this is meant to be a one-shot upload. Re-runs upload
all files again (HF will deduplicate at the storage layer via xet
content hashing). For incremental updates use `huggingface_hub`
directly with `add=[...]` for just the changed files.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi


BUCKET_ID = "open-athena/MarinFold"
REMOTE_PREFIX = "data/protenix-foldbench-monomers"


HF_README = """\
# Protenix v2 on FoldBench monomers (single-seq + MSA)

Outputs of running Protenix v2 on the first 100 of FoldBench's
[`monomer_protein.csv`](https://github.com/BEAM-Labs/FoldBench/blob/main/targets/monomer_protein.csv),
in both single-sequence mode (no MSA) and with pre-computed MSAs from
ColabFold's MMseqs2 API. Source experiment: [Open-Athena/MarinFold
exp12](https://github.com/Open-Athena/MarinFold/tree/main/experiments/exp12_data_protenix_foldbench_monomers)
(tracking issue: [Open-Athena/MarinFold#12](https://github.com/Open-Athena/MarinFold/issues/12)).

For per-protein metric reproduction details (cutoffs, atom
conventions, CASP precision spec, etc.) see the experiment README in
the repo above; the upload here is the data side only.

## Layout

```
data/protenix-foldbench-monomers/
├── README.md                      # this file
├── scores.csv                     # 200 rows, top-1 sample per (protein × mode)
├── scores_all_samples.csv         # 8000 rows, every (protein × mode × seed × sample)
├── scores_summary.csv             # per-mode mean / median / min / max
├── manifest.csv                   # 100 proteins (pdb_id, chain, n_residues, ...)
├── gt/                            # GT biological-assembly mmCIFs (RCSB)
│   └── {pdb}_{chain}.cif
├── best/                          # top-1 sample by Protenix's ranking_score
│   ├── single_seq/{pdb}_{chain}/
│   │   ├── structure.cif
│   │   ├── confidence.json        # Protenix summary_confidence for that sample
│   │   ├── distogram.npz          # [N, N, 64] softmaxed probs from the chosen seed
│   │   └── provenance.json        # which seed/sample_idx + ranking_score
│   └── msa/{pdb}_{chain}/...
└── msa/                           # pre-computed ColabFold MSAs (per protein)
    └── {pdb}_{chain}/0/
        ├── pairing.a3m            # stub for monomers (just the query)
        ├── 0/non_pairing.a3m      # the real unpaired MSA
        ├── bfd.mgnify30.metaeuk30.smag30.a3m
        └── uniref.a3m
```

Settings used: 5 seeds × 8 diffusion samples per (protein × mode) on
H100; Protenix v2 default N_cycle=10, N_step=200. Top-1 sample per
(protein × mode) chosen by Protenix's built-in `ranking_score`. See
the experiment README for the full reproducibility pins (FoldBench
commit, Protenix / torch / gemmi versions, etc.).

## Not in this upload

The full 40-sample raw outputs (~50 GB) live on the source
`foldbench-protenix-runs` Modal Volumes split across two workspaces.
The per-sample metrics in `scores_all_samples.csv` capture everything
downstream eval typically needs; ask the Open-Athena team if you need
the raw structures themselves.
"""


def gather_local_files(exp_dir: Path) -> list[tuple[str, str]]:
    """Build (local_path, remote_path) tuples for the experiment-dir contents."""
    add: list[tuple[str, str]] = []

    # Scoring CSVs.
    for name in ("scores.csv", "scores_all_samples.csv", "scores_summary.csv"):
        p = exp_dir / "data" / name
        if p.exists():
            add.append((str(p), f"{REMOTE_PREFIX}/{name}"))

    # Manifest (lives under inputs/ in the repo).
    p = exp_dir / "inputs" / "manifest.csv"
    if p.exists():
        add.append((str(p), f"{REMOTE_PREFIX}/manifest.csv"))

    # GT CIFs.
    gt_dir = exp_dir / "inputs" / "gt"
    for cif in sorted(gt_dir.glob("*.cif")):
        add.append((str(cif), f"{REMOTE_PREFIX}/gt/{cif.name}"))

    # best/ tree.
    best_dir = exp_dir / "best"
    for f in sorted(best_dir.rglob("*")):
        if f.is_file():
            rel = f.relative_to(best_dir)
            add.append((str(f), f"{REMOTE_PREFIX}/best/{rel}"))

    return add


def sync_msa_volume(local_dir: Path) -> None:
    """``modal volume get`` the entire MSA volume into ``local_dir``.

    Uses the currently-active Modal profile (should be open-athena,
    since that's where the MSAs live).
    """
    local_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "modal", "volume", "get",
        "protenix-foldbench-msa", "/", str(local_dir),
    ]
    subprocess.run(cmd, check=True)


def gather_msa_files(msa_local_dir: Path) -> list[tuple[str, str]]:
    """Walk the synced MSA tree and build upload tuples for every file."""
    out: list[tuple[str, str]] = []
    for f in sorted(msa_local_dir.rglob("*")):
        if not f.is_file():
            continue
        rel = f.relative_to(msa_local_dir)
        out.append((str(f), f"{REMOTE_PREFIX}/msa/{rel}"))
    return out


def chunked(iterable, size: int):
    chunk: list = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--exp-dir", type=Path, default=Path("."),
                        help="Path to the experiment dir (default: cwd).")
    parser.add_argument("--include-msa", action="store_true", default=True,
                        help="Sync + upload the MSAs (default: True).")
    parser.add_argument("--no-msa", action="store_false", dest="include_msa")
    parser.add_argument("--chunk-size", type=int, default=200,
                        help="Files per batch_bucket_files call.")
    parser.add_argument("--dry-run", action="store_true",
                        help="List what would be uploaded; don't upload.")
    args = parser.parse_args()

    exp_dir = args.exp_dir.resolve()
    api = HfApi()

    # Verify bucket.
    info = api.bucket_info(BUCKET_ID)
    print(f"Target bucket: {BUCKET_ID} (private={info.private}, files={info.total_files}, size={info.size})")

    # Stage README locally so it gets uploaded by the local-files pass.
    staging_dir = Path(tempfile.mkdtemp(prefix="exp12_hf_stage_"))
    readme_path = staging_dir / "README.md"
    readme_path.write_text(HF_README)
    files = [(str(readme_path), f"{REMOTE_PREFIX}/README.md")]

    files += gather_local_files(exp_dir)
    print(f"Local-side files: {len(files)} (incl. README)")

    msa_synced_dir: Path | None = None
    if args.include_msa:
        msa_synced_dir = Path(tempfile.mkdtemp(prefix="exp12_msa_sync_"))
        print(f"\nSyncing MSA volume to {msa_synced_dir} ...")
        sync_msa_volume(msa_synced_dir)
        msa_files = gather_msa_files(msa_synced_dir)
        print(f"MSA files: {len(msa_files)}")
        files += msa_files

    total_bytes = sum(os.path.getsize(local) for local, _ in files)
    print(f"\nTotal: {len(files)} files, ~{total_bytes / 1e9:.2f} GB")

    if args.dry_run:
        print("\n[dry-run] first 10:")
        for local, remote in files[:10]:
            print(f"  {os.path.getsize(local):>10}  {remote}")
        return

    # Upload in chunks.
    for i, chunk in enumerate(chunked(files, args.chunk_size), start=1):
        n = len(chunk)
        bytes_in_chunk = sum(os.path.getsize(local) for local, _ in chunk)
        print(f"[batch {i}] uploading {n} files, ~{bytes_in_chunk / 1e6:.1f} MB ...")
        api.batch_bucket_files(BUCKET_ID, add=chunk)
    print("\nDone. View at: https://huggingface.co/buckets/" + BUCKET_ID)

    # Cleanup temp dirs.
    if msa_synced_dir is not None:
        shutil.rmtree(msa_synced_dir, ignore_errors=True)
    shutil.rmtree(staging_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
