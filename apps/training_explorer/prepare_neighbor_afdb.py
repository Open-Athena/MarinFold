"""Stage exact AFDB v4 source backbones for neighbor extraction."""

import argparse
import gzip
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from extract_neighbor_structures import compact_backbone, selected_sources

HERE = Path(__file__).resolve().parent
AFDB_BUCKET = "public-datasets-deepmind-alphafold-v4"


def fetch_backbone(entry_id: str, billing_project: str, output: Path) -> int:
    """Fetch and compact one requester-pays AFDB v4 mmCIF."""
    target = output / f"{entry_id}.pdb"
    if target.is_file() and target.stat().st_size:
        return target.stat().st_size
    uri = f"gs://{AFDB_BUCKET}/{entry_id}-model_v4.cif"
    content = subprocess.check_output(
        ["gcloud", "storage", "cat", uri, f"--billing-project={billing_project}"]
    )
    if content[:2] == b"\x1f\x8b":
        content = gzip.decompress(content)
    target.write_text(compact_backbone(content.decode(), entry_id))
    return target.stat().st_size


def main() -> None:
    """Materialize the selected AFDB v4 backbones for the CoreWeave job bundle."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--billing-project", required=True)
    parser.add_argument("--output", type=Path, default=HERE / "legacy_afdb")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    sources, _ = selected_sources()
    entries = sorted(
        source["entryId"] for source in sources.values() if source["kind"] == "afdb"
    )
    args.output.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        sizes = pool.map(
            lambda entry: fetch_backbone(entry, args.billing_project, args.output),
            entries,
        )
        total = sum(sizes)
    print(f"Staged {len(entries)} AFDB v4 backbones ({total:,} bytes)")


if __name__ == "__main__":
    main()
