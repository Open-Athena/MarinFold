"""Build a portable visual curation gallery from measured AFDB sample pairs.

The gallery embeds its data and a pinned 3Dmol.js runtime. All displayed traces
are C-alpha coordinates from source predictions, not reconstructed structures.
The 8-angstrom contact maps are geometry diagnostics, not pyconfind labels.
"""

import argparse
import json
from pathlib import Path

import requests
from structure_audit import aligned_indices, load_protein, read_csv
from tmtools import tm_align


def make_payload(sample_dir: Path, cache_dir: Path) -> dict:
    """Serialize candidates, their nearest anchors, and alignment diagnostics."""
    candidates = read_csv(sample_dir / "candidates.csv")
    rows = read_csv(sample_dir / "sample.csv")
    proteins = {}
    for row in rows:
        protein = load_protein(cache_dir, row["entry_id"])
        proteins[row["entry_id"]] = {
            "sequence": protein.sequence,
            "coords": protein.coords.round(3).tolist(),
            "plddt": protein.plddt.round(2).tolist(),
            "anchor": row["is_anchor"].lower() == "true",
            "cluster": row["struct_cluster_id"],
        }
    for row in candidates:
        a = load_protein(cache_dir, row["entry_id"])
        b = load_protein(cache_dir, row["nearest_anchor"])
        aligned = tm_align(a.coords, b.coords, a.sequence, b.sequence)
        ia, ib = aligned_indices(aligned.seqxA, aligned.seqyA)
        row["aligned_candidate"] = (
            (a.coords @ aligned.u.T + aligned.t).round(3).tolist()
        )
        row["aligned_indices_a"] = ia.tolist()
        row["aligned_indices_b"] = ib.tolist()
        row["tm"] = float(row["nearest_anchor_tm"])
        row["selected"] = int(row["selected_order"]) > 0
    candidates.sort(key=lambda r: (not r["selected"], r["tm"], r["entry_id"]))
    return {
        "candidates": candidates,
        "proteins": proteins,
        "pairs": read_csv(sample_dir / "pairs.csv"),
        "summary": json.loads((sample_dir / "audit.json").read_text()),
    }


def main() -> None:
    """Generate the offline HTML artifact and preserve its runtime license."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = make_payload(args.sample_dir, args.cache)
    runtime_path = args.cache.parent / "3Dmol-2.4.2.min.js"
    if not runtime_path.exists():
        response = requests.get(
            "https://cdn.jsdelivr.net/npm/3dmol@2.4.2/build/3Dmol-min.js", timeout=60
        )
        response.raise_for_status()
        runtime_path.write_bytes(response.content)
    template = Path(__file__).with_name("gallery_template.html").read_text()
    html = template.replace("__RUNTIME__", runtime_path.read_text()).replace(
        "__DATA__", json.dumps(payload, separators=(",", ":")).replace("<", "\\u003c")
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html)
    license_response = requests.get(
        "https://cdn.jsdelivr.net/npm/3dmol@2.4.2/LICENSE", timeout=30
    )
    license_response.raise_for_status()
    args.output.with_name("3Dmol-LICENSE.txt").write_text(license_response.text)
    print(
        f"{args.output}: {len(payload['candidates'])} candidates, {len(html):,} bytes"
    )


if __name__ == "__main__":
    main()
