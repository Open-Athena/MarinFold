# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Fetch CAMEO ``hard`` modeling targets as a rolling difficult-monomer set.

CAMEO (https://cameo3d.org) runs weekly, blind, automated structure-prediction
benchmarking and labels every target's post-prediction difficulty
(easy/medium/hard). The hard tail is a temporally-honest stream of difficult
monomers (``notes/eval-dataset-design.md`` section 2,
``notes/low-msa-eval-curation.md`` section 4.3). Difficulty is template-based,
**not** purely MSA depth, so these get re-labelled with measured Neff
downstream — they probe the fold-novelty axis, and we keep the MSA-depth axis
separate.

CAMEO's site is dynamic; the durable handle is the targets table's DataTables
endpoint ``/modeling/targets/<period>/ajax/``, which returns one JSON row per
target with its PDB id, chain, sequence length, experimental method,
resolution, date, and a ``diff`` field (0=easy, 1=medium, 2=hard). We pull
that, keep the requested difficulty, and download each target's reference
structure straight from RCSB by PDB id (cheaper and more standard than
CAMEO's 90 MB ``raw_targets`` tarball). Structures land under
``structures/cameo_hard/`` (gitignored); the manifest in
``data/cameo_hard_manifest.csv`` (committed).
"""

import argparse
from pathlib import Path

import requests

from _pdb_io import ManifestRow, download_cif, http_get, polymer_chains, write_manifest

HERE = Path(__file__).resolve().parent

DEFAULT_BASE_URL = "https://cameo3d.org"
TARGETS_AJAX = "/modeling/targets/{period}/ajax/"
# CAMEO's ``diff`` integer -> difficulty label (verified against the site's
# easy/medium/hard target-count split).
DIFF_LABELS = {"0": "easy", "1": "medium", "2": "hard"}
LABEL_TO_DIFF = {v: k for k, v in DIFF_LABELS.items()}

NOVELTY_AXIS = "CAMEO hard: difficult monomer (template-based difficulty; re-label with measured Neff)"


def fetch_target_rows(base_url: str, period: str, to_date: str | None) -> list[dict]:
    """Fetch CAMEO's target table for ``period`` as a list of row dicts.

    Raises if the endpoint doesn't return the expected ``aaData`` envelope, so
    a CAMEO layout change fails loudly instead of yielding an empty set.
    """
    url = base_url.rstrip("/") + TARGETS_AJAX.format(period=period)
    params = {"to_date": to_date} if to_date else {}
    resp = http_get(url, params=params, timeout=90)
    try:
        payload = resp.json()
    except requests.exceptions.JSONDecodeError as exc:
        raise RuntimeError(
            f"CAMEO targets endpoint {url} did not return JSON "
            f"(layout may have changed): {exc}"
        ) from exc
    if "aaData" not in payload:
        raise RuntimeError(
            f"CAMEO targets JSON from {url} has no 'aaData' key "
            f"(keys: {list(payload)}); layout may have changed."
        )
    return payload["aaData"]


def manifest_row(row: dict, cif_path: Path) -> ManifestRow:
    """Build a manifest row from a CAMEO target row + its downloaded mmCIF."""
    pdb_id = str(row["pdbid"]).lower()
    chain = str(row["pdbid_chain"])
    chains = polymer_chains(cif_path)
    # Prefer the named chain's length; fall back to the whole entry.
    length = chains.get(chain) or sum(chains.values())
    res = row.get("res")
    return ManifestRow(
        source="cameo_hard",
        stem=f"{pdb_id}_{chain}",
        pdb_id=pdb_id,
        chain=chain,
        length=length,
        resolution=f"{res:.2f}" if isinstance(res, (int, float)) and res > 0 else "",
        deposit_date=str(row.get("date", "")),  # CAMEO target (evaluation) date
        category="hard",
        novelty_axis=NOVELTY_AXIS,
        local_path=str(cif_path.relative_to(HERE)),
    )


def run(
    *,
    base_url: str,
    period: str,
    to_date: str | None,
    difficulty: str,
    out_dir: Path,
    manifest_path: Path,
    limit: int | None,
    download: bool,
) -> list[ManifestRow]:
    rows = fetch_target_rows(base_url, period, to_date)
    want = LABEL_TO_DIFF[difficulty]
    hard = [r for r in rows if str(r.get("diff")) == want]
    print(
        f"CAMEO {period}: {len(rows)} targets, {len(hard)} '{difficulty}' "
        f"(diff={want})"
    )
    if limit is not None:
        hard = hard[:limit]
        print(f"  (limited to first {len(hard)} for this run)")

    out_rows: list[ManifestRow] = []
    for i, row in enumerate(hard, start=1):
        pdb_id = str(row["pdbid"]).lower()
        chain = str(row["pdbid_chain"])
        if not download:
            out_rows.append(
                ManifestRow(
                    source="cameo_hard",
                    stem=f"{pdb_id}_{chain}",
                    pdb_id=pdb_id,
                    chain=chain,
                    category="hard",
                    novelty_axis=NOVELTY_AXIS,
                )
            )
            continue
        print(f"[{i}/{len(hard)}] {pdb_id}_{chain}: downloading mmCIF ...", flush=True)
        cif = download_cif(pdb_id, out_dir)
        out_rows.append(manifest_row(row, cif))

    write_manifest(out_rows, manifest_path)
    print(f"Wrote {len(out_rows)} rows to {manifest_path}; structures in {out_dir}/")
    return out_rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--period", choices=("3-months", "6-months", "1-year"), default="1-year",
        help="CAMEO rolling window to pull targets from (default: 1-year).",
    )
    p.add_argument(
        "--to-date", default=None,
        help="End date (YYYY-MM-DD) of the window; default: CAMEO's latest.",
    )
    p.add_argument(
        "--difficulty", choices=("easy", "medium", "hard"), default="hard",
        help="Which difficulty class to keep (default: hard).",
    )
    p.add_argument(
        "--out", type=Path, default=HERE / "structures" / "cameo_hard",
        help="Where to write reference mmCIFs (default: ./structures/cameo_hard/).",
    )
    p.add_argument(
        "--manifest", type=Path, default=HERE / "data" / "cameo_hard_manifest.csv",
        help="Manifest CSV path (default: ./data/cameo_hard_manifest.csv).",
    )
    p.add_argument(
        "--cameo-base-url", default=DEFAULT_BASE_URL,
        help=f"CAMEO base URL (default: {DEFAULT_BASE_URL}).",
    )
    p.add_argument("--limit", type=int, default=None, help="Take first N targets (smoke test).")
    p.add_argument(
        "--no-download", dest="download", action="store_false",
        help="Write the target list only; skip mmCIF downloads.",
    )
    args = p.parse_args()

    run(
        base_url=args.cameo_base_url,
        period=args.period,
        to_date=args.to_date,
        difficulty=args.difficulty,
        out_dir=args.out,
        manifest_path=args.manifest,
        limit=args.limit,
        download=args.download,
    )


if __name__ == "__main__":
    main()
