# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Fetch CASP14/15 free-modeling (FM) target domains as hard, novel-fold GT.

CASP FM domains are the community gold standard for "no usable template, few
homologs" (``notes/low-msa-eval-curation.md`` section 4.3,
``notes/eval-dataset-design.md`` section 7): expert-curated, blind, and
temporally honest by construction. They are small in number (tens) but
high-signal.

Two inputs:

1. The per-domain FM/FM-TBM classification, committed as
   ``data/casp_fm_domains.csv`` (built once from the assessors' tables on
   ``predictioncenter.org/casp{14,15}/results.cgi``; columns
   ``casp,target,domain,category``).
2. The target coordinates, published as public tarballs under
   ``predictioncenter.org/download_area/CASP{14,15}/targets/``. CASP14 splits
   them into a domain-level tarball (multi-domain targets only) plus a
   full-target tarball; CASP15 ships a single domain-level tarball.

For each FM domain we resolve its coordinate file (prefer the domain-split
``<domain>.pdb``, else the full ``<target>.pdb``), copy it under
``structures/casp_fm/`` (gitignored), and write the shared manifest.

Some FM domains are *not* in the public monomer tarballs (their targets are
oligomeric or were released late, so predictioncenter only ships them in the
``oligo`` sets or not at all). For those we fall back to the deposited PDB
entry via the committed ``data/casp_fm_pdb_fallback.csv`` map (domain ->
``pdb_id, chain, casp_range``): the entry is downloaded from RCSB and the FM
domain is clipped out by residue range on the target chain. This is safe
because the deposited CASP-target chains carry author residue numbers equal to
the CASP target numbering (verified per entry, e.g. 8h2n chain A is numbered
327..1117, matching T1125's domain boundaries). Domains with no released PDB
at all (and the one whose domain is only partially modelled, T1125-D6) stay in
the manifest with an empty ``local_path`` rather than being silently dropped.

Each resolved domain is also stamped with a ``deposit_date``: the RCSB
deposition date of its answer structure (resolved from the CASP targetlist's
answer-PDB code, or the fallback map's explicit ``pdb_id``). This is the
temporal axis the maintainer needs to reason about training-set cutoffs, and is
the same ``deposit_date`` semantics the de novo source uses. Note a CASP
target's deposit date can predate its public release (structures are often
deposited then embargoed through the prediction season), so for strict
leakage-cutoff reasoning the release date is the more conservative bound.
"""

import argparse
import csv
import io
import re
import shutil
import tarfile
from pathlib import Path

import gemmi

from _pdb_io import (
    ManifestRow,
    count_residues,
    download_cif,
    download_to,
    http_get,
    write_manifest,
)

HERE = Path(__file__).resolve().parent
FM_DOMAINS_CSV = HERE / "data" / "casp_fm_domains.csv"
FM_PDB_FALLBACK_CSV = HERE / "data" / "casp_fm_pdb_fallback.csv"

# Targetlist (``;``-delimited CSV) maps each CASP target to its deposited answer
# structure: the trailing 4-char PDB code in the Description column. RCSB's
# entry metadata endpoint then gives that structure's deposition date. We date a
# CASP domain by its answer structure's RCSB ``deposit_date`` so the manifest's
# temporal axis is apples-to-apples with the de novo source.
CASP_TARGETLIST_URL = "https://predictioncenter.org/{casp}/targetlist.cgi?type=csv"
RCSB_ENTRY_URL = "https://data.rcsb.org/rest/v1/core/entry/{pdb}"

# Per-CASP download config. ``*_prefix`` are matched against the live targets
# directory index so a re-dated tarball (the filenames carry a release date)
# still resolves. ``target_prefix`` is the full-target fallback (CASP14 only).
CASP_CONFIG = {
    "CASP14": {
        "base": "https://predictioncenter.org/download_area/CASP14/targets/",
        "domain_prefix": "casp14.targets.T-dom.public",
        "target_prefix": "casp14.targets.T.public",
    },
    "CASP15": {
        "base": "https://predictioncenter.org/download_area/CASP15/targets/",
        "domain_prefix": "casp15.targets.TS-domains.public",
        "target_prefix": None,
    },
}

NOVELTY_AXIS = "CASP free-modeling: no usable template, few homologs (blind, temporally honest)"


def load_fm_domains(categories: set[str], casps: list[str]) -> list[dict]:
    """Read the committed FM classification, filtered by category + CASP."""
    with FM_DOMAINS_CSV.open() as fh:
        rows = [r for r in csv.DictReader(fh)]
    return [
        r for r in rows if r["category"] in categories and r["casp"] in casps
    ]


def load_pdb_fallback() -> dict[str, dict]:
    """domain -> {pdb_id, chain, casp_range, status, note} from the committed map."""
    if not FM_PDB_FALLBACK_CSV.exists():
        return {}
    with FM_PDB_FALLBACK_CSV.open() as fh:
        return {r["domain"]: r for r in csv.DictReader(fh)}


def parse_pdb_code(description: str) -> str | None:
    """Extract the answer PDB code from a targetlist Description cell.

    A PDB code is a digit followed by three alphanumerics, word-bounded; we
    return the last such token (the answer code trails the protein name, e.g.
    ``GLuc 7d2o`` -> ``7d2o``, ``S0A2C3d1 6vr4`` -> ``6vr4``) lower-cased.
    Cells with no released structure carry no code -- ``g3873``, ``Q858F5.1``
    have no word-bounded digit-led 4-char token -- and return ``None``.
    """
    codes = re.findall(r"\b[0-9][a-zA-Z0-9]{3}\b", description)
    return codes[-1].lower() if codes else None


def casp_target_pdb_map(casp: str) -> dict[str, str]:
    """Map CASP target id -> deposited answer PDB code, from the targetlist.

    ``targetlist.cgi?type=csv`` is a ``;``-delimited table whose Description
    column ends with the answer structure's 4-char PDB code for released targets;
    canceled / unreleased targets carry no code and are absent from the map.
    """
    text = http_get(CASP_TARGETLIST_URL.format(casp=casp.lower()), timeout=60).text
    out: dict[str, str] = {}
    for row in csv.DictReader(io.StringIO(text), delimiter=";"):
        target = (row.get("Target") or "").strip()
        code = parse_pdb_code(row.get("Description") or "")
        if target and code:
            out[target] = code
    return out


def rcsb_deposit_date(pdb_id: str, cache: dict[str, str]) -> str:
    """RCSB initial deposition date (``YYYY-MM-DD``) for ``pdb_id``; cached.

    The cache dedupes repeat lookups when several domains share one answer
    structure (e.g. T1125's four domains all map to ``8h2n``).
    """
    pid = pdb_id.lower()
    if pid in cache:
        return cache[pid]
    info = http_get(RCSB_ENTRY_URL.format(pdb=pid), timeout=60).json()
    raw = info.get("rcsb_accession_info", {}).get("deposit_date", "") or ""
    cache[pid] = raw.split("T")[0]  # ISO timestamp -> calendar date
    return cache[pid]


def parse_range(spec: str) -> list[tuple[int, int]]:
    """Parse a CASP residue range like ``327-460`` or ``7-209,216-226``."""
    segs: list[tuple[int, int]] = []
    for part in spec.split(","):
        lo, hi = part.split("-")
        segs.append((int(lo), int(hi)))
    return segs


def extract_domain_from_pdb(
    pdb_id: str, chain_id: str, casp_range: str, dest: Path, cif_dir: Path
) -> Path:
    """Clip an FM domain out of a deposited RCSB entry by residue range.

    Downloads ``pdb_id``'s mmCIF, keeps only ``chain_id``'s polymer residues
    whose author ``seqid.num`` falls inside ``casp_range`` (which is in CASP
    target numbering == author numbering for these entries), and writes a
    single-chain PDB to ``dest``. Idempotent.
    """
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    cif = download_cif(pdb_id, cif_dir)
    st = gemmi.read_structure(str(cif))
    st.setup_entities()
    src_chain = next((c for c in st[0] if c.name == chain_id), None)
    if src_chain is None:
        raise RuntimeError(f"{pdb_id}: chain {chain_id!r} not found")
    segs = parse_range(casp_range)
    out = gemmi.Structure()
    out.name = pdb_id.upper()
    model = gemmi.Model("1")
    chain = gemmi.Chain("A")
    for res in src_chain.get_polymer():
        if any(lo <= res.seqid.num <= hi for lo, hi in segs):
            chain.add_residue(res)
    if len(chain) == 0:
        raise RuntimeError(f"{pdb_id} chain {chain_id}: no residues in range {casp_range}")
    model.add_chain(chain)
    out.add_model(model)
    out.setup_entities()
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.write_pdb(str(dest))
    return dest


def find_tarball(base_url: str, prefix: str) -> str:
    """Resolve the newest tarball whose filename starts with ``prefix``.

    Lists the directory index and matches ``href="<prefix>...tar.gz"``. Raises
    if nothing matches (so a layout change fails loudly, not silently).
    """
    idx = http_get(base_url, timeout=60).text
    names = re.findall(r'href="(' + re.escape(prefix) + r'[^"]*\.tar\.gz)"', idx)
    if not names:
        raise RuntimeError(
            f"no tarball matching {prefix!r}*.tar.gz in {base_url} "
            f"(predictioncenter layout may have changed)"
        )
    return sorted(set(names))[-1]  # lexically-last == newest date stamp


def download_and_extract(base_url: str, name: str, raw_dir: Path) -> Path:
    """Download tarball ``name`` and extract it into ``raw_dir/<name stem>/``."""
    archive = download_to(base_url + name, raw_dir / name)
    dest = raw_dir / name.replace(".tar.gz", "")
    if not dest.exists():
        with tarfile.open(archive) as tf:
            try:
                tf.extractall(dest, filter="data")
            except TypeError:
                tf.extractall(dest)
    return dest


def index_pdbs(root: Path) -> dict[str, Path]:
    """Map ``<stem>`` -> path for every ``*.pdb`` under ``root`` (recursive)."""
    return {p.stem: p for p in root.rglob("*.pdb")}


def run(
    *,
    casps: list[str],
    categories: set[str],
    out_dir: Path,
    manifest_path: Path,
    raw_dir: Path,
    limit: int | None,
) -> list[ManifestRow]:
    fm = load_fm_domains(categories, casps)
    if limit is not None:
        fm = fm[:limit]
    print(
        f"{len(fm)} FM domains to fetch "
        f"(casps={casps}, categories={sorted(categories)})"
    )

    # Build a per-CASP lookup of available coordinate files.
    available: dict[str, dict[str, Path]] = {}
    for casp in casps:
        cfg = CASP_CONFIG[casp]
        files: dict[str, Path] = {}
        dom_name = find_tarball(cfg["base"], cfg["domain_prefix"])
        files.update(index_pdbs(download_and_extract(cfg["base"], dom_name, raw_dir / casp)))
        if cfg["target_prefix"]:
            tgt_name = find_tarball(cfg["base"], cfg["target_prefix"])
            # Full-target files only fill gaps; don't shadow domain files.
            for stem, path in index_pdbs(
                download_and_extract(cfg["base"], tgt_name, raw_dir / casp)
            ).items():
                files.setdefault(stem, path)
        available[casp] = files
        print(f"  {casp}: {len(files)} coordinate files available")

    pdb_fallback = load_pdb_fallback()
    cif_dir = raw_dir / "pdb_cif"

    # Temporal axis: resolve each target's answer PDB (targetlist code, or the
    # fallback map's explicit pdb_id) and date the domain by that structure's
    # RCSB deposit date. ``date_cache`` dedupes shared answer structures.
    target_pdb: dict[str, str] = {}
    for casp in casps:
        target_pdb.update(casp_target_pdb_map(casp))
    date_cache: dict[str, str] = {}

    rows: list[ManifestRow] = []
    n_unresolved = 0
    n_fallback = 0
    n_dated = 0
    for r in fm:
        casp, target, domain, category = r["casp"], r["target"], r["domain"], r["category"]
        files = available[casp]
        src = files.get(domain) or files.get(target)
        source = f"{casp.lower()}_fm"
        fb = pdb_fallback.get(domain)

        answer_pdb = (fb.get("pdb_id") if fb else "") or target_pdb.get(target, "")
        deposit_date = rcsb_deposit_date(answer_pdb, date_cache) if answer_pdb else ""
        if deposit_date:
            n_dated += 1

        # Not in the public monomer tarballs: try the deposited-PDB fallback.
        if src is None:
            if fb and fb.get("pdb_id") and fb.get("status") == "pdb_fallback":
                dest = out_dir / casp / f"{domain}.pdb"
                extract_domain_from_pdb(fb["pdb_id"], fb["chain"], fb["casp_range"], dest, cif_dir)
                n_fallback += 1
                print(f"  {domain} ({category}): clipped from PDB {fb['pdb_id']} chain {fb['chain']} {fb['casp_range']}")
                rows.append(
                    ManifestRow(
                        source=source, stem=domain, pdb_id=fb["pdb_id"],
                        chain=fb["chain"], length=count_residues(dest),
                        deposit_date=deposit_date,
                        category=category, novelty_axis=NOVELTY_AXIS,
                        local_path=str(dest.relative_to(HERE)),
                    )
                )
                continue
            n_unresolved += 1
            why = fb.get("note", "not in public tarballs") if fb else "not in public tarballs"
            print(f"  WARNING: {domain} ({category}) unresolved — {why}")
            rows.append(
                ManifestRow(
                    source=source, stem=domain, pdb_id=target,
                    deposit_date=deposit_date,
                    category=category, novelty_axis=NOVELTY_AXIS,
                )
            )
            continue

        dest = out_dir / casp / f"{domain}.pdb"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copyfile(src, dest)
        rows.append(
            ManifestRow(
                source=source,
                stem=domain,
                pdb_id=target,
                chain="",  # CASP domain coords are single-chain
                length=count_residues(dest),
                deposit_date=deposit_date,
                category=category,
                novelty_axis=NOVELTY_AXIS,
                local_path=str(dest.relative_to(HERE)),
            )
        )

    write_manifest(rows, manifest_path)
    n_ok = len(rows) - n_unresolved
    print(
        f"Wrote {len(rows)} rows to {manifest_path} "
        f"({n_ok} with structures [{n_fallback} via PDB fallback], "
        f"{n_unresolved} unresolved, {n_dated} dated); structures in {out_dir}/"
    )
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--casp", choices=("14", "15", "both"), default="both",
        help="Which CASP edition(s) to fetch (default: both).",
    )
    p.add_argument(
        "--categories", default="FM,FM/TBM",
        help="Comma-separated categories to keep (default: 'FM,FM/TBM').",
    )
    p.add_argument(
        "--out", type=Path, default=HERE / "structures" / "casp_fm",
        help="Where to write resolved domain PDBs (default: ./structures/casp_fm/).",
    )
    p.add_argument(
        "--manifest", type=Path, default=HERE / "data" / "casp_fm_manifest.csv",
        help="Manifest CSV path (default: ./data/casp_fm_manifest.csv).",
    )
    p.add_argument(
        "--raw-dir", type=Path, default=HERE / "casp_raw",
        help="Where to cache + extract the target tarballs (default: ./casp_raw/).",
    )
    p.add_argument("--limit", type=int, default=None, help="Take first N FM domains (smoke test).")
    args = p.parse_args()

    casps = ["CASP14", "CASP15"] if args.casp == "both" else [f"CASP{args.casp}"]
    categories = {c.strip() for c in args.categories.split(",") if c.strip()}

    run(
        casps=casps,
        categories=categories,
        out_dir=args.out,
        manifest_path=args.manifest,
        raw_dir=args.raw_dir,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
