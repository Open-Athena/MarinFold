# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""What each non-FoldBench eval protein actually *is*, and where it came from.

A stem like ``9deb_E`` says nothing about the molecule. This fetches the
deposited entry's title, the description of the specific polymer entity our
chain belongs to, its source organism, the experimental method and resolution,
and the release date — plus the link a reader should follow to check any of it.

Chain matching goes through ``auth_asym_ids``: RCSB's polymer entities carry
both label and auth chain identifiers and #226 was already bitten by the
difference, so the entity is chosen by the auth id the structure mapping
settled on, falling back to the first entity when an entry has only one.

The two CASP free-modeling domains with no released PDB entry get their CASP
target page instead, which is the primary source for those.

    uv run python dashboard/build_annotations.py
"""

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import upstream as U  # noqa: E402
from build_structures import structure_sources  # noqa: E402

RCSB_GRAPHQL = "https://data.rcsb.org/graphql"
RCSB_ENTRY = "https://www.rcsb.org/structure/{pdb_id}"
CASP_TARGET = "https://predictioncenter.org/{casp}/target.cgi?target={target}&view=all"

QUERY = """
query($ids: [String!]!) {
  entries(entry_ids: $ids) {
    rcsb_id
    struct { title }
    struct_keywords { pdbx_keywords }
    exptl { method }
    rcsb_entry_info { resolution_combined }
    rcsb_accession_info { initial_release_date }
    polymer_entities {
      rcsb_polymer_entity { pdbx_description }
      rcsb_polymer_entity_container_identifiers { auth_asym_ids }
      rcsb_entity_source_organism { ncbi_scientific_name }
    }
  }
}
"""


def fetch_entries(pdb_ids: list[str]) -> dict[str, dict]:
    """Query RCSB for every entry at once, with one retry."""

    payload = json.dumps({"query": QUERY, "variables": {"ids": pdb_ids}}).encode()
    request = urllib.request.Request(
        RCSB_GRAPHQL, data=payload, headers={"Content-Type": "application/json"}
    )
    for attempt in range(2):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body = json.loads(response.read())
            break
        except urllib.error.URLError:
            if attempt == 1:
                raise
            time.sleep(3)
    if "errors" in body:
        raise RuntimeError(f"RCSB GraphQL errors: {body['errors']}")
    return {entry["rcsb_id"].upper(): entry for entry in body["data"]["entries"]}


def entity_for_chain(entry: dict, chain: str | None) -> dict:
    """Pick the polymer entity our chain belongs to."""

    entities = entry.get("polymer_entities") or []
    if not entities:
        return {}
    for entity in entities:
        identifiers = entity.get("rcsb_polymer_entity_container_identifiers") or {}
        if chain and chain in (identifiers.get("auth_asym_ids") or []):
            return entity
    return entities[0]


def describe(entry: dict, chain: str | None) -> dict:
    """Flatten one entry into the fields the dashboard shows."""

    entity = entity_for_chain(entry, chain)
    organisms = entity.get("rcsb_entity_source_organism") or []
    resolutions = (entry.get("rcsb_entry_info") or {}).get("resolution_combined") or []
    methods = [m.get("method") for m in (entry.get("exptl") or []) if m.get("method")]
    return {
        "title": (entry.get("struct") or {}).get("title"),
        "molecule": (entity.get("rcsb_polymer_entity") or {}).get("pdbx_description"),
        "organism": organisms[0].get("ncbi_scientific_name") if organisms else None,
        "keywords": (entry.get("struct_keywords") or {}).get("pdbx_keywords"),
        "method": methods[0] if methods else None,
        "resolution": round(resolutions[0], 2) if resolutions else None,
        "released": ((entry.get("rcsb_accession_info") or {}).get("initial_release_date") or "")[:10]
        or None,
    }


#: Anything RCSB calls a de novo protein, or whose source organism is a
#: synthetic construct, is a design regardless of which dataset it arrived in.
#: #241 found the same failure in eval2-natural: a "natural" label inherited
#: from the collection rather than checked against the entry.
DESIGN_KEYWORDS = ("DE NOVO",)
SYNTHETIC_ORGANISMS = ("synthetic construct",)


def is_designed(annotation: dict) -> bool:
    """True when the deposited entry says this is a designed protein."""

    keywords = (annotation.get("keywords") or "").upper()
    organism = (annotation.get("organism") or "").lower()
    return any(k in keywords for k in DESIGN_KEYWORDS) or organism in SYNTHETIC_ORGANISMS


def all_sources() -> pd.DataFrame:
    """Every non-FoldBench natural eval protein, not just the shallow ones.

    The design contamination is a property of the collection, so it has to be
    measured over all 58 CAMEO-hard and CASP-FM targets — otherwise the tier
    table is corrected in one bin and wrong in the others.
    """

    exp65 = (
        U.EXPERIMENTS / "exp65_evals_low_msa_depth_proteins" / "data"
    )
    universe = pd.read_csv(U.DATA / "universe.csv")
    other = universe[universe.subset == "nonfoldbench_natural"][["dataset", "stem"]]
    cameo = pd.read_csv(exp65 / "cameo_hard_manifest.csv")[["stem", "pdb_id", "chain"]]
    casp = pd.read_csv(exp65 / "casp_fm_pdb_fallback.csv")
    casp = casp[casp.status == "pdb_fallback"].rename(columns={"domain": "stem"})
    casp = casp[["stem", "pdb_id", "chain", "casp_range"]]
    merged = other.merge(
        pd.concat([cameo, casp], ignore_index=True), on="stem", how="left"
    )
    if "casp_range" not in merged:
        merged["casp_range"] = None
    return merged


def main() -> None:
    sources = all_sources()
    casp = pd.read_csv(
        U.EXPERIMENTS
        / "exp65_evals_low_msa_depth_proteins"
        / "data/casp_fm_domains.csv"
    ).set_index("domain")

    with_entries = sources[sources.pdb_id.notna()]
    entries = fetch_entries(sorted({p.upper() for p in with_entries.pdb_id}))

    out: dict[str, dict] = {}
    for record in sources.itertuples(index=False):
        key = f"{record.dataset}__{record.stem}"
        if isinstance(record.pdb_id, str) and record.pdb_id:
            entry = entries.get(record.pdb_id.upper())
            if entry is None:
                raise KeyError(f"RCSB returned no entry for {record.pdb_id}")
            annotation = describe(
                entry, record.chain if isinstance(record.chain, str) else None
            )
            annotation["source_name"] = f"RCSB {record.pdb_id.upper()}"
            annotation["source_url"] = RCSB_ENTRY.format(pdb_id=record.pdb_id.lower())
        else:
            row = casp.loc[record.stem]
            annotation = {
                "title": f"{row.casp} free-modeling target {row.target}, domain "
                f"{record.stem.split('-')[-1]} ({row.category})",
                "molecule": None,
                "organism": None,
                "keywords": f"CASP {row.category} domain",
                "method": None,
                "resolution": None,
                "released": None,
                "source_name": f"{row.casp} target {row.target}",
                "source_url": CASP_TARGET.format(
                    casp=str(row.casp).lower(), target=row.target
                ),
            }
        # A CASP domain clipped out of a deposited entry keeps both links.
        if isinstance(record.casp_range, str) and record.casp_range:
            annotation["casp_range"] = record.casp_range
        annotation["designed"] = is_designed(annotation)
        out[key] = annotation
        flag = " [DESIGNED]" if annotation["designed"] else ""
        print(f"[annotate] {key}{flag}: {annotation['title']}", flush=True)

    destination = U.DATA / "nonfoldbench_annotations.json"
    destination.write_text(json.dumps(out, indent=1, sort_keys=True))
    print(
        json.dumps(
            {
                "proteins": len(out),
                "with_titles": sum(1 for a in out.values() if a["title"]),
                "with_molecule": sum(1 for a in out.values() if a["molecule"]),
                "with_organism": sum(1 for a in out.values() if a["organism"]),
                "designed": sorted(k for k, a in out.items() if a["designed"]),
                "out": str(destination),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
