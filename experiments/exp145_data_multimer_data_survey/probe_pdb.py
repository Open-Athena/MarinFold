# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Census of multimeric content in the PDB via the RCSB Search API.

Every number in the report's PDB section comes from this script. It only
issues faceted *count* queries (``paginate.rows = 0``) against
``search.rcsb.org``, so it downloads kilobytes, not the ~250 GB the full
archive would cost.

Key attributes used (see https://search.rcsb.org/rcsbsearch/v2/metadata/schema):

- ``rcsb_assembly_info.polymer_entity_instance_count_protein`` — number of
  protein *chains* in the biological assembly (copies included).
- ``rcsb_assembly_info.polymer_entity_count_protein`` — number of *distinct*
  protein entities in the assembly. 1 → homomeric, >= 2 → heteromeric.
- ``rcsb_entry_info.structure_determination_methodology`` — ``experimental``
  vs ``computational``; RCSB also indexes ~1M computed structure models
  (AFDB + ModelArchive) which must be excluded from any "the PDB contains"
  claim.

Usage::

    python probe_pdb.py --out data
"""

import argparse
import csv
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"

# Restrict to experimentally determined entries. Without this the counts fold in
# RCSB's computed-structure-model mirror of AFDB/ModelArchive.
EXPERIMENTAL: dict[str, Any] = {
    "type": "terminal",
    "service": "text",
    "parameters": {
        "attribute": "rcsb_entry_info.structure_determination_methodology",
        "operator": "exact_match",
        "value": "experimental",
    },
}


def _and(*clauses: dict[str, Any]) -> dict[str, Any]:
    kept = [c for c in clauses if c]
    if len(kept) == 1:
        return kept[0]
    return {"type": "group", "logical_operator": "and", "nodes": kept}


def gte(attribute: str, value: float) -> dict[str, Any]:
    return {
        "type": "terminal",
        "service": "text",
        "parameters": {"attribute": attribute, "operator": "greater_or_equal", "value": value},
    }


def exact(attribute: str, value: str) -> dict[str, Any]:
    return {
        "type": "terminal",
        "service": "text",
        "parameters": {"attribute": attribute, "operator": "exact_match", "value": value},
    }


def run(
    query: dict[str, Any],
    return_type: str,
    facets: list[dict[str, Any]] | None = None,
    content_type: list[str] | None = None,
) -> dict[str, Any]:
    """POST a search and return the parsed response.

    ``content_type`` defaults to experimental-only on RCSB's side; pass
    ``["computational"]`` to reach the computed-structure-model mirror.
    Retries on transient 5xx / rate limiting; anything else propagates.
    """
    body: dict[str, Any] = {
        "query": query,
        "return_type": return_type,
        "request_options": {"paginate": {"start": 0, "rows": 0}},
    }
    if facets:
        body["request_options"]["facets"] = facets
    if content_type:
        body["request_options"]["results_content_type"] = content_type
    payload = json.dumps(body).encode()
    req = urllib.request.Request(
        SEARCH_URL, data=payload, headers={"Content-Type": "application/json"}
    )
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                raw = resp.read()
            # RCSB answers a zero-hit query with 204 No Content and an empty body.
            return json.loads(raw) if raw.strip() else {"total_count": 0}
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 500, 502, 503, 504) and attempt < 4:
                time.sleep(2**attempt)
                continue
            raise
    raise RuntimeError("unreachable")


def count(query: dict[str, Any], return_type: str, content_type: list[str] | None = None) -> int:
    return int(run(query, return_type, content_type=content_type).get("total_count", 0))


def buckets(resp: dict[str, Any], name: str) -> list[tuple[str, int]]:
    for facet in resp.get("facets", []):
        if facet["name"] == name:
            return [(str(b["label"]), int(b["population"])) for b in facet["buckets"]]
    raise KeyError(f"facet {name!r} not in response")


def write_csv(path: Path, header: list[str], rows: list[tuple]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def chain_count_histogram(out: Path) -> None:
    """Assemblies by number of protein chains in the biological assembly."""
    resp = run(
        _and(EXPERIMENTAL, gte("rcsb_assembly_info.polymer_entity_instance_count_protein", 1)),
        "assembly",
        [
            {
                "name": "chains",
                "aggregation_type": "histogram",
                "attribute": "rcsb_assembly_info.polymer_entity_instance_count_protein",
                "interval": 1,
                "min_interval_population": 1,
            }
        ],
    )
    rows = sorted(((int(float(label)), n) for label, n in buckets(resp, "chains")))
    write_csv(out / "pdb_assembly_protein_chain_count.csv", ["n_protein_chains", "n_assemblies"], rows)


def homo_vs_hetero(out: Path) -> None:
    """Split multi-chain assemblies into homomeric and heteromeric."""
    rows: list[tuple] = []
    multi = gte("rcsb_assembly_info.polymer_entity_instance_count_protein", 2)
    for label, clause in [
        ("all_protein_assemblies", gte("rcsb_assembly_info.polymer_entity_instance_count_protein", 1)),
        ("single_chain", exact_int("rcsb_assembly_info.polymer_entity_instance_count_protein", 1)),
        ("multi_chain", multi),
        (
            "multi_chain_homomeric",
            _and(multi, exact_int("rcsb_assembly_info.polymer_entity_count_protein", 1)),
        ),
        (
            "multi_chain_heteromeric",
            _and(multi, gte("rcsb_assembly_info.polymer_entity_count_protein", 2)),
        ),
        (
            "multi_chain_with_nucleic_acid",
            _and(multi, gte("rcsb_assembly_info.polymer_entity_instance_count_nucleic_acid", 1)),
        ),
    ]:
        rows.append((label, count(_and(EXPERIMENTAL, clause), "assembly")))
    write_csv(out / "pdb_assembly_homo_hetero.csv", ["category", "n_assemblies"], rows)


def exact_int(attribute: str, value: int) -> dict[str, Any]:
    return {
        "type": "terminal",
        "service": "text",
        "parameters": {"attribute": attribute, "operator": "equals", "value": value},
    }


def symmetry(out: Path) -> None:
    """Point-group symmetry of multi-chain assemblies (global symmetry only)."""
    resp = run(
        _and(EXPERIMENTAL, gte("rcsb_assembly_info.polymer_entity_instance_count_protein", 2)),
        "assembly",
        [
            {
                "name": "symbol",
                "aggregation_type": "terms",
                "attribute": "rcsb_struct_symmetry.symbol",
                "max_num_intervals": 40,
            }
        ],
    )
    rows = sorted(buckets(resp, "symbol"), key=lambda r: -r[1])
    write_csv(out / "pdb_assembly_symmetry.csv", ["symmetry_symbol", "n_assemblies"], rows)


def by_year(out: Path) -> None:
    """Entries released per year, total vs. those with a multimeric assembly."""
    def year_counts(extra: dict[str, Any] | None) -> dict[str, int]:
        resp = run(
            _and(EXPERIMENTAL, gte("rcsb_assembly_info.polymer_entity_instance_count_protein", 1), extra),
            "entry",
            [
                {
                    "name": "year",
                    "aggregation_type": "date_histogram",
                    "attribute": "rcsb_accession_info.initial_release_date",
                    "interval": "year",
                    "min_interval_population": 1,
                }
            ],
        )
        return {label[:4]: n for label, n in buckets(resp, "year")}

    total = year_counts(None)
    multi = year_counts(gte("rcsb_assembly_info.polymer_entity_instance_count_protein", 2))
    hetero = year_counts(
        _and(
            gte("rcsb_assembly_info.polymer_entity_instance_count_protein", 2),
            gte("rcsb_assembly_info.polymer_entity_count_protein", 2),
        )
    )
    years = sorted(set(total) | set(multi))
    rows = [
        (y, total.get(y, 0), multi.get(y, 0), hetero.get(y, 0))
        for y in years
    ]
    write_csv(
        out / "pdb_entries_by_year.csv",
        ["year", "n_entries_protein", "n_entries_multichain", "n_entries_heteromeric"],
        rows,
    )


def by_method(out: Path) -> None:
    """Experimental method, for all protein entries and for multimeric ones."""
    def method_counts(extra: dict[str, Any] | None) -> dict[str, int]:
        resp = run(
            _and(EXPERIMENTAL, gte("rcsb_assembly_info.polymer_entity_instance_count_protein", 1), extra),
            "entry",
            [
                {
                    "name": "method",
                    "aggregation_type": "terms",
                    "attribute": "rcsb_entry_info.experimental_method",
                    "max_num_intervals": 20,
                }
            ],
        )
        return dict(buckets(resp, "method"))

    total = method_counts(None)
    multi = method_counts(gte("rcsb_assembly_info.polymer_entity_instance_count_protein", 2))
    big = method_counts(gte("rcsb_assembly_info.polymer_entity_instance_count_protein", 6))
    rows = [
        (m, total.get(m, 0), multi.get(m, 0), big.get(m, 0))
        for m in sorted(total, key=lambda k: -total[k])
    ]
    write_csv(
        out / "pdb_entries_by_method.csv",
        ["method", "n_entries_protein", "n_entries_multichain", "n_entries_ge6_chains"],
        rows,
    )


def redundancy(out: Path) -> None:
    """Distinct protein entities vs. distinct sequence clusters.

    Gives the deduplication factor: how much smaller a non-redundant complex
    corpus is than the raw entry count.
    """
    rows: list[tuple] = []
    protein = exact("rcsb_entry_info.selected_polymer_entity_types", "Protein (only)")
    for label, query, rtype in [
        ("protein_entities_total", _and(EXPERIMENTAL, gte("rcsb_assembly_info.polymer_entity_instance_count_protein", 1)), "polymer_entity"),
        ("protein_entities_in_multimers", _and(EXPERIMENTAL, gte("rcsb_assembly_info.polymer_entity_instance_count_protein", 2)), "polymer_entity"),
        ("entries_protein_only", _and(EXPERIMENTAL, protein), "entry"),
    ]:
        rows.append((label, count(query, rtype)))
    write_csv(out / "pdb_entity_counts.csv", ["category", "n"], rows)

    # Sequence-cluster counts at each RCSB identity cutoff, over entities that
    # participate in a multi-chain assembly. ``group_by`` returns the number of
    # groups in ``group_by_count``, which is the deduplicated entity count.
    cluster_rows: list[tuple] = []
    # 40 is not an accepted ``sequence_identity`` cutoff for group_by, unlike the
    # older downloadable cluster files.
    for cutoff in (30, 50, 70, 90, 95, 100):
        body = {
            "query": _and(
                EXPERIMENTAL, gte("rcsb_assembly_info.polymer_entity_instance_count_protein", 2)
            ),
            "return_type": "polymer_entity",
            "request_options": {
                "paginate": {"start": 0, "rows": 0},
                "group_by": {
                    "aggregation_method": "sequence_identity",
                    "similarity_cutoff": cutoff,
                },
                "group_by_return_type": "representatives",
            },
        }
        req = urllib.request.Request(
            SEARCH_URL, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=180) as resp:
            payload = json.loads(resp.read())
        cluster_rows.append((cutoff, int(payload["total_count"]), int(payload["group_by_count"])))
    write_csv(
        out / "pdb_multimer_sequence_clusters.csv",
        ["identity_cutoff_pct", "n_entities", "n_distinct_clusters"],
        cluster_rows,
    )


def computed_models(out: Path) -> None:
    """How many of RCSB's computed structure models are multimeric.

    RCSB mirrors AFDB and ModelArchive under ``structure_determination_methodology
    = computational``. If any of those are multi-chain they are a bulk source of
    predicted complexes.
    """
    computational = exact("rcsb_entry_info.structure_determination_methodology", "computational")
    csm = ["computational"]
    rows = [
        ("computed_models_total", count(computational, "entry", csm)),
        (
            "computed_models_multichain",
            count(
                _and(computational, gte("rcsb_assembly_info.polymer_entity_instance_count_protein", 2)),
                "entry",
                csm,
            ),
        ),
        (
            "computed_models_heteromeric",
            count(
                _and(
                    computational,
                    gte("rcsb_assembly_info.polymer_entity_instance_count_protein", 2),
                    gte("rcsb_assembly_info.polymer_entity_count_protein", 2),
                ),
                "entry",
                csm,
            ),
        ),
    ]
    write_csv(out / "pdb_computed_models.csv", ["category", "n_entries"], rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data"))
    args = parser.parse_args()

    chain_count_histogram(args.out)
    homo_vs_hetero(args.out)
    symmetry(args.out)
    by_year(args.out)
    by_method(args.out)
    redundancy(args.out)
    computed_models(args.out)


if __name__ == "__main__":
    main()
