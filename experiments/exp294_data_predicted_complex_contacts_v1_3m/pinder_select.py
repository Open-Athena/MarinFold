# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage E — select the PINDER heterodimer arm and resolve it to byte offsets.

AFCDB's heterodimers are exhausted at 183,809 (see README), so the 500,000
target needs an external source. PINDER supplies it: experimental PDB dimers,
Apache-2.0, and **nearly disjoint from AFCDB** — only 7.8% of its eligible
UniProt pairs overlap ours, because one source is predicted and the other
measured.

This arm is deliberately **per structure, not per sequence pair**. PINDER holds
~12 crystal forms of a typical pair, and those are genuinely different
conformations — the structural diversity the issue asks to preserve in the
corpus and control in sampling. That is the opposite of AFCDB, whose duplicate
sequence pairs are the same prediction under a different accession label (92%
agree on ipSAE to within 0.01), which is why the AFCDB arm stays deduplicated.

The mechanics are also far kinder than AFCDB's. ``pdbs.zip`` is 169 GB but
carries a real central directory, so a member is a byte offset rather than a
28,000-header walk, and the host is a CDN rather than EBI. The 449,835 eligible
heterodimers come to 35.3 GB.

    uv run python pinder_select.py \\
        --index /data/exp294_stageE/index.parquet \\
        --metadata /data/exp294_stageE/metadata.parquet \\
        --zip-index /data/exp294_stageE/zip_index.parquet \\
        --afcdb /data/exp294_predicted_complexes/selection/selected.parquet \\
        --out /data/exp294_stageE/selection
"""

import argparse
import json
from pathlib import Path
from typing import Any

import duckdb

ZIP_URL = "https://pinderdata.org/2024-02/pdbs.zip"
#: The contacts-v1 ring budget, identical to the AFCDB arm so documents from the
#: two sources are directly comparable.
DEFAULT_MAX_TOTAL_RESIDUES = 1998
#: PINDER's own quality verdict. ``invalid`` systems are ones it excludes from
#: every split; trusting that is cheaper and better founded than inventing a
#: second opinion.
EXCLUDED_SPLITS = ("invalid",)


def _sql_literal(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def run(
    index: Path,
    metadata: Path,
    zip_index: Path,
    afcdb: Path,
    out_dir: Path,
    *,
    max_total_residues: int = DEFAULT_MAX_TOTAL_RESIDUES,
) -> dict[str, Any]:
    """Write the PINDER heterodimer manifest plus a terminal-status ledger."""
    out_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    splits = ", ".join(_sql_literal(s) for s in EXCLUDED_SPLITS)

    con.execute(
        f"""
        CREATE OR REPLACE TABLE joined AS
        SELECT
            i.id, i.split, i.cluster_id, i.cluster_id_R, i.cluster_id_L,
            i.uniprot_R, i.uniprot_L, i.pdb_id,
            least(i.uniprot_R, i.uniprot_L) || '|' ||
                greatest(i.uniprot_R, i.uniprot_L) AS uniprot_pair,
            m.length1, m.length2, m.length1 + m.length2 AS total_residues,
            m.intermolecular_contacts, m.buried_sasa, m.resolution, m.method,
            m.release_date, m.complex_type,
            z.lho, z.csize, z.usize,
            CASE
                WHEN i.uniprot_R = i.uniprot_L THEN 'homodimer_not_this_arm'
                WHEN i.split IN ({splits}) THEN 'pinder_invalid_split'
                WHEN i.uniprot_R IN ('UNDEFINED', '') OR i.uniprot_L IN ('UNDEFINED', '')
                    THEN 'no_uniprot_mapping'
                WHEN m.length1 + m.length2 < 4
                     OR m.length1 + m.length2 > {int(max_total_residues)}
                    THEN 'does_not_fit_contacts_ring'
                WHEN coalesce(m.intermolecular_contacts, 0) < 1
                    THEN 'no_reported_interaction'
                WHEN z.name IS NULL THEN 'not_in_archive'
                ELSE NULL
            END AS rejection
        FROM read_parquet({_sql_literal(index)}) i
        JOIN read_parquet({_sql_literal(metadata)}) m USING (id)
        LEFT JOIN read_parquet({_sql_literal(zip_index)}) z
            ON z.name = 'pdbs/' || i.id || '.pdb'
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TABLE afcdb_pairs AS
        SELECT DISTINCT least(accession_a, accession_b) || '|' ||
               greatest(accession_a, accession_b) AS uniprot_pair
        FROM read_parquet({_sql_literal(afcdb)})
        WHERE complex_type = 'heterodimer'
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE selected AS
        SELECT j.*,
            (a.uniprot_pair IS NOT NULL) AS pair_also_in_afcdb,
            'pinder'::VARCHAR AS source_arm
        FROM joined j LEFT JOIN afcdb_pairs a USING (uniprot_pair)
        WHERE j.rejection IS NULL
        """
    )

    manifest = out_dir / "pinder_selected.parquet"
    con.execute(
        f"""
        COPY (
            SELECT id AS system_id, source_arm, split, cluster_id, cluster_id_R,
                   cluster_id_L, uniprot_R, uniprot_L, uniprot_pair, pdb_id,
                   length1, length2, total_residues, intermolecular_contacts,
                   buried_sasa, resolution, method, release_date,
                   pair_also_in_afcdb,
                   {_sql_literal(ZIP_URL)} AS zip_url,
                   'pdbs/' || id || '.pdb' AS member_name,
                   lho AS local_header_offset, csize AS compressed_bytes,
                   usize AS uncompressed_bytes
            FROM selected ORDER BY lho
        ) TO {_sql_literal(manifest)}
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
        """
    )
    ledger = out_dir / "pinder_ledger.parquet"
    con.execute(
        f"""
        COPY (
            SELECT id AS system_id,
                   CASE WHEN rejection IS NULL THEN 'selected' ELSE 'rejected' END AS status,
                   coalesce(rejection, 'selected') AS reason
            FROM joined ORDER BY id
        ) TO {_sql_literal(ledger)}
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
        """
    )

    rows = con.execute(
        "SELECT coalesce(rejection, 'selected'), count(*) FROM joined GROUP BY 1 ORDER BY 2 DESC"
    ).fetchall()
    totals = con.execute(
        """
        SELECT count(*), count(DISTINCT uniprot_pair), count(DISTINCT cluster_id),
               sum(csize), count(*) FILTER (WHERE NOT pair_also_in_afcdb),
               count(DISTINCT uniprot_pair) FILTER (WHERE NOT pair_also_in_afcdb),
               count(DISTINCT cluster_id) FILTER (WHERE NOT pair_also_in_afcdb)
        FROM selected
        """
    ).fetchone()
    stats = {
        "manifest": str(manifest.resolve()),
        "ledger": str(ledger.resolve()),
        "zip_url": ZIP_URL,
        "max_total_residues": max_total_residues,
        "source_rows": int(con.execute("SELECT count(*) FROM joined").fetchone()[0]),
        "selected_systems": int(totals[0]),
        "unique_uniprot_pairs": int(totals[1]),
        "interface_clusters": int(totals[2]),
        "download_bytes": int(totals[3] or 0),
        "download_gb": round((totals[3] or 0) / 1e9, 2),
        "systems_on_pairs_absent_from_afcdb": int(totals[4]),
        "pairs_absent_from_afcdb": int(totals[5]),
        "clusters_on_pairs_absent_from_afcdb": int(totals[6]),
        "ledger_breakdown": [{"reason": r, "rows": int(n)} for r, n in rows],
    }
    (out_dir / "pinder_selection.json").write_text(json.dumps(stats, indent=2) + "\n")
    if stats["selected_systems"] != sum(
        e["rows"] for e in stats["ledger_breakdown"] if e["reason"] == "selected"
    ):
        raise RuntimeError("manifest and ledger disagree on the selected count")
    return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--zip-index", type=Path, required=True)
    parser.add_argument("--afcdb", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--max-total-residues", type=int, default=DEFAULT_MAX_TOTAL_RESIDUES
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    stats = run(
        args.index, args.metadata, args.zip_index, args.afcdb, args.out,
        max_total_residues=args.max_total_residues,
    )
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
