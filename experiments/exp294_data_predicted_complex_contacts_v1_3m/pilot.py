# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage B — a deterministic, stratified ~50k pilot drawn from the selected frontier.

The pilot exists to set the production Tier-B quality floor, which the metadata
census cannot do on its own: the census can say how many candidates sit above a
floor, but only generated documents say whether those candidates are usable.
So the sample is deliberately **not** proportional to the selected mixture. It
over-samples the bins the decision actually turns on — the low-confidence tail
that a 0.30 floor admits and a 0.50 floor would not — because a proportional
draw would spend most of its 50k on Tier A, which nobody is arguing about.

Strata are the four axes the issue names: confidence bin, homo/heterodimer,
length band, and taxon breadth. Within a stratum, rows are ordered by a hash of
``model_id`` so the draw is reproducible from the manifest alone, with no stored
random seed and no dependence on row order. The same ``model_id`` is drawn every
time the manifest is regenerated with the same contents.

Allocation is equalised across non-empty strata and then redistributed: a
stratum smaller than its share contributes its shortfall to the others, so a
thin stratum never silently shrinks the pilot below the requested size.

It also emits a second, differently-shaped sample: a **throughput probe**.
The stratified draw answers "are these documents any good", and for that it
deliberately spreads across strata -- which spreads it across ~12,800 of the
~17,800 source tars at about 4 models per tar. Extraction cost is not per
model, though: walking a tar's headers costs the same whether you take 4
members from it or 400. Timing the stratified draw would therefore overestimate
the production cost per document by roughly the ratio of those densities. The
probe instead takes *every* selected model from a handful of whole tars, which
is the density the real run sees, so the two samples answer the two questions
without either contaminating the other.

    uv run python pilot.py --selected /data/exp294/selection/selected.parquet \\
        --out /data/exp294/pilot --size 50000 --throughput-tars 20
"""

import argparse
import json
from pathlib import Path
from typing import Any

import duckdb

DEFAULT_PILOT_SIZE = 50_000
#: Whole tars taken for the throughput probe. Enough to average over tar-size
#: and member-size variation without being a meaningful fraction of the run.
DEFAULT_THROUGHPUT_TARS = 20
#: Confidence bins. The boundaries are the decision points: 1.0 is the nominal
#: Tier-A gate, 0.5 was pilot-v0's floor, 0.3 is the census-derived floor.
QUALITY_BINS = ((1.0, "A_ge_1.0"), (0.5, "B_0.5_1.0"), (0.3, "B_0.3_0.5"))
#: Residue bands. 1998 is the contacts-v1 ring budget.
LENGTH_BINS = ((400, "len_lt_400"), (800, "len_400_800"), (1400, "len_800_1400"))


def _sql_literal(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _quality_bin_sql(column: str = "quality_ratio") -> str:
    clauses = " ".join(
        f"WHEN {column} >= {edge} THEN '{label}'" for edge, label in QUALITY_BINS
    )
    return f"CASE {clauses} ELSE 'B_lt_0.3' END"


def _length_bin_sql(column: str = "total_residues") -> str:
    clauses = " ".join(
        f"WHEN {column} < {edge} THEN '{label}'" for edge, label in LENGTH_BINS
    )
    return f"CASE {clauses} ELSE 'len_ge_1400' END"


def _prepare(con: duckdb.DuckDBPyConnection, selected: Path) -> None:
    """Label every selected row with its stratum and a deterministic rank."""
    con.execute(
        f"""
        CREATE OR REPLACE TABLE strata AS
        SELECT *,
            {_quality_bin_sql()} AS quality_bin,
            {_length_bin_sql()} AS length_bin,
            complex_type || '|' || {_quality_bin_sql()} || '|' || {_length_bin_sql()}
                AS stratum
        FROM read_parquet({_sql_literal(selected)})
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE ranked AS
        SELECT *,
            row_number() OVER (
                PARTITION BY stratum
                -- Hash of the key, not random(): reproducible from the
                -- manifest alone, and independent of row order. The key is
                -- (complex_type, model_id) because AFCDB reuses modelEntityId
                -- across its two source tables.
                ORDER BY hash(source_model_key), source_model_key
            ) AS stratum_rank,
            count(*) OVER (PARTITION BY stratum) AS stratum_size,
            -- Taxon rank within the stratum spreads the draw across organisms
            -- rather than letting one proteome dominate a thin stratum.
            row_number() OVER (
                PARTITION BY stratum, tax_id_a
                ORDER BY hash(source_model_key), source_model_key
            ) AS taxon_rank
        FROM strata
        """
    )


def _allocate(sizes: dict[str, int], size: int) -> dict[str, int]:
    """Equal shares, with each thin stratum's shortfall spread over the rest."""
    quotas = {name: 0 for name in sizes}
    remaining = size
    open_strata = {name for name, n in sizes.items() if n > 0}
    while remaining > 0 and open_strata:
        share = max(1, remaining // len(open_strata))
        progressed = False
        for name in sorted(open_strata):
            if remaining <= 0:
                break
            take = min(share, sizes[name] - quotas[name], remaining)
            if take <= 0:
                continue
            quotas[name] += take
            remaining -= take
            progressed = True
        open_strata = {n for n in open_strata if quotas[n] < sizes[n]}
        if not progressed:
            break
    return quotas


def _throughput_probe(
    con: duckdb.DuckDBPyConnection, out_dir: Path, tars: int
) -> dict[str, Any]:
    """Every selected model from ``tars`` whole source tars, at run density."""
    con.execute(
        f"""
        CREATE OR REPLACE TABLE probe_tars AS
        SELECT source_tar_uri FROM (
            SELECT source_tar_uri, count(*) AS models
            FROM strata GROUP BY source_tar_uri
        )
        -- Deterministic, and spread over the tar-size distribution rather than
        -- clustered at one end of it.
        ORDER BY hash(source_tar_uri), source_tar_uri
        LIMIT {int(tars)}
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE probe AS
        SELECT s.* FROM strata s JOIN probe_tars USING (source_tar_uri)
        """
    )
    path = out_dir / "throughput_probe.parquet"
    con.execute(
        f"""
        COPY (SELECT * FROM probe ORDER BY source_tar_uri, source_model_key)
        TO {_sql_literal(path)}
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
        """
    )
    models, uris, residues = con.execute(
        "SELECT count(*), count(DISTINCT source_tar_uri), sum(total_residues) FROM probe"
    ).fetchone()
    return {
        "throughput_probe": str(path.resolve()),
        "probe_tars": int(uris),
        "probe_models": int(models),
        "probe_models_per_tar": round(int(models) / max(int(uris), 1), 1),
        "probe_total_residues": int(residues or 0),
    }


def build(
    selected: Path,
    out_dir: Path,
    *,
    size: int = DEFAULT_PILOT_SIZE,
    throughput_tars: int = DEFAULT_THROUGHPUT_TARS,
) -> dict[str, Any]:
    """Write the pilot manifest, the throughput probe, and stratum accounting."""
    out_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    _prepare(con, selected)

    sizes = {
        name: int(n)
        for name, n in con.execute(
            "SELECT stratum, count(*) FROM ranked GROUP BY stratum"
        ).fetchall()
    }
    if not sizes:
        raise ValueError(f"{selected}: no rows to sample")
    quotas = _allocate(sizes, size)
    drawn = sum(quotas.values())

    values = ",".join(
        f"({_sql_literal(name)}, {quota})" for name, quota in sorted(quotas.items())
    )
    con.execute(
        f"""
        CREATE OR REPLACE TABLE pilot AS
        SELECT r.* FROM ranked r
        JOIN (VALUES {values}) AS q(stratum, quota) USING (stratum)
        WHERE r.stratum_rank <= q.quota
        """
    )
    manifest = out_dir / "pilot_manifest.parquet"
    con.execute(
        f"""
        COPY (SELECT * FROM pilot ORDER BY source_tar_uri, source_model_key)
        TO {_sql_literal(manifest)}
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
        """
    )

    breakdown = [
        {
            "stratum": name,
            "complex_type": name.split("|")[0],
            "quality_bin": name.split("|")[1],
            "length_bin": name.split("|")[2],
            "available": sizes[name],
            "drawn": quotas[name],
        }
        for name in sorted(sizes)
    ]
    tars, taxa, residues = con.execute(
        "SELECT count(DISTINCT source_tar_uri), count(DISTINCT tax_id_a), "
        "sum(total_residues) FROM pilot"
    ).fetchone()
    probe = _throughput_probe(con, out_dir, throughput_tars) if throughput_tars else {}
    corpus_tars, corpus_models = con.execute(
        "SELECT count(DISTINCT source_tar_uri), count(*) FROM strata"
    ).fetchone()
    stats = {
        "selected_manifest": str(selected.resolve()),
        "corpus_models": int(corpus_models),
        "corpus_tars": int(corpus_tars),
        "corpus_models_per_tar": round(int(corpus_models) / max(int(corpus_tars), 1), 1),
        "pilot_manifest": str(manifest.resolve()),
        "requested": size,
        "drawn": int(drawn),
        "strata": len(sizes),
        "distinct_source_tars": int(tars),
        "distinct_taxa": int(taxa),
        "total_residues": int(residues or 0),
        "pilot_models_per_tar": round(int(drawn) / max(int(tars), 1), 1),
        **probe,
        "breakdown": breakdown,
    }
    (out_dir / "pilot.json").write_text(json.dumps(stats, indent=2) + "\n")
    if drawn < size:
        raise RuntimeError(
            f"pilot drew {drawn:,} of the requested {size:,}; the selected "
            f"frontier has only {sum(sizes.values()):,} rows. Wrote {out_dir} "
            "for inspection"
        )
    return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--size", type=int, default=DEFAULT_PILOT_SIZE)
    parser.add_argument(
        "--throughput-tars", type=int, default=DEFAULT_THROUGHPUT_TARS
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(
        json.dumps(
            build(
                args.selected,
                args.out,
                size=args.size,
                throughput_tars=args.throughput_tars,
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
