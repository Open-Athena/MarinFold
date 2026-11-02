# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Step 1 -- cut FoldBench's 334 monomers into eval-val / eval-test / eval-denovo.

The three sets partition the monomer universe exactly, with no protein in two of
them and none left over:

``eval-val`` (97)
    The natural monomers inside the historical FoldBench-100 -- the slice every
    published contact number of ours has been computed on. It is *97*, not 100:
    three of exp12's first hundred rows are de novo designs and belong in
    ``eval-denovo``.
``eval-test`` (218)
    Every other natural monomer. Nothing here has ever been scored by any model
    or baseline in this repo, and #225 decontaminated the #232 training corpora
    against all of it, so it is a genuine held-out set for those checkpoints.
``eval-denovo`` (19)
    Every de novo designed monomer, wherever it sits in the file.

**Designed vs natural** is decided by two independent signals, following #241:
RCSB's ``synthetic construct`` source taxon (32630), and the PDB's own
``DE NOVO PROTEIN`` structural class in ``struct_keywords.pdbx_keywords``. On
these 334 the two agree on every protein -- 19 flagged by both, 315 by neither --
so the verdict does not depend on which one you trust. Where #241 also annotated
a protein, its verdict is asserted to match.

**Viral** comes from the NCBI taxonomy lineage of the entity's source organism,
not from a name match: a lineage containing ``Viruses`` is viral. #241 measured
that MarinFold ties ESMFold on viral eval proteins and loses by 0.145 on natural
non-viral ones, so every set carries the flag and every report splits on it.

    uv run python build_eval_sets.py            # RCSB sweep, ~20 s
    uv run python build_eval_sets.py --offline  # reuse data/rcsb_annotation.csv
"""
import argparse
import csv
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

import upstream as U

DATA = U.DATA
ANNOTATION = DATA / "rcsb_annotation.csv"
EVAL_SETS = DATA / "eval_sets.csv"
SUMMARY = DATA / "eval_sets.summary.json"

RCSB_GRAPHQL = "https://data.rcsb.org/graphql"
BATCH_SIZE = 50

#: Everything the designed verdict, the kingdom split and the provenance columns
#: need, in one entry-level query. Entity resolution (which polymer entity is
#: this monomer) is already settled in #226's ``foldbench_targets.csv``, so this
#: only has to find the matching entity id and read its annotations.
ENTRY_QUERY = """
query($ids: [String!]!) {
  entries(entry_ids: $ids) {
    rcsb_id
    struct { title }
    rcsb_accession_info { deposit_date initial_release_date }
    struct_keywords { pdbx_keywords text }
    polymer_entities {
      rcsb_id
      rcsb_polymer_entity_container_identifiers {
        reference_sequence_identifiers { database_accession database_name }
      }
      rcsb_entity_source_organism {
        ncbi_taxonomy_id
        ncbi_scientific_name
        taxonomy_lineage { id name }
      }
    }
  }
}
"""

#: NCBI taxon the PDB assigns an entity with no natural source organism.
SYNTHETIC_TAXID = "32630"
#: The PDB's own structural class for designed protein. Tested against both the
#: curated ``pdbx_keywords`` class and the depositor's free-text keyword list --
#: 8gac_A ("high affinity CTLA-4 binder") is classed ``PROTEIN BINDING`` and
#: says "De novo protein design" only in the free text, and #241 catches it for
#: exactly that reason.
DENOVO_KEYWORD = "DE NOVO PROTEIN"

#: Lineage buckets, tested in order. ``Viruses`` must come before the cellular
#: clades because a virus lineage contains none of them. Same table as #241's.
KINGDOMS = (
    ("synthetic", ("artificial sequences",)),
    ("virus", ("Viruses",)),
    ("archaea", ("Archaea",)),
    ("bacteria", ("Bacteria",)),
    ("eukaryote", ("Eukaryota",)),
)

#: Set names. ``eval-val`` is a validation set only in the sense that it is the
#: set we have been iterating against; nothing here fits anything to it.
SET_VAL = "eval-val"
SET_TEST = "eval-test"
SET_DENOVO = "eval-denovo"

#: Assertions, not inputs: a rebuild that disagrees with any of these has
#: changed the eval sets and must say so.
EXPECTED_MONOMERS = 334
EXPECTED_HISTORICAL = 100
EXPECTED_SIZES = {SET_VAL: 97, SET_TEST: 218, SET_DENOVO: 19}


def post(payload: dict, *, retries: int = 5) -> dict:
    """POST to RCSB's GraphQL API with backoff. Errors propagate after retries."""
    body = json.dumps(payload).encode()
    for attempt in range(retries):
        request = urllib.request.Request(
            RCSB_GRAPHQL, data=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                return json.loads(response.read().decode())
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def kingdom_of(lineage: list[dict]) -> str:
    """Bucket an NCBI lineage. ``unclassified`` when it matches no clade."""
    names = {node.get("name") for node in lineage if node}
    for label, markers in KINGDOMS:
        if any(marker in names for marker in markers):
            return label
    return "unclassified" if names else "unknown"


def annotate(targets: pd.DataFrame) -> pd.DataFrame:
    """One row per monomer: designed signals, kingdom, dates, provenance."""
    entry_ids = sorted({pdb.upper() for pdb in targets["pdb_id"]})
    entries: dict[str, dict] = {}
    for start in range(0, len(entry_ids), BATCH_SIZE):
        batch = entry_ids[start:start + BATCH_SIZE]
        payload = post({"query": ENTRY_QUERY, "variables": {"ids": batch}})
        if "errors" in payload:
            raise RuntimeError(f"RCSB GraphQL errors: {payload['errors']}")
        for entry in payload["data"]["entries"] or []:
            entries[entry["rcsb_id"].upper()] = entry
        print(f"  [rcsb] {min(start + BATCH_SIZE, len(entry_ids))}/{len(entry_ids)}",
              flush=True)

    missing = sorted(set(entry_ids) - set(entries))
    if missing:
        raise RuntimeError(f"RCSB returned no entry for: {missing}")

    rows = []
    for row in targets.itertuples():
        entry = entries[row.pdb_id.upper()]
        entity = next(
            (e for e in entry["polymer_entities"] or []
             if e["rcsb_id"].upper() == str(row.entity_id).upper()), None,
        )
        if entity is None:
            raise RuntimeError(f"{row.stem}: entity {row.entity_id} not in {entry['rcsb_id']}")
        sources = entity.get("rcsb_entity_source_organism") or []
        lineage = [node for source in sources
                   for node in (source.get("taxonomy_lineage") or [])]
        taxids = {str(source.get("ncbi_taxonomy_id")) for source in sources}
        struct_keywords = entry.get("struct_keywords") or {}
        keywords = struct_keywords.get("pdbx_keywords") or ""
        keywords_text = struct_keywords.get("text") or ""
        xrefs = [
            xref for e in [entity]
            for xref in
            ((e.get("rcsb_polymer_entity_container_identifiers") or {})
             .get("reference_sequence_identifiers") or [])
            if (xref or {}).get("database_name") == "UniProt"
        ]
        accession = entry.get("rcsb_accession_info") or {}
        rows.append({
            "stem": row.stem,
            "entry_id": entry["rcsb_id"],
            "entity_id": entity["rcsb_id"],
            "title": (entry.get("struct") or {}).get("title", ""),
            "pdbx_keywords": keywords,
            "struct_keywords_text": keywords_text,
            "source_organisms": ";".join(
                sorted({str(s.get("ncbi_scientific_name")) for s in sources})),
            "source_taxids": ";".join(sorted(taxids)),
            "kingdom": kingdom_of(lineage),
            "is_synthetic_taxon": int(SYNTHETIC_TAXID in taxids),
            "has_denovo_keyword": int(
                DENOVO_KEYWORD in keywords.upper()
                or DENOVO_KEYWORD in keywords_text.upper()
            ),
            "n_uniprot_xrefs": len(xrefs),
            "uniprot_accessions": ";".join(
                sorted({x["database_accession"] for x in xrefs})),
            "deposit_date": accession.get("deposit_date", ""),
            "initial_release_date": accession.get("initial_release_date", ""),
        })
    return pd.DataFrame(rows)


def cross_check_exp241(annotation: pd.DataFrame) -> dict:
    """Assert #241's independent annotation agrees where the two overlap."""
    path = U.exp241_annotation()
    if path is None:
        return {"available": False}
    other = pd.read_csv(path)
    # #241 annotated the 776-protein eval universe; a monomer can appear there
    # under two dataset labels with the same entry, so collapse on stem.
    other = other.drop_duplicates("stem").set_index("stem")
    shared = [s for s in annotation["stem"] if s in other.index]
    merged = annotation.set_index("stem").loc[shared]
    disagreements = []
    for column in ("kingdom", "is_synthetic_taxon", "has_denovo_keyword"):
        mismatched = merged.index[
            merged[column].astype(str).values != other.loc[shared, column].astype(str).values
        ]
        if len(mismatched):
            disagreements.append({"column": column, "stems": sorted(mismatched)})
    if disagreements:
        raise AssertionError(
            f"exp245's RCSB annotation disagrees with #241's: {disagreements}"
        )
    return {"available": True, "stems_compared": len(shared), "disagreements": 0}


def build_sets(targets: pd.DataFrame, annotation: pd.DataFrame) -> pd.DataFrame:
    """Join the annotation onto the monomer list and assign each protein a set."""
    # Three columns exist on both sides (#226 resolved the entity too). Assert
    # they agree, then keep one copy rather than carrying `_x`/`_y` suffixes.
    shared = ["entity_id", "title", "source_taxids"]
    joined = targets.set_index("stem")[shared].join(
        annotation.set_index("stem")[shared], rsuffix="_rcsb")
    for column in shared:
        left = joined[column].astype(str).str.upper()
        right = joined[f"{column}_rcsb"].astype(str).str.upper()
        if column == "source_taxids":
            # Both sides join multiple source entities with ";", and #226 keeps
            # duplicates where an entity lists the same organism twice, so this
            # compares the sets rather than the strings.
            disagree = [s for s, a, b in zip(joined.index, left, right)
                        if set(a.split(";")) != set(b.split(";"))]
        else:
            disagree = [s for s, a, b in zip(joined.index, left, right) if a != b]
        if disagree:
            raise AssertionError(f"{column} disagrees with #226 for {disagree[:10]}")

    frame = targets.drop(columns=shared).merge(
        annotation, on="stem", how="left", validate="one_to_one")
    if frame["kingdom"].isna().any():
        raise AssertionError("annotation is missing rows")

    # #226 resolved source organisms independently; the taxon flag must agree.
    if (frame["synthetic"].astype(int) != frame["is_synthetic_taxon"]).any():
        mismatch = frame.loc[
            frame["synthetic"].astype(int) != frame["is_synthetic_taxon"], "stem"].tolist()
        raise AssertionError(f"synthetic flag disagrees with #226 for {mismatch}")

    frame["designed"] = (
        (frame["is_synthetic_taxon"] == 1) | (frame["has_denovo_keyword"] == 1)
    ).astype(int)
    disagree = frame.loc[
        frame["is_synthetic_taxon"] != frame["has_denovo_keyword"], "stem"].tolist()
    if disagree:
        # Not fatal -- either signal alone is sufficient to call a design -- but
        # it is exactly the kind of thing that should never pass unremarked.
        print(f"[sets] designed signals disagree for {disagree}; union taken", flush=True)

    frame["foldbench_row"] = range(len(frame))
    frame["in_historical_100"] = (frame["foldbench_row"] < EXPECTED_HISTORICAL).astype(int)
    frame["is_viral"] = (frame["kingdom"] == "virus").astype(int)
    frame["eval_set"] = [
        SET_DENOVO if designed else (SET_VAL if historical else SET_TEST)
        for designed, historical in zip(frame["designed"], frame["in_historical_100"])
    ]
    return frame


def attach_training_identity(frame: pd.DataFrame) -> pd.DataFrame:
    """Carry #226's measured identity to the (contaminated) #199 training set.

    This is the *pre*-decontamination number, and it is what makes eval-val and
    eval-test interpretable for the #199 reference checkpoint: it is the identity
    the model that trained on the unfiltered corpora could have exploited. The
    #232 checkpoints' exposure is a separate, verified-zero quantity and lives in
    ``confirm_decontamination.py``.
    """
    identity = pd.read_csv(U.EXPANDED_IDENTITY)
    identity = identity[identity["dataset"].isin(("foldbench100", "foldbench_rest"))]
    identity = identity.set_index("stem")
    columns = {
        "best_identity_covered": "exp199_best_identity",
        "afdb_best_identity_covered": "exp199_afdb_best_identity",
        "esm_atlas_best_identity_covered": "exp199_esm_atlas_best_identity",
        "stratum": "exp199_stratum",
    }
    for source, name in columns.items():
        frame[name] = [
            identity[source].get(stem) if stem in identity.index else None
            for stem in frame["stem"]
        ]
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--offline", action="store_true",
                        help="reuse data/rcsb_annotation.csv instead of calling RCSB")
    args = parser.parse_args()

    targets = pd.read_csv(U.FOLDBENCH_TARGETS)
    if len(targets) != EXPECTED_MONOMERS:
        raise AssertionError(f"expected {EXPECTED_MONOMERS} monomers, got {len(targets)}")

    if args.offline:
        annotation = pd.read_csv(ANNOTATION)
    else:
        annotation = annotate(targets)
        DATA.mkdir(parents=True, exist_ok=True)
        annotation.to_csv(ANNOTATION, index=False)
        print(f"[sets] annotation -> {ANNOTATION}", flush=True)

    check = cross_check_exp241(annotation)
    print(f"[sets] #241 cross-check: {check}", flush=True)

    frame = attach_training_identity(build_sets(targets, annotation))
    sizes = frame["eval_set"].value_counts().to_dict()
    if sizes != EXPECTED_SIZES:
        raise AssertionError(f"set sizes {sizes} != expected {EXPECTED_SIZES}")

    keep = [
        "eval_set", "stem", "pdb_id", "chain_id", "entity_id", "entry_id",
        "foldbench_row", "in_historical_100", "designed", "is_viral", "kingdom",
        "seq_len", "sequence", "auth_asym_ids", "chain_match", "title",
        "pdbx_keywords", "struct_keywords_text", "source_organisms", "source_taxids", "n_uniprot_xrefs",
        "deposit_date", "initial_release_date", "exp199_best_identity",
        "exp199_afdb_best_identity", "exp199_esm_atlas_best_identity",
        "exp199_stratum",
    ]
    frame = frame[keep].sort_values(["eval_set", "foldbench_row"])
    frame.to_csv(EVAL_SETS, index=False)

    summary = {
        "n_monomers": len(frame),
        "sets": {
            name: {
                "n": int(len(group)),
                "n_viral": int(group["is_viral"].sum()),
                "n_nonviral": int(len(group) - group["is_viral"].sum()),
                "median_length": float(group["seq_len"].median()),
                "max_length": int(group["seq_len"].max()),
                "kingdoms": group["kingdom"].value_counts().to_dict(),
            }
            for name, group in frame.groupby("eval_set")
        },
        "exp241_cross_check": check,
        "foldbench_targets_sha256": U.sha256(U.FOLDBENCH_TARGETS),
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["sets"], indent=2))
    print(f"[sets] {len(frame)} monomers -> {EVAL_SETS}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
