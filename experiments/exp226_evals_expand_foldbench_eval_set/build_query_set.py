# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Step 1 — assemble the 222 net-new FoldBench monomers as MMseqs2 queries.

Our contact eval set uses **100** of FoldBench's 334 monomers (exp12 took the
first 100 rows of ``targets/monomer_protein.csv``). This step recovers the other
234, subtracts the 12 that are already in the eval set under a different dataset
label, and writes the remaining **222** as a query FASTA that
:mod:`search_expanded` appends to exp213's 554.

Everything here is derived and then checked against the issue's stated numbers,
rather than hard-coded from them — the constants below are assertions, not
inputs. A rebuild that disagrees with any of them fails loudly.

**Sequences** come from RCSB's GraphQL data API, one canonical
``entity_poly.pdbx_seq_one_letter_code_can`` string per polymer entity — the
same mmCIF field exp12 pulled out of the assembly CIF for the original 100.

**The chain gotcha:** FoldBench's ``chain_id`` is sometimes the mmCIF *label*
asym id rather than the *auth* chain (``8ork_A`` is auth ``AAA``; ``5sbj_A``,
already one of our 100, is auth ``C``). The API returns both id lists per
entity, so :func:`select_entity` matches auth first and falls back to label,
with no string parsing of FASTA headers and no silent mismatch.

The method is validated the strongest way available: all 100 sequences we
already use are re-fetched through this exact path and compared byte-for-byte
against exp213's committed query FASTA.

    uv run python build_query_set.py
    uv run python build_query_set.py --offline   # reuse data/foldbench_targets.csv
"""
import argparse
import csv
import hashlib
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from exp213_link import check_exp213_queries, read_exp213_queries

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

# --- the pinned FoldBench target list ---------------------------------------

#: exp12's pinned FoldBench commit; the eval set's 100 monomers are the first
#: 100 rows of this revision's ``targets/monomer_protein.csv``.
FOLDBENCH_COMMIT = "4273f6877d82bd0b2fa476d1b2f34d121cbccc70"
FOLDBENCH_CSV_URL = (
    "https://raw.githubusercontent.com/BEAM-Labs/FoldBench/"
    f"{FOLDBENCH_COMMIT}/targets/monomer_protein.csv"
)
FOLDBENCH_CSV_SHA256 = (
    "43c2a5e9a73e84e00afb8d0108761547a8f9d6e52865e122792748c9c32bf595"
)
N_FOLDBENCH_MONOMERS = 334

#: The dataset label exp12's 100 carry in the eval manifests, and the label the
#: net-new ones get here.
DATASET_USED = "foldbench100"
DATASET_NEW = "foldbench_rest"

#: FoldBench monomers that are *already* in the 554 eval set under exp65's de
#: novo PDB dataset. They must not be double-counted as new proteins. Derived
#: from the eval set at run time; listed here only to assert the derivation.
EXPECTED_ALREADY_IN_EVAL = frozenset({
    "8eov_A", "8fjf_A", "8ju8_A", "8k7o_A", "8k7z_A", "8ka7_A",
    "8kac_A", "8kcj_A", "8oys_A", "8qkd_A", "8qup_A", "8vc8_A",
})

#: Residue-count checksums for a faithful rebuild (issue #226).
EXPECTED_N_NEW = 234              # FoldBench monomers we have never used
EXPECTED_AA_NEW = 66_692
EXPECTED_N_NET_NEW = 222          # ... minus the 12 already in the eval set
EXPECTED_AA_NET_NEW = 64_624

# --- RCSB ------------------------------------------------------------------

RCSB_GRAPHQL = "https://data.rcsb.org/graphql"
#: One request per batch of entries; RCSB's API handles this size comfortably
#: and 334 entries then cost 7 round trips instead of 334.
BATCH_SIZE = 50
ENTRY_QUERY = """
query($ids: [String!]!) {
  entries(entry_ids: $ids) {
    rcsb_id
    struct { title }
    polymer_entities {
      rcsb_id
      entity_poly { type pdbx_seq_one_letter_code_can }
      rcsb_polymer_entity_container_identifiers { auth_asym_ids asym_ids }
      rcsb_entity_source_organism { ncbi_taxonomy_id ncbi_scientific_name }
      rcsb_polymer_entity { pdbx_description }
    }
  }
}
"""

#: NCBI taxon for "synthetic construct" — what the PDB assigns an entity with no
#: natural source organism, including every de novo designed protein.
#:
#: exp213 splits designed from natural on the *dataset* label (`denovo_pdb`),
#: which is right for the 554 but says nothing about FoldBench monomers. It has
#: to say something: 12 of the 234 unused monomers are themselves in exp65's de
#: novo set, so FoldBench demonstrably contains designed protein. This taxon is
#: the available proxy, and it is a deliberately *conservative* one — it also
#: catches engineered variants of natural proteins, so it over-flags rather than
#: under-flags, making the natural survivor count a lower bound. Validated in
#: :func:`main` against those 12 known designs.
SYNTHETIC_TAXID = "32630"


@dataclass(frozen=True)
class MonomerTarget:
    """One row of FoldBench's ``monomer_protein.csv``."""

    pdb_id: str    # lowercase 4-character entry id, e.g. "5sbj"
    chain_id: str  # FoldBench's chain, which may be an auth OR a label asym id

    @property
    def stem(self) -> str:
        return f"{self.pdb_id}_{self.chain_id}"


@dataclass(frozen=True)
class ResolvedTarget:
    """A :class:`MonomerTarget` joined to the RCSB entity it names."""

    target: MonomerTarget
    entity_id: str
    sequence: str
    auth_asym_ids: tuple[str, ...]
    asym_ids: tuple[str, ...]
    chain_match: str  # "auth" or "label" — which id list FoldBench's chain hit
    source_taxids: tuple[str, ...]
    source_names: tuple[str, ...]
    title: str
    description: str

    @property
    def synthetic(self) -> bool:
        """True when the entity has no natural source organism.

        See :data:`SYNTHETIC_TAXID`. An entity with an empty source list is
        counted as synthetic too — the PDB leaves it empty for designs that
        predate the ``synthetic construct`` convention.
        """
        return not self.source_taxids or all(
            t == SYNTHETIC_TAXID for t in self.source_taxids
        )


def fetch_foldbench_csv(cache: Path) -> str:
    """Download the pinned monomer list, verifying its sha256.

    Cached on disk because it is an immutable input pinned to a commit; a
    cached copy whose digest disagrees is deleted rather than trusted.
    """
    if cache.exists():
        text = cache.read_text()
        if hashlib.sha256(text.encode()).hexdigest() == FOLDBENCH_CSV_SHA256:
            return text
        print(f"[foldbench] cached {cache} has the wrong digest; re-downloading",
              flush=True)
    with urllib.request.urlopen(FOLDBENCH_CSV_URL, timeout=60) as fh:
        raw = fh.read()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != FOLDBENCH_CSV_SHA256:
        raise SystemExit(
            f"{FOLDBENCH_CSV_URL} has sha256 {digest}, expected "
            f"{FOLDBENCH_CSV_SHA256}. The pinned commit's file changed under us."
        )
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(raw)
    return raw.decode()


def parse_targets(text: str) -> list[MonomerTarget]:
    """``pdb_id,chain_id`` rows -> targets. ``pdb_id`` is ``<pdb>-assembly<n>``."""
    targets: list[MonomerTarget] = []
    for row in csv.DictReader(text.splitlines()):
        pdb_id = row["pdb_id"].strip().split("-")[0].lower()
        targets.append(MonomerTarget(pdb_id=pdb_id, chain_id=row["chain_id"].strip()))
    if len(targets) != N_FOLDBENCH_MONOMERS:
        raise SystemExit(
            f"expected {N_FOLDBENCH_MONOMERS} FoldBench monomers, parsed {len(targets)}"
        )
    stems = [t.stem for t in targets]
    if len(set(stems)) != len(stems):
        raise SystemExit("FoldBench monomer list has duplicate (pdb, chain) rows")
    return targets


def graphql_entries(entry_ids: list[str], *, retries: int = 4) -> dict[str, dict]:
    """Fetch polymer entities for a batch of entries, keyed by uppercase id."""
    payload = json.dumps({"query": ENTRY_QUERY,
                          "variables": {"ids": entry_ids}}).encode()
    request = urllib.request.Request(
        RCSB_GRAPHQL, data=payload,
        headers={"Content-Type": "application/json"},
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=120) as fh:
                body = json.load(fh)
            break
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == retries - 1:
                raise RuntimeError(
                    f"RCSB GraphQL failed after {retries} attempts for {entry_ids}"
                ) from exc
            time.sleep(2 ** attempt)
    if body.get("errors"):
        raise RuntimeError(f"RCSB GraphQL returned errors: {body['errors']}")
    entries = {e["rcsb_id"].upper(): e for e in body["data"]["entries"] or []}
    missing = [i for i in entry_ids if i not in entries]
    if missing:
        raise RuntimeError(f"RCSB returned no entry for {missing}")
    return entries


def select_entity(entry: dict, target: MonomerTarget) -> ResolvedTarget:
    """Resolve FoldBench's chain to exactly one protein entity of ``entry``.

    Auth chain ids are tried first and label asym ids second, because FoldBench
    is inconsistent about which it stores and auth is the common case. Anything
    that resolves to zero or more than one entity is an error, not a guess.
    """
    proteins = [
        e for e in entry["polymer_entities"] or []
        if "polypeptide" in ((e["entity_poly"] or {}).get("type") or "")
    ]
    if not proteins:
        raise ValueError(f"{target.stem}: entry {entry['rcsb_id']} has no protein entity")

    def ids(entity: dict, field: str) -> tuple[str, ...]:
        return tuple(entity["rcsb_polymer_entity_container_identifiers"][field] or ())

    for kind, field in (("auth", "auth_asym_ids"), ("label", "asym_ids")):
        matches = [e for e in proteins if target.chain_id in ids(e, field)]
        if len(matches) > 1:
            raise ValueError(
                f"{target.stem}: chain {target.chain_id!r} matches "
                f"{len(matches)} protein entities on the {kind} axis"
            )
        if matches:
            entity = matches[0]
            sources = entity["rcsb_entity_source_organism"] or []
            return ResolvedTarget(
                target=target,
                entity_id=entity["rcsb_id"],
                sequence=entity["entity_poly"]["pdbx_seq_one_letter_code_can"].strip().upper(),
                auth_asym_ids=ids(entity, "auth_asym_ids"),
                asym_ids=ids(entity, "asym_ids"),
                chain_match=kind,
                source_taxids=tuple(str(s["ncbi_taxonomy_id"]) for s in sources
                                    if s.get("ncbi_taxonomy_id") is not None),
                source_names=tuple(s["ncbi_scientific_name"] for s in sources
                                   if s.get("ncbi_scientific_name")),
                title=((entry["struct"] or {}).get("title") or "").strip(),
                description=((entity["rcsb_polymer_entity"] or {}).get("pdbx_description")
                             or "").strip(),
            )
    seen = {e["rcsb_id"]: (ids(e, "auth_asym_ids"), ids(e, "asym_ids")) for e in proteins}
    raise ValueError(
        f"{target.stem}: chain {target.chain_id!r} matches no protein entity "
        f"by auth or label id; entry has {seen}"
    )


def resolve_all(targets: list[MonomerTarget]) -> list[ResolvedTarget]:
    """Fetch and resolve every target, in batches, preserving input order."""
    by_entry: dict[str, dict] = {}
    unique = sorted({t.pdb_id.upper() for t in targets})
    for start in range(0, len(unique), BATCH_SIZE):
        batch = unique[start:start + BATCH_SIZE]
        by_entry.update(graphql_entries(batch))
        print(f"[rcsb] {min(start + BATCH_SIZE, len(unique))}/{len(unique)} entries",
              flush=True)
    return [select_entity(by_entry[t.pdb_id.upper()], t) for t in targets]


TARGETS_COLUMNS = ["stem", "pdb_id", "chain_id", "entity_id", "chain_match",
                   "auth_asym_ids", "asym_ids", "synthetic", "source_taxids",
                   "source_names", "title", "description", "seq_len", "sequence"]


def write_targets_csv(resolved: list[ResolvedTarget], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(TARGETS_COLUMNS)
        for r in resolved:
            writer.writerow([
                r.target.stem, r.target.pdb_id, r.target.chain_id, r.entity_id,
                r.chain_match, ";".join(r.auth_asym_ids), ";".join(r.asym_ids),
                int(r.synthetic), ";".join(r.source_taxids), ";".join(r.source_names),
                r.title, r.description, len(r.sequence), r.sequence,
            ])


def read_targets_csv(path: Path) -> list[ResolvedTarget]:
    with path.open() as fh:
        return [
            ResolvedTarget(
                target=MonomerTarget(pdb_id=row["pdb_id"], chain_id=row["chain_id"]),
                entity_id=row["entity_id"],
                sequence=row["sequence"],
                auth_asym_ids=tuple(filter(None, row["auth_asym_ids"].split(";"))),
                asym_ids=tuple(filter(None, row["asym_ids"].split(";"))),
                chain_match=row["chain_match"],
                source_taxids=tuple(filter(None, row["source_taxids"].split(";"))),
                source_names=tuple(filter(None, row["source_names"].split(";"))),
                title=row["title"],
                description=row["description"],
            )
            for row in csv.DictReader(fh)
        ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets-csv", type=Path, default=DATA / "foldbench_targets.csv",
                    help="resolved 334-row table (written, or re-read with --offline)")
    ap.add_argument("--foldbench-cache", type=Path,
                    default=DATA / "foldbench_monomer_protein.csv")
    ap.add_argument("--out-fasta", type=Path,
                    default=DATA / "foldbench_rest_queries.fasta")
    ap.add_argument("--out-validation", type=Path,
                    default=DATA / "query_set_validation.json")
    ap.add_argument("--offline", action="store_true",
                    help="reuse --targets-csv instead of calling RCSB")
    args = ap.parse_args()

    targets = parse_targets(fetch_foldbench_csv(args.foldbench_cache))
    print(f"[foldbench] {len(targets)} monomers @ {FOLDBENCH_COMMIT[:8]}", flush=True)

    # --- what the eval set already has ---------------------------------------
    check_exp213_queries()
    eval_records = read_exp213_queries()
    eval_by_dataset: dict[str, dict[str, str]] = {}
    for header, sequence in eval_records:
        dataset, stem = header.split("__", 1)
        eval_by_dataset.setdefault(dataset, {})[stem] = sequence
    used = eval_by_dataset[DATASET_USED]
    print(f"[eval] {len(eval_records)} eval proteins; {len(used)} are {DATASET_USED}",
          flush=True)

    all_stems = [t.stem for t in targets]
    outside = sorted(set(used) - set(all_stems))
    if outside:
        raise SystemExit(
            f"{len(outside)} of our {DATASET_USED} stems are not FoldBench monomers "
            f"at this commit ({outside[:5]}); the 'strict subset' premise is wrong."
        )
    if set(used) != set(all_stems[:len(used)]):
        raise SystemExit(
            "our foldbench100 is not the first 100 rows of the monomer list; "
            "exp12's provenance claim no longer holds."
        )

    new_targets = [t for t in targets if t.stem not in used]
    if len(new_targets) != EXPECTED_N_NEW:
        raise SystemExit(f"expected {EXPECTED_N_NEW} unused monomers, got {len(new_targets)}")

    # A dozen of the unused monomers are already eval proteins, carried under
    # exp65's de novo PDB dataset. Counting them again would inflate the
    # expanded set and double-weight them in every survival number.
    elsewhere: dict[str, str] = {}
    for dataset, stems in eval_by_dataset.items():
        if dataset == DATASET_USED:
            continue
        for stem in stems:
            if stem in {t.stem for t in new_targets}:
                elsewhere[stem] = dataset
    if set(elsewhere) != EXPECTED_ALREADY_IN_EVAL:
        raise SystemExit(
            "the set of unused monomers already in the eval set changed: "
            f"got {sorted(elsewhere)}, expected {sorted(EXPECTED_ALREADY_IN_EVAL)}"
        )
    net_new = [t for t in new_targets if t.stem not in elsewhere]
    if len(net_new) != EXPECTED_N_NET_NEW:
        raise SystemExit(f"expected {EXPECTED_N_NET_NEW} net-new, got {len(net_new)}")
    print(f"[split] {len(new_targets)} unused - {len(elsewhere)} already in the eval "
          f"set (as {sorted(set(elsewhere.values()))}) = {len(net_new)} net-new",
          flush=True)

    # --- sequences ------------------------------------------------------------
    if args.offline:
        resolved = read_targets_csv(args.targets_csv)
        if [r.target.stem for r in resolved] != all_stems:
            raise SystemExit(f"{args.targets_csv} does not match the pinned target list")
    else:
        resolved = resolve_all(targets)
        write_targets_csv(resolved, args.targets_csv)
        print(f"[rcsb] {len(resolved)} sequences -> {args.targets_csv}", flush=True)
    by_stem = {r.target.stem: r for r in resolved}

    label_matched = sorted(r.target.stem for r in resolved if r.chain_match == "label")
    print(f"[chains] {len(label_matched)} targets matched on the label asym id, "
          f"not auth: {label_matched}", flush=True)

    # --- validation: our existing 100, re-fetched through this exact path -----
    mismatches = [
        {"stem": stem, "eval_len": len(seq), "rcsb_len": len(by_stem[stem].sequence)}
        for stem, seq in sorted(used.items())
        if by_stem[stem].sequence != seq
    ]
    if mismatches:
        raise SystemExit(
            f"{len(mismatches)}/{len(used)} re-fetched FoldBench-100 sequences differ "
            f"from the eval set's: {mismatches[:5]}. The fetch method is not exp12's."
        )
    print(f"[validate] {len(used)}/{len(used)} existing FoldBench sequences reproduced "
          "byte-for-byte from RCSB", flush=True)

    # --- validation: the designed-protein proxy, calibrated on known designs ---
    # The 12 unused monomers that are already eval proteins are there *because*
    # exp65 curated them as de novo designs. If the source-organism proxy does
    # not flag all 12, it is not usable for splitting the net-new set.
    missed = sorted(s for s in EXPECTED_ALREADY_IN_EVAL if not by_stem[s].synthetic)
    if missed:
        raise SystemExit(
            f"the synthetic-source proxy missed {len(missed)} known de novo designs "
            f"({missed}); it cannot be trusted to split designed from natural."
        )
    synthetic_new = sorted(t.stem for t in net_new if by_stem[t.stem].synthetic)
    print(f"[validate] synthetic-source proxy flags {len(EXPECTED_ALREADY_IN_EVAL)}/"
          f"{len(EXPECTED_ALREADY_IN_EVAL)} known de novo designs", flush=True)
    print(f"[designed] {len(synthetic_new)}/{len(net_new)} net-new monomers have no "
          "natural source organism", flush=True)

    aa_new = sum(len(by_stem[t.stem].sequence) for t in new_targets)
    aa_net_new = sum(len(by_stem[t.stem].sequence) for t in net_new)
    lengths = [len(by_stem[t.stem].sequence) for t in new_targets]
    for label, got, want in (("234 unused", aa_new, EXPECTED_AA_NEW),
                             ("222 net-new", aa_net_new, EXPECTED_AA_NET_NEW)):
        if got != want:
            raise SystemExit(f"{label} total is {got} aa, expected {want}")
    print(f"[validate] {aa_new} aa over {len(new_targets)} unused monomers "
          f"(min {min(lengths)}, max {max(lengths)}); {aa_net_new} aa over "
          f"{len(net_new)} net-new — both match the issue's checksums", flush=True)

    # --- outputs --------------------------------------------------------------
    args.out_fasta.parent.mkdir(parents=True, exist_ok=True)
    args.out_fasta.write_text("".join(
        f">{DATASET_NEW}__{t.stem}\n{by_stem[t.stem].sequence}\n" for t in net_new
    ))
    print(f"[queries] {len(net_new)} records -> {args.out_fasta}", flush=True)

    args.out_validation.write_text(json.dumps({
        "foldbench_commit": FOLDBENCH_COMMIT,
        "foldbench_csv_sha256": FOLDBENCH_CSV_SHA256,
        "n_foldbench_monomers": len(targets),
        "n_already_used": len(used),
        "n_unused": len(new_targets),
        "n_already_in_eval_as_other_dataset": len(elsewhere),
        "already_in_eval": {s: elsewhere[s] for s in sorted(elsewhere)},
        "n_net_new": len(net_new),
        "aa_unused": aa_new,
        "aa_net_new": aa_net_new,
        "unused_len_min": min(lengths),
        "unused_len_max": max(lengths),
        "label_chain_matches": label_matched,
        "foldbench100_sequences_reproduced": f"{len(used)}/{len(used)}",
        "expanded_eval_set_size": len(eval_records) + len(net_new),
        "synthetic_taxid": SYNTHETIC_TAXID,
        "known_designs_flagged_synthetic": f"{len(EXPECTED_ALREADY_IN_EVAL)}/"
                                           f"{len(EXPECTED_ALREADY_IN_EVAL)}",
        "n_net_new_synthetic": len(synthetic_new),
        "net_new_synthetic": synthetic_new,
    }, indent=2))
    print(f"[validate] -> {args.out_validation}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
