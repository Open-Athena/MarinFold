# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Annotate each eval protein with its source organism + a viral flag (RCSB).

Motivation: AFDB (the contacts-v1 training source) **excludes viruses**, so viral
eval proteins are genuinely out-of-distribution. This script labels all 554 eval
proteins so the KNN/MarinFold analysis can be split by viral vs non-viral.

Two paths (both via RCSB, needs network — this is an offline-pipeline exception,
run once to regenerate `data/eval_taxonomy.csv`):

* **PDB-id stems** (foldbench100, cameo_hard, denovo_pdb): GraphQL fetch of every
  polymer entity's `taxonomy_lineage`, matched to the eval `gt_chain`. Viral iff
  the chain's entity has "Viruses" in its lineage.
* **CASP14 FM T-numbers** (casp_fm): no direct PDB code, so we sequence-search the
  target's `input_seq` against RCSB, take the best polymer-entity hit (all match
  their released structures at >=0.87 identity), and read its lineage.

Writes `data/eval_taxonomy.csv` (554 rows) and `data/viral_proteins.csv` (the viral
subset). Bacteriophages count as viral (also excluded from AFDB).

    uv run python annotate_taxonomy.py
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

EXP78 = Path("/home/bizon/git/MarinFold-exp78/experiments/exp78_evals_esmfold_contacts/data")
MANIFESTS = [EXP78 / "eval_manifest_foldbench.csv", EXP78 / "eval_manifest_exp65.csv"]
GRAPHQL = "https://data.rcsb.org/graphql"
SEARCH = "https://search.rcsb.org/rcsbsearch/v2/query"

ENTRIES_Q = """query($ids:[String!]!){ entries(entry_ids:$ids){ rcsb_id
 polymer_entities{ rcsb_polymer_entity_container_identifiers{ auth_asym_ids }
 rcsb_entity_source_organism{ ncbi_scientific_name taxonomy_lineage{ name } } } } }"""


def pdb_id(stem: str) -> str | None:
    m = re.match(r"^([0-9][a-zA-Z0-9]{3})_", stem)
    return m.group(1).upper() if m else None


def load_eval() -> list[dict]:
    rows: list[dict] = []
    for man in MANIFESTS:
        with man.open() as fh:
            for r in csv.DictReader(fh):
                rows.append({"dataset": r["dataset"], "stem": r["stem"],
                             "gt_chain": r["gt_chain"], "input_seq": r["input_seq"]})
    return rows


def _post_graphql(query: str, variables: dict) -> dict:
    data = json.dumps({"query": query, "variables": variables}).encode()
    req = urllib.request.Request(GRAPHQL, data=data, headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=60))["data"]


def fetch_pdb_entities(pdb_ids: list[str]) -> dict[str, list[dict]]:
    """pdb_id -> [{chains, organisms, viral}] for every polymer entity."""
    out: dict[str, list[dict]] = {}
    for i in range(0, len(pdb_ids), 80):
        for e in _post_graphql(ENTRIES_Q, {"ids": pdb_ids[i:i + 80]})["entries"]:
            ents = []
            for pe in (e.get("polymer_entities") or []):
                ids = pe.get("rcsb_polymer_entity_container_identifiers") or {}
                chains = set(ids.get("auth_asym_ids") or [])
                orgs = pe.get("rcsb_entity_source_organism") or []
                names = [o.get("ncbi_scientific_name") for o in orgs]
                lineage = [n.get("name") for o in orgs for n in (o.get("taxonomy_lineage") or [])]
                ents.append({"chains": chains, "organisms": names, "viral": "Viruses" in lineage})
            out[e["rcsb_id"]] = ents
        time.sleep(0.3)
    return out


def search_entity(seq: str) -> tuple[str, float] | None:
    """Best polymer-entity hit for a sequence -> (identifier, sequence_identity)."""
    q = {"query": {"type": "terminal", "service": "sequence", "parameters": {
        "evalue_cutoff": 10, "identity_cutoff": 0.0, "sequence_type": "protein", "value": seq}},
        "request_options": {"scoring_strategy": "sequence",
                            "paginate": {"start": 0, "rows": 1}, "results_verbosity": "verbose"},
        "return_type": "polymer_entity"}
    url = SEARCH + "?json=" + urllib.parse.quote(json.dumps(q))
    d = json.load(urllib.request.urlopen(url, timeout=60))
    res = d.get("result_set", [])
    if not res:
        return None
    ctx = res[0]["services"][0]["nodes"][0]["match_context"][0]
    return res[0]["identifier"], float(ctx.get("sequence_identity", 0.0))


def entity_organism(identifier: str) -> tuple[list[str], bool]:
    eid, ent = identifier.split("_")
    q = ('query{ polymer_entity(entry_id:"%s",entity_id:"%s"){ rcsb_entity_source_organism{'
         ' ncbi_scientific_name taxonomy_lineage{ name } } } }' % (eid, ent))
    pe = _post_graphql(q, {})["polymer_entity"] or {}
    orgs = pe.get("rcsb_entity_source_organism") or []
    names = [o.get("ncbi_scientific_name") for o in orgs]
    lineage = [n.get("name") for o in orgs for n in (o.get("taxonomy_lineage") or [])]
    return names, "Viruses" in lineage


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/eval_taxonomy.csv"))
    ap.add_argument("--viral-out", type=Path, default=Path("data/viral_proteins.csv"))
    args = ap.parse_args()

    rows = load_eval()
    pdb_ids = sorted({pdb_id(r["stem"]) for r in rows if pdb_id(r["stem"])})
    print(f"[tax] {len(rows)} eval proteins, {len(pdb_ids)} unique PDB ids", flush=True)
    pdb_info = fetch_pdb_entities(pdb_ids)

    out_rows: list[dict] = []
    for r in rows:
        pid = pdb_id(r["stem"])
        if pid:
            ents = pdb_info.get(pid, [])
            chosen = next((e for e in ents if r["gt_chain"] in e["chains"]), None) or (ents[0] if ents else None)
            organisms = ";".join(str(n) for n in chosen["organisms"]) if chosen else ""
            viral = bool(chosen["viral"]) if chosen else None
            evidence = pid
        else:  # CASP14 FM target: sequence-search to the released structure
            hit = search_entity(r["input_seq"])
            time.sleep(0.3)
            if hit:
                organisms_list, viral = entity_organism(hit[0])
                organisms = ";".join(str(n) for n in organisms_list)
                evidence = f"{hit[0]}@{hit[1]:.2f}id"
            else:
                organisms, viral, evidence = "", None, ""
        out_rows.append({"dataset": r["dataset"], "stem": r["stem"], "gt_chain": r["gt_chain"],
                         "pdb_evidence": evidence, "source_organism": organisms,
                         "is_viral": "" if viral is None else viral})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "stem", "gt_chain", "pdb_evidence", "source_organism", "is_viral"]
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)

    viral = [r for r in out_rows if r["is_viral"] is True]
    with args.viral_out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(viral)

    import collections
    by_ds = collections.Counter(r["dataset"] for r in viral)
    print(f"[tax] viral: {len(viral)}/{len(out_rows)} -> {dict(by_ds)}", flush=True)
    print(f"[tax] wrote {args.out} and {args.viral_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
