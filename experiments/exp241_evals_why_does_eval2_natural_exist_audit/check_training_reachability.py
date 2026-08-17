# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Step 2 — is an eval2-natural protein *unknown*, or just *unsampled*? (A3)

The premise of issue #241 is that a natural protein deposited to the PDB has a
sequence that was determined years earlier, sits in UniProt, and therefore has an
AlphaFold model. If that premise holds, then "no homolog in our training set"
cannot mean "no homolog exists" — it has to mean our corpora do not contain the
homologs that do exist. This step separates those two readings per protein, with
four checks that get progressively closer to the counterfactual we care about:

1. **Is the sequence in UniProt at all?** The RCSB entity's UniProt
   cross-reference (from :mod:`annotate_rcsb`). No cross-reference is itself an
   answer — it means the deposited entity is not a UniProt sequence.
2. **Does full AFDB have a model for that accession?** EBI's AFDB API. This is
   the difference between "AlphaFold never folded it" and "AlphaFold folded it
   and we did not train on it".
3. **Is that accession in our AFDB training arm?** An exact set-membership test
   against the 4,129,682 accessions in exp213's ``train_afdb.fasta`` headers.
   Free, local, and exact — no search sensitivity to argue about.
4. **How many relatives of it exist in UniProt, and how many did we train on?**
   UniRef50 and UniRef90 cluster sizes give the number of sequences in UniProt at
   roughly >=50 % / >=90 % identity; intersecting each cluster's UniProtKB member
   accessions with the training arm gives how many of those we actually hold.
   ``uniref50_size`` = 400 with ``uniref50_in_arm`` = 0 is the finding: the
   protein has hundreds of close relatives in the reference databases and our
   1.9 %-of-AFDB sample contains none of them.

Check 3 doubles as the **positive control** (A5): run over all 776 eval
proteins, every protein whose accession *is* in the arm must show a near-100 %
identity in exp226's table. If that fails, the audit is measuring a broken
search rather than a sampled corpus.

    uv run python check_training_reachability.py              # the 78 + control
    uv run python check_training_reachability.py --cohort all # every accession
    uv run python check_training_reachability.py --offline    # cache only
"""
import argparse
import csv
import json
import time
import urllib.error
import urllib.parse
import urllib.request

import upstream as U

DATA = U.HERE / "data"
ANNOTATION = DATA / "rcsb_annotation.csv"
CACHE = DATA / "reachability_cache.json"
OUT = DATA / "training_reachability.csv"
CONTROL_OUT = DATA / "arm_membership_control.csv"

AFDB_API = "https://alphafold.ebi.ac.uk/api/prediction/{acc}"
UNIREF_SEARCH = "https://rest.uniprot.org/uniref/search"
UNIPROTKB_SEARCH = "https://rest.uniprot.org/uniprotkb/search"
UNIPROTKB_ENTRY = "https://rest.uniprot.org/uniprotkb/{acc}.json"

#: Page size for cluster-member listing, and the cap on pages. A UniRef50
#: cluster with more than PAGE * MAX_PAGES UniProtKB members is reported as
#: censored rather than truncated silently — the intersection count is then a
#: lower bound, which is the safe direction for this argument (it can only
#: understate how much of the cluster we trained on).
PAGE = 500
MAX_PAGES = 20

#: Identity a hit must reach for the positive control to count the arm's own
#: sequence as "found". Anything below this on a protein whose exact accession is
#: in the training arm means the search lost a self-hit.
CONTROL_MIN_IDENTITY = 0.90


def get(url: str, *, retries: int = 5, accept_404: bool = False):
    """GET with backoff. Returns ``(status, body_text, headers)``."""
    for attempt in range(retries):
        req = urllib.request.Request(url, headers={
            "Accept": "application/json",
            "User-Agent": "MarinFold-exp241 (github.com/Open-Athena/MarinFold)",
        })
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                return resp.status, resp.read().decode(), dict(resp.headers)
        except urllib.error.HTTPError as exc:
            if exc.code == 404 and accept_404:
                return 404, "", {}
            if exc.code not in (429, 500, 502, 503, 504) or attempt == retries - 1:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries - 1:
                raise
        time.sleep(2 ** attempt)
    raise RuntimeError(f"unreachable: {url}")


def afdb_has_model(acc: str, cache: dict) -> bool:
    """Does full AFDB hold a prediction for this accession?"""
    key = f"afdb:{acc}"
    if key not in cache:
        status, _, _ = get(AFDB_API.format(acc=acc), accept_404=True)
        cache[key] = status == 200
        time.sleep(0.1)
    return cache[key]


def uniref_clusters(acc: str, cache: dict) -> dict:
    """``{"UniRef50": (id, member_count), ...}`` for the accession's clusters."""
    key = f"uniref:{acc}"
    if key not in cache:
        url = f"{UNIREF_SEARCH}?" + urllib.parse.urlencode({
            "query": f"uniprot_id:{acc}", "fields": "id,count", "size": 10,
            "format": "json",
        })
        _, body, _ = get(url)
        # The search returns the accession's UniRef100, UniRef90 and UniRef50
        # clusters in one response. ``entryType`` is only populated when the
        # default field set is requested, so the level is taken from the cluster
        # id prefix instead — otherwise all three records collapse onto one key.
        out = {}
        for rec in json.loads(body).get("results", []):
            level = rec["id"].split("_")[0]
            out[level] = (rec["id"], rec.get("memberCount"))
        cache[key] = out
        time.sleep(0.15)
    return cache[key]


def cluster_member_accessions(cluster_id: str, cache: dict) -> tuple[list[str], bool]:
    """UniProtKB accessions in a UniRef cluster; ``(accessions, censored)``."""
    key = f"members:{cluster_id}"
    if key not in cache:
        level = cluster_id.split("_")[0].replace("UniRef", "")
        field = f"uniref_cluster_{level}"
        accessions: list[str] = []
        cursor = None
        censored = True
        for _page in range(MAX_PAGES):
            params = {"query": f"{field}:{cluster_id}", "fields": "accession",
                      "format": "list", "size": PAGE}
            if cursor:
                params["cursor"] = cursor
            _, body, headers = get(
                f"{UNIPROTKB_SEARCH}?{urllib.parse.urlencode(params)}")
            accessions.extend(a for a in body.split("\n") if a.strip())
            link = headers.get("Link") or headers.get("link") or ""
            if 'rel="next"' not in link:
                censored = False
                break
            cursor = urllib.parse.parse_qs(
                urllib.parse.urlparse(link.split(">")[0].lstrip("<")).query
            ).get("cursor", [None])[0]
            if cursor is None:
                censored = False
                break
            time.sleep(0.15)
        cache[key] = {"accessions": accessions, "censored": censored}
        time.sleep(0.15)
    rec = cache[key]
    return rec["accessions"], rec["censored"]


def uniprot_entry(acc: str, cache: dict) -> dict:
    """Creation date, protein-existence evidence and organism for an accession."""
    key = f"entry:{acc}"
    if key not in cache:
        status, body, _ = get(UNIPROTKB_ENTRY.format(acc=acc), accept_404=True)
        if status == 404 or not body:
            cache[key] = {}
        else:
            rec = json.loads(body)
            audit = rec.get("entryAudit") or {}
            cache[key] = {
                "date_created": audit.get("firstPublicDate", ""),
                "sequence_date": audit.get("lastSequenceUpdateDate", ""),
                "protein_existence": rec.get("proteinExistence", ""),
                "organism": (rec.get("organism") or {}).get("scientificName", ""),
                "reviewed": rec.get("entryType", ""),
            }
        time.sleep(0.15)
    return cache[key]


def read_annotation() -> dict[tuple[str, str], dict]:
    if not ANNOTATION.exists():
        raise SystemExit("run annotate_rcsb.py first")
    with ANNOTATION.open() as fh:
        return {(r["dataset"], r["stem"]): r for r in csv.DictReader(fh)}


def accessions_of(ann: dict) -> list[str]:
    return [a for a in (ann.get("uniprot_accessions") or "").split("|") if a]


def write_control(proteins: list[U.Protein], annotation: dict,
                  arm: set[str]) -> list[dict]:
    """A5 — every eval protein whose exact accession is in the training arm."""
    rows = []
    for p in proteins:
        ann = annotation.get((p.dataset, p.stem))
        if ann is None:
            continue
        hits = [a for a in accessions_of(ann) if a in arm]
        if not hits:
            continue
        rows.append({
            "dataset": p.dataset, "stem": p.stem,
            "accession_in_arm": "|".join(hits),
            "afdb_best_identity": "" if p.afdb_best_identity is None
                                  else f"{p.afdb_best_identity:.3f}",
            "best_identity": "" if p.best_identity is None
                             else f"{p.best_identity:.3f}",
            "in_eval2": int(p.in_eval2),
            "passes_control": int(
                (p.afdb_best_identity or 0) >= CONTROL_MIN_IDENTITY),
        })
    with CONTROL_OUT.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cohort", choices=["natural", "all"], default="natural",
                    help="which proteins get the (networked) UniRef treatment")
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args(argv)

    proteins = U.read_proteins()
    natural = U.eval2_natural(proteins)
    annotation = read_annotation()

    print("[arm] reading AFDB training-arm accessions", flush=True)
    arm = U.afdb_arm_accessions()
    print(f"[arm] {len(arm):,} accessions", flush=True)

    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}

    control = write_control(proteins, annotation, arm)
    n_pass = sum(r["passes_control"] for r in control)
    print(f"[control] {n_pass}/{len(control)} eval proteins whose exact accession "
          f"is in the AFDB arm are found at >= {CONTROL_MIN_IDENTITY:.0%} identity "
          f"-> {CONTROL_OUT.name}", flush=True)

    cohort = natural if args.cohort == "natural" else proteins
    rows = []
    for i, p in enumerate(cohort, 1):
        ann = annotation[(p.dataset, p.stem)]
        accs = accessions_of(ann)
        row = {
            "dataset": p.dataset, "stem": p.stem, "length": p.length,
            "kingdom": ann["kingdom"],
            "designed_signal": int(ann["is_synthetic_taxon"] == "1"
                                   or ann["has_denovo_keyword"] == "1"),
            "uniprot_accessions": "|".join(accs),
            "best_identity": "" if p.best_identity is None
                             else f"{p.best_identity:.3f}",
            "afdb_best_identity": "" if p.afdb_best_identity is None
                                  else f"{p.afdb_best_identity:.3f}",
            "esm_atlas_best_identity": "" if p.esm_atlas_best_identity is None
                                       else f"{p.esm_atlas_best_identity:.3f}",
            "n_hits_significant": p.n_hits_significant,
            "msa_neff": "" if p.msa_neff is None else f"{p.msa_neff:.1f}",
            "in_afdb_arm": "", "in_afdb_full": "",
            "uniref100_size": "", "uniref90_id": "", "uniref90_size": "",
            "uniref50_id": "", "uniref50_size": "",
            "uniref90_in_arm": "", "uniref50_in_arm": "",
            "uniref50_censored": "", "uniprot_first_public": "",
            "uniprot_sequence_date": "", "protein_existence": "",
        }
        if accs:
            row["in_afdb_arm"] = int(any(a in arm for a in accs))
        if accs and not args.offline:
            row["in_afdb_full"] = int(any(afdb_has_model(a, cache) for a in accs))
            # Cluster statistics come from the first accession that resolves. A
            # chimeric entity with several xrefs is rare in this cohort and is
            # listed in the README rather than averaged into a meaningless mean.
            for acc in accs:
                clusters = uniref_clusters(acc, cache)
                if not clusters:
                    continue
                for level, col in (("UniRef100", "uniref100"),
                                   ("UniRef90", "uniref90"),
                                   ("UniRef50", "uniref50")):
                    if level not in clusters:
                        continue
                    cid, count = clusters[level]
                    if f"{col}_id" in row:
                        row[f"{col}_id"] = cid
                    row[f"{col}_size"] = count
                    if col == "uniref100":
                        continue
                    members, censored = cluster_member_accessions(cid, cache)
                    row[f"{col}_in_arm"] = sum(1 for m in members if m in arm)
                    if col == "uniref50":
                        row["uniref50_censored"] = int(censored)
                meta = uniprot_entry(acc, cache)
                row["uniprot_first_public"] = meta.get("date_created", "")
                row["uniprot_sequence_date"] = meta.get("sequence_date", "")
                row["protein_existence"] = meta.get("protein_existence", "")
                break
        rows.append(row)
        if i % 10 == 0 or i == len(cohort):
            print(f"[reach] {i}/{len(cohort)}", flush=True)
            CACHE.write_text(json.dumps(cache, sort_keys=True))

    CACHE.write_text(json.dumps(cache, sort_keys=True))
    with OUT.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[out] {OUT} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
