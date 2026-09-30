"""Search public source databases separately, then reduce cached hits into counts.

The ColabFold UniRef source is documented as 2023_02. Its environmental source
is an older mixed database. EBI's UniProt and MGnify searches are newer proxies,
whose exact versions are recorded in each job. None identifies the final ESMC
training subset. Submission/collection and offline analysis are separate phases.
"""

import argparse
import csv
import hashlib
import io
import json
import shutil
import tarfile
import urllib.parse
import urllib.request
from pathlib import Path

from audit_low_depth_msas import records, write_csv

ROOT = Path(__file__).resolve().parent.parent
SCRATCH = ROOT / "scratch/training_source_search"
INPUTS = ROOT / "data/inputs/training_source_search"
QUERY_FILE = ROOT / "data/inputs/low_depth_training_audit/queries.csv"
HMMER = "https://www.ebi.ac.uk/Tools/hmmer/api/v1"
COLABFOLD = "https://api.colabfold.com"
EVALUE = 0.001
HISTORY = "https://github.com/sokrypton/ColabFold/wiki/MSA-Server-Database-History"
MGNIFY_RELEASE = "https://ftp.ebi.ac.uk/pub/databases/metagenomics/peptide_database/2023_02"


def queries() -> list[dict]:
    """Load the fixed five resolved-residue sequences used in the earlier plots."""
    with QUERY_FILE.open() as stream:
        return list(csv.DictReader(stream))


def fetch(url: str, payload: bytes | None = None, content_type: str = "application/json") -> bytes:
    """Make one checked request; callers persist responses before interpretation."""
    request = urllib.request.Request(url, data=payload, headers={
        "Content-Type": content_type, "Accept": "application/json", "User-Agent": "MarinFold-exp325/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def submit() -> None:
    """Submit at most ten small HMMER searches and one five-sequence MSA job."""
    SCRATCH.mkdir(parents=True, exist_ok=True)
    for database in ("uniprot", "mgnify30_c2"):
        for query in queries():
            prefix = SCRATCH / f"hmmer_{database}_{query['stem']}"
            request_path = Path(f"{prefix}_request.json")
            submission = Path(f"{prefix}_submission.json")
            request = {"database": database, "input": f">{query['stem']}\n{query['sequence']}",
                       "E": EVALUE, "domE": EVALUE, "incE": EVALUE, "incdomE": EVALUE,
                       "with_taxonomy": False, "with_architecture": False}
            if submission.exists():
                if json.loads(request_path.read_text()) != request:
                    raise ValueError(f"Saved request differs: {request_path}")
                continue
            request_path.write_text(json.dumps(request, indent=2) + "\n")
            submission.write_bytes(fetch(f"{HMMER}/search/phmmer", json.dumps(request).encode()))
            print(submission.name, submission.read_text(), flush=True)
    request = urllib.parse.urlencode({
        "q": "".join(f">{101+i}\n{row['sequence']}\n" for i, row in enumerate(queries())),
        "mode": "env-nofilter"}).encode()
    request_path, submission = SCRATCH / "colabfold_request.txt", SCRATCH / "colabfold_submission.json"
    if submission.exists():
        if request_path.read_bytes() != request:
            raise ValueError("Saved ColabFold request differs")
        return
    request_path.write_bytes(request)
    submission.write_bytes(fetch(f"{COLABFOLD}/ticket/msa", request, "application/x-www-form-urlencoded"))
    print(submission.read_text(), flush=True)


def collect() -> None:
    """Require completed jobs, then freeze raw evidence; retry this phase if pending."""
    for filename, url in {
        "hmmer_databases.json": f"{HMMER}/search/databases",
        "README.txt": f"{MGNIFY_RELEASE}/README.txt",
        "md5sum.txt": f"{MGNIFY_RELEASE}/md5sum.txt",
        "colabfold_database_history.md": "https://raw.githubusercontent.com/wiki/sokrypton/ColabFold/MSA-Server-Database-History.md",
    }.items():
        path = SCRATCH / filename
        if not path.exists():
            path.write_bytes(fetch(url))
    for database in ("uniprot", "mgnify30_c2"):
        for query in queries():
            prefix = f"hmmer_{database}_{query['stem']}"
            submission = json.loads((SCRATCH / f"{prefix}_submission.json").read_text())
            details_path = SCRATCH / f"{prefix}_details.json"
            results_path = SCRATCH / f"{prefix}_results.json"
            if not details_path.exists() or json.loads(details_path.read_text())["task"]["status"] != "SUCCESS":
                details_path.write_bytes(fetch(f"{HMMER}/search/{submission['id']}"))
            details = json.loads(details_path.read_text())
            if details["task"]["status"] != "SUCCESS":
                raise RuntimeError(f"{prefix}: {details['task']['status']}; collect again later")
            if not results_path.exists() or json.loads(results_path.read_text())["status"] != "SUCCESS":
                results_path.write_bytes(fetch(f"{HMMER}/result/{submission['id']}?page_size=1000&with_domains=true"))
            result = json.loads(results_path.read_text())
            if result["status"] != "SUCCESS" or result["page_count"] > 1:
                raise ValueError(f"{prefix}: incomplete result pages")
    job = json.loads((SCRATCH / "colabfold_submission.json").read_text())["id"]
    archive_path = SCRATCH / "colabfold_results.tar.gz"
    if not archive_path.exists():
        status = json.loads(fetch(f"{COLABFOLD}/ticket/{job}"))
        if status["status"] != "COMPLETE":
            raise RuntimeError(f"ColabFold: {status['status']}; collect again later")
        archive_path.write_bytes(fetch(f"{COLABFOLD}/result/download/{job}"))
    INPUTS.mkdir(parents=True, exist_ok=True)
    for path in sorted(SCRATCH.glob("hmmer_*.json")):
        shutil.copyfile(path, INPUTS / path.name)
    for name in ("colabfold_request.txt", "colabfold_submission.json", "README.txt", "md5sum.txt",
                 "colabfold_database_history.md"):
        shutil.copyfile(SCRATCH / name, INPUTS / name)
    with tarfile.open(fileobj=io.BytesIO(archive_path.read_bytes()), mode="r:gz") as archive:
        archive.extractall(INPUTS / "colabfold", filter="data")
    for source, filename in (("uniref", "uniref.a3m"), ("environment", "bfd.mgnify30.metaeuk30.smag30.a3m")):
        blocks = [block for block in (INPUTS / "colabfold" / filename).read_text().split("\0") if block.strip()]
        if len(blocks) != len(queries()):
            raise ValueError("Incomplete ColabFold target blocks")
        for index, (query, block) in enumerate(zip(queries(), blocks, strict=True)):
            if block.splitlines()[0] != f">{101+index}":
                raise ValueError("ColabFold returned targets in a different order")
            (INPUTS / f"colabfold_{source}_{query['stem']}.a3m").write_text(block)


def domain_alignment(hit: dict, length: int) -> str:
    """Map significant HMMER domains into query columns without counting insertions."""
    row = ["-"] * length
    for domain in sorted(hit["domains"], key=lambda item: -item["bitscore"]):
        if domain["ievalue"] > EVALUE or not domain["is_reported"]:
            continue
        display = domain["alignment_display"]
        position = display["hmmfrom"] - 1
        if display["m"] != length:
            raise ValueError("HMMER query length differs from the submitted sequence")
        for model, residue in zip(display["model"], display["aseq"], strict=True):
            if model == ".":
                continue
            if residue != "-" and row[position] == "-":
                row[position] = residue.upper()
            position += 1
        if position != display["hmmto"]:
            raise ValueError("HMMER alignment coordinates do not reproduce domain endpoints")
    return "".join(row)


def prepare() -> None:
    """Count significant source hits separately from query rows and full-query hits."""
    hits, summaries = [], []
    for query in queries():
        stem, sequence = query["stem"], query["sequence"]
        for source, version in (("uniref", "2023_02 (server documentation)"), ("environment", "ColabFoldDB 202108 (mixed sources)")):
            path = INPUTS / f"colabfold_{source}_{stem}.a3m"
            alignment = records(path)
            if alignment[0][1] != sequence:
                raise ValueError(f"{stem}: ColabFold query changed")
            rows = []
            for index, (header, raw) in enumerate(alignment[1:], start=1):
                fields = header.split()
                aligned = "".join(aa for aa in raw if not aa.islower())
                if len(fields) != 10 or len(aligned) != len(sequence):
                    raise ValueError("Unexpected ColabFold hit format")
                rows.append({"stem": stem, "source": f"colabfold_{source}", "version": version,
                             "accession": fields[0], "evalue": float(fields[3]),
                             "query_coverage": sum(aa != "-" for aa in aligned) / len(sequence),
                             "aligned_sequence": aligned, "raw_file": str(path.relative_to(ROOT)), "raw_record": index})
            hits.extend(rows)
            summaries.append(summarize(rows, stem, f"colabfold_{source}", version, len(alignment)-1))
        for database in ("uniprot", "mgnify30_c2"):
            prefix = f"hmmer_{database}_{stem}"
            path = INPUTS / f"{prefix}_results.json"
            results = json.loads(path.read_text())
            details = json.loads((INPUTS / f"{prefix}_details.json").read_text())
            request = json.loads((INPUTS / f"{prefix}_request.json").read_text())
            result = results["result"]
            if (results["status"] != "SUCCESS" or details["task"]["status"] != "SUCCESS"
                    or results["page_count"] > 1 or len(result["hits"]) != result["stats"]["nreported"]
                    or details["number_of_hits"] != len(result["hits"])):
                raise ValueError(f"{prefix}: incomplete hit retrieval")
            if (request["input"] != f">{stem}\n{sequence}" or details["input"] != request["input"]
                    or request["database"] != database or details["database"]["id"] != database
                    or details["algo"] != "phmmer"
                    or any(details[key] != EVALUE for key in ("E", "domE", "incE", "incdomE"))):
                raise ValueError(f"{prefix}: search protocol changed")
            version, rows = details["database"]["version"], []
            for index, hit in enumerate(result["hits"]):
                aligned = domain_alignment(hit, len(sequence))
                rows.append({"stem": stem, "source": f"hmmer_{database}", "version": version,
                             "accession": hit["metadata"]["accession"], "evalue": hit["evalue"],
                             "query_coverage": sum(aa != "-" for aa in aligned) / len(sequence),
                             "aligned_sequence": aligned, "raw_file": str(path.relative_to(ROOT)), "raw_record": index})
            hits.extend(rows)
            summaries.append(summarize(rows, stem, f"hmmer_{database}", version, len(rows)))
    write_csv(ROOT / "data/training_source_hits.csv", hits)
    write_csv(ROOT / "data/training_source_depths.csv", summaries)
    manifest = {
        "queried_date": "2026-09-30", "query_file": str(QUERY_FILE.relative_to(ROOT)),
        "query_sha256": hashlib.sha256(QUERY_FILE.read_bytes()).hexdigest(),
        "evalue_max": EVALUE, "coverage_cuts": [0.5, 0.8],
        "depth_definition": "Database hit records excluding the explicitly added query; unique aligned rows also reported",
        "hmmer_coverage": "Union of query positions occupied by residues in reported domains with independent E<=0.001",
        "colabfold_version_source": HISTORY,
        "limitations": "UniRef source is searched through profiles/cluster expansion. The environment mix is not MGnify 2023_02. HMMER UniProt/MGnify are newer proxies; MGnify30-C2 excludes singleton clusters. JGI not searched.",
        "inputs": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in sorted(INPUTS.rglob("*")) if path.is_file()},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (ROOT / "data/training_source_search_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for row in summaries:
        print(row["stem"], row["source"], row["hits_e001"], row["hits_e001_cov50"], row["hits_e001_cov80"])


def summarize(rows: list[dict], stem: str, source: str, version: str, returned: int) -> dict:
    """Keep zero-hit searches explicit and distinguish sequence/domain evidence."""
    significant = [row for row in rows if row["evalue"] <= EVALUE]
    result = {"stem": stem, "source": source, "version": version, "returned_hits": returned,
              "hits_e001": len(significant)}
    for coverage in (0.5, 0.8):
        covered = [row for row in significant if row["query_coverage"] >= coverage]
        result[f"hits_e001_cov{int(100*coverage)}"] = len(covered)
        result[f"unique_aligned_e001_cov{int(100*coverage)}"] = len({row["aligned_sequence"] for row in covered})
    return result


def main() -> None:
    """Expose costly requests separately from fast, deterministic table reduction."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["submit", "collect", "prepare"])
    args = parser.parse_args()
    {"submit": submit, "collect": collect, "prepare": prepare}[args.phase]()


if __name__ == "__main__":
    main()
