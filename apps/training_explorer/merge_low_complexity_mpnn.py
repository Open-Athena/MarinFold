"""Merge an unmasked full-MPNN search for an exceptional low-complexity query."""

import configparser
import csv
import gzip
import json
from pathlib import Path

import s3fs

from assemble_neighbors import EXPECTED, FULL_CORPUS, SHARDS, make_hit, self_target


HERE = Path(__file__).resolve().parent
DATA = HERE / "data/latest.json"
PREFIX = "marin-us-east-02a/MarinFold/training_explorer/2026-09-22/low_complexity"
QUERY_ID = "latest:148268521"


def main() -> None:
    """Check all MPNN rows and rank every reported native and MPNN hit."""
    credentials = configparser.ConfigParser()
    credentials.read(Path.home() / ".aws/credentials")
    cw = credentials["cw"]
    fs = s3fs.S3FileSystem(
        key=cw["aws_access_key_id"],
        secret=cw["aws_secret_access_key"],
        endpoint_url="https://cwobject.com",
        config_kwargs={"s3": {"addressing_style": "virtual"}},
    )
    manifests = []
    for index in range(SHARDS):
        path = f"{PREFIX}/manifest-{index:02d}-of-{SHARDS:02d}.json"
        with fs.open(path, "rb") as stream:
            manifest = json.load(stream)
        if manifest["shard"] != index or not manifest["unmasked"]:
            raise ValueError(f"Invalid unmasked manifest {path}")
        manifests.append(manifest)
    for arm, expected in EXPECTED.items():
        actual = sum(manifest["sourceCounts"][arm] for manifest in manifests)
        if actual != expected:
            raise ValueError(f"{arm}: expected {expected:,} rows, got {actual:,}")

    snapshot = json.loads(DATA.read_text())
    matches = [p for p in snapshot["proteins"] if p["id"] == QUERY_ID]
    if len(matches) != 1:
        raise ValueError(f"Missing query {QUERY_ID}")
    protein = matches[0]
    own = self_target(protein)
    merged = {hit["id"]: hit for hit in protein["neighbors"]}
    for index, manifest in enumerate(manifests):
        path = f"{PREFIX}/hits-{index:02d}-of-{SHARDS:02d}.tsv.gz"
        with fs.open(path, "rb") as remote, gzip.open(remote, "rt") as stream:
            for values in csv.reader(stream, delimiter="\t"):
                if len(values) != 9:
                    raise ValueError(f"Malformed result in {path}")
                query, hit = make_hit(values, manifest["count"])
                if query != QUERY_ID:
                    raise ValueError(f"Unexpected query {query}")
                if hit["id"] == own:
                    continue
                hit["weakFallback"] = True
                hit["lowComplexityRescue"] = True
                merged[hit["id"]] = hit
        print(f"Merged unmasked MPNN shard {index + 1}/{SHARDS}", flush=True)
    protein["neighbors"] = sorted(
        merged.values(), key=lambda hit: (-hit["bitscore"], hit["evalue"], hit["id"])
    )[:10]
    if len(protein["neighbors"]) != 10:
        raise ValueError(f"Only {len(protein['neighbors'])} low-complexity neighbors")
    protein["lowComplexitySearch"] = True
    snapshot["search"]["unmaskedMpnnRescue"] = {
        "queryId": QUERY_ID,
        "rowsVerified": sum(m["count"] for m in manifests),
        "sourceCounts": EXPECTED,
        "evalueDisplay": f"approximately scaled to the {FULL_CORPUS:,}-document search space",
    }
    DATA.write_text(json.dumps(snapshot, separators=(",", ":")))
    print(
        f"Merged {len(merged):,} distinct unmasked candidates for {QUERY_ID}",
        flush=True,
    )


if __name__ == "__main__":
    main()
