"""Merge complete CoreWeave MPNN and local native searches for 216 queries."""

import configparser
import csv
import gzip
import json
import re
from collections import defaultdict
from pathlib import Path

import s3fs


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
PREFIX = "marin-us-east-02a/MarinFold/training_explorer/2026-09-22"
SHARDS = 16
EXPECTED = {"mpnn-afdb": 31_702_680, "mpnn-esm": 130_872_044}
FULL_CORPUS = 232_090_905
ORIGINAL_CORPUS = 70_889_604


def self_target(protein: dict) -> str | None:
    """Return the exact target key of a sampled MPNN document."""
    source = protein["source"]
    if not source.startswith("MPNN"):
        return None
    filename = protein["sourceFile"].rsplit("/", 1)[-1]
    if source == "MPNN AFDB":
        match = re.match(r"documents-backbones-(\d+)-of-", filename)
        arm = "mpnn-afdb"
    else:
        match = re.match(r"documents-(\d+)-of-", filename)
        arm = "mpnn-esm"
    if match is None:
        raise ValueError(f"Unrecognized source file {filename}")
    return f"{arm}|{int(match.group(1))}_{protein['sourceRow']}_{protein['entryId']}#{protein['designIndex']}"


def make_hit(values: list[str], shard_count: int) -> tuple[str, dict]:
    """Decode one MMseqs result and normalize its search-space E-value."""
    query, target, identity, _, qcov, tcov, evalue, bits, tlen = values
    arm, local = target.split("|", 1)
    _, _, entry = local.split("_", 2)
    return query, {
        "id": target,
        "label": entry,
        "source": "MPNN AFDB" if arm == "mpnn-afdb" else "MPNN ESM-Atlas",
        "length": int(tlen),
        "identity": float(identity),
        "queryCoverage": float(qcov),
        "targetCoverage": float(tcov),
        "evalue": float(evalue) * FULL_CORPUS / shard_count,
        "bitscore": float(bits),
    }


def main() -> None:
    """Require all shard manifests, then attach global top-10 hits."""
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
        if manifest["shard"] != index or manifest["shards"] != SHARDS:
            raise ValueError(f"Bad search manifest: {path}")
        manifests.append(manifest)
    for arm, expected in EXPECTED.items():
        observed = sum(manifest["sourceCounts"][arm] for manifest in manifests)
        if observed != expected:
            raise ValueError(f"{arm}: indexed {observed:,}, expected {expected:,}")
    print(
        f"Verified all {sum(m['count'] for m in manifests):,} MPNN sequences",
        flush=True,
    )

    mpnn: dict[str, list[dict]] = defaultdict(list)
    for index, manifest in enumerate(manifests):
        path = f"{PREFIX}/hits-{index:02d}-of-{SHARDS:02d}.tsv.gz"
        shard_hits: dict[str, list[dict]] = defaultdict(list)
        with fs.open(path, "rb") as remote, gzip.open(remote, "rt") as stream:
            for values in csv.reader(stream, delimiter="\t"):
                if len(values) != 9:
                    raise ValueError(f"Malformed alignment in {path}: {values[:2]}")
                query, hit = make_hit(values, manifest["count"])
                shard_hits[query].append(hit)
        for query, hits in shard_hits.items():
            hits.sort(key=lambda hit: (-hit["bitscore"], hit["evalue"], hit["id"]))
            mpnn[query].extend(hits[:30])
        print(f"Merged shard {index + 1}/{SHARDS}", flush=True)

    native = json.loads((DATA / "native_hits.json").read_text())
    for name in ("latest", "eval"):
        path = DATA / f"{name}.json"
        snapshot = json.loads(path.read_text())
        for protein in snapshot["proteins"]:
            own = self_target(protein) if name == "latest" else None
            native_hits = [
                {**hit, "evalue": hit["evalue"] * FULL_CORPUS / ORIGINAL_CORPUS}
                for hit in native[protein["id"]]
            ]
            hits = [
                hit for hit in native_hits + mpnn[protein["id"]] if hit["id"] != own
            ]
            hits.sort(key=lambda hit: (-hit["bitscore"], hit["evalue"], hit["id"]))
            protein["neighbors"] = hits[:10]
            protein["reportedAtE10"] = bool(protein["neighbors"])
        snapshot["neighborsComplete"] = True
        snapshot["search"] = {
            "tool": "MMseqs2",
            "rank": "local alignment bit score",
            "sensitivity": 7.5,
            "evalueLimitPerShard": 10,
            "mpnnShards": SHARDS,
            "mpnnRowsVerified": sum(m["count"] for m in manifests),
            "nativeRows": 3_963_003 + 65_553_178,
            "nativeMethod": "exp213 full DB, exact exp225 drop-list exclusion",
            "evalueDisplay": "scaled approximately to the 232,090,905-row search space",
        }
        path.write_text(json.dumps(snapshot, separators=(",", ":")))
        counts = [len(p["neighbors"]) for p in snapshot["proteins"]]
        print(
            f"{name}: {len(counts)} queries, {sum(c == 10 for c in counts)} have 10 neighbors; minimum {min(counts)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
