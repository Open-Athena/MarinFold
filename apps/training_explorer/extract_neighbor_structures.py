"""Publish compact source backbones for every displayed sequence neighbor.

Run this on CoreWeave next to the mirrored ESMFold2 source parts. The script
reads only row groups containing displayed neighbors, compacts each structure
to backbone atoms, and publishes browser-sized PDB files to the shared public
Hugging Face bucket.
"""

import argparse
import gzip
import hashlib
import json
import os
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

import gemmi
import pyarrow.parquet as pq
import s3fs
from huggingface_hub import HfFileSystem

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
ESM_SHARD_MAP = HERE / "esm_shard_map.json.gz"
LEGACY_AFDB = HERE / "legacy_afdb"
if not ESM_SHARD_MAP.is_file():
    ESM_SHARD_MAP = (
        HERE.parents[1]
        / "experiments/exp266_data_mpnn_redesign_contacts_v1/esm_shard_map.json.gz"
    )
SOURCE_PREFIX = (
    "marin-us-east-02a/protein-structure/MarinFold/"
    "exp139_esm_atlas_contacts_v1/source/structures/parts"
)
WORK_PREFIX = (
    "marin-us-east-02a/protein-structure/MarinFold/"
    "training_explorer/2026-09-22/neighbor_structures"
)
HF_BUCKET = "open-athena/MarinFold"
HF_PREFIX = "data/training-explorer/2026-09-22/neighbor-structures"
HF_RESOLVE = f"https://huggingface.co/buckets/{HF_BUCKET}/resolve/"
ESM_PARTS = 3338
BACKBONE = {"N", "CA", "C", "O"}


def canonical_source(hit: dict) -> tuple[str, str, str, int]:
    """Return source kind, arm, parent entry ID, and encoded shard number."""
    arm, local = hit["id"].split("|", 1)
    shard, _, entry = local.split("_", 2)
    entry = entry.split("#", 1)[0]
    if arm in {"afdb", "mpnn-afdb"}:
        return "afdb", arm, entry, int(shard)
    if arm in {"esm_atlas", "mpnn-esm"}:
        return "esm", arm, entry, int(shard)
    raise ValueError(f"Unknown neighbor source arm: {arm}")


def selected_sources() -> tuple[dict[str, dict], dict[str, str]]:
    """Collect unique parent backbones and map every hit ID to its parent."""
    sources: dict[str, dict] = {}
    hit_sources: dict[str, str] = {}
    for name in ("latest", "original", "eval"):
        snapshot = json.loads((DATA / f"{name}.json").read_text())
        for protein in snapshot["proteins"]:
            for hit in protein.get("neighbors", []):
                kind, arm, entry, shard = canonical_source(hit)
                key = f"{kind}:{entry}"
                source = sources.setdefault(
                    key,
                    {
                        "sourceKey": key,
                        "kind": kind,
                        "entryId": entry,
                        "parts": [],
                        "corpusShards": [],
                    },
                )
                field = "parts" if arm == "mpnn-esm" else "corpusShards"
                if shard not in source[field]:
                    source[field].append(shard)
                hit_sources[hit["id"]] = key
    with gzip.open(ESM_SHARD_MAP, "rt") as stream:
        mapping = json.load(stream)
    reverse: dict[int, list[int]] = defaultdict(list)
    for part, shards in mapping.items():
        for shard in shards:
            reverse[int(shard)].append(int(part))
    for source in sources.values():
        if source["kind"] != "esm":
            continue
        for shard in source.pop("corpusShards"):
            candidates = reverse.get(shard)
            if not candidates:
                raise ValueError(f"ESM corpus shard {shard} has no source-part mapping")
            for part in candidates:
                if part not in source["parts"]:
                    source["parts"].append(part)
    return sources, hit_sources


def compact_backbone(cif: str, entry_id: str) -> str:
    """Convert one prediction to a small PDB containing backbone atoms only."""
    structure = gemmi.read_structure_string(cif, format=gemmi.CoorFormat.Mmcif)
    structure.name = entry_id
    structure.remove_alternative_conformations()
    structure.remove_hydrogens()
    structure.remove_ligands_and_waters()
    ca_count = 0
    for model in structure:
        for chain in model:
            for residue in chain:
                for index in range(len(residue) - 1, -1, -1):
                    if residue[index].name not in BACKBONE:
                        del residue[index]
                if any(atom.name == "CA" for atom in residue):
                    ca_count += 1
    if ca_count < 2:
        raise ValueError(f"{entry_id}: compact structure has {ca_count} C-alpha atoms")
    return "".join(
        f"{line.rstrip()}\n" for line in structure.make_pdb_string().splitlines()
    )


def afdb_cif(entry_id: str) -> str:
    """Fetch the current public AlphaFold DB model for an AFDB entity ID."""
    accession = entry_id.split("-")[1]
    request = urllib.request.Request(
        f"https://alphafold.ebi.ac.uk/api/prediction/{accession}",
        headers={"User-Agent": "MarinFold-training-explorer/1.1"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        predictions = json.load(response)
    match = next(
        (
            prediction
            for prediction in predictions
            if prediction["modelEntityId"] == entry_id
        ),
        None,
    )
    if match is None:
        raise ValueError(f"No AlphaFold DB model for {entry_id}")
    with urllib.request.urlopen(match["cifUrl"], timeout=120) as response:
        return response.read().decode()


class EsmSource:
    """Range-read the in-region ESMFold2 source mirror with cached footers."""

    def __init__(self, fs: s3fs.S3FileSystem) -> None:
        self.fs = fs
        self._metadata: dict[int, pq.FileMetaData] = {}

    def path(self, part: int) -> str:
        if not 0 <= part < ESM_PARTS:
            raise ValueError(f"Invalid ESM source part {part}")
        return f"{SOURCE_PREFIX}/part_{part:05d}.parquet"

    def metadata(self, part: int) -> pq.FileMetaData:
        cached = self._metadata.get(part)
        if cached is not None:
            return cached
        with self.fs.open(self.path(part), "rb") as stream:
            metadata = pq.read_metadata(stream)
        self._metadata[part] = metadata
        return metadata

    def bounds(self, part: int) -> tuple[str, str]:
        metadata = self.metadata(part)
        column = metadata.schema.names.index("entry_id")
        first = metadata.row_group(0).column(column).statistics.min
        last = (
            metadata.row_group(metadata.num_row_groups - 1)
            .column(column)
            .statistics.max
        )
        return first, last

    def group_in_part(self, part: int, entry_id: str) -> int | None:
        metadata = self.metadata(part)
        column = metadata.schema.names.index("entry_id")
        candidates = []
        for group in range(metadata.num_row_groups):
            statistics = metadata.row_group(group).column(column).statistics
            if statistics and statistics.min <= entry_id <= statistics.max:
                candidates.append(group)
        if not candidates:
            return None
        with self.fs.open(self.path(part), "rb") as stream:
            parquet = pq.ParquetFile(stream)
            for group in candidates:
                ids = parquet.read_row_group(group, columns=["entry_id"]).column(0)
                if entry_id in ids.to_pylist():
                    return group
        return None

    def locate(self, source: dict) -> tuple[int, int]:
        entry_id = source["entryId"]
        for part in source["parts"]:
            group = self.group_in_part(part, entry_id)
            if group is not None:
                return part, group
        low, high = 0, ESM_PARTS - 1
        while low <= high:
            part = (low + high) // 2
            first, last = self.bounds(part)
            if entry_id < first:
                high = part - 1
            elif entry_id > last:
                low = part + 1
            else:
                group = self.group_in_part(part, entry_id)
                if group is not None:
                    return part, group
                break
        raise ValueError(f"No ESMFold2 source structure for {entry_id}")

    def fetch_group(
        self, part: int, group: int, sources: list[dict]
    ) -> list[tuple[str, str]]:
        with self.fs.open(self.path(part), "rb") as stream:
            table = pq.ParquetFile(stream).read_row_group(
                group, columns=["entry_id", "cif_content"]
            )
        rows = {row["entry_id"]: row["cif_content"] for row in table.to_pylist()}
        output = []
        for source in sources:
            entry_id = source["entryId"]
            cif = rows.get(entry_id)
            if cif is None:
                raise ValueError(f"Source group {part}/{group} lacks {entry_id}")
            output.append((source["sourceKey"], compact_backbone(cif, entry_id)))
        return output


def filename(source_key: str) -> str:
    """Return a stable content namespace filename for one source backbone."""
    return hashlib.sha256(source_key.encode()).hexdigest()[:20] + ".pdb"


def write_structure(root: Path, source_key: str, pdb: str) -> dict:
    """Write one compact PDB and return its checked manifest row."""
    path = root / filename(source_key)
    path.write_text(pdb)
    return record_for_path(source_key, path)


def record_for_path(source_key: str, path: Path) -> dict:
    """Return the checked manifest row for an existing compact PDB."""
    content = path.read_bytes()
    structure = gemmi.read_structure_string(
        content.decode(), format=gemmi.CoorFormat.Pdb
    )
    ca_count = sum(
        atom.name == "CA"
        for model in structure
        for chain in model
        for residue in chain
        for atom in residue
    )
    if ca_count < 2:
        raise ValueError(f"{source_key}: persisted PDB has {ca_count} C-alpha atoms")
    return {
        "sourceKey": source_key,
        "file": path.name,
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "url": HF_RESOLVE + quote(f"{HF_PREFIX}/{path.name}", safe=""),
    }


def upload_with_retry(hffs: HfFileSystem, path: Path) -> None:
    """Upload one artifact, retrying only transient remote failures."""
    destination = f"buckets/{HF_BUCKET}/{HF_PREFIX}/{path.name}"
    error = None
    for attempt in range(9):
        try:
            with path.open("rb") as source, hffs.open(destination, "wb") as remote:
                while chunk := source.read(8 << 20):
                    remote.write(chunk)
            return
        except Exception as exc:  # noqa: BLE001 - retry boundary re-raises failures
            error = exc
            if attempt == 8:
                break
            time.sleep(min(5 * (2**attempt), 120))
    raise RuntimeError(f"Failed to upload {path.name}") from error


def main() -> None:
    """Extract, verify, persist, and publish all selected neighbor backbones."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--upload-workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("/tmp/neighbor-structures"))
    parser.add_argument(
        "--publish", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()

    sources, hit_sources = selected_sources()
    ordered = [sources[key] for key in sorted(sources)]
    if args.limit is not None:
        ordered = [
            source
            for kind in ("afdb", "esm")
            for source in [item for item in ordered if item["kind"] == kind][
                : args.limit
            ]
        ]
        keep = {source["sourceKey"] for source in ordered}
        hit_sources = {hit: key for hit, key in hit_sources.items() if key in keep}
    args.output.mkdir(parents=True, exist_ok=True)
    fs = s3fs.S3FileSystem(anon=False)
    existing = (
        {Path(path).name for path in fs.ls(WORK_PREFIX)}
        if fs.exists(WORK_PREFIX)
        else set()
    )
    records: dict[str, dict] = {}
    pending = []
    for source in ordered:
        path = args.output / filename(source["sourceKey"])
        remote = f"{WORK_PREFIX}/{path.name}"
        if not path.is_file() and path.name in existing:
            fs.get_file(remote, str(path))
        if path.is_file():
            records[source["sourceKey"]] = record_for_path(source["sourceKey"], path)
        else:
            pending.append(source)
    if records:
        print(f"Restored {len(records)} persisted backbones", flush=True)

    esm_source = EsmSource(fs)
    esm = [source for source in pending if source["kind"] == "esm"]
    afdb = [source for source in pending if source["kind"] == "afdb"]
    print(
        f"Selected {len(ordered)} backbones; pending {len(esm)} ESMFold2, "
        f"{len(afdb)} AFDB",
        flush=True,
    )

    checkpoint_futures = []
    with ThreadPoolExecutor(max_workers=args.workers) as checkpoint_pool:
        if esm:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                locations = list(pool.map(esm_source.locate, esm))
            groups: dict[tuple[int, int], list[dict]] = defaultdict(list)
            for source, location in zip(esm, locations, strict=True):
                groups[location].append(source)
            print(f"Reading {len(groups)} selected ESMFold2 row groups", flush=True)
            keys = sorted(groups)
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                results = pool.map(
                    lambda key: esm_source.fetch_group(*key, groups[key]), keys
                )
                for index, structures in enumerate(results, 1):
                    for source_key, pdb in structures:
                        record = write_structure(args.output, source_key, pdb)
                        records[source_key] = record
                        checkpoint_futures.append(
                            checkpoint_pool.submit(
                                fs.put_file,
                                str(args.output / record["file"]),
                                f"{WORK_PREFIX}/{record['file']}",
                            )
                        )
                    if index % 100 == 0 or index == len(keys):
                        print(f"ESMFold2 groups {index}/{len(keys)}", flush=True)

        def fetch_afdb(source: dict) -> tuple[str, str]:
            legacy = LEGACY_AFDB / f"{source['entryId']}.pdb"
            if legacy.is_file():
                return source["sourceKey"], legacy.read_text()
            return source["sourceKey"], compact_backbone(
                afdb_cif(source["entryId"]), source["entryId"]
            )

        if afdb:
            with ThreadPoolExecutor(max_workers=min(8, args.workers)) as pool:
                results = pool.map(fetch_afdb, afdb)
                for index, (source_key, pdb) in enumerate(results, 1):
                    record = write_structure(args.output, source_key, pdb)
                    records[source_key] = record
                    checkpoint_futures.append(
                        checkpoint_pool.submit(
                            fs.put_file,
                            str(args.output / record["file"]),
                            f"{WORK_PREFIX}/{record['file']}",
                        )
                    )
                    if index % 50 == 0 or index == len(afdb):
                        print(f"AFDB structures {index}/{len(afdb)}", flush=True)

        for future in checkpoint_futures:
            future.result()

    if len(records) != len(ordered):
        raise ValueError(f"Expected {len(ordered)} structures, wrote {len(records)}")
    manifest = {
        "sourcePrefix": SOURCE_PREFIX,
        "workingPrefix": f"s3://{WORK_PREFIX}",
        "publicPrefix": f"hf://buckets/{HF_BUCKET}/{HF_PREFIX}",
        "sources": [records[key] for key in sorted(records)],
        "hitSources": hit_sources,
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, separators=(",", ":")))

    fs.put_file(str(manifest_path), f"{WORK_PREFIX}/{manifest_path.name}")
    print(
        f"Persisted {len(records)} structures and manifest to s3://{WORK_PREFIX}",
        flush=True,
    )

    if args.publish:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise ValueError("HF_TOKEN is required with --publish")
        hffs = HfFileSystem(token=token)
        paths = sorted(args.output / row["file"] for row in records.values())
        paths.append(manifest_path)
        with ThreadPoolExecutor(max_workers=args.upload_workers) as pool:
            list(pool.map(lambda path: upload_with_retry(hffs, path), paths))
        anonymous = HfFileSystem(token=False)
        prefix = f"buckets/{HF_BUCKET}/{HF_PREFIX}"
        remote = {
            Path(item["name"]).name: int(item["size"])
            for item in anonymous.ls(prefix, detail=True)
        }
        local = {path.name: path.stat().st_size for path in paths}
        if remote != local:
            raise ValueError(
                f"Public bucket mismatch: {len(remote)} remote != {len(local)} local"
            )
        print(
            f"Published and anonymously verified {len(records)} structures", flush=True
        )


if __name__ == "__main__":
    main()
