"""Split downloaded dimers into one structure file per distinct chain.

PINDER clusters at the level of the individual chain, not the complex: it runs
Foldseek chain-against-chain, builds a graph over chains, finds communities, and
only then labels a complex by the pair of communities its two chains fall into.
Our fetch produced whole two-chain mmCIFs, so this is the step in between.

The unit of identity here is the UniProt accession. A homodimer's two chains are
the same accession and contribute one entry; a heterodimer contributes two. Over
the 881,010 val-eligible complexes that is 858,360 distinct chains, less those
reachable only through a complex the fetch could not retrieve.

Each accession is emitted once, from the lexicographically smallest complex that
contains it, so a rerun with the same inputs produces byte-identical output and a
partial run can be resumed by simply skipping accessions already on disk.
"""

import argparse
import csv
import gzip
import io
import json
import multiprocessing
import os
import tarfile
import time

import pyarrow.fs
from pathlib import Path

# Zero-based positions in an _atom_site row. The AFDB writes these columns in a
# fixed order (verified against the release), so a positional read is safe and is
# roughly twenty times faster than a real mmCIF parser at this volume. Field 6 is
# label_asym_id; do not use field 7, which is label_entity_id and is legitimately
# "1" for both chains of a homodimer.
ASYM_FIELD = 6

ASSIGNMENT = "dimer_split_assignment_id30_cov50.csv.gz"
MISSING = "fetch_missing_404.csv"


def load_missing(path: Path | None) -> set[str]:
    """Model ids the fetch could not retrieve, as written by curate_fetch_structures."""
    if path is None:
        return set()
    with open(path) as handle:
        return {row["model_entity_id"] for row in csv.DictReader(handle)}


def choose_sources(path: Path, split: str, missing: set[str]) -> dict[str, tuple[str, str]]:
    """Map each accession to the (complex, chain) it will be extracted from.

    Returns ``{accession: (model_entity_id, "A" | "B")}``. Chain A carries
    ``uniprot_1`` and chain B carries ``uniprot_2``; for a homodimer the two are
    equal and only chain A is ever requested.

    Models in ``missing`` are not candidates. 762 of the 881,010 val complexes
    404'd during the fetch, and an accession whose smallest complex is one of them
    falls through to its next-smallest instead of going unextractable.
    """
    opener = gzip.open if path.suffix == ".gz" else open
    sources: dict[str, tuple[str, str]] = {}
    with opener(path, "rt") as handle:
        for row in csv.DictReader(handle):
            if row["split"] != split:
                continue
            model = row["model_entity_id"]
            if model in missing:
                continue
            for accession, chain in ((row["uniprot_1"], "A"), (row["uniprot_2"], "B")):
                prior = sources.get(accession)
                if prior is None or model < prior[0]:
                    sources[accession] = (model, chain)
    return sources


def split_chains(text: str, wanted: set[str]) -> dict[str, str]:
    """Pull the requested chains out of one dimer mmCIF.

    Emits a minimal file per chain: the ``data_`` header, the ``_atom_site`` loop
    header, and that chain's ATOM records. Foldseek reads only the coordinates and
    the residue names, so carrying the entity and reference blocks across would
    cost bytes without changing any alignment.
    """
    lines = text.splitlines()
    header = next(line for line in lines if line.startswith("data_"))
    loop = [line for line in lines if line.startswith("_atom_site.")]
    if not loop:
        raise ValueError(f"{header}: no _atom_site loop")

    rows: dict[str, list[str]] = {chain: [] for chain in wanted}
    for line in lines:
        if not line.startswith("ATOM"):
            continue
        chain = line.split()[ASYM_FIELD]
        if chain in rows:
            rows[chain].append(line)

    out: dict[str, str] = {}
    for chain, atoms in rows.items():
        if not atoms:
            raise ValueError(f"{header}: chain {chain} has no ATOM records")
        out[chain] = "\n".join([header, "loop_", *loop, *atoms, "#", ""])
    return out


def iter_tar(stream, name: str, wanted: set[str] | None = None):
    """Yield (model_entity_id, mmCIF text) from one open tar stream.

    Mode "r|" is the streaming reader: it never seeks, so the same code path works
    on a local file and on a tar already in memory.

    `wanted` skips members before they are decompressed. A caller that needs a
    small subset of a shard pays about 12 s of gzip per shard otherwise, which is
    most of the CPU in a pass that only wants a few dozen of the 2,000 members.
    """
    with tarfile.open(fileobj=stream, mode="r|") as tar:
        for member in tar:
            if not member.name.endswith(".cif.gz"):
                continue
            model = member.name.removesuffix(".cif.gz")
            if wanted is not None and model not in wanted:
                continue
            handle = tar.extractfile(member)
            if handle is None:
                raise ValueError(f"{name}: {member.name} is not a regular file")
            yield model, gzip.decompress(handle.read()).decode()


def list_shards(source: str, shard_stride: int) -> list[str]:
    """List the tar shards under a local directory or an s3:// prefix.

    ``shard_stride`` takes every Nth tar rather than the first N. Shards are
    contiguous bands of a sorted ID list, and the survey found AFDB ID windows are
    organism-blocked, so a subsample must stride rather than truncate.
    """
    if source.startswith("s3://"):
        path = source[len("s3://"):].rstrip("/")
        # pyarrow, never boto3: boto3 wedges ephemeral EC2 workers about a minute
        # in. resolve_s3_region avoids a cross-region read of every shard.
        fs = pyarrow.fs.S3FileSystem(region=pyarrow.fs.resolve_s3_region(path.split("/")[0]))
        listing = fs.get_file_info(pyarrow.fs.FileSelector(path, recursive=False))
        return sorted(f.path for f in listing if f.path.endswith(".tar"))[::shard_stride]
    return [str(p) for p in sorted(Path(source).glob("*.tar"))[::shard_stride]]


def read_shard(name: str, fs) -> bytes:
    """Read one tar shard whole.

    Whole-object, never a stream: tarfile pulls 10240-byte records, and pyarrow's
    open_input_stream turns every one of those into its own round trip. Measured
    on shard_000_00000, 0.08 MB/s that way against 12.06 MB/s here, a difference
    that showed up in production as an extraction pass on track to take 74 hours.
    Shards are ~250 MB, so holding one in memory costs nothing.
    """
    if fs is None:
        return Path(name).read_bytes()
    with fs.open_input_file(name) as handle:
        return handle.read()


# Per-worker state. Workers are spawned rather than forked, so each one builds its
# own assignment table and its own S3 filesystem: a pyarrow S3FileSystem created
# before a fork is not safe to use in the child.
_STATE: dict = {}


def init_worker(assignment: str, split: str, missing: str, out: str, source: str) -> None:
    sources = choose_sources(Path(assignment), split, load_missing(Path(missing) if missing else None))
    # A list of (chain, accession), never a dict keyed by chain: one model id can
    # carry two different accessions on the same chain letter, because a model id
    # can appear in the assignment as two different complexes. AF-0000000208803286
    # is both the homodimer A0A3Q1MTM2/A0A3Q1MTM2 and the heterodimer
    # A0A044RNL9/A0A044S3F6, and it is the only such id in the val assignment. A
    # chain-keyed dict dropped the first of those, which is the one chain the first
    # full run came up short.
    by_model: dict[str, list[tuple[str, str]]] = {}
    for accession, (model, chain) in sources.items():
        by_model.setdefault(model, []).append((chain, accession))
    _STATE["by_model"] = by_model
    _STATE["out"] = Path(out)
    _STATE["fs"] = (pyarrow.fs.S3FileSystem(
        region=pyarrow.fs.resolve_s3_region(source[len("s3://"):].split("/")[0]))
        if source.startswith("s3://") else None)


def extract_shard(name: str) -> tuple[str, int, int]:
    """Extract every wanted chain out of one shard. Returns (name, written, skipped)."""
    by_model = _STATE["by_model"]
    out = _STATE["out"]
    written = skipped = 0
    for model, text in iter_tar(io.BytesIO(read_shard(name, _STATE["fs"])), name):
        wanted = by_model.get(model)
        if not wanted:
            continue
        need = [(chain, acc) for chain, acc in wanted
                if not (out / f"{acc}.cif.gz").exists()]
        skipped += len(wanted) - len(need)
        if not need:
            continue
        chains = split_chains(text, {chain for chain, _ in need})
        for chain, acc in need:
            chain_text = chains.get(chain)
            if chain_text is None:
                # The file does not hold the chain the assignment promised. Not
                # raised: chains_absent.csv names every accession that ends up
                # missing and --max-absent is the threshold that fails the run.
                continue
            # Level 6, not the default 9: measured on this data, level 9 costs
            # 25.8 ms per structure against 6.1 and produces a marginally larger
            # file. mtime=0 keeps the output byte-identical across reruns.
            (out / f"{acc}.cif.gz").write_bytes(
                gzip.compress(chain_text.encode(), compresslevel=6, mtime=0))
            written += 1
    return name, written, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True,
                        help="a directory of .tar shards or an s3:// prefix holding them")
    parser.add_argument("--shard-stride", type=int, default=1,
                        help="take every Nth shard; shards are contiguous ID bands and "
                             "AFDB ID windows are organism-blocked, so a subsample must "
                             "stride rather than truncate")
    parser.add_argument("--assignment", type=Path,
                        default=Path("data/curation") / ASSIGNMENT)
    parser.add_argument("--missing", type=Path,
                        default=Path("data/curation") / MISSING,
                        help="models the fetch could not retrieve; accessions sourced "
                             "only from these are not expected on disk")
    parser.add_argument("--split", default="val")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--max-absent", type=int, default=10,
                        help="tolerated shortfall. Reconciling the fetch against the "
                             "assignment accounts for every val complex but one, so a "
                             "residue of a few accessions is known and a systematic "
                             "failure would be thousands. Every absent accession is "
                             "written to <out>_absent.csv either way")
    parser.add_argument("--max-shards", type=int, default=0,
                        help="stop after this many shards (0 = all); skips the "
                             "completeness check, since a partial run is short by design")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    missing = load_missing(args.missing)
    print(f"{len(missing):,} models excluded as unfetchable ({args.missing})")
    sources = choose_sources(args.assignment, args.split, missing)
    print(f"{len(sources):,} distinct chains expected in split {args.split!r}")

    shards = list_shards(args.source, args.shard_stride)
    if args.max_shards:
        shards = shards[:args.max_shards]
    print(f"{len(shards)} tar shards from {args.source} "
          f"(stride {args.shard_stride}), {args.workers} workers", flush=True)

    # Spawn, not fork: a pyarrow S3FileSystem built in the parent to list the
    # shards cannot be used across a fork. Each worker rebuilds the assignment
    # table itself, which costs a few seconds once and avoids pickling 858,360
    # entries to every process.
    ctx = multiprocessing.get_context("spawn")
    written = skipped = 0
    started = time.time()
    with ctx.Pool(args.workers, initializer=init_worker,
                  initargs=(str(args.assignment), args.split,
                            str(args.missing) if args.missing else "",
                            str(args.out), args.source)) as pool:
        for done, (name, shard_written, shard_skipped) in enumerate(
                pool.imap_unordered(extract_shard, shards), start=1):
            written += shard_written
            skipped += shard_skipped
            # A run this long is unobservable without a progress line: the first
            # attempt spent 8 hours in here with nothing in the log but heartbeats.
            print(f"[{time.time() - started:8.0f}s] {done}/{len(shards)} "
                  f"{name.rsplit('/', 1)[-1]} +{shard_written} "
                  f"({written:,} written, {skipped:,} duplicate)", flush=True)

    elapsed = time.time() - started
    on_disk = sorted(p.stem.removesuffix(".cif") for p in args.out.glob("*.cif.gz"))
    total = sum(p.stat().st_size for p in args.out.glob("*.cif.gz"))
    summary = {
        "split": args.split,
        "source": args.source,
        "shard_stride": args.shard_stride,
        "n_shards": len(shards),
        "n_workers": args.workers,
        "n_models_unfetchable": len(missing),
        "n_chains_expected": len(sources),
        "n_written": written,
        "n_duplicate_models": skipped,
        "n_on_disk": len(on_disk),
        "bytes_on_disk": total,
        "mean_kb": round(total / max(len(on_disk), 1) / 1024, 1),
        "elapsed_seconds": round(elapsed, 1),
    }
    print(json.dumps(summary, indent=2))
    (args.out.parent / f"{args.out.name}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")

    # The completeness check lives here rather than in the caller, so it can name
    # what is missing instead of reporting a count that has to be re-derived. The
    # first EC2 run died on a hard-coded expectation that did not subtract the 762
    # unfetchable models.
    if args.max_shards:
        return
    absent = sorted(set(sources) - set(on_disk))
    absent_path = args.out.parent / f"{args.out.name}_absent.csv"
    with open(absent_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["accession", "source_model_entity_id", "source_chain"])
        for accession in absent:
            writer.writerow([accession, *sources[accession]])
    print(f"{len(absent):,} expected accessions absent, listed in {absent_path}")
    if len(absent) > args.max_absent:
        raise SystemExit(f"{len(absent):,} absent exceeds --max-absent "
                         f"{args.max_absent}; see {absent_path}")


if __name__ == "__main__":
    main()
