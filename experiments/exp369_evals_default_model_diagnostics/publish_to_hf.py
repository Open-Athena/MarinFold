"""Package reproducible diagnostics and publish them to the public project bucket."""

import argparse
import gzip
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DESTINATION = "hf://buckets/open-athena/MarinFold/data/exp369-default-model-diagnostics"


def build_archive(files: dict[str, Path], archive: Path) -> dict:
    """Write a deterministic archive with a checksum manifest for every member."""
    manifest = {name: {"size": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                for name, path in sorted(files.items())}
    archive.parent.mkdir(parents=True, exist_ok=True)
    with archive.open("wb") as output, gzip.GzipFile(fileobj=output, mode="wb", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w|") as tar:
            for name, path in sorted(files.items()):
                info = tarfile.TarInfo(name)
                info.size = path.stat().st_size
                with path.open("rb") as stream:
                    tar.addfile(info, stream)
            payload = (json.dumps(manifest, indent=2) + "\n").encode()
            info = tarfile.TarInfo("manifest.json")
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"sha256": digest, "size_bytes": archive.stat().st_size, "n_files": len(files)}


def main() -> None:
    """Build a deterministic archive of public inputs, results, and source code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=Path("/data/exp369"))
    parser.add_argument("--run-dir", type=Path, action="append", default=[])
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    files = {str(path.relative_to(HERE)): path for path in HERE.rglob("*")
             if path.is_file() and not any(part.startswith(".") or part == "__pycache__" for part in path.relative_to(HERE).parts)
             and path.name != "public_artifacts.json"}
    files["inputs/spiking_plan.json"] = args.work / "spiking_plan.json"
    files.update({f"execution_bundles/{path.name}": path for path in (args.work / "execution_bundles").glob("*.tar.gz")})
    plan = json.loads((HERE / "data/afdb_inference_plan.json").read_text())
    for target in plan["targets"]:
        accession, pdb, chain = target["uniprot"], target["pdb"], target["chain"]
        for relative in (f"AF-{accession}-F1-model_v4.cif", f"AF-{accession}-F1-source-info.json",
                         f"pdb_chains/{pdb}_{chain}.cif", f"metadata/{pdb}.json"):
            files[f"inputs/{relative}"] = args.work / relative
    for run_dir in args.run_dir:
        files.update({f"predictions/{run_dir.name}/{path.relative_to(run_dir)}": path
                      for path in run_dir.rglob("*") if path.is_file() and not path.name.endswith(".partial")})
    archive = args.work / "public-results.tar.gz"
    metadata = build_archive(files, archive)
    uri = f"{DESTINATION}/{metadata['sha256']}/results.tar.gz"
    publication = {"uri": uri, **metadata, "uploaded": False}
    print(json.dumps(publication, indent=2), flush=True)
    if args.upload:
        subprocess.run(["hf", "buckets", "cp", str(archive), uri], check=True)
        publication["uploaded"] = True
        (HERE / "data/public_artifacts.json").write_text(json.dumps(publication, indent=2) + "\n")


if __name__ == "__main__":
    main()
