"""Publish contact-ranking figures plus their exact coordinates and pair ranks."""

import argparse
import hashlib
import io
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DESTINATION = "hf://buckets/open-athena/MarinFold/data/exp325-writeup-analysis/contact-confidence-2026-10-13"


def digest(path: Path) -> str:
    """Hash an artifact without retaining its bytes."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def main() -> None:
    """Validate and bundle preprocessing inputs, derived rows and display assets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upload",action="store_true")
    args=parser.parse_args()
    manifest=json.loads((HERE/"data/contact_confidence_analysis.json").read_text())
    for name,pin in manifest["files"].items():
        if digest(HERE/"data"/name)!=pin:
            raise ValueError(f"Changed analysis: {name}")
    for section in ["sources","code"]:
        for name,pin in manifest[section].items():
            if digest(HERE/name)!=pin:
                raise ValueError(f"Changed {section}: {name}")
    stage=HERE/"scratch/public_contact_confidence"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    for pattern in ["data/*.csv","data/*.csv.gz","data/*.json","data/*.txt","data/*.md","data/inputs/Lato-*","plots/*","site/*","*.py","*.md",
                    "pyproject.toml","uv.lock","generation/*contact_confidence*.py"]:
        for source in HERE.glob(pattern):
            if not source.is_file() or source.name.endswith("_publication.json"):
                continue
            target=stage/source.relative_to(HERE)
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,target)
    sources=pd.read_csv(HERE/"data/contact_confidence_sources.csv",keep_default_na=False)
    portable=sources.copy()
    portable["original_structure_file"]=portable.structure_file
    total=0
    archive_path=stage/"contact_confidence_inputs_and_pairs.tar.gz"
    with tarfile.open(archive_path,"w:gz",compresslevel=3) as archive:
        for index,row in sources.iterrows():
            source=Path(row.structure_file)
            if digest(source)!=row.structure_sha256:
                raise ValueError(f"Structure changed: {source}")
            name=f"scratch/contact_confidence/structures/{row.method}/{row.stem}/{source.name}"
            portable.loc[index,"structure_file"]=name
            archive.add(source,arcname=name)
            total+=source.stat().st_size
        payload=portable.to_csv(index=False).encode()
        info=tarfile.TarInfo("scratch/contact_confidence/sources.csv")
        info.size=len(payload)
        archive.addfile(info,io.BytesIO(payload))
        for name in ["scratch/contact_confidence/archived_contacts.parquet", "scratch/contact_confidence/source_pins.json",
                     "scratch/contacts/inputs/gt_universe_scored.jsonl"]:
            archive.add(HERE/name,arcname=name)
        archive.add(HERE/"scratch/contact_confidence/scored",arcname="scratch/contact_confidence/scored")
    # Keep exact scoring helpers beside the frozen inputs, including the source
    # sequence table used to map predictor coordinates to input positions.
    for relative in ["experiments/exp245_evals_foldbench_held_out_monomers/data/eval_sets.csv",
                     "experiments/exp78_evals_esmfold_contacts/pyconfind_contacts.py",
                     "experiments/exp89_evals_contacts_v1_model_on_eval_set/compute_metrics.py",
                     "marinfold/marinfold/document_structures/contacts_v1/parse.py"]:
        target=stage/"source_snapshots"/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(HERE.parents[1]/relative,target)
    instructions=("Extract contact_confidence_inputs_and_pairs.tar.gz into the experiment directory. "
        "It restores scratch/contact_confidence/{sources.csv,archived_contacts.parquet,structures,scored} "
        "and the frozen contact truth. The input sources.csv has portable relative coordinate paths, "
        "with original paths retained in original_structure_file; data/contact_confidence_sources.csv "
        "remains the original analysis provenance. From the published experiment branch, run the "
        "scorer and prepare_contact_confidence.py commands in CONTACT_CONFIDENCE.md. To repeat "
        "coordinate extraction rather than reuse validated caches, move the restored scored/ "
        "directory aside first. Rendering reads only committed analysis CSVs.\n")
    (stage/"RESTORE.txt").write_text(instructions)
    inventory={str(path.relative_to(stage)):dict(bytes=path.stat().st_size,sha256=digest(path))
               for path in sorted(stage.rglob("*")) if path.is_file()}
    publication=dict(destination=DESTINATION,files=inventory,selected_structures=len(sources),
                     original_coordinate_bytes=total,raw_archive=archive_path.name,
                     contents="Exact selected coordinates, archived degrees, mapped residue pLDDTs, all emitted eligible pair features, selected ranks and validation results")
    content=json.dumps(publication,indent=2)+"\n"
    (stage/"publication_manifest.json").write_text(content)
    (HERE/"data/contact_confidence_publication.json").write_text(content)
    print(f"{len(inventory)} files; {sum(r['bytes'] for r in inventory.values())/1e6:.1f} MB; {len(sources)} structures → {DESTINATION}",flush=True)
    if args.upload:
        subprocess.run(["hf","buckets","sync",str(stage),DESTINATION,"--quiet"],check=True)


if __name__=="__main__":
    main()
