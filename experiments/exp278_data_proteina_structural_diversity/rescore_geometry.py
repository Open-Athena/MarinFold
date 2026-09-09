"""Apply cis-aware geometry checks to saved predictions without GPU inference.

Original outputs stay immutable. Unchanged documents are copied byte-for-byte;
only newly accepted structures require contact serialization. Every changed
decision is recorded, including the original trans-only geometry verdict.
"""

import argparse
import io
import json
import time
from pathlib import Path

import gemmi
import pyarrow as pa
import pyarrow.parquet as pq
from marinfold.document_structures.contacts_v1 import (
    GenerationConfig,
    generate_document,
)
from marinfold.document_structures.contacts_v1.read import sequence_from_document
from pyconfind import load_library
from pyconfind.data import cached_rotamer_library

from analyze_screen import write_csv
from launch import storage_filesystem
from quality import backbone_geometry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    fs = storage_filesystem("cw-rno2a")
    source, target = args.input.removeprefix("s3://"), args.output.removeprefix("s3://")
    if source == target or fs.exists(target + "/complete.json"):
        raise ValueError("Use a new output prefix for the geometry correction")
    if not fs.exists(source + "/complete.json"):
        raise ValueError("Wait for refolding to complete before rescoring")
    rotamers = load_library(cached_rotamer_library())
    began = time.perf_counter()
    audit = []
    for path in sorted(fs.glob(source + "/candidates/*.parquet")):
        part_id = Path(path).stem
        old_documents = {}
        old_doc_path = source + f"/documents-provisional/{part_id}.parquet"
        if fs.exists(old_doc_path):
            with fs.open(old_doc_path, "rb") as handle:
                old_documents = {
                    row["entry_id"]: row for row in pq.read_table(handle).to_pylist()
                }
        with fs.open(path, "rb") as handle:
            records = pq.read_table(handle).to_pylist()
        revised, documents = [], []
        for row in records:
            start = time.perf_counter()
            structure = gemmi.read_pdb_string(row["pdb_content"])
            geometry = backbone_geometry(structure)
            passed = bool(
                row["scrmsd"] <= 2
                and row["plddt"] >= 70
                and geometry["ca_geometry_pass"]
            )
            revised.append(
                {
                    **row,
                    **geometry,
                    "quality_pass": passed,
                    "previous_quality_pass": row["quality_pass"],
                    "geometry_version": "cis-proline-v2",
                }
            )
            if passed:
                if row["quality_pass"]:
                    document = old_documents[row["stem"]]
                else:
                    result = generate_document(
                        structure,
                        entry_id=row["stem"],
                        config=GenerationConfig(),
                        rotamer_library=rotamers,
                    )
                    if result is None:
                        raise ValueError(
                            f"Serializer rejected corrected candidate {row['stem']}"
                        )
                    document = {
                        **result.metadata_row(),
                        "sequence": row["sequence"],
                        "source": "proteina-exp278",
                        "label_geometry": "esmfold_refold",
                        "release_status": "provisional",
                    }
                if (
                    sequence_from_document(
                        document["document"],
                        len(row["sequence"]),
                        document["n_term_index"],
                    )
                    != row["sequence"]
                ):
                    raise ValueError(
                        "Corrected document sequence failed round-trip verification"
                    )
                documents.append(document)
            audit.append(
                {
                    "stem": row["stem"],
                    "length": len(row["sequence"]),
                    "previous_quality_pass": row["quality_pass"],
                    "quality_pass": passed,
                    "previous_chain_breaks": row["ca_chain_breaks"],
                    "ca_chain_breaks": geometry["ca_chain_breaks"],
                    "cis_proline_bonds": geometry["cis_proline_bonds"],
                    "elapsed_cpu_seconds": time.perf_counter() - start,
                }
            )
        for kind, rows in [
            ("candidates", revised),
            ("documents-provisional", documents),
        ]:
            if rows:
                buffer = io.BytesIO()
                pq.write_table(pa.Table.from_pylist(rows), buffer, compression="zstd")
                fs.pipe_file(f"{target}/{kind}/{part_id}.parquet", buffer.getvalue())
    fs.cp_file(source + "/timings.csv", target + "/timings.csv")
    summary = {
        "input": args.input,
        "output": args.output,
        "geometry_version": "cis-proline-v2",
        "candidates": len(audit),
        "quality_pass": sum(row["quality_pass"] for row in audit),
        "previous_quality_pass": sum(row["previous_quality_pass"] for row in audit),
        "changed": sum(
            row["quality_pass"] != row["previous_quality_pass"] for row in audit
        ),
        "elapsed_cpu_wall_seconds": time.perf_counter() - began,
        "gpu_inference_repeated": False,
    }
    write_csv(args.report / "geometry-correction.csv", audit)
    (args.report / "geometry-correction.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    fs.pipe_file(target + "/complete.json", json.dumps(summary).encode())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
