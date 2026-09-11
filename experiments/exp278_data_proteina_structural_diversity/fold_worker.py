"""Design sequences, refold them, and serialize quality-passing pilot candidates.

Outputs are explicitly provisional: decontamination and cross-shard diversity
selection happen subsequently, before any training corpus is published.
"""

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import random
import socket
import time
from datetime import datetime, timezone
from pathlib import Path

import fsspec
import gemmi
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from marinfold.document_structures.contacts_v1 import (
    GenerationConfig,
    generate_document,
)
from marinfold.document_structures.contacts_v1.read import sequence_from_document
from protein_mpnn_utils import ProteinMPNN
from pyconfind import load_library
from pyconfind.data import cached_rotamer_library
from transformers import AutoTokenizer, EsmForProteinFolding

from prepare_assets import download
from quality import aligned_rmsd, backbone_geometry, confidence_percent
from scale_common import paused, write_json

MPNN_REVISION = "8907e6671bfbfc92303b5f79c4b5e6ce47cdef57"
ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"


def put_bytes(uri: str, content: bytes) -> None:
    """Persist completed output through the injected filesystem configuration."""
    with fsspec.open(uri, "wb") as handle:
        handle.write(content)


def design_sequences(
    model: ProteinMPNN, coordinates: np.ndarray, seed: int
) -> list[str]:
    """Sample one sequence for each complete single-chain C-alpha backbone."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    batch_size, length, _ = coordinates.shape
    x = torch.as_tensor(coordinates, device="cuda", dtype=torch.float32)
    mask = torch.ones((batch_size, length), device="cuda")
    indices = torch.arange(length, device="cuda")[None, :].expand(batch_size, -1)
    with torch.inference_mode():
        result = model.sample(
            x,
            torch.randn_like(mask),
            torch.zeros_like(mask, dtype=torch.long),
            mask,
            torch.ones_like(mask, dtype=torch.long),
            indices,
            mask=mask,
            temperature=0.1,
            omit_AAs_np=np.array([aa == "X" for aa in ALPHABET], dtype=np.float32),
            bias_AAs_np=np.zeros(21, dtype=np.float32),
            chain_M_pos=mask,
            bias_by_res=torch.zeros((batch_size, length, 21), device="cuda"),
        )
    return [
        "".join(ALPHABET[index] for index in row) for row in result["S"].cpu().tolist()
    ]


def structure_coordinates(structure: gemmi.Structure, sequence: str) -> np.ndarray:
    """Require a complete, single-chain backbone matching the designed sequence."""
    if len(structure) != 1 or len(structure[0]) != 1:
        raise ValueError("ESMFold output must contain exactly one model and chain")
    residues = list(structure[0][0])
    if len(residues) != len(sequence):
        raise ValueError("ESMFold changed the sequence length")
    actual = "".join(
        gemmi.find_tabulated_residue(residue.name).one_letter_code
        for residue in residues
    )
    if actual != sequence:
        raise ValueError("ESMFold output sequence differs from ProteinMPNN sequence")
    coords = []
    for residue in residues:
        for name in ("N", "CA", "C", "O"):
            if residue.find_atom(name, "*") is None:
                raise ValueError(f"Incomplete ESMFold backbone: missing {name}")
        coords.append(list(residue["CA"][0].pos))
    return np.asarray(coords)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", default=[])
    parser.add_argument("--manifest", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--esm-revision", required=True)
    parser.add_argument("--seed", type=int, default=278)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--control", default="")
    args = parser.parse_args()
    prefixes = list(args.input)
    for manifest in args.manifest:
        with fsspec.open(manifest, "rt") as handle:
            prefixes.extend(case["output"] for case in json.load(handle))
    if not prefixes:
        parser.error("Provide at least one input prefix or case manifest")
    if not 0 <= args.shard_index < args.shard_count:
        parser.error("Shard index must be between zero and shard count minus one")
    prefixes = prefixes[args.shard_index :: args.shard_count]
    for prefix in prefixes:
        source_fs, source_marker = fsspec.core.url_to_fs(prefix + "/complete.json")
        if not source_fs.exists(source_marker):
            raise ValueError(f"Generation is incomplete under {prefix}")
    configuration = {**vars(args), "resolved_inputs": prefixes}
    output_fs, configuration_path = fsspec.core.url_to_fs(
        args.output + "/configuration.json"
    )
    if output_fs.exists(configuration_path):
        with output_fs.open(configuration_path, "rt") as handle:
            if json.load(handle) != configuration:
                raise ValueError(
                    "Output prefix already belongs to a different configuration"
                )
    else:
        put_bytes(
            args.output + "/configuration.json", json.dumps(configuration).encode()
        )
    completion_fs, completion_path = fsspec.core.url_to_fs(
        args.output + "/complete.json"
    )
    if completion_fs.exists(completion_path):
        print(f"Already complete: {args.output}", flush=True)
        return
    began = time.perf_counter()
    data_root = Path(os.environ["DATA_PATH"])
    weight = data_root / "mpnn-ca-v_48_020.pt"
    download(
        f"https://raw.githubusercontent.com/dauparas/ProteinMPNN/{MPNN_REVISION}/ca_model_weights/v_48_020.pt",
        weight,
    )
    checkpoint = torch.load(weight, map_location="cpu", weights_only=False)
    mpnn = ProteinMPNN(
        ca_only=True,
        num_letters=21,
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        augment_eps=0.0,
        k_neighbors=checkpoint["num_edges"],
    )
    mpnn.load_state_dict(checkpoint["model_state_dict"], strict=True)
    mpnn = mpnn.cuda().eval()
    torch.cuda.synchronize()
    mpnn_load = time.perf_counter() - began
    start = time.perf_counter()
    esm_path = data_root / "esmfold" / args.esm_revision
    for filename in (
        "config.json",
        "pytorch_model.bin",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "vocab.txt",
    ):
        download(
            f"https://huggingface.co/facebook/esmfold_v1/resolve/{args.esm_revision}/{filename}",
            esm_path / filename,
        )
    tokenizer = AutoTokenizer.from_pretrained(esm_path)
    folder = EsmForProteinFolding.from_pretrained(esm_path).cuda().eval()
    folder.esm.half()
    folder.trunk.set_chunk_size(128)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.synchronize()
    esm_load = time.perf_counter() - start
    rotamers = load_library(cached_rotamer_library())
    props = torch.cuda.get_device_properties(0)
    worker = {
        "runner_tag": "iris",
        "gpu_name": props.name,
        "gpu_total_memory_gb": props.total_memory / 1e9,
        "gpu_compute_capability": f"{props.major}.{props.minor}",
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "esm_revision": args.esm_revision,
    }
    records = []
    timings = []
    documents = []
    processed = 0
    quality_pass_count = 0
    print(
        json.dumps(
            {
                "event": "models_loaded",
                "esm_load_seconds": esm_load,
                "mpnn_load_seconds": mpnn_load,
            }
        ),
        flush=True,
    )
    for prefix in prefixes:
        fs, path = fsspec.core.url_to_fs(prefix)
        files = sorted(fs.glob(path.rstrip("/") + "/batch-*.npz"))
        if not files:
            raise ValueError(f"No generated batches found under {prefix}")
        for batch_path in files:
            if paused(args.control):
                print("Paused at durable refolding batch boundary", flush=True)
                return
            records = []
            documents = []
            part_id = hashlib.sha256(f"{prefix}/{batch_path}".encode()).hexdigest()[:20]
            marker_uri = f"{args.output}/completed-parts/{part_id}.json"
            output_fs, marker_path = fsspec.core.url_to_fs(marker_uri)
            if output_fs.exists(marker_path):
                with output_fs.open(marker_path, "rt") as handle:
                    previous = json.load(handle)
                processed += previous["candidates"]
                quality_pass_count += previous["quality_pass"]
                with fsspec.open(
                    f"{args.output}/timings/{part_id}.csv", "rt"
                ) as handle:
                    timings.extend(csv.DictReader(handle))
                print(f"Already complete: {part_id}", flush=True)
                if args.limit and processed >= args.limit:
                    break
                continue
            with fs.open(batch_path, "rb") as handle:
                coordinates = np.load(io.BytesIO(handle.read()))["ca"]
            if args.limit:
                coordinates = coordinates[: args.limit - processed]
            if not len(coordinates):
                break
            batch_seed = (args.seed + int(part_id[:8], 16)) % (2**32)
            first_timing = len(timings)
            sequence_uri = f"{args.output}/sequences/{part_id}.parquet"
            sequence_fs, sequence_path = fsspec.core.url_to_fs(sequence_uri)
            if sequence_fs.exists(sequence_path):
                with sequence_fs.open(sequence_path, "rb") as handle:
                    designed = pq.read_table(handle).to_pylist()
                if len(designed) != len(coordinates) or any(
                    row["seed"] != batch_seed for row in designed
                ):
                    raise ValueError(
                        "Saved sequence attempts disagree with the input batch"
                    )
                sequences = [row["sequence"] for row in designed]
                design_seconds = sum(row["elapsed_seconds"] for row in designed)
            else:
                torch.cuda.synchronize()
                start = time.perf_counter()
                sequences = design_sequences(mpnn, coordinates, batch_seed)
                torch.cuda.synchronize()
                design_seconds = time.perf_counter() - start
                designed = [
                    {
                        "stem": "proteina-"
                        + hashlib.sha256(
                            f"{prefix}/{Path(batch_path).name}/{index}".encode()
                        ).hexdigest()[:20],
                        "sequence": sequence,
                        "source_prefix": prefix,
                        "source_batch": Path(batch_path).name,
                        "sample_in_batch": index,
                        "seed": batch_seed,
                        "attempt": 1,
                        "mpnn_revision": MPNN_REVISION,
                        "elapsed_seconds": design_seconds / len(sequences),
                        "model_load_seconds": mpnn_load,
                        "n_residues": len(sequence),
                        **worker,
                    }
                    for index, sequence in enumerate(sequences)
                ]
                buffer = io.BytesIO()
                pq.write_table(
                    pa.Table.from_pylist(designed), buffer, compression="zstd"
                )
                # Save EVERY designed sequence before any folding or rejection.
                # On preemption these attempts are reused, never regenerated.
                put_bytes(sequence_uri, buffer.getvalue())
            for index, (original, sequence) in enumerate(
                zip(coordinates, sequences, strict=True)
            ):
                stem = (
                    "proteina-"
                    + hashlib.sha256(
                        f"{prefix}/{Path(batch_path).name}/{index}".encode()
                    ).hexdigest()[:20]
                )
                pair_start = time.perf_counter()
                tokenized = tokenizer(
                    [sequence], return_tensors="pt", add_special_tokens=False
                )["input_ids"].cuda()
                torch.cuda.synchronize()
                start = time.perf_counter()
                with torch.inference_mode():
                    output = folder(tokenized, num_recycles=4)
                torch.cuda.synchronize()
                fold_seconds = time.perf_counter() - start
                # Transformers 4.48.3 returns ESMFold pLDDT in [0, 1],
                # including in output_to_pdb's B-factor fields. The corpus
                # confidence convention and the preregistered cutoff use 0–100.
                confidence = float(output["plddt"][0, :, 1].mean().item())
                plddt = confidence_percent(confidence)
                pdb = folder.output_to_pdb({**output, "plddt": 100 * output["plddt"]})[
                    0
                ]
                structure = gemmi.read_pdb_string(pdb)
                refolded = structure_coordinates(structure, sequence)
                rmsd = aligned_rmsd(original, refolded)
                geometry = backbone_geometry(structure)
                passed = bool(
                    rmsd <= 2.0 and plddt >= 70 and geometry["ca_geometry_pass"]
                )
                contact_start = time.perf_counter()
                if passed:
                    result = generate_document(
                        structure,
                        entry_id=stem,
                        config=GenerationConfig(),
                        rotamer_library=rotamers,
                    )
                    if result is None:
                        raise ValueError(
                            f"Serializer rejected a supposedly valid designed monomer: {stem}"
                        )
                    metadata = result.metadata_row()
                    reconstructed = sequence_from_document(
                        metadata["document"], len(sequence), metadata["n_term_index"]
                    )
                    if reconstructed != sequence:
                        raise ValueError(
                            "Serialized document does not preserve the designed sequence"
                        )
                    documents.append(
                        {
                            **metadata,
                            "sequence": sequence,
                            "source": "proteina-exp278",
                            "label_geometry": "esmfold_refold",
                            "release_status": "provisional",
                        }
                    )
                    quality_pass_count += 1
                contact_seconds = time.perf_counter() - contact_start
                records.append(
                    {
                        "stem": stem,
                        "source_prefix": prefix,
                        "source_batch": Path(batch_path).name,
                        "sample_in_batch": index,
                        "sequence": sequence,
                        "pdb_content": pdb,
                        "original_ca": original.tolist(),
                        "scrmsd": rmsd,
                        "plddt": plddt,
                        "quality_pass": passed,
                        "decontamination_status": "pending",
                        "geometry_version": "cis-proline-v2",
                        **geometry,
                    }
                )
                common = {
                    **worker,
                    "stem": stem,
                    "n_residues": len(sequence),
                    "n_pairs": len(sequence) * (len(sequence) - 1) // 2,
                    "seed": batch_seed,
                    "total_seconds": time.perf_counter() - pair_start,
                    "pyconfind_seconds": contact_seconds,
                }
                timings.extend(
                    [
                        {
                            **common,
                            "mode": "sequence_design",
                            "model_nickname": "proteinmpnn-ca",
                            "elapsed_seconds": design_seconds / len(sequences),
                            "model_load_seconds": mpnn_load,
                            "batch_size": len(sequences),
                        },
                        {
                            **common,
                            "mode": "refold",
                            "model_nickname": "esmfold",
                            "elapsed_seconds": fold_seconds,
                            "model_load_seconds": esm_load,
                            "batch_size": 1,
                        },
                    ]
                )
                processed += 1
                del output
                print(
                    json.dumps(
                        {
                            "event": "refold_done",
                            "count": processed,
                            "stem": stem,
                            "scrmsd": rmsd,
                            "plddt": plddt,
                            "quality_pass": passed,
                        }
                    ),
                    flush=True,
                )
            # Checkpoint each input batch, including rejected candidates.
            buffer = io.BytesIO()
            pq.write_table(pa.Table.from_pylist(records), buffer, compression="zstd")
            put_bytes(f"{args.output}/candidates/{part_id}.parquet", buffer.getvalue())
            if documents:
                buffer = io.BytesIO()
                pq.write_table(
                    pa.Table.from_pylist(documents), buffer, compression="zstd"
                )
                put_bytes(
                    f"{args.output}/documents-provisional/{part_id}.parquet",
                    buffer.getvalue(),
                )
            text = io.StringIO()
            writer = csv.DictWriter(
                text, fieldnames=list(timings[0]), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(timings)
            put_bytes(args.output + "/timings.csv", text.getvalue().encode())
            part_text = io.StringIO()
            part_writer = csv.DictWriter(
                part_text, fieldnames=list(timings[0]), lineterminator="\n"
            )
            part_writer.writeheader()
            part_writer.writerows(timings[first_timing:])
            put_bytes(
                f"{args.output}/timings/{part_id}.csv", part_text.getvalue().encode()
            )
            # Publish the marker last, so interrupted partial writes are retried.
            put_bytes(
                marker_uri,
                json.dumps(
                    {
                        "candidates": len(records),
                        "quality_pass": len(documents),
                        "sequences_saved": len(sequences),
                    }
                ).encode(),
            )
            write_json(
                args.output + "/progress.json",
                {
                    "candidates": processed,
                    "quality_pass": quality_pass_count,
                    "updated_utc": datetime.now(timezone.utc).isoformat(),
                    "last_part": part_id,
                },
            )
            if args.limit and processed >= args.limit:
                break
        if args.limit and processed >= args.limit:
            break
    put_bytes(
        args.output + "/complete.json",
        json.dumps(
            {"candidates": processed, "quality_pass": quality_pass_count, **worker}
        ).encode(),
    )


if __name__ == "__main__":
    main()
