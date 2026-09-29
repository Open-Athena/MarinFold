"""Compute per-token teacher-forced CE split by document phase."""

import argparse
import os
import platform
import socket
import time
from collections.abc import Iterable
from pathlib import Path

import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROLE_NAMES = {
    0: "other",
    1: "amino_acid",
    2: "sequence_other",
    3: "contact_content",
    4: "contact_stop",
}

TOKEN_SCHEMA = pa.schema(
    [
        ("model", pa.string()),
        ("document_format", pa.string()),
        ("dataset", pa.string()),
        ("stem", pa.string()),
        ("document_id", pa.string()),
        ("source_row", pa.int32()),
        ("n_tokens", pa.int32()),
        ("target_token_ids", pa.list_(pa.int16())),
        ("role_ids", pa.list_(pa.int8())),
        ("token_ce", pa.list_(pa.float32())),
        ("amino_acid_count", pa.int32()),
        ("amino_acid_mean_ce", pa.float64()),
        ("sequence_phase_count", pa.int32()),
        ("sequence_phase_mean_ce", pa.float64()),
        ("contact_phase_count", pa.int32()),
        ("contact_phase_mean_ce", pa.float64()),
    ]
)


def stage_model(source: str, destination: Path) -> tuple[Path, float]:
    """Download a flat HF model directory to local disk."""
    destination.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    fs, root = fsspec.core.url_to_fs(source)
    files = [entry for entry in fs.ls(root, detail=True) if entry["type"] == "file"]
    if not files:
        raise ValueError(f"no model files at {source}")
    for entry in files:
        fs.get_file(entry["name"], str(destination / os.path.basename(entry["name"])))
    return destination, time.monotonic() - started


def load_contacts_v1_documents(
    source: str, shard_index: int, num_shards: int
) -> list[tuple[int, str, list[int], str | None, str | None]]:
    """Load selected rows from a Levanter jagged token cache."""
    import zarr

    fs, root = fsspec.core.url_to_fs(source)
    offsets_array = zarr.open_array(fs.get_mapper(f"{root}/offsets"), mode="r")
    num_rows = int(offsets_array[0])
    ends = np.asarray(offsets_array[1 : num_rows + 1], dtype=np.int64)
    total_tokens = int(ends[-1])
    data_array = zarr.open_array(fs.get_mapper(f"{root}/data"), mode="r")
    data = np.asarray(data_array[:total_tokens], dtype=np.int32)
    starts = np.concatenate([np.zeros(1, dtype=np.int64), ends[:-1]])
    return [
        (row, f"cache-row-{row:05d}", data[starts[row] : ends[row]].tolist(), None, None)
        for row in range(shard_index, num_rows, num_shards)
    ]


def load_parquet_documents(
    source: str, shard_index: int, num_shards: int
) -> list[tuple[int, str, list[int], str | None, str | None]]:
    """Load selected token documents from a parquet file."""
    with fsspec.open(source, "rb") as handle:
        table = pq.read_table(handle)
    required = {"entry_id", "token_ids"}
    if missing := required - set(table.column_names):
        raise ValueError(f"parquet document input is missing columns: {sorted(missing)}")
    rows = table.to_pylist()
    return [
        (
            row,
            str(record["entry_id"]),
            [int(token) for token in record["token_ids"]],
            str(record["dataset"]) if record.get("dataset") is not None else None,
            str(record["stem"]) if record.get("stem") is not None else None,
        )
        for row, record in enumerate(rows)
        if row % num_shards == shard_index
    ]


def classify_targets(input_ids: list[int], document_format: str) -> np.ndarray:
    """Assign semantic roles to target tokens (all input tokens except the first)."""
    roles = np.zeros(len(input_ids) - 1, dtype=np.int8)
    if document_format == "contacts-v1":
        begin_sequence = input_ids.index(8)
        begin_contacts = input_ids.index(9)
        end_document = input_ids.index(10)
        for target_position in range(begin_sequence + 1, begin_contacts):
            token_id = input_ids[target_position]
            roles[target_position - 1] = 1 if 86 <= token_id <= 105 or token_id == 2844 else 2
        roles[begin_contacts:end_document - 1] = 3
        return roles
    if document_format == "delta-v2":
        begin_contacts = input_ids.index(22)
        end_document = input_ids.index(23)
        roles[: begin_contacts - 1] = 1
        contact_ids = np.asarray(input_ids[begin_contacts + 1 : end_document], dtype=np.int32)
        roles[begin_contacts:end_document - 1] = np.where(contact_ids == 0, 4, 3)
        return roles
    raise ValueError(f"unsupported document format: {document_format}")


def batches_by_token_budget(
    documents: list[tuple[int, str, list[int], str | None, str | None]], token_budget: int
) -> Iterable[list[tuple[int, str, list[int], str | None, str | None]]]:
    """Group length-sorted documents while bounding padded batch tokens."""
    batch: list[tuple[int, str, list[int], str | None, str | None]] = []
    max_length = 0
    for document in sorted(documents, key=lambda item: len(item[2])):
        candidate_max = max(max_length, len(document[2]))
        if batch and candidate_max * (len(batch) + 1) > token_budget:
            yield batch
            batch = []
            max_length = 0
        batch.append(document)
        max_length = max(max_length, len(document[2]))
    if batch:
        yield batch


def masked_mean(losses: np.ndarray, roles: np.ndarray, selected_roles: set[int]) -> tuple[int, float]:
    """Return selected token count and loss mean."""
    mask = np.isin(roles, list(selected_roles))
    count = int(mask.sum())
    return count, float(losses[mask].mean()) if count else float("nan")


def write_rows(rows: list[dict], destination: str) -> None:
    with fsspec.open(destination, "wb") as handle:
        pq.write_table(pa.Table.from_pylist(rows, schema=TOKEN_SCHEMA), handle, compression="zstd")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--document-format", choices=("contacts-v1", "delta-v2"), required=True)
    parser.add_argument("--documents", required=True)
    parser.add_argument("--documents-kind", choices=("auto", "cache", "parquet"), default="auto")
    parser.add_argument("--out", required=True)
    parser.add_argument("--shard", required=True, help="INDEX/COUNT")
    parser.add_argument("--token-budget", type=int, default=8192)
    parser.add_argument("--rows-per-part", type=int, default=256)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    shard_index, num_shards = (int(value) for value in args.shard.split("/"))
    documents_kind = args.documents_kind
    if documents_kind == "auto":
        documents_kind = "cache" if args.document_format == "contacts-v1" else "parquet"
    if documents_kind == "cache":
        documents = load_contacts_v1_documents(args.documents, shard_index, num_shards)
    else:
        documents = load_parquet_documents(args.documents, shard_index, num_shards)
    if args.limit is not None:
        documents = documents[: args.limit]
    if not documents:
        raise ValueError(f"shard {args.shard} selected no documents")
    if max(len(document[2]) for document in documents) > 8192:
        raise ValueError("document exceeds model context length 8192")

    import torch
    from torch.nn import functional
    from transformers import AutoModelForCausalLM

    model_path, stage_seconds = stage_model(args.model, Path(f"/tmp/teacher-ce-model-{shard_index}"))
    load_started = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cuda",
    )
    model.eval()
    load_seconds = time.monotonic() - load_started

    output_root = f"{args.out.rstrip('/')}/{args.model_label}"
    role_loss_sums = {role: 0.0 for role in ROLE_NAMES}
    role_counts = {role: 0 for role in ROLE_NAMES}
    output_rows: list[dict] = []
    part_index = 0
    documents_done = 0
    inference_started = time.monotonic()

    for batch in batches_by_token_budget(documents, args.token_budget):
        max_length = max(len(document[2]) for document in batch)
        input_ids = torch.zeros((len(batch), max_length), dtype=torch.long, device="cuda")
        attention_mask = torch.zeros_like(input_ids)
        for batch_index, (_, _, ids, _, _) in enumerate(batch):
            input_ids[batch_index, : len(ids)] = torch.tensor(ids, dtype=torch.long, device="cuda")
            attention_mask[batch_index, : len(ids)] = 1
        with torch.inference_mode():
            logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits[:, :-1]
            targets = input_ids[:, 1:]
            token_losses = functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none"
            ).reshape(len(batch), max_length - 1)

        for batch_index, (source_row, document_id, ids, dataset, stem) in enumerate(batch):
            losses = token_losses[batch_index, : len(ids) - 1].float().cpu().numpy()
            roles = classify_targets(ids, args.document_format)
            if len(losses) != len(roles):
                raise AssertionError(f"loss/role length mismatch for {document_id}")
            for role in ROLE_NAMES:
                selected = roles == role
                role_loss_sums[role] += float(losses[selected].sum(dtype=np.float64))
                role_counts[role] += int(selected.sum())
            amino_count, amino_mean = masked_mean(losses, roles, {1})
            sequence_count, sequence_mean = masked_mean(losses, roles, {1, 2})
            contact_count, contact_mean = masked_mean(losses, roles, {3, 4})
            output_rows.append(
                {
                    "model": args.model_label,
                    "document_format": args.document_format,
                    "dataset": dataset,
                    "stem": stem,
                    "document_id": document_id,
                    "source_row": source_row,
                    "n_tokens": len(ids),
                    "target_token_ids": ids[1:],
                    "role_ids": roles.tolist(),
                    "token_ce": losses.tolist(),
                    "amino_acid_count": amino_count,
                    "amino_acid_mean_ce": amino_mean,
                    "sequence_phase_count": sequence_count,
                    "sequence_phase_mean_ce": sequence_mean,
                    "contact_phase_count": contact_count,
                    "contact_phase_mean_ce": contact_mean,
                }
            )
            documents_done += 1
            if len(output_rows) >= args.rows_per_part:
                destination = f"{output_root}/tokens/shard-{shard_index:03d}-part-{part_index:04d}.parquet"
                write_rows(output_rows, destination)
                output_rows = []
                part_index += 1
                print(f"[{documents_done}/{len(documents)}] -> {destination}", flush=True)
        del logits, targets, token_losses, input_ids, attention_mask

    if output_rows:
        destination = f"{output_root}/tokens/shard-{shard_index:03d}-part-{part_index:04d}.parquet"
        write_rows(output_rows, destination)
        print(f"[{documents_done}/{len(documents)}] -> {destination}", flush=True)

    properties = torch.cuda.get_device_properties(0)
    summary_rows = []
    for role, role_name in ROLE_NAMES.items():
        count = role_counts[role]
        summary_rows.append(
            {
                "model": args.model_label,
                "document_format": args.document_format,
                "shard": shard_index,
                "num_shards": num_shards,
                "role_id": role,
                "role": role_name,
                "loss_sum": role_loss_sums[role],
                "token_count": count,
                "mean_ce": role_loss_sums[role] / count if count else float("nan"),
                "document_count": len(documents),
                "model_stage_seconds": stage_seconds,
                "model_load_seconds": load_seconds,
                "inference_seconds": time.monotonic() - inference_started,
                "gpu_name": properties.name,
                "gpu_total_memory_gb": properties.total_memory / 2**30,
                "gpu_compute_capability": f"{properties.major}.{properties.minor}",
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "torch_version": torch.__version__,
            }
        )
    summary_destination = f"{output_root}/summary/shard-{shard_index:03d}.parquet"
    with fsspec.open(summary_destination, "wb") as handle:
        pq.write_table(pa.Table.from_pylist(summary_rows), handle, compression="zstd")
    print(f"DONE {len(documents)} documents -> {summary_destination}", flush=True)


if __name__ == "__main__":
    main()
