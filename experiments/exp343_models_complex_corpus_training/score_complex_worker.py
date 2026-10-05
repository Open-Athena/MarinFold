# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Score one checkpoint's token-level loss on the held-out complex shard.

Runs standalone on a single CoreWeave H100 inside a vendor GPU image. It takes
no repo imports -- the section labeller travels with it, base64'd by the driver --
so the whole worker is one file the driver can ship into any image.

What it measures, per document and aggregated:

* mean next-token NLL over the whole document, which is what the training
  validation curve reports;
* the same, split by **token role** -- sequence, terminus, intra-chain contact,
  inter-chain contact -- because a model can score well on a complex document by
  predicting only the contacts a monomer model already knows how to predict. The
  inter-chain number is the one the experiment turns on.

Two loading traps this guards against, both of which fail *silently* and cost
about 0.76 nats/token if missed: `rope_parameters` is ignored by transformers 4.x
(MarinFold #163), and an export whose tokenizer class the image cannot construct
loads a different tokenizer rather than erroring. Both are asserted, not hoped
for.
"""

import argparse
import base64
import csv
import importlib.util
import json
import math
import platform
import socket
import sys
import time
import types
from collections import defaultdict
from pathlib import Path

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

#: The trained rotary configuration. A checkpoint that loads with anything else
#: is being evaluated as a different model than the one that was trained.
EXPECTED_ROPE_THETA = 500_000
EXPECTED_ROPE_TYPE = "llama3"
EXPECTED_VOCAB_SIZE = 2845
#: Documents per forward pass. They are padded to the longest in the batch, and
#: the corpus is length-sorted first, so padding waste stays small.
BATCH_DOCUMENTS = 8
#: The model's context length, and the corpus's document cap. A document already
#: at the cap has its appended `<eos>` trimmed -- exactly what the training
#: packer does to it -- rather than being scored one position past the context.
MAX_SEQ_LEN = 8192
#: Documents per durable output part. Batch priority is preemptible and this
#: worker has no other state, so a restart replays at most one part -- about
#: 90 seconds -- instead of the whole shard. A completed run's first attempt
#: already wrote every part, so a retry is then almost free.
DOCUMENTS_PER_PART = 512


def log(message: str) -> None:
    print(json.dumps({"event": "log", "message": message}), flush=True)


def load_sections_module(encoded: str) -> types.ModuleType:
    """Materialize the section labeller the driver shipped with this worker."""
    source = base64.b64decode(encoded).decode()
    path = Path("/tmp/complex_sections.py")
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("complex_sections", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the section labeller")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fetch_checkpoint(uri: str, destination: Path) -> dict[str, int]:
    """Copy an HF export to local disk and report what arrived."""
    destination.mkdir(parents=True, exist_ok=True)
    fs, root = fsspec.core.url_to_fs(uri)
    sizes = {}
    for remote in fs.ls(root, detail=False):
        name = remote.rsplit("/", 1)[-1]
        if not name or fs.isdir(remote):
            continue
        local = destination / name
        with fs.open(remote, "rb") as source, local.open("wb") as sink:
            while chunk := source.read(32 << 20):
                sink.write(chunk)
        sizes[name] = local.stat().st_size
    missing = [
        name
        for name in ("config.json", "tokenizer.json", "model.safetensors.index.json")
        if name not in sizes
    ]
    if missing:
        raise ValueError(f"{uri} is missing {missing}")
    log(f"fetched {len(sizes)} files, {sum(sizes.values()) / 1e9:.2f} GB")
    return sizes


def load_model(directory: Path):
    """Load the export, asserting the rotary and tokenizer contracts hold."""
    config = AutoConfig.from_pretrained(directory)
    theta = getattr(config, "rope_theta", None)
    scaling = getattr(config, "rope_scaling", None) or {}
    parameters = getattr(config, "rope_parameters", None) or {}
    theta = theta if theta is not None else parameters.get("rope_theta")
    rope_type = scaling.get("rope_type") or parameters.get("rope_type")
    if theta != EXPECTED_ROPE_THETA or rope_type != EXPECTED_ROPE_TYPE:
        raise ValueError(
            f"Loaded rope theta={theta} type={rope_type}; the checkpoint was "
            f"trained with theta={EXPECTED_ROPE_THETA} type={EXPECTED_ROPE_TYPE}. "
            "This transformers build is dropping the exported rotary config."
        )
    tokenizer = AutoTokenizer.from_pretrained(directory)
    if len(tokenizer) != EXPECTED_VOCAB_SIZE:
        raise ValueError(
            f"Tokenizer has {len(tokenizer)} tokens, expected {EXPECTED_VOCAB_SIZE}"
        )
    # `torch_dtype`, not `dtype`: the pinned image has transformers 4.53, where an
    # unknown `dtype` kwarg lands on the config as a `torch.dtype` and the next
    # config log line dies trying to JSON-serialize it. The load is asserted
    # afterwards, so a silent float32 load fails rather than quietly doubling
    # memory and changing the numbers.
    model = AutoModelForCausalLM.from_pretrained(
        directory, torch_dtype="bfloat16", device_map="cuda"
    )
    if model.dtype != torch.bfloat16:
        raise ValueError(f"Model loaded as {model.dtype}, expected bfloat16")
    if model.config.vocab_size != EXPECTED_VOCAB_SIZE:
        raise ValueError(f"Model vocabulary is {model.config.vocab_size}")
    model.eval()
    log(
        f"loaded: rope_theta={theta} rope_type={rope_type} "
        f"vocab={model.config.vocab_size} params={sum(p.numel() for p in model.parameters())}"
    )
    return model, tokenizer


def read_documents(uri: str, limit: int | None) -> list[dict]:
    """Read the held-out shard, keeping only what the scorer needs."""
    columns = ["document_id", "document", "source_arm", "complex_type",
               "confidence_tier", "num_tokens", "num_chains", "truncated",
               "seq_len", "chain_lengths",
               "contacts_emitted", "contacts_emitted_inter_chain"]
    with fsspec.open(uri, "rb") as handle:
        table = pq.ParquetFile(handle).read(columns=columns)
    rows = table.to_pylist()
    if limit is not None:
        rows = rows[:limit]
    log(f"read {len(rows)} documents from {uri}")
    return rows


#: Residues closer than this along a chain are never candidate contacts
#: (contacts-v1 `min_seq_separation`). Inter-chain pairs have no such floor.
MIN_SEQ_SEPARATION = 6


def candidate_pairs(chain_lengths: list[int]) -> int:
    """The contacts-v1 candidate-pair universe for one complex.

    Intra-chain: every pair at least `MIN_SEQ_SEPARATION` apart. Inter-chain:
    every cross-chain pair, at any index distance. This is `n_pairs` in the
    repo's standard predictor-timing schema, computed exactly rather than
    approximated by L^2.
    """
    total = 0
    for length in chain_lengths:
        if length > MIN_SEQ_SEPARATION:
            full = length * (length - 1) // 2
            near = sum(length - sep for sep in range(1, MIN_SEQ_SEPARATION))
            total += full - near
    for index, length in enumerate(chain_lengths):
        for other in chain_lengths[index + 1 :]:
            total += length * other
    return total


def score_batch(model, tokenizer, rows: list[dict], chunk: list[int],
                sections_module) -> list[dict]:
    """Score one batch of documents, one result row each.

    Every document is scored in full -- no truncation and no sliding window --
    because the corpus caps documents at 8192 tokens, which is the model's own
    context length.
    """
    role_enum = sections_module.Role
    eos_id = tokenizer.convert_tokens_to_ids("<eos>")
    pad_id = tokenizer.convert_tokens_to_ids("<pad>")
    labelled = [sections_module.label(rows[index]["document"]) for index in chunk]
    # The training cache is the document's tokens plus a trailing `<eos>`, so the
    # scored sequence must be too, or the last real token is never predicted from
    # and this is not the training loss.
    encoded = [
        (tokenizer.convert_tokens_to_ids(list(section.tokens)) + [eos_id])[:MAX_SEQ_LEN]
        for section in labelled
    ]
    width = max(len(ids) for ids in encoded)
    batch = torch.full((len(chunk), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(chunk), width), dtype=torch.long)
    for row, ids in enumerate(encoded):
        batch[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        mask[row, : len(ids)] = 1
    batch, mask = batch.to("cuda"), mask.to("cuda")
    # Timed around the forward pass only, then divided across the batch, so each
    # document carries its share of the work that produced its loss. Documents
    # are scored in batches, so this is a per-document share of a batch rather
    # than an isolated single-document measurement -- recorded as such.
    started = time.perf_counter()
    with torch.no_grad():
        logits = model(input_ids=batch, attention_mask=mask).logits.float()
    log_probs = torch.log_softmax(logits[:, :-1], dim=-1)
    nll = -log_probs.gather(2, batch[:, 1:].unsqueeze(-1)).squeeze(-1).cpu()
    torch.cuda.synchronize()
    batch_seconds = time.perf_counter() - started
    per_document_seconds = batch_seconds / len(chunk)
    scored_rows = []
    for row, (index, section, ids) in enumerate(
        zip(chunk, labelled, encoded, strict=True)
    ):
        # Target t is predicted from position t-1, so target position k of `nll`
        # is token k+1 of the sequence and carries roles[k+1]. The final target
        # is the appended `<eos>`, scored under the END role.
        totals: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)
        roles = list(section.roles) + [role_enum.END]
        for target in range(len(ids) - 1):
            role = roles[target + 1].value
            totals[role] += float(nll[row, target])
            counts[role] += 1
        total = sum(totals.values())
        scored = sum(counts.values())
        if scored != len(ids) - 1:
            raise ValueError("role accounting lost a scored position")
        if not math.isfinite(total):
            raise ValueError(f"non-finite loss on {rows[index]['document_id']}")
        source = rows[index]
        scored_rows.append(
            {
                "document_index": index,
                "document_id": source["document_id"],
                "source_arm": source["source_arm"],
                "complex_type": source["complex_type"],
                "confidence_tier": source["confidence_tier"],
                "num_chains": source["num_chains"],
                "truncated": source["truncated"],
                "num_tokens": source["num_tokens"],
                "n_residues": source["seq_len"],
                "n_pairs": candidate_pairs(list(source["chain_lengths"])),
                "elapsed_seconds": per_document_seconds,
                "batch_seconds": batch_seconds,
                "batch_documents": len(chunk),
                "scored_positions": scored,
                "nll_total": total,
                "contacts_intra": section.contacts_intra,
                "contacts_inter": section.contacts_inter,
                "contacts_unresolved": section.contacts_unresolved,
                "published_contacts": source["contacts_emitted"],
                "published_contacts_inter": source["contacts_emitted_inter_chain"],
                **{
                    f"nll_{member.value}": totals.get(member.value, 0.0)
                    for member in role_enum
                },
                **{
                    f"n_{member.value}": counts.get(member.value, 0)
                    for member in role_enum
                },
            }
        )
    return scored_rows


def part_order(rows: list[dict]) -> list[int]:
    """Deterministic scoring order: longest documents first.

    Longest-first makes the first batch the memory-heaviest, so an
    out-of-memory failure happens in the first ninety seconds rather than three
    hours in. `sorted` is stable, so ties keep the shard's own order and the
    partition into parts is reproducible across restarts -- which is what makes
    a completed part safe to skip.
    """
    return sorted(range(len(rows)), key=lambda index: -rows[index]["num_tokens"])


def score(model, tokenizer, rows: list[dict], sections_module, *, parts_prefix: str
          ) -> list[dict]:
    """Score every document, writing durable parts and resuming from them.

    Batch priority is preemptible and this worker holds no other state, so each
    part is written as soon as it is complete and an already-present part is
    read back instead of recomputed. A restart replays at most one part.
    """
    order = part_order(rows)
    fs, root = fsspec.core.url_to_fs(parts_prefix)
    existing = set()
    if fs.exists(root):
        existing = {path.rsplit("/", 1)[-1] for path in fs.ls(root, detail=False)}
    scored: list[dict] = []
    started = time.perf_counter()
    computed = 0
    for part, start in enumerate(range(0, len(order), DOCUMENTS_PER_PART)):
        name = f"part-{part:05d}.parquet"
        target = f"{parts_prefix.rstrip('/')}/{name}"
        indices = order[start : start + DOCUMENTS_PER_PART]
        if name in existing:
            with fsspec.open(target, "rb") as handle:
                recovered = pq.ParquetFile(handle).read().to_pylist()
            if len(recovered) != len(indices):
                raise ValueError(
                    f"{name} holds {len(recovered)} rows, expected {len(indices)}"
                )
            scored.extend(recovered)
            log(f"resumed {name} ({len(recovered)} documents)")
            continue
        part_rows: list[dict] = []
        for batch_start in range(0, len(indices), BATCH_DOCUMENTS):
            part_rows.extend(
                score_batch(
                    model,
                    tokenizer,
                    rows,
                    indices[batch_start : batch_start + BATCH_DOCUMENTS],
                    sections_module,
                )
            )
            computed += min(BATCH_DOCUMENTS, len(indices) - batch_start)
        with fsspec.open(target, "wb") as handle:
            pq.write_table(pa.Table.from_pylist(part_rows), handle, compression="zstd")
        scored.extend(part_rows)
        rate = computed / max(time.perf_counter() - started, 1e-9)
        log(f"wrote {name}; {len(scored)}/{len(order)} documents, {rate:.1f} docs/s")
    if len(scored) != len(rows):
        raise ValueError(f"scored {len(scored)} of {len(rows)} documents")
    if len({row["document_index"] for row in scored}) != len(rows):
        raise ValueError("the recovered parts do not cover every document exactly once")
    scored.sort(key=lambda row: row["document_index"])
    return scored


def aggregate(rows: list[dict], role_values: list[str]) -> dict:
    """Token-weighted means overall, per arm and per complex type."""
    def summarize(subset: list[dict]) -> dict:
        positions = sum(row["scored_positions"] for row in subset)
        summary = {
            "documents": len(subset),
            "scored_positions": positions,
            "nll_per_token": sum(row["nll_total"] for row in subset) / positions,
            "nll_per_document": sum(row["nll_total"] for row in subset) / len(subset),
        }
        for role in role_values:
            count = sum(row[f"n_{role}"] for row in subset)
            summary[f"n_{role}"] = count
            summary[f"nll_per_token_{role}"] = (
                sum(row[f"nll_{role}"] for row in subset) / count if count else None
            )
        return summary

    groups = {"all": summarize(rows)}
    for key in ("source_arm", "complex_type", "confidence_tier"):
        for value in sorted({row[key] for row in rows}):
            subset = [row for row in rows if row[key] == value]
            groups[f"{key}={value}"] = summarize(subset)
    return groups


#: The repo's standard predictor-timing schema (root `AGENTS.md`), so these rows
#: join with exp12/exp20's on `(stem, n_residues)`.
TIMING_COLUMNS = (
    "stem", "n_residues", "n_pairs", "mode",
    "elapsed_seconds", "model_load_seconds", "total_seconds",
    "model_nickname", "runner_tag",
    "gpu_name", "gpu_total_memory_gb", "gpu_compute_capability",
    "hostname", "platform", "torch_version", "timestamp_utc",
    "batch_documents", "batch_seconds", "scored_positions", "num_tokens",
)


def write_timings(uri: str, scored: list[dict], summary: dict, label: str) -> None:
    """One row per scored document, written at evaluation time.

    `elapsed_seconds` is that document's share of its batch's forward pass;
    `batch_seconds` and `batch_documents` are carried so the sharing is visible
    rather than implied. `model_load_seconds` is the run-level weight-load cost,
    repeated per row as the schema expects.
    """
    worker = summary["worker"]
    timing = summary["timing"]
    with fsspec.open(uri, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(TIMING_COLUMNS),
                                lineterminator="\n")
        writer.writeheader()
        for row in scored:
            writer.writerow({
                "stem": row["document_id"],
                "n_residues": row["n_residues"],
                "n_pairs": row["n_pairs"],
                "mode": "lm_loss",
                "elapsed_seconds": f"{row['elapsed_seconds']:.6f}",
                "model_load_seconds": f"{timing['model_load_seconds']:.6f}",
                "total_seconds": f"{row['elapsed_seconds']:.6f}",
                "model_nickname": label,
                "runner_tag": "iris",
                "gpu_name": worker["gpu_name"],
                "gpu_total_memory_gb": f"{worker['gpu_total_memory_gb']:.2f}",
                "gpu_compute_capability": worker["gpu_compute_capability"],
                "hostname": worker["hostname"],
                "platform": worker["platform"],
                "torch_version": worker["torch_version"],
                "timestamp_utc": worker["timestamp_utc"],
                "batch_documents": row["batch_documents"],
                "batch_seconds": f"{row['batch_seconds']:.6f}",
                "scored_positions": row["scored_positions"],
                "num_tokens": row["num_tokens"],
            })
    log(f"wrote {len(scored)} timing rows to {uri}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HF export URI")
    parser.add_argument("--label", required=True)
    parser.add_argument("--shard", required=True, help="held-out parquet shard URI")
    parser.add_argument("--out", required=True, help="output prefix")
    parser.add_argument("--sections-b64", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if args.model.startswith("gs://") or args.shard.startswith("gs://"):
        raise ValueError("GCS is unreachable from CoreWeave jobs")

    sections_module = load_sections_module(args.sections_b64)
    role_values = [member.value for member in sections_module.Role]
    directory = Path("/tmp/checkpoint")
    load_started = time.perf_counter()
    files = fetch_checkpoint(args.model, directory)
    model, tokenizer = load_model(directory)
    model_load_seconds = time.perf_counter() - load_started

    prefix = args.out.rstrip("/")
    rows = read_documents(args.shard, args.limit)
    inference_started = time.perf_counter()
    scored = score(
        model,
        tokenizer,
        rows,
        sections_module,
        parts_prefix=f"{prefix}/{args.label}/parts",
    )
    elapsed_seconds = time.perf_counter() - inference_started

    properties = torch.cuda.get_device_properties(0)
    summary = {
        "label": args.label,
        "model": args.model,
        "shard": args.shard,
        "checkpoint_files": files,
        "aggregate": aggregate(scored, role_values),
        "timing": {
            "elapsed_seconds": elapsed_seconds,
            "model_load_seconds": model_load_seconds,
            "total_seconds": time.perf_counter() - load_started,
            "documents": len(scored),
            "scored_positions": sum(row["scored_positions"] for row in scored),
        },
        "worker": {
            "gpu_name": properties.name,
            "gpu_total_memory_gb": properties.total_memory / 2**30,
            "gpu_compute_capability": f"{properties.major}.{properties.minor}",
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "python_version": sys.version.split()[0],
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    }
    with fsspec.open(f"{prefix}/{args.label}/summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    with fsspec.open(f"{prefix}/{args.label}/per_document.parquet", "wb") as handle:
        pq.write_table(pa.Table.from_pylist(scored), handle, compression="zstd")
    write_timings(f"{prefix}/{args.label}/timings.csv", scored, summary, args.label)
    print(json.dumps({"event": "done", **summary["aggregate"]["all"]}), flush=True)
    log(f"wrote {prefix}/{args.label}/")


if __name__ == "__main__":
    main()
