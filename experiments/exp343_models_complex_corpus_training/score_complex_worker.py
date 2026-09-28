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
               "contacts_emitted", "contacts_emitted_inter_chain"]
    with fsspec.open(uri, "rb") as handle:
        table = pq.ParquetFile(handle).read(columns=columns)
    rows = table.to_pylist()
    if limit is not None:
        rows = rows[:limit]
    log(f"read {len(rows)} documents from {uri}")
    return rows


def score(model, tokenizer, rows: list[dict], sections_module) -> list[dict]:
    """Per-document NLL totals, overall and by token role.

    Every document is scored in full -- no truncation and no sliding window --
    because the corpus caps documents at 8192 tokens, which is the model's own
    context length.
    """
    role_enum = sections_module.Role
    # Longest first: the batch's peak memory is set by its longest document, and
    # a failure then happens in the first batch rather than three hours in.
    order = sorted(range(len(rows)), key=lambda i: -rows[i]["num_tokens"])
    results: list[dict | None] = [None] * len(rows)
    eos_id = tokenizer.convert_tokens_to_ids("<eos>")
    pad_id = tokenizer.convert_tokens_to_ids("<pad>")
    started = time.perf_counter()
    for start in range(0, len(order), BATCH_DOCUMENTS):
        chunk = order[start : start + BATCH_DOCUMENTS]
        labelled = [sections_module.label(rows[i]["document"]) for i in chunk]
        # The training cache is the document's tokens plus a trailing `<eos>`,
        # so the scored sequence must be too, or the last real token is never
        # predicted from and the loss is not the training loss.
        encoded = [
            (tokenizer.convert_tokens_to_ids(list(section.tokens)) + [eos_id])[
                :MAX_SEQ_LEN
            ]
            for section in labelled
        ]
        width = max(len(ids) for ids in encoded)
        batch = torch.full((len(chunk), width), pad_id, dtype=torch.long)
        for row, ids in enumerate(encoded):
            batch[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        batch = batch.to("cuda")
        mask = torch.zeros_like(batch, dtype=torch.long)
        for row, ids in enumerate(encoded):
            mask[row, : len(ids)] = 1
        with torch.no_grad():
            logits = model(input_ids=batch, attention_mask=mask).logits.float()
        log_probs = torch.log_softmax(logits[:, :-1], dim=-1)
        targets = batch[:, 1:]
        nll = -log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1).cpu()
        for row, (index, section, ids) in enumerate(
            zip(chunk, labelled, encoded, strict=True)
        ):
            # Target t is predicted from position t-1, so target position k of
            # `nll` is token k+1 of the sequence, whose role is roles[k+1]. The
            # final target is the appended `<eos>`, scored under the END role.
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
            results[index] = {
                "document_id": source["document_id"],
                "source_arm": source["source_arm"],
                "complex_type": source["complex_type"],
                "confidence_tier": source["confidence_tier"],
                "num_chains": source["num_chains"],
                "truncated": source["truncated"],
                "num_tokens": source["num_tokens"],
                "scored_positions": scored,
                "nll_total": total,
                "contacts_intra": section.contacts_intra,
                "contacts_inter": section.contacts_inter,
                "contacts_unresolved": section.contacts_unresolved,
                "published_contacts": source["contacts_emitted"],
                "published_contacts_inter": source["contacts_emitted_inter_chain"],
                **{f"nll_{role}": totals.get(role, 0.0) for role in
                   (member.value for member in role_enum)},
                **{f"n_{role}": counts.get(role, 0) for role in
                   (member.value for member in role_enum)},
            }
        if start % (BATCH_DOCUMENTS * 25) == 0:
            done = start + len(chunk)
            rate = done / (time.perf_counter() - started)
            log(f"{done}/{len(order)} documents, {rate:.1f} docs/s")
    if any(result is None for result in results):
        raise ValueError("a document was not scored")
    return [result for result in results if result is not None]


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

    rows = read_documents(args.shard, args.limit)
    inference_started = time.perf_counter()
    scored = score(model, tokenizer, rows, sections_module)
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
    prefix = args.out.rstrip("/")
    with fsspec.open(f"{prefix}/{args.label}/summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    with fsspec.open(f"{prefix}/{args.label}/per_document.parquet", "wb") as handle:
        pq.write_table(pa.Table.from_pylist(scored), handle, compression="zstd")
    print(json.dumps({"event": "done", **summary["aggregate"]["all"]}), flush=True)
    log(f"wrote {prefix}/{args.label}/")


if __name__ == "__main__":
    main()
