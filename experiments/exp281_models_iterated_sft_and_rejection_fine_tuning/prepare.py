"""Prepare immutable target shards from decontaminated contacts-v1 documents.

The input manifest must identify its provenance and decontamination reference.
Sequence/reference tokens are retained exactly, including the position ring.
Shards are streamed; targets are assigned to train/validation by stable identity.
"""

import argparse
from pathlib import Path

import fsspec
import torch
from marinfold.document_structures.contacts_v1_multi import BEGIN, END, FINAL, MULTI, parse_history
from marinfold.inference._config import load_config
from marinfold.inference._tokenizer import load_tokenizer
from transformers import AutoModelForCausalLM

from common import identity, publish_directory, read_json, rows, seed_for, stage_model, write_json, write_rows


def target_from_document(document: str, target_id: str) -> dict:
    """Extract a monomer target without regenerating or truncating contacts."""
    tokens = document.split()
    if tokens[0] != "<contacts-v1>" or tokens.count(BEGIN) != 1 or tokens[-1] != END:
        raise ValueError(f"{target_id}: expected one plain contacts-v1 document")
    offset = tokens.index(BEGIN)
    header, reference = tokens[:offset], tokens[offset + 1:-1]
    parse_history([FINAL, *reference, END])
    nterm = header.index("<n-term>")
    start = int(header[nterm + 1][2:-1])
    positions = [int(t[2:-1]) for t in header[2::2] if t.startswith("<p")]
    if len(set(positions)) != len(positions) or not positions:
        raise ValueError(f"{target_id}: invalid residue positions")
    positions.sort(key=lambda p: (p - start) % 2000)
    if positions != [(start + i) % 2000 for i in range(len(positions))]:
        raise ValueError(f"{target_id}: only contiguous monomers are supported")
    residues = {int(header[i][2:-1]): header[i + 1] for i in range(2, len(header), 2)
                if header[i].startswith("<p")}
    return {"target_id": target_id, "header": header, "reference": reference,
            "positions": positions, "n_residues": len(positions),
            "sequence_key": identity([residues[p] for p in positions])}


def prepare_targets(manifest: dict, output: str, shard_size: int, limit: int | None,
                    max_residues: int = 512, context: int = 8192) -> None:
    """Validate provenance, stream targets, and publish a manifest last."""
    if not manifest.get("decontamination_reference") or not manifest.get("source_revision"):
        raise ValueError("source manifest must pin revision and decontamination reference")
    seen: set[str] = set()
    buffers = {"train": [], "validation": []}
    paths = {"train": [], "validation": []}
    counts = {"train": 0, "validation": 0}
    excluded = {"length": 0, "answer_budget": 0}
    for source in manifest["sources"]:
        if "size_bytes" in source:
            fs, path = fsspec.core.url_to_fs(source["uri"])
            info = fs.info(path)
            if info["size"] != source["size_bytes"] or info.get("ETag") != source["etag"]:
                raise ValueError(f"source object changed since preflight: {source['uri']}")
        for row in rows(source["uri"]):
            target_id = f"{source['name']}:{row[source['id_column']]}"
            if target_id in seen:
                raise ValueError(f"duplicate target identity {target_id}; use one document per protein")
            seen.add(target_id)
            target = target_from_document(row[source["document_column"]], target_id)
            target["source"] = source["name"]
            # Explicit eligibility filters, counted rather than truncated. The
            # generation budget depends only on sequence length, never labels.
            if not 8 <= target["n_residues"] <= max_residues:
                excluded["length"] += 1
                continue
            reserve = 6 * target["n_residues"] + 128
            if len(target["reference"]) + 1 > reserve or len(target["header"]) + reserve + 1 > context:
                excluded["answer_budget"] += 1
                continue
            split = "validation" if seed_for(target["sequence_key"], 281) % 100 == 0 else "train"
            buffers[split].append(target)
            counts[split] += 1
            if len(buffers[split]) == shard_size:
                path = f"{output}/{split}-{len(paths[split]):05d}.parquet"
                write_rows(path, buffers[split])
                paths[split].append(path)
                buffers[split] = []
            if limit and sum(counts.values()) >= limit:
                break
        if limit and sum(counts.values()) >= limit:
            break
    for split, buffer in buffers.items():
        if buffer:
            path = f"{output}/{split}-{len(paths[split]):05d}.parquet"
            write_rows(path, buffer)
            paths[split].append(path)
    write_json(f"{output}/manifest.json", {"source": manifest, "source_hash": identity(manifest),
                                         "counts": counts, "excluded": excluded, "shards": paths,
                                         "max_residues": max_residues, "context": context})


def prepare_model(source: str, output: Path) -> None:
    """Append missing mode/final tokens without repurposing existing vocabulary ids."""
    if output.exists():
        raise FileExistsError(output)
    local = stage_model(source, output.parent / "model-cache")
    tokenizer = load_tokenizer(local)
    before = tokenizer.get_vocab()
    tokenizer.add_tokens([t for t in (MULTI, FINAL) if t not in before], special_tokens=True)
    if any(tokenizer.get_vocab()[t] != index for t, index in before.items()):
        raise ValueError("original token ids changed")
    # Keep fp32 optimizer parameters in the trainer; storage/generation use bf16.
    torch.manual_seed(281)
    model = AutoModelForCausalLM.from_pretrained(local, config=load_config(local), dtype="bfloat16")
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    output.mkdir(parents=True)
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    write_json(str(output / "initialization.json"), {"source": source, "old_vocab": len(before),
               "new_vocab": len(tokenizer), "tokens": {t: tokenizer.convert_tokens_to_ids(t) for t in (MULTI, FINAL)}})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    target = sub.add_parser("targets")
    target.add_argument("--manifest", required=True)
    target.add_argument("--output", required=True)
    target.add_argument("--shard-size", type=int, default=1024)
    target.add_argument("--limit", type=int)
    target.add_argument("--max-residues", type=int, default=512)
    target.add_argument("--context", type=int, default=8192)
    model = sub.add_parser("model")
    model.add_argument("--source", required=True)
    model.add_argument("--output", type=Path, required=True)
    model.add_argument("--publish", help="co-located durable checkpoint prefix")
    args = parser.parse_args()
    if args.command == "model":
        prepare_model(args.source, args.output)
        if args.publish:
            publish_directory(args.output, args.publish)
    else:
        prepare_targets(read_json(args.manifest), args.output, args.shard_size, args.limit,
                        args.max_residues, args.context)


if __name__ == "__main__":
    main()
