# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Log raw contacts-v1 rollout completions for exp124 think-token analysis."""

import argparse
import os
import re
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


BEGIN, NUM_POS = "<begin_statements>", 2000
CONTACT_RE = re.compile(r"<contact>\s+<p(\d+)>\s+<p(\d+)>")
TOKEN_RE = re.compile(r"<[^>]+>|\S+")
THINK_TOKEN = "<think>"

SCHEMA = pa.schema(
    [
        ("dataset", pa.string()),
        ("stem", pa.string()),
        ("L", pa.int32()),
        ("input_seq", pa.string()),
        ("rollout_idx", pa.int32()),
        ("prompt_tokens", pa.int32()),
        ("max_new_tokens", pa.int32()),
        ("generated_tokens", pa.int32()),
        ("finish_reason", pa.string()),
        ("text", pa.string()),
        ("first_generated_token", pa.string()),
        ("first_token_is_think", pa.bool_()),
        ("think_token_count", pa.int32()),
        ("parsed_i", pa.list_(pa.int16())),
        ("parsed_j", pa.list_(pa.int16())),
        ("raw_pos_i", pa.list_(pa.int16())),
        ("raw_pos_j", pa.list_(pa.int16())),
    ]
)


def stage_model(src: str, dst: Path) -> Path:
    """Copy a remote model directory to local disk for vLLM."""
    import fsspec

    if "://" not in src:
        return Path(src)
    dst.mkdir(parents=True, exist_ok=True)
    fs, root = fsspec.core.url_to_fs(src)
    files = [f for f in fs.ls(root, detail=True) if f["type"] == "file"]
    if not files:
        raise FileNotFoundError(f"no files under {src}")
    start = time.time()
    for f in files:
        fs.get_file(f["name"], str(dst / os.path.basename(f["name"])))
    size = sum(f["size"] for f in files)
    print(
        f"[raw-rollout] staged model {src} -> {dst} "
        f"({len(files)} files, {size / 2**30:.2f} GiB, {time.time() - start:.0f}s)",
        flush=True,
    )
    return dst


def read_parquet(uri: str, **kwargs):
    import fsspec

    with fsspec.open(uri, "rb") as handle:
        return pq.read_table(handle, **kwargs)


def write_parquet(table: pa.Table, uri: str) -> None:
    import fsspec

    with fsspec.open(uri, "wb") as handle:
        pq.write_table(table, handle, compression="zstd")


def load_targets(path: str) -> list[dict]:
    records = read_parquet(path).to_pylist()
    records.sort(key=lambda r: r["L"])
    return records


def done_stems(out_dir: str, shard_i: int, num_shards: int) -> tuple[set[str], int]:
    """Return completed stems for this shard and the next part number."""
    import fsspec

    fs, _ = fsspec.core.url_to_fs(out_dir)
    pattern = f"{out_dir.rstrip('/')}/shard-{shard_i:03d}-of-{num_shards:03d}-part-*.parquet"
    try:
        parts = fs.glob(pattern)
    except FileNotFoundError:
        return set(), 0
    seen: set[str] = set()
    for part in parts:
        try:
            table = read_parquet(fs.unstrip_protocol(part), columns=["dataset", "stem"])
            seen |= {
                f"{dataset}__{stem}"
                for dataset, stem in zip(table.column("dataset").to_pylist(), table.column("stem").to_pylist())
            }
        except Exception as exc:
            print(f"[raw-rollout] ignoring unreadable part {part}: {exc}", flush=True)
    return seen, len(parts)


def first_token(text: str) -> str | None:
    match = TOKEN_RE.search(text)
    return match.group(0) if match else None


def parse_contacts(text: str, pos_to_seq: dict[int, int]) -> tuple[list[int], list[int], list[int], list[int]]:
    parsed_i: list[int] = []
    parsed_j: list[int] = []
    raw_i: list[int] = []
    raw_j: list[int] = []
    seen: set[tuple[int, int]] = set()
    for x, y in CONTACT_RE.findall(text):
        px, py = int(x), int(y)
        sx, sy = pos_to_seq.get(px), pos_to_seq.get(py)
        if sx is None or sy is None or sx == sy:
            continue
        lo, hi = sorted((sx, sy))
        if (lo, hi) in seen:
            continue
        seen.add((lo, hi))
        parsed_i.append(lo)
        parsed_j.append(hi)
        raw_i.append(px)
        raw_j.append(py)
    return parsed_i, parsed_j, raw_i, raw_j


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--out", required=True, help="S3 prefix; parts land under <out>/<label>/")
    parser.add_argument("--label", required=True)
    parser.add_argument("--shard", required=True, help="i/n")
    parser.add_argument("--n-rollouts", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--contact-mult", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-per-request-seed",
        dest="per_request_seed",
        action="store_false",
        help=(
            "Required on TPU: the JAX backend rejects SamplingParams.seed. "
            "The engine-level --seed still applies; only bitwise replay of one "
            "specific rollout set is lost."
        ),
    )
    parser.add_argument("--gpu-frac", type=float, default=0.90)
    parser.add_argument("--chunk", type=int, default=8)
    parser.add_argument("--max-num-seqs", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    shard_i, num_shards = (int(x) for x in args.shard.split("/"))
    out_dir = f"{args.out.rstrip('/')}/{args.label}"

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from marinfold.document_structures.contacts_v1 import GenerationConfig, build_document, residues_from_sequence
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    records = load_targets(args.targets)
    mine = [r for idx, r in enumerate(records) if idx % num_shards == shard_i]
    skip, part = done_stems(out_dir, shard_i, num_shards)
    todo = [r for r in mine if f"{r['dataset']}__{r['stem']}" not in skip]
    if args.limit is not None:
        todo = todo[: args.limit]
    print(
        f"[raw-rollout] shard {shard_i}/{num_shards}: {len(mine)} assigned, "
        f"{len(skip)} done, {len(todo)} todo | label={args.label} n_rollouts={args.n_rollouts}",
        flush=True,
    )
    if not todo:
        return 0

    model_dir = stage_model(args.model, Path("/tmp/marinfold_model"))
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    end_id = tokenizer.convert_tokens_to_ids("<end>")
    if end_id is None or end_id < 0:
        raise ValueError("model tokenizer has no <end> token")
    llm = LLM(
        model=str(model_dir),
        dtype="bfloat16",
        max_model_len=8192,
        gpu_memory_utilization=args.gpu_frac,
        enable_prefix_caching=False,
        generation_config="vllm",
        max_num_seqs=args.max_num_seqs,
        seed=args.seed,
    )

    start_all = time.time()
    for offset in range(0, len(todo), args.chunk):
        group = todo[offset : offset + args.chunk]
        prompts: list[str] = []
        sampling_params = []
        per_protein = []
        for record in group:
            residues = residues_from_sequence(record["input_seq"])
            first = len(prompts)
            pos_maps = []
            prompt_token_count = None
            max_new = None
            for rollout_idx in range(args.n_rollouts):
                doc = build_document(f"{record['stem']}:r{rollout_idx}", residues, [], config=GenerationConfig())
                if doc is None:
                    raise ValueError(f"could not serialize {record['stem']} with L={len(residues)}")
                prompt = doc.document[: doc.document.index(BEGIN) + len(BEGIN)]
                prompts.append(prompt)
                pos_maps.append({(doc.n_term_index + seq_idx) % NUM_POS: seq_idx for seq_idx in range(doc.seq_len)})
                if prompt_token_count is None:
                    prompt_token_count = len(tokenizer(prompt, add_special_tokens=False).input_ids)
                    max_new = min(8192 - prompt_token_count, args.contact_mult * record["L"] + 128)
                sampling_params.append(
                    SamplingParams(
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        max_tokens=max_new,
                        stop_token_ids=[end_id],
                        skip_special_tokens=False,
                        **(
                            {"seed": args.seed * 1_000_003 + first + rollout_idx}
                            if args.per_request_seed
                            else {}
                        ),
                    )
                )
            per_protein.append((record, first, pos_maps, int(prompt_token_count), int(max_new)))

        start = time.time()
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        elapsed = time.time() - start

        rows = {name: [] for name in SCHEMA.names}
        for record, first, pos_maps, prompt_token_count, max_new in per_protein:
            for local_idx, (output, pos_map) in enumerate(zip(outputs[first : first + args.n_rollouts], pos_maps)):
                text = output.outputs[0].text
                first_generated = first_token(text)
                parsed_i, parsed_j, raw_i, raw_j = parse_contacts(text, pos_map)
                rows["dataset"].append(record["dataset"])
                rows["stem"].append(record["stem"])
                rows["L"].append(int(record["L"]))
                rows["input_seq"].append(record["input_seq"])
                rows["rollout_idx"].append(local_idx)
                rows["prompt_tokens"].append(prompt_token_count)
                rows["max_new_tokens"].append(max_new)
                rows["generated_tokens"].append(len(output.outputs[0].token_ids))
                rows["finish_reason"].append(str(output.outputs[0].finish_reason))
                rows["text"].append(text)
                rows["first_generated_token"].append(first_generated)
                rows["first_token_is_think"].append(first_generated == THINK_TOKEN)
                rows["think_token_count"].append(text.count(THINK_TOKEN))
                rows["parsed_i"].append(parsed_i)
                rows["parsed_j"].append(parsed_j)
                rows["raw_pos_i"].append(raw_i)
                rows["raw_pos_j"].append(raw_j)

        dest = f"{out_dir}/shard-{shard_i:03d}-of-{num_shards:03d}-part-{part:04d}.parquet"
        write_parquet(pa.table(rows, schema=SCHEMA), dest)
        part += 1
        n_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
        print(
            f"[raw-rollout] [{offset + len(group)}/{len(todo)}] L={group[0]['L']}-{group[-1]['L']} "
            f"{elapsed:.1f}s {n_tokens / max(elapsed, 1e-6):.0f} tok/s -> {dest} "
            f"(elapsed {(time.time() - start_all) / 60:.1f}m)",
            flush=True,
        )

    print(
        f"[raw-rollout] DONE shard {shard_i}/{num_shards}: {len(todo)} proteins in "
        f"{(time.time() - start_all) / 60:.1f} min",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
