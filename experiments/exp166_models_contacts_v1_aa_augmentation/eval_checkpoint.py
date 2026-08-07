# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate one HF checkpoint with the settled contacts-v1 rollout recipe.

The runner is intentionally single-model. Each checkpoint gets a separate Iris
job, local scratch directory, and durable output prefix. A timing part is the
commit marker for its matching sparse-vote part, making whole-chunk retries
safe after preemption.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import socket
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from huggingface_hub import HfFileSystem, snapshot_download
from marinfold.document_structures.contacts_v1 import (
    CONTEXT_LENGTH,
    NUM_POSITION_INDICES,
    GenerationConfig,
    build_document,
    residues_from_sequence,
)
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoTokenizer

from checkpoint_specs import (
    CHECKPOINTS,
    GROUND_TRUTH_SHA256,
    GROUND_TRUTH_URL,
    HF_BUCKET_ROOT,
    MARINFOLD_REVISION,
    TARGETS_URL,
    CheckpointSpec,
)

BEGIN_STATEMENTS = "<begin_statements>"
END_TOKEN = "<end>"
CONTACT_RE = re.compile(r"<contact>\s+<p(\d+)>\s+<p(\d+)>")
MIN_SEQUENCE_SEPARATION = 6
SCHEMA_VERSION = 1

VOTES_SCHEMA = pa.schema(
    [
        ("dataset", pa.string()),
        ("stem", pa.string()),
        ("n_residues", pa.int32()),
        ("i", pa.int16()),
        ("j", pa.int16()),
        ("votes", pa.int16()),
    ]
)

TIMINGS_SCHEMA = pa.schema(
    [
        ("dataset", pa.string()),
        ("stem", pa.string()),
        ("n_residues", pa.int32()),
        ("n_pairs", pa.int32()),
        ("mode", pa.string()),
        ("elapsed_seconds", pa.float64()),
        ("model_load_seconds", pa.float64()),
        ("total_seconds", pa.float64()),
        ("model_nickname", pa.string()),
        ("runner_tag", pa.string()),
        ("gpu_name", pa.string()),
        ("gpu_total_memory_gb", pa.float64()),
        ("gpu_compute_capability", pa.string()),
        ("hostname", pa.string()),
        ("platform", pa.string()),
        ("torch_version", pa.string()),
        ("timestamp_utc", pa.string()),
        ("checkpoint_prepare_seconds", pa.float64()),
        ("output_seconds", pa.float64()),
        ("n_rollouts", pa.int32()),
        ("generated_tokens", pa.int32()),
        ("stopped_rollouts", pa.int32()),
        ("parsed_contacts", pa.int32()),
        ("valid_contacts", pa.int32()),
        ("complete", pa.bool_()),
    ]
)


@dataclass(frozen=True)
class Target:
    """One contacts-v1 sequence evaluation target."""

    dataset: str
    stem: str
    n_residues: int
    input_sequence: str

    @property
    def key(self) -> tuple[str, str]:
        return self.dataset, self.stem


@dataclass(frozen=True)
class RolloutBatch:
    """The prompts and position maps for one protein's rollouts."""

    prompts: tuple[str, ...]
    position_maps: tuple[dict[int, int], ...]


@dataclass(frozen=True)
class Prediction:
    """Sparse vote rows plus per-input inference statistics."""

    vote_rows: tuple[dict[str, Any], ...]
    elapsed_seconds: float
    generated_tokens: int
    stopped_rollouts: int
    parsed_contacts: int
    valid_contacts: int


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_file(url: str, destination: Path) -> Path:
    """Download one small public input file, streaming it to scratch."""

    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.partial")
    with httpx.stream("GET", url, follow_redirects=True, timeout=120.0) as response:
        response.raise_for_status()
        with temporary.open("wb") as output:
            for block in response.iter_bytes(1024 * 1024):
                output.write(block)
    temporary.replace(destination)
    return destination


def load_targets(path: Path, *, limit: int | None = None) -> list[Target]:
    """Load and length-sort the fixed exp169/exp166 target set."""

    rows = pq.read_table(path).to_pylist()
    targets = [
        Target(
            dataset=str(row["dataset"]),
            stem=str(row["stem"]),
            n_residues=int(row["L"]),
            input_sequence=str(row["input_seq"]),
        )
        for row in rows
    ]
    for target in targets:
        if len(target.input_sequence) != target.n_residues:
            raise ValueError(
                f"{target.key}: sequence length {len(target.input_sequence)} "
                f"does not match L={target.n_residues}"
            )
    targets.sort(key=lambda target: (target.n_residues, *target.key))
    return targets if limit is None else targets[:limit]


def download_checkpoint(spec: CheckpointSpec, scratch: Path) -> Path:
    """Download only the selected HF subtree with one file worker."""

    repository = scratch / "download"
    started = time.monotonic()
    snapshot_download(
        repo_id=spec.repo_id,
        allow_patterns=[f"{spec.subfolder}/*"],
        local_dir=repository,
        max_workers=1,
    )
    source = repository / spec.subfolder
    files = [path for path in source.iterdir() if path.is_file()]
    if not files:
        raise FileNotFoundError(
            f"no checkpoint files under {spec.repo_id}/{spec.subfolder}"
        )
    size = sum(path.stat().st_size for path in files)
    print(
        f"[prepare] downloaded {len(files)} files ({size / 2**30:.2f} GiB) "
        f"in {time.monotonic() - started:.1f}s",
        flush=True,
    )
    return source


def recast_checkpoint(source: Path, destination: Path) -> int:
    """Copy an HF checkpoint while recasting floating tensors shardwise to BF16."""

    index_path = source / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"missing {index_path}")
    index = json.loads(index_path.read_text())
    shards = sorted(set(index["weight_map"].values()))
    destination.mkdir(parents=True, exist_ok=True)

    for path in source.iterdir():
        if path.is_file() and path.name not in {*shards, index_path.name}:
            shutil.copy2(path, destination / path.name)

    total = 0
    for shard in shards:
        tensors = load_file(source / shard, device="cpu")
        recast = {
            name: tensor.to(torch.bfloat16) if tensor.is_floating_point() else tensor
            for name, tensor in tensors.items()
        }
        save_file(recast, destination / shard, metadata={"format": "pt"})
        size = (destination / shard).stat().st_size
        total += size
        print(f"[prepare] recast {shard}: {size / 2**30:.2f} GiB", flush=True)
        del tensors, recast

    index.setdefault("metadata", {})["total_size"] = total
    (destination / index_path.name).write_text(json.dumps(index, sort_keys=True))
    return total


def validate_checkpoint(path: Path) -> tuple[Any, Any]:
    """Load checkpoint metadata and require dedicated contacts-v1 tokens."""

    config = AutoConfig.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(path)
    required = ("<end>", "<contact>", "<p0>")
    missing = []
    for token in required:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None or (
            tokenizer.unk_token_id is not None
            and token_id == tokenizer.unk_token_id
            and token != tokenizer.unk_token
        ):
            missing.append(token)
    if missing:
        raise ValueError(f"prepared tokenizer is missing tokens: {missing}")
    return config, tokenizer


def prepare_checkpoint(spec: CheckpointSpec, scratch: Path) -> tuple[Path, float]:
    """Download, stream-recast, and validate one checkpoint in local scratch."""

    started = time.monotonic()
    destination = scratch / "prepared-bf16"
    marker = destination / ".complete.json"
    marker_value = {"repo_id": spec.repo_id, "subfolder": spec.subfolder}
    if marker.exists() and json.loads(marker.read_text()) == marker_value:
        validate_checkpoint(destination)
        print(f"[prepare] reusing {destination}", flush=True)
        return destination, time.monotonic() - started

    source = download_checkpoint(spec, scratch)
    total = recast_checkpoint(source, destination)
    config, tokenizer = validate_checkpoint(destination)
    marker.write_text(json.dumps(marker_value, sort_keys=True))
    elapsed = time.monotonic() - started
    print(
        f"[prepare] validated {type(config).__name__}, vocab={len(tokenizer)}, "
        f"weights={total / 2**30:.2f} GiB in {elapsed:.1f}s",
        flush=True,
    )
    return destination, elapsed


def build_rollout_batch(target: Target, n_rollouts: int) -> RolloutBatch:
    """Build independently resampled contacts-v1 prompts for one protein."""

    residues = residues_from_sequence(target.input_sequence)
    prompts: list[str] = []
    position_maps: list[dict[int, int]] = []
    for rollout in range(n_rollouts):
        document = build_document(
            f"{target.stem}:r{rollout}",
            residues,
            [],
            config=GenerationConfig(),
        )
        if document is None:
            raise ValueError(f"could not serialize {target.key}")
        end = document.document.index(BEGIN_STATEMENTS) + len(BEGIN_STATEMENTS)
        prompts.append(document.document[:end])
        position_maps.append(
            {
                (document.n_term_index + offset) % NUM_POSITION_INDICES: offset
                for offset in range(document.seq_len)
            }
        )
    return RolloutBatch(tuple(prompts), tuple(position_maps))


def vote_matrix(
    texts: Sequence[str], position_maps: Sequence[dict[int, int]], n_residues: int
) -> tuple[np.ndarray, int, int]:
    """Parse completions into one symmetric-free upper-triangle vote matrix."""

    if len(texts) != len(position_maps):
        raise ValueError("completion and position-map counts differ")
    votes = np.zeros((n_residues, n_residues), dtype=np.int16)
    parsed_contacts = 0
    valid_contacts = 0
    for text, position_map in zip(texts, position_maps, strict=True):
        seen: set[tuple[int, int]] = set()
        contacts = CONTACT_RE.findall(text)
        parsed_contacts += len(contacts)
        for raw_i, raw_j in contacts:
            seq_i = position_map.get(int(raw_i))
            seq_j = position_map.get(int(raw_j))
            if seq_i is None or seq_j is None or seq_i == seq_j:
                continue
            i, j = sorted((seq_i, seq_j))
            if j - i < MIN_SEQUENCE_SEPARATION or (i, j) in seen:
                continue
            seen.add((i, j))
            votes[i, j] += 1
            valid_contacts += 1
    return votes, parsed_contacts, valid_contacts


def sparse_vote_rows(target: Target, votes: np.ndarray) -> tuple[dict[str, Any], ...]:
    """Return nonzero upper-triangle votes in the published sparse schema."""

    ii, jj = np.nonzero(np.triu(votes, k=1))
    return tuple(
        {
            "dataset": target.dataset,
            "stem": target.stem,
            "n_residues": target.n_residues,
            "i": int(i),
            "j": int(j),
            "votes": int(votes[i, j]),
        }
        for i, j in zip(ii, jj, strict=True)
    )


def load_llm(model_path: Path, *, tensor_parallel_size: int, max_num_seqs: int):
    """Load the pinned TPU vLLM backend lazily so CPU unit tests stay lightweight."""

    # recast_checkpoint exercises PyTorch's GNU OpenMP runtime. Forking after
    # that can leave EngineCore with corrupted OpenMP state while it copies BF16
    # weights. Spawn gives the engine a clean runtime.
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    from vllm import LLM

    started = time.monotonic()
    llm = LLM(
        model=str(model_path),
        dtype="bfloat16",
        max_model_len=CONTEXT_LENGTH,
        gpu_memory_utilization=0.90,
        enable_prefix_caching=False,
        generation_config="vllm",
        max_num_seqs=max_num_seqs,
        seed=0,
        tensor_parallel_size=tensor_parallel_size,
    )
    elapsed = time.monotonic() - started
    print(f"[model] loaded in {elapsed:.1f}s", flush=True)
    return llm, elapsed


def predict_target(llm, target: Target, *, n_rollouts: int) -> Prediction:
    """Run all resampled rollouts for one target as one vLLM request batch."""

    from vllm import SamplingParams

    batch = build_rollout_batch(target, n_rollouts)
    tokenizer = llm.get_tokenizer()
    end_id = tokenizer.convert_tokens_to_ids(END_TOKEN)
    if end_id is None:
        raise ValueError(f"tokenizer has no {END_TOKEN!r} token")
    prompt_tokens = len(tokenizer(batch.prompts[0], add_special_tokens=False).input_ids)
    max_tokens = min(CONTEXT_LENGTH - prompt_tokens, 6 * target.n_residues + 128)

    started = time.monotonic()
    outputs = llm.generate(
        list(batch.prompts),
        SamplingParams(
            temperature=1.0,
            top_p=0.95,
            top_k=-1,
            max_tokens=max_tokens,
            stop_token_ids=[end_id],
            skip_special_tokens=False,
        ),
        use_tqdm=False,
    )
    elapsed = time.monotonic() - started
    texts = [output.outputs[0].text for output in outputs]
    votes, parsed_contacts, valid_contacts = vote_matrix(
        texts, batch.position_maps, target.n_residues
    )
    return Prediction(
        vote_rows=sparse_vote_rows(target, votes),
        elapsed_seconds=elapsed,
        generated_tokens=sum(len(output.outputs[0].token_ids) for output in outputs),
        stopped_rollouts=sum(
            output.outputs[0].finish_reason == "stop" for output in outputs
        ),
        parsed_contacts=parsed_contacts,
        valid_contacts=valid_contacts,
    )


class BucketOutput:
    """Durable HF bucket parts and their timing-file completion markers."""

    def __init__(self, prefix: str, token: str):
        self.prefix = prefix.removeprefix("hf://").rstrip("/")
        self.fs = HfFileSystem(token=token)

    def _path(self, relative: str) -> str:
        return f"{self.prefix}/{relative}"

    def ensure_manifest(self, manifest: dict[str, Any]) -> None:
        path = self._path("manifest.json")
        self.fs.invalidate_cache(path)
        if self.fs.exists(path):
            with self.fs.open(path, "r") as source:
                existing = json.load(source)
            if existing != manifest:
                raise ValueError(
                    f"existing output manifest does not match this run: {path}"
                )
            return
        with self.fs.open(path, "w") as output:
            json.dump(manifest, output, indent=2, sort_keys=True)

    def part_complete(self, part: int, expected: Sequence[Target]) -> bool:
        path = self._path(f"parts/timings-{part:04d}.parquet")
        self.fs.invalidate_cache(path)
        if not self.fs.exists(path):
            return False
        try:
            with self.fs.open(path, "rb") as source:
                rows = pq.read_table(
                    source, columns=["dataset", "stem", "complete"]
                ).to_pylist()
        except (FileNotFoundError, OSError, pa.ArrowInvalid) as error:
            print(f"[resume] ignoring unreadable completion marker {path}: {error}")
            return False
        observed = {(row["dataset"], row["stem"]) for row in rows if row["complete"]}
        return observed == {target.key for target in expected}

    def upload(self, source: Path, relative: str) -> None:
        with (
            source.open("rb") as local,
            self.fs.open(self._path(relative), "wb") as remote,
        ):
            shutil.copyfileobj(local, remote, length=1024 * 1024)


def _table(rows: Sequence[dict[str, Any]], schema: pa.Schema) -> pa.Table:
    return pa.Table.from_pylist(list(rows), schema=schema)


def _timing_row(
    target: Target,
    prediction: Prediction,
    spec: CheckpointSpec,
    *,
    prepare_seconds: float,
    model_load_seconds: float,
    n_rollouts: int,
) -> dict[str, Any]:
    n_pairs = max(0, (target.n_residues - 6) * (target.n_residues - 5) // 2)
    return {
        "dataset": target.dataset,
        "stem": target.stem,
        "n_residues": target.n_residues,
        "n_pairs": n_pairs,
        "mode": "rollout_resample",
        "elapsed_seconds": prediction.elapsed_seconds,
        "model_load_seconds": model_load_seconds,
        "total_seconds": prepare_seconds
        + model_load_seconds
        + prediction.elapsed_seconds,
        "model_nickname": spec.output_name,
        "runner_tag": "iris",
        "gpu_name": os.environ.get("MARINFOLD_ACCELERATOR", "v6e-4"),
        "gpu_total_memory_gb": None,
        "gpu_compute_capability": "",
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "checkpoint_prepare_seconds": prepare_seconds,
        "output_seconds": 0.0,
        "n_rollouts": n_rollouts,
        "generated_tokens": prediction.generated_tokens,
        "stopped_rollouts": prediction.stopped_rollouts,
        "parsed_contacts": prediction.parsed_contacts,
        "valid_contacts": prediction.valid_contacts,
        "complete": True,
    }


def write_part(
    output: BucketOutput,
    scratch: Path,
    part: int,
    vote_rows: Sequence[dict[str, Any]],
    timing_rows: list[dict[str, Any]],
) -> None:
    """Upload votes first and timings last; timings are the completion marker."""

    local = scratch / "output"
    local.mkdir(parents=True, exist_ok=True)
    votes_path = local / f"votes-{part:04d}.parquet"
    timings_path = local / f"timings-{part:04d}.parquet"
    started = time.monotonic()
    pq.write_table(_table(vote_rows, VOTES_SCHEMA), votes_path, compression="zstd")
    output.upload(votes_path, f"parts/{votes_path.name}")
    output_seconds = time.monotonic() - started
    share = output_seconds / max(len(timing_rows), 1)
    for row in timing_rows:
        row["output_seconds"] = share
        row["total_seconds"] += share
    pq.write_table(
        _table(timing_rows, TIMINGS_SCHEMA), timings_path, compression="zstd"
    )
    output.upload(timings_path, f"parts/{timings_path.name}")
    print(
        f"[output] part {part:04d}: {len(timing_rows)} targets, "
        f"{len(vote_rows)} nonzero pairs -> {output.prefix}",
        flush=True,
    )


def run(args: argparse.Namespace) -> int:
    """Run or resume one independently submitted checkpoint evaluation."""

    spec = CHECKPOINTS[args.checkpoint]
    args.scratch.mkdir(parents=True, exist_ok=True)
    if args.limit is not None and args.output_prefix is None:
        raise ValueError("--limit requires a separate --output-prefix")
    targets_path = download_file(
        args.targets_url, args.scratch / "eval_targets.parquet"
    )
    ground_truth_path = download_file(
        args.ground_truth_url, args.scratch / "gt_universe.jsonl"
    )
    targets = load_targets(targets_path, limit=args.limit)
    target_digest = _sha256(targets_path)
    ground_truth_digest = _sha256(ground_truth_path)
    if ground_truth_digest != GROUND_TRUTH_SHA256:
        raise ValueError(
            "ground-truth digest mismatch: "
            f"expected {GROUND_TRUTH_SHA256}, got {ground_truth_digest}"
        )
    output_prefix = args.output_prefix or f"{HF_BUCKET_ROOT}/scores/{spec.output_name}"
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN is required for durable result writes")
    output = BucketOutput(output_prefix, token)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "checkpoint": asdict(spec),
        "marinfold_revision": MARINFOLD_REVISION,
        "targets_url": args.targets_url,
        "targets_sha256": target_digest,
        "ground_truth_url": args.ground_truth_url,
        "ground_truth_sha256": ground_truth_digest,
        "n_targets": len(targets),
        "n_rollouts": args.n_rollouts,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": -1,
        "max_tokens": "min(8192-prompt_tokens, 6*L+128)",
        "min_sequence_separation": MIN_SEQUENCE_SEPARATION,
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_num_seqs": args.max_num_seqs,
        "versions": {
            name: _package_version(name)
            for name in ("marinfold", "vllm", "tpu-inference", "transformers", "torch")
        },
    }
    output.ensure_manifest(manifest)
    output.upload(targets_path, "inputs/eval_targets.parquet")
    output.upload(ground_truth_path, "inputs/gt_universe.jsonl")

    chunks = [
        targets[i : i + args.part_size] for i in range(0, len(targets), args.part_size)
    ]
    pending = [
        (part, chunk)
        for part, chunk in enumerate(chunks)
        if not output.part_complete(part, chunk)
    ]
    if not pending:
        print(f"[eval] all {len(targets)} targets already complete", flush=True)
        return 0
    print(
        f"[eval] {len(targets)} targets, {len(pending)}/{len(chunks)} parts pending; "
        f"checkpoint={spec.key}",
        flush=True,
    )

    model_path, prepare_seconds = prepare_checkpoint(spec, args.scratch)
    llm, model_load_seconds = load_llm(
        model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        max_num_seqs=args.max_num_seqs,
    )
    charge_setup = True
    completed = 0
    for part, chunk in pending:
        votes: list[dict[str, Any]] = []
        timings: list[dict[str, Any]] = []
        for target in chunk:
            prediction = predict_target(llm, target, n_rollouts=args.n_rollouts)
            votes.extend(prediction.vote_rows)
            timings.append(
                _timing_row(
                    target,
                    prediction,
                    spec,
                    prepare_seconds=prepare_seconds if charge_setup else 0.0,
                    model_load_seconds=model_load_seconds if charge_setup else 0.0,
                    n_rollouts=args.n_rollouts,
                )
            )
            charge_setup = False
            completed += 1
            rate = prediction.generated_tokens / max(prediction.elapsed_seconds, 1e-9)
            print(
                f"[eval] {completed}/{sum(len(chunk) for _, chunk in pending)} "
                f"{target.dataset}/{target.stem} L={target.n_residues} "
                f"{prediction.elapsed_seconds:.2f}s {rate:.0f} tok/s "
                f"stop={prediction.stopped_rollouts}/{args.n_rollouts}",
                flush=True,
            )
        write_part(output, args.scratch, part, votes, timings)
    print(f"[eval] complete: {len(targets)} targets -> {output.prefix}", flush=True)
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", choices=sorted(CHECKPOINTS), required=True)
    parser.add_argument(
        "--scratch", type=Path, default=Path("/app/scratch/exp166-eval")
    )
    parser.add_argument("--targets-url", default=TARGETS_URL)
    parser.add_argument("--ground-truth-url", default=GROUND_TRUTH_URL)
    parser.add_argument("--output-prefix")
    parser.add_argument("--n-rollouts", type=int, default=100)
    parser.add_argument("--part-size", type=int, default=16)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--max-num-seqs", type=int, default=128)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    if args.n_rollouts < 1 or args.part_size < 1:
        parser.error("--n-rollouts and --part-size must be positive")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
