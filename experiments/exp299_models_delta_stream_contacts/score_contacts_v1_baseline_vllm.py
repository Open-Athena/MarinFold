# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Score a contacts-v1 HF model on the exp89 contact eval set with vLLM."""

import argparse
import csv
import hashlib
import io
import platform
import random
import shutil
import socket
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import fsspec
import numpy as np
import pandas as pd

NUM_POS = 2000
BEGIN = "<begin_statements>"
DOC_TYPE = "<contacts-v1>"
BEGIN_SEQUENCE = "<begin_sequence>"
N_TERM = "<n-term>"
C_TERM = "<c-term>"
CONTACT = "<contact>"
AA_1_TO_3 = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}
DEFAULT_MANIFESTS = [
    "experiments/exp78_evals_esmfold_contacts/data/eval_manifest_foldbench.csv",
    "experiments/exp78_evals_esmfold_contacts/data/eval_manifest_exp65.csv",
]


def load_eval_proteins(manifests: list[str]) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    for manifest in manifests:
        df = pd.read_csv(manifest)
        for _, row in df.iterrows():
            rows.append((str(row["dataset"]), str(row["stem"]), str(row["input_seq"])))
    return rows


def _generation_seed(entry_id: str) -> int:
    return int(hashlib.sha1(entry_id.encode()).hexdigest()[:8], 16)


def prefix_and_positions(stem: str, input_seq: str):
    cleaned = "".join(input_seq.split()).upper()
    if not (2 <= len(cleaned) <= NUM_POS):
        return None
    rng = random.Random(_generation_seed(stem))
    start = rng.randrange(NUM_POS)
    seq_positions = [(start + k) % NUM_POS for k in range(len(cleaned))]
    statements: list[tuple[str, str]] = [
        (f"<p{pos}>", f"<{AA_1_TO_3.get(code, 'UNK')}>")
        for pos, code in zip(seq_positions, cleaned)
    ]
    statements.append((N_TERM, f"<p{seq_positions[0]}>"))
    statements.append((C_TERM, f"<p{seq_positions[-1]}>"))
    rng.shuffle(statements)
    tokens = [DOC_TYPE, BEGIN_SEQUENCE]
    for statement in statements:
        tokens.extend(statement)
    tokens.append(BEGIN)
    return " ".join(tokens), seq_positions, len(cleaned)


def stage_model(model: str, local: Path) -> str:
    if not (model.startswith("s3://") or model.startswith("gs://") or model.startswith("hf://")):
        return model
    if local.exists():
        shutil.rmtree(local)
    local.mkdir(parents=True)
    fs, root = fsspec.core.url_to_fs(model)
    files = fs.find(root)
    for src in files:
        rel = Path(src).relative_to(root)
        dst = local / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        fs.get(src, str(dst))
    return str(local)


def save_npz(path: str, score: np.ndarray) -> None:
    buf = io.BytesIO()
    np.savez_compressed(buf, score=score.astype(np.float16))
    with fsspec.open(path, "wb") as fh:
        fh.write(buf.getvalue())


def write_timings(path: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    with fsspec.open(path, "wt") as fh:
        fh.write(buf.getvalue())


def worker_metadata() -> dict[str, Any]:
    import torch

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu_name = props.name
        gpu_total_memory_gb: float | str = props.total_memory / 1e9
        gpu_compute_capability = f"{props.major}.{props.minor}"
    else:
        gpu_name = ""
        gpu_total_memory_gb = ""
        gpu_compute_capability = ""
    return {
        "model_nickname": "contacts-v1-baseline",
        "runner_tag": "iris-vllm",
        "gpu_name": gpu_name,
        "gpu_total_memory_gb": gpu_total_memory_gb,
        "gpu_compute_capability": gpu_compute_capability,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--timings-out", default=None)
    parser.add_argument("--manifest", action="append", default=DEFAULT_MANIFESTS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-logprobs", type=int, default=4096)
    args = parser.parse_args()

    from vllm import LLM, SamplingParams, TokensPrompt

    timings_out = args.timings_out or f"{args.out_dir.rstrip('/')}/timings.csv"
    model_load_start = time.perf_counter()
    model_path = stage_model(args.model, Path("/tmp/exp299_contacts_v1_baseline_model"))
    llm = LLM(
        model=model_path,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        enforce_eager=True,
        max_logprobs=args.max_logprobs,
        trust_remote_code=True,
    )
    model_load_seconds = time.perf_counter() - model_load_start
    tok = llm.get_tokenizer()
    contact_id = tok.convert_tokens_to_ids(CONTACT)

    def ptoken(pos: int) -> int:
        token_id = tok.convert_tokens_to_ids(f"<p{pos}>")
        if token_id is None:
            raise ValueError(f"missing token <p{pos}>")
        return int(token_id)

    params = SamplingParams(max_tokens=1, temperature=0.0, logprobs=args.max_logprobs)

    def next_logprobs(prompt_ids: list[list[int]]) -> list[dict[int, float]]:
        outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompt_ids], params)
        dists: list[dict[int, float]] = []
        for out in outs:
            logprobs = out.outputs[0].logprobs[0]
            dists.append({int(token_id): float(value.logprob) for token_id, value in logprobs.items()})
        return dists

    proteins = load_eval_proteins(args.manifest)
    if args.limit is not None:
        proteins = proteins[: args.limit]
    print(f"[contacts-v1-score] scoring {len(proteins)} proteins -> {args.out_dir}", flush=True)
    neg = float(np.log(1e-12))
    n_ok = 0
    timing_rows: list[dict[str, Any]] = []
    meta = worker_metadata()
    for idx, (dataset, stem, seq) in enumerate(proteins):
        built = prefix_and_positions(stem, seq)
        if built is None:
            continue
        total_start = time.perf_counter()
        prefix, seq_positions, length = built
        pos_ids = [ptoken(pos) for pos in seq_positions]
        base = list(tok(prefix, add_special_tokens=False).input_ids) + [contact_id]
        inference_start = time.perf_counter()
        first = next_logprobs([base])[0]
        lp1 = np.asarray([first.get(pos_id, neg) for pos_id in pos_ids], dtype=np.float64)
        second = next_logprobs([base + [pos_id] for pos_id in pos_ids])
        lp2 = np.asarray([[dist.get(pos_j, neg) for pos_j in pos_ids] for dist in second], dtype=np.float64)
        elapsed_seconds = time.perf_counter() - inference_start
        fwd = lp1[:, None] + lp2
        sym = 0.5 * (fwd + fwd.T)
        save_npz(f"{args.out_dir.rstrip('/')}/{dataset}__{stem}.npz", sym)
        timing_rows.append({
            "stem": stem,
            "n_residues": length,
            "n_pairs": int(length * (length - 1) // 2),
            "mode": "contacts_v1_logprob",
            "elapsed_seconds": elapsed_seconds,
            "model_load_seconds": model_load_seconds,
            "total_seconds": time.perf_counter() - total_start,
            **meta,
            "timestamp_utc": datetime.now(UTC).isoformat(),
        })
        n_ok += 1
        if (idx + 1) % 25 == 0:
            print(f"[contacts-v1-score] ...{idx + 1}/{len(proteins)} last={dataset}/{stem} L={length}", flush=True)
    write_timings(timings_out, timing_rows)
    print(f"[contacts-v1-score] scored={n_ok} out={args.out_dir} timings={timings_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
