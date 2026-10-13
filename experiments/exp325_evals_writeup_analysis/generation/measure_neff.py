"""Measure exact Neff80 on the frozen alignments used by exp325's predictors.

Neff = sum_i 1 / count_j(identity(i,j) >= 0.8). Identity counts matching
non-gap symbols divided by columns non-gap in both sequences. Insertions are
dropped by exp74's parser. All original records, including duplicate queries,
contribute. Unique rows compress the computation with multiplicity weights;
CUDA binary matrix products count exact matches/overlaps in blocks. Each MSA
also checks sampled rows against direct integer NumPy comparisons.

Run with the existing CUDA environment:
uv run --project /home/bizon/git/helico --no-sync python generation/measure_neff.py
"""

import argparse
import csv
import gzip
import hashlib
import json
import platform
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
REFERENCE = HERE.parent / "exp74_evals_protenix_pyconfind_contacts/msa_depth.py"
sys.path.insert(0, str(REFERENCE.parent))
from msa_depth import a3m_to_match_array, parse_a3m_sequences


def digest(path: Path) -> str:
    """Return the SHA256 of an input or output file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def neighbor_counts(arr: np.ndarray, *, block_size: int = 256) -> np.ndarray:
    """Return exact Neff80 neighbor counts in original alignment record order."""
    if arr.ndim != 2 or min(arr.shape) == 0 or np.any((arr != ord("-")).sum(axis=1) == 0):
        raise ValueError("Neff requires nonempty sequences with at least one non-gap column")
    unique, inverse, multiplicity = np.unique(arr, axis=0, return_inverse=True, return_counts=True)
    symbols, encoded = np.unique(unique, return_inverse=True)
    indices = torch.as_tensor(encoded.reshape(unique.shape), device="cuda", dtype=torch.int64)
    one_hot = torch.zeros((*unique.shape, len(symbols)), device="cuda", dtype=torch.float32)
    one_hot.scatter_(2, indices.unsqueeze(-1), 1.0)
    gap = np.flatnonzero(symbols == ord("-"))
    if len(gap):
        one_hot[:, :, int(gap[0])] = 0
    features = one_hot.reshape(len(unique), -1)
    nongap = torch.as_tensor(unique != ord("-"), device="cuda", dtype=torch.float32)
    weights = torch.as_tensor(multiplicity, device="cuda", dtype=torch.int64)
    counts = torch.empty(len(unique), device="cuda", dtype=torch.int64)
    for start in range(0, len(unique), block_size):
        end = start + block_size
        matches = features[start:end] @ features.T
        overlap = nongap[start:end] @ nongap.T
        neighbors = (overlap > 0) & (5 * matches >= 4 * overlap)
        counts[start:end] = (neighbors.to(torch.int64) * weights).sum(dim=1)
    result = counts.cpu().numpy()[inverse]
    if np.any(result < 1):
        raise ValueError("Every sequence must count itself")
    return result


def audit_counts(arr: np.ndarray, counts: np.ndarray) -> int:
    """Cross-check sampled original rows against direct integer comparisons."""
    indices = np.unique(np.linspace(0, len(arr) - 1, min(8, len(arr)), dtype=int))
    for index in indices:
        both = (arr != ord("-")) & (arr[index] != ord("-"))
        overlap = both.sum(axis=1)
        matches = ((arr == arr[index]) & both).sum(axis=1)
        expected = np.count_nonzero((overlap > 0) & (5 * matches >= 4 * overlap))
        if counts[index] != expected:
            raise ValueError(f"CUDA/reference disagreement at row {index}: {counts[index]} != {expected}")
    return len(indices)


def main() -> None:
    """Validate frozen MSAs, cache per-record counts, and publish the small table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=HERE / "scratch/alphafold/inputs")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    pins = json.loads((HERE / "data/alphafold_inputs.json").read_text())
    targets_path = args.inputs / "targets.json"
    if digest(targets_path) != pins["inputs_sha256"]:
        raise ValueError("Frozen predictor input manifest changed")
    targets = json.loads(targets_path.read_text())
    if len(targets) != 333 or len({r["stem"] for r in targets}) != 333:
        raise ValueError("Expected all 333 original predictor inputs")
    # Start shallow, so the first outputs provide an immediate definition check.
    targets = sorted(targets, key=lambda r: (r["msa_depth"], r["stem"]))
    if args.limit:
        targets = targets[:args.limit]
    cache = HERE / "scratch/neff/counts"
    cache.mkdir(parents=True, exist_ok=True)
    worker_sha = digest(Path(__file__))
    metadata = dict(runner_tag="local", gpu_name=torch.cuda.get_device_name(),
                    gpu_total_memory_gb=torch.cuda.get_device_properties(0).total_memory / 2**30,
                    hostname=socket.gethostname(), platform=platform.platform(), torch_version=torch.__version__)
    records = []
    for i, target in enumerate(targets):
        stem = target["stem"]
        source = args.inputs / f"{stem}.a3m.gz"
        if digest(source) != target["msa_sha256"]:
            raise ValueError(f"Frozen alignment changed: {stem}")
        cached = cache / f"{stem}.json"
        count_path = cache / f"{stem}.npz"
        if cached.exists():
            record = json.loads(cached.read_text())
            if (record["worker_sha256"] != worker_sha or record["msa_sha256"] != target["msa_sha256"]
                    or digest(count_path) != record["counts_sha256"]):
                raise ValueError(f"Stale cached Neff measurement: {stem}")
            records.append(record)
            continue
        started = time.perf_counter()
        text = gzip.decompress(source.read_bytes()).decode()
        sequences = parse_a3m_sequences(text)
        if sequences[0] != target["sequence"] or len(sequences) != target["msa_depth"]:
            raise ValueError(f"Alignment query/count mismatch: {stem}")
        arr = a3m_to_match_array(sequences)
        if arr.shape[1] != target["n_residues"]:
            raise ValueError(f"Alignment query length mismatch: {stem}")
        counts = neighbor_counts(arr)
        validated = audit_counts(arr, counts)
        neff = float(np.sum(1.0 / counts))
        if not 1 - 1e-10 <= neff <= len(arr):
            raise ValueError(f"Neff outside [1, depth]: {stem}: {neff}")
        np.savez_compressed(count_path, neighbor_counts=counts)
        record = dict(stem=stem, msa_depth=len(arr), msa_query_length=arr.shape[1], msa_neff=neff,
                      identity_threshold=0.8, msa_sha256=target["msa_sha256"],
                      msa_text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                      counts_sha256=digest(count_path), validated_rows=validated,
                      elapsed_seconds=time.perf_counter() - started, worker_sha256=worker_sha,
                      reference_sha256=digest(REFERENCE), timestamp_utc=datetime.now(timezone.utc).isoformat(), **metadata)
        cached.write_text(json.dumps(record, indent=2) + "\n")
        records.append(record)
        print(f"{i + 1}/{len(targets)} {stem}: depth={len(arr)} Neff80={neff:.3f} ({record['elapsed_seconds']:.1f}s)", flush=True)
    output = HERE / ("scratch/neff/smoke.csv" if args.limit else "data/msa_neff.csv")
    with output.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(sorted(records, key=lambda r: r["stem"]))
    print(output, flush=True)


if __name__ == "__main__":
    main()
