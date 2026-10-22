# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Fold and score an ESMFold2 ensemble on the fly for one manifest shard.

For each protein this worker samples ``n_samples`` ESMFold2 structures, runs
pyconfind on each transient mmCIF, averages predicted contact degrees over the
samples, and scores the resulting contact-ranking matrix against emitted
contacts-v1 ground-truth pairs. It writes metrics only; per-sample structures are
not persisted.
"""

import argparse
import csv
import difflib
import json
import math
import platform
import socket
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fsspec
import gemmi
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from esm.models.esmfold2 import ESMFold2InputBuilder, EsmFold2Model, ProteinInput, StructurePredictionInput
from marinfold.document_structures.contacts_v1 import analyze_structure

HF_MODEL_ID = "biohub/ESMFold2"
MIN_SEQ_SEP = 6
RANGES: dict[str, tuple[int, int | None]] = {
    "all": (6, None),
    "short": (6, 11),
    "medium": (12, 23),
    "long": (24, None),
}
CUTS = ("L", "L/2", "L/5", "R")
PYCONFIND_KWARGS = dict(native_only=True, contact_distance=3.0, dcut=25.0, clash_distance=2.0, assembly=None)
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


@dataclass(frozen=True)
class ContactResult:
    chain: str
    n_resolved_residues: int
    n_mapped_residues: int
    alignment_identity: float
    contacts: tuple[tuple[int, int, float], ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gt-parquet", type=Path, required=True)
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--num-loops", type=int, default=20)
    parser.add_argument("--num-sampling-steps", type=int, default=100)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def one_letter(canonical_resname: str) -> str:
    return THREE_TO_ONE.get(canonical_resname, "X")


def align_obs_to_ref(obs: str, ref: str) -> list[int | None]:
    if obs == ref:
        return list(range(len(obs)))
    matcher = difflib.SequenceMatcher(a=obs, b=ref, autojunk=False)
    mapping: list[int | None] = [None] * len(obs)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                mapping[i1 + k] = j1 + k
        elif tag == "replace":
            for k in range(min(i2 - i1, j2 - j1)):
                mapping[i1 + k] = j1 + k
    return mapping


def extract_single_chain(structure: gemmi.Structure) -> tuple[gemmi.Structure, str]:
    st = structure.clone()
    st.setup_entities()
    while len(st) > 1:
        del st[1]
    model = st[0]

    def pep_len(chain: gemmi.Chain) -> int:
        try:
            return len(chain.get_polymer())
        except Exception:
            return 0

    candidates = [(chain.name, pep_len(chain)) for chain in model]
    candidates = [(name, n) for name, n in candidates if n > 0]
    if not candidates:
        raise ValueError("no polymer peptide chain found")
    chosen = max(candidates, key=lambda item: item[1])[0]
    for name in [chain.name for chain in list(model)]:
        if name != chosen:
            model.remove_chain(name)
    st.remove_ligands_and_waters()
    st.remove_empty_chains()
    return st, chosen


def compute_pred_contacts_from_mmcif(mmcif: str, input_seq: str, stem: str, tmp_path: Path) -> ContactResult:
    cif_path = tmp_path / f"{stem}.cif"
    cif_path.write_text(mmcif)
    st, chain = extract_single_chain(gemmi.read_structure(str(cif_path)))
    analyzed = analyze_structure(st, entry_id=stem, **PYCONFIND_KWARGS)
    obs = "".join(one_letter(res.resname) for res in analyzed.residues)
    mapping = align_obs_to_ref(obs, input_seq)
    matched = sum(1 for idx, seq_idx in enumerate(mapping) if seq_idx is not None and obs[idx] == input_seq[seq_idx])
    identity = matched / len(obs) if obs else 0.0
    contacts: list[tuple[int, int, float]] = []
    for contact in analyzed.contacts:
        ci = mapping[contact.seq_i] if contact.seq_i < len(mapping) else None
        cj = mapping[contact.seq_j] if contact.seq_j < len(mapping) else None
        if ci is None or cj is None or ci == cj:
            continue
        i, j = (ci, cj) if ci < cj else (cj, ci)
        contacts.append((i, j, float(contact.degree)))
    contacts.sort()
    return ContactResult(
        chain=chain,
        n_resolved_residues=len(analyzed.residues),
        n_mapped_residues=sum(1 for value in mapping if value is not None),
        alignment_identity=identity,
        contacts=tuple(contacts),
    )


def score_confidence(result: Any) -> float:
    for attr in ("ptm", "mean_plddt", "plddt", "confidence"):
        value = getattr(result, attr, None)
        if value is None and hasattr(result, "complex"):
            value = getattr(result.complex, attr, None)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(np.asarray(value, dtype=float).mean())
    return math.nan


def target_for_cut(cut: str, seq_len: int, n_true: int) -> int:
    if cut == "L":
        return seq_len
    if cut == "L/2":
        return max(1, seq_len // 2)
    if cut == "L/5":
        return max(1, seq_len // 5)
    if cut == "R":
        return n_true
    raise ValueError(cut)


def auc_from_scores(scores: np.ndarray, labels: np.ndarray) -> float:
    n_pos = int(labels.sum())
    n = int(labels.size)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return math.nan
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(n, dtype=np.float64)
    start = 0
    while start < n:
        end = start + 1
        while end < n and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    sum_pos_ranks = float(ranks[labels.astype(bool)].sum())
    return (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def metric_rows(stem: str, seq_len: int, gt_i: list[int], gt_j: list[int], score: np.ndarray) -> list[dict[str, Any]]:
    true = np.zeros((seq_len, seq_len), dtype=bool)
    for i, j in zip(gt_i, gt_j, strict=True):
        if 0 <= i < j < seq_len and (j - i) >= MIN_SEQ_SEP:
            true[i, j] = True
    pair_i, pair_j = np.triu_indices(seq_len, k=1)
    pair_sep = pair_j - pair_i
    cand_score_all = score[pair_i, pair_j]
    cand_true_all = true[pair_i, pair_j]
    rows: list[dict[str, Any]] = []
    for range_name, (lo, hi) in RANGES.items():
        in_range = pair_sep >= lo
        if hi is not None:
            in_range &= pair_sep <= hi
        scores = cand_score_all[in_range]
        labels = cand_true_all[in_range].astype(np.int64)
        n_candidate = int(scores.size)
        n_true = int(labels.sum())
        auc = auc_from_scores(scores, labels) if n_candidate else math.nan
        order = np.argsort(-scores, kind="mergesort") if n_candidate else np.array([], dtype=np.int64)
        labels_sorted = labels[order] if n_candidate else labels
        for cut in CUTS:
            target = target_for_cut(cut, seq_len, n_true)
            if n_candidate == 0 or target <= 0:
                precision = math.nan
                n_top = 0
            else:
                n_top = min(target, n_candidate)
                precision = float(labels_sorted[:n_top].sum()) / n_top
            rows.append(
                {
                    "stem": stem,
                    "model": "esmfold2",
                    "ensemble": "mean_contact_degree",
                    "range": range_name,
                    "cut": cut,
                    "precision": precision,
                    "auc": auc,
                    "n_candidate": n_candidate,
                    "n_true": n_true,
                    "n_top": n_top,
                }
            )
    return rows


def load_manifest(path: Path, shard_index: int, num_shards: int, limit: int | None) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for idx, row in enumerate(rows) if idx % num_shards == shard_index]
    if limit is not None:
        rows = rows[:limit]
    return rows


def worker_meta() -> dict[str, Any]:
    meta: dict[str, Any] = {"hostname": socket.gethostname(), "platform": platform.platform(), "torch_version": torch.__version__, "model": HF_MODEL_ID}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        meta.update({"gpu_name": props.name, "gpu_total_memory_gb": round(props.total_memory / 1e9, 2), "gpu_compute_capability": f"{props.major}.{props.minor}"})
    return meta


def write_dataframe(uri: str, df: pd.DataFrame) -> None:
    with fsspec.open(uri, "wt") as handle:
        df.to_csv(handle, index=False)


def main() -> None:
    args = parse_args()
    rows = load_manifest(args.manifest, args.shard_index, args.num_shards, args.limit)
    gt_df = pq.read_table(args.gt_parquet).to_pandas()
    gt_by_stem = {row.stem: row for row in gt_df.itertuples(index=False)}
    print(f"[esmfold2-ensemble] shard {args.shard_index}/{args.num_shards}; rows={len(rows)}; n_samples={args.n_samples}", flush=True)
    print(f"[esmfold2-ensemble] torch cuda={torch.cuda.is_available()}", flush=True)
    model_load_start = time.time()
    model = EsmFold2Model.from_pretrained(HF_MODEL_ID).cuda().eval()
    model_load_seconds = time.time() - model_load_start
    builder = ESMFold2InputBuilder()
    meta = worker_meta()
    print(f"[esmfold2-ensemble] meta={meta}; model_load_seconds={model_load_seconds:.2f}", flush=True)

    metric_out: list[dict[str, Any]] = []
    meta_out: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for idx, row in enumerate(rows, start=1):
            stem = row["stem"]
            seq = row["sequence"]
            gt = gt_by_stem[stem]
            t0 = time.time()
            print(f"[esmfold2-ensemble] {idx}/{len(rows)} {stem} L={len(seq)}", flush=True)
            try:
                score_sum = np.zeros((len(seq), len(seq)), dtype=np.float64)
                sample_scores: list[dict[str, Any]] = []
                align_identities: list[float] = []
                pred_contact_counts: list[int] = []
                spi = StructurePredictionInput(sequences=[ProteinInput(id="A", sequence=seq)])
                for seed in range(args.n_samples):
                    result = builder.fold(
                        model,
                        spi,
                        num_loops=args.num_loops,
                        num_sampling_steps=args.num_sampling_steps,
                        num_diffusion_samples=1,
                        seed=seed,
                    )
                    confidence = score_confidence(result)
                    pred = compute_pred_contacts_from_mmcif(result.complex.to_mmcif(), seq, stem, tmp_path)
                    sample_scores.append({"seed": seed, "confidence": confidence})
                    align_identities.append(pred.alignment_identity)
                    pred_contact_counts.append(len(pred.contacts))
                    for i, j, degree in pred.contacts:
                        if 0 <= i < j < len(seq):
                            score_sum[i, j] += max(0.0, degree)
                    if (seed + 1) % 10 == 0 or seed + 1 == args.n_samples:
                        print(f"[esmfold2-ensemble] {stem} sample {seed + 1}/{args.n_samples}", flush=True)
                score = score_sum / args.n_samples
                metrics = metric_rows(stem, len(seq), list(gt.gt_contact_i), list(gt.gt_contact_j), score)
                metric_out.extend(metrics)
                r_all = next(m["precision"] for m in metrics if m["range"] == "all" and m["cut"] == "R")
                meta_out.append(
                    {
                        "stem": stem,
                        "seq_len": len(seq),
                        "n_gt_contacts": int(gt.n_gt_contacts),
                        "n_samples": args.n_samples,
                        "mean_sample_confidence": float(np.nanmean([s["confidence"] for s in sample_scores])),
                        "mean_pred_align_identity": float(np.nanmean(align_identities)),
                        "mean_pred_contacts_raw": float(np.mean(pred_contact_counts)),
                        "model_load_seconds": round(model_load_seconds, 4),
                        "elapsed_seconds": round(time.time() - t0, 4),
                        "status": "ok",
                        **meta,
                    }
                )
                print(f"[esmfold2-ensemble] done {stem}: R_all={r_all:.4f} elapsed={time.time() - t0:.1f}s", flush=True)
            except Exception as error:
                failure = {"stem": stem, "seq_len": len(seq), "n_samples": args.n_samples, "elapsed_seconds": round(time.time() - t0, 4), "status": "failed", "error": repr(error)}
                failures.append(failure)
                meta_out.append(failure)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"[esmfold2-ensemble] FAILED {stem}: {error!r}", flush=True)

    out_prefix = args.out_prefix.rstrip("/")
    shard = f"shard-{args.shard_index:03d}-of-{args.num_shards:03d}"
    metric_df = pd.DataFrame(metric_out)
    meta_df = pd.DataFrame(meta_out)
    failure_df = pd.DataFrame(failures)
    write_dataframe(f"{out_prefix}/score_shards/{shard}.contact_precision.csv", metric_df)
    write_dataframe(f"{out_prefix}/score_shards/{shard}.meta.csv", meta_df)
    write_dataframe(f"{out_prefix}/score_shards/{shard}.failures.csv", failure_df)
    summary = {"timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "shard_index": args.shard_index, "num_shards": args.num_shards, "n_rows": len(rows), "n_ok": int((meta_df.get("status") == "ok").sum()) if len(meta_df) else 0, "n_failed": len(failures)}
    if len(metric_df):
        r_all = metric_df[(metric_df["range"] == "all") & (metric_df["cut"] == "R")]["precision"]
        summary["mean_r_precision_all"] = float(r_all.mean())
    with fsspec.open(f"{out_prefix}/score_shards/{shard}.summary.json", "wt") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(f"[esmfold2-ensemble] summary {summary}", flush=True)


if __name__ == "__main__":
    main()
