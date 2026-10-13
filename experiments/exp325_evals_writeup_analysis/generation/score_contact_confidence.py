"""Rescore archived structural contacts using residue pLDDT, without inference.

For atom-level confidence heads, residue confidence is the arithmetic mean over
the residue's deposited non-hydrogen atoms. ESMFold stores confidence on [0,1];
the other archives store [0,100]. All are normalized to [0,1]. Only ConFind
degree>0 pairs enter the new rankings; truth is consulted after selection.
The original all-candidate degree ranking must reproduce both archived P@L/5
and R-precision for every input and range before alternative scores are saved.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import platform
import socket
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
EXPERIMENTS = HERE.parent
CACHE = HERE / "scratch/contact_confidence"
sys.path.insert(0, str(EXPERIMENTS / "exp78_evals_esmfold_contacts"))
sys.path.insert(0, str(EXPERIMENTS / "exp89_evals_contacts_v1_model_on_eval_set"))
sys.path.insert(0, str(HERE.parents[1] / "marinfold"))
from compute_metrics import degree_matrix, metric_rows, resolved_pairs, true_matrix
from pyconfind_contacts import PYCONFIND_KWARGS, _one_letter, align_obs_to_ref, extract_single_chain
from marinfold.document_structures.contacts_v1 import analyze_structure

RANKINGS = ("degree", "plddt_product", "plddt_product_degree")


def digest(path: Path) -> str:
    """Hash a small artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def residue_confidence(path: Path, sequence: str, method: str) -> tuple[np.ndarray, list, float]:
    """Map pLDDT using the exact same analyzed residues as ConFind contacts."""
    structure, chain = extract_single_chain(path)
    analyzed = analyze_structure(structure, entry_id=path.parent.name, **PYCONFIND_KWARGS)
    observed = "".join(_one_letter(r.resname) for r in analyzed.residues)
    mapping = align_obs_to_ref(observed, sequence)
    residues = {int(r.seqid.num): r for r in structure[0][chain]}
    if len(residues) != len(structure[0][chain]):
        raise ValueError(f"Ambiguous residue numbering: {path}")
    confidence = np.full(len(sequence), np.nan)
    scale = 1.0 if method == "esmfold" else 100.0
    for record, position in zip(analyzed.residues, mapping, strict=True):
        if position is None:
            continue
        atoms = [a.b_iso for a in residues[record.resnum] if not a.element.is_hydrogen]
        values = np.asarray(atoms, dtype=float) / scale
        if not len(values) or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError(f"Invalid pLDDT: {path}/{record.resnum}: {values}")
        confidence[position] = values.mean()
    contacts = []
    for contact in analyzed.contacts:
        i, j = mapping[contact.seq_i], mapping[contact.seq_j]
        if i is not None and j is not None and i != j:
            contacts.append((min(i, j), max(i, j), float(contact.degree)))
    identity = sum(p is not None and observed[i] == sequence[p] for i, p in enumerate(mapping)) / len(observed)
    if identity < 0.9:
        raise ValueError(f"Prediction sequence mismatch: {path}: {identity}")
    return confidence, contacts, identity


def rank_contacts(i: np.ndarray, j: np.ndarray, degree: np.ndarray, confidence: np.ndarray,
                  *, minimum: int, k: int, ranking: str) -> np.ndarray:
    """Choose up to k ConFind-positive pairs; stable ties retain (i,j) order."""
    valid = (j - i >= minimum) & (degree > 0)
    candidates = np.flatnonzero(valid)
    if ranking == "degree":
        score = degree[candidates]
    else:
        score = confidence[i[candidates]] * confidence[j[candidates]]
        if ranking == "plddt_product_degree":
            score = score * degree[candidates]
        elif ranking != "plddt_product":
            raise ValueError(ranking)
    if not np.isfinite(score).all():
        raise ValueError("A contact endpoint lacks mapped confidence")
    return candidates[np.argsort(-score, kind="stable")[:k]]


def score_one(task: tuple) -> dict:
    """Validate an archived predictor, then cache all pair features and rankings."""
    source, truth, sequence, archived, expected, worker_hash = task
    stem, method = source["stem"], source["method"]
    directory = CACHE / "scored" / method / stem
    marker = directory / "result.json"
    if marker.exists():
        result = json.loads(marker.read_text())
        if result["worker_sha256"] != worker_hash or result["structure_sha256"] != source["structure_sha256"]:
            raise ValueError(f"Stale score cache: {method}/{stem}")
        for name, pin in result["files"].items():
            if digest(directory / name) != pin:
                raise ValueError(f"Corrupt cache: {directory}/{name}")
        return result
    started = time.perf_counter()
    path = Path(source["structure_file"])
    if digest(path) != source["structure_sha256"]:
        raise ValueError(f"Changed structure: {path}")
    length = truth["L"]
    if length != len(sequence):
        raise ValueError(f"Sequence-length mismatch: {stem}")
    confidence, recomputed, identity = residue_confidence(path, sequence, method)
    degree = degree_matrix(length, archived)
    reconstructed = degree_matrix(length, recomputed)
    tmat = true_matrix(length, truth["contacts"])
    pi, pj, sep = resolved_pairs(np.asarray(truth["resolved"]))
    original = metric_rows(degree, tmat, pi, pj, sep, length, with_precision=True)
    repeated = metric_rows(reconstructed, tmat, pi, pj, sep, length, with_precision=True)
    for kind, rows in [("archived", original), ("recomputed", repeated)]:
        for row in rows:
            key = (row["range"], row["cut"])
            if key not in expected:
                continue
            if not np.isclose(row["precision"], expected[key], atol=1e-12, rtol=0):
                raise ValueError(f"{method}/{stem}/{kind}/{key}: {row['precision']} != {expected[key]}")
    # Use archived degrees bit-for-bit, after checking that their coordinates
    # reproduce the published metrics. Keep only the shared resolved universe.
    positive = degree[pi, pj] > 0
    i, j, d = pi[positive], pj[positive], degree[pi[positive], pj[positive]]
    if not np.isfinite(confidence[np.concatenate([i, j])]).all():
        raise ValueError(f"Unmapped contact confidence: {method}/{stem}")
    features = pd.DataFrame(dict(i=i, j=j, degree=d, plddt_i=confidence[i], plddt_j=confidence[j],
                                 true_contact=tmat[i, j], sep=j-i))
    features["pair_row"] = np.arange(len(features))
    metrics, selections = [], []
    k = max(1, length // 5)
    for region, minimum in [("all", 6), ("long", 24)]:
        old_score = expected[(region, "L/5")]
        for ranking in RANKINGS:
            indices = rank_contacts(i, j, d, confidence, minimum=minimum, k=k, ranking=ranking)
            correct = int(tmat[i[indices], j[indices]].sum())
            metrics.append(dict(stem=stem, method=method, range=region, ranking=ranking, L=length,
                                k=k, n_selected=len(indices), n_positive=int(((j-i)>=minimum).sum()),
                                n_correct=correct, precision=correct/k, original_precision=old_score,
                                structure_sha256=source["structure_sha256"]))
            selections.extend(dict(range=region, ranking=ranking, rank=rank+1, pair_row=int(idx))
                              for rank, idx in enumerate(indices))
    directory.mkdir(parents=True, exist_ok=True)
    features.to_parquet(directory / "pairs.parquet", index=False)
    pd.DataFrame(dict(position=np.flatnonzero(np.isfinite(confidence)), plddt=confidence[np.isfinite(confidence)])).to_parquet(
        directory / "residues.parquet", index=False)
    pd.DataFrame(selections).to_parquet(directory / "selected.parquet", index=False)
    result = dict(stem=stem, method=method, metrics=metrics, structure_sha256=source["structure_sha256"],
                  alignment_identity=identity, contact_degree_max_difference=float(np.max(np.abs(degree-reconstructed))),
                  contact_support_difference=int(np.count_nonzero((degree>0)!=(reconstructed>0))),
                  worker_sha256=worker_hash, elapsed_seconds=time.perf_counter()-started,
                  hostname=socket.gethostname(), platform=platform.platform(),
                  files={name:digest(directory/name) for name in ["pairs.parquet","residues.parquet","selected.parquet"]})
    marker.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    """Precompute comparisons and pair-level evidence, independently of plots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    sources = pd.read_csv(CACHE / "sources.csv", keep_default_na=False)
    raw = pd.read_parquet(CACHE / "archived_contacts.parquet")
    pairs = {(method, stem): frame[["i", "j", "degree"]].to_numpy().tolist()
             for (method, stem), frame in raw.groupby(["method", "stem"])}
    truth_path = HERE / "scratch/contacts/inputs/gt_universe_scored.jsonl"
    if digest(truth_path) != "f30c23e3d2fbab245755fc01548388b41730ddfa45da87325539698cadb153e5":
        raise ValueError("Frozen truth changed")
    truth = {r["stem"]:r for r in map(json.loads, truth_path.read_text().splitlines())}
    prompts = pd.read_csv(EXPERIMENTS / "exp245_evals_foldbench_held_out_monomers/data/eval_sets.csv").set_index("stem")
    old = pd.concat([pd.read_csv(HERE / f"data/{name}", float_precision="round_trip")
                     for name in ["figure_rows.csv", "pl5_figure_rows.csv"]])
    old = old[old.figure.isin(["04_contacts", "04_contacts_pl5"])]
    expectations = {}
    for (method, stem), frame in old.groupby(["method", "stem"]):
        expectations[(method, stem)] = {("long" if r.metric.endswith("_long") else "all",
                                        "R" if r.metric.startswith("r_precision") else "L/5"):r.value for r in frame.itertuples()}
    worker_hash = digest(Path(__file__))
    tasks = [(row, truth[row["stem"]], prompts.loc[row["stem"],"sequence"], pairs[(row["method"],row["stem"])],
              expectations[(row["method"],row["stem"])], worker_hash) for row in sources.to_dict("records")]
    if args.limit:
        tasks = [task for method in sources.method.unique() for task in sorted(
            (t for t in tasks if t[0]["method"] == method), key=lambda t:t[1]["L"])[:args.limit]]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, result in enumerate(pool.map(score_one, tasks)):
            results.append(result)
            if index % 25 == 0 or index+1 == len(tasks):
                print(f"{index+1}/{len(tasks)} verified: {result['method']}/{result['stem']}", flush=True)
    if args.limit:
        print("Smoke complete", flush=True)
        return
    metrics = pd.DataFrame([r for result in results for r in result["metrics"]])
    metrics.to_csv(HERE / "data/contact_confidence_metrics.csv", index=False)
    sources.to_csv(HERE / "data/contact_confidence_sources.csv", index=False)
    pd.DataFrame([{k:v for k,v in r.items() if k not in ["metrics", "files"]} for r in results]).to_csv(
        HERE / "data/contact_confidence_validation.csv", index=False)
    print(metrics.groupby(["method","ranking","range"]).precision.mean().to_string(), flush=True)


if __name__ == "__main__":
    main()
