"""Score Boltz-2 with the exact frozen structure/contact universe used by AF2/3."""

import argparse
import concurrent.futures
import hashlib
import importlib.metadata
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from score_alphafold import (
    EXPERIMENTS, HELICO, HELICO_SHA, ROOT, compute_contacts, degree_matrix,
    metric_rows, resolved_pairs, structure_scores, true_matrix,
)


def score_one(task: tuple[dict, dict, dict, str]) -> dict:
    """Verify every candidate and the confidence argmax before reading the truth."""
    record, truth, target, protocol_hash = task
    stem = record["stem"]
    directory = ROOT / "scratch/boltz2/results/boltz2" / stem
    marker = json.loads((directory / "complete.json").read_text())
    if any(marker[k] != v for k, v in record.items()) or marker["protocol_sha256"] != protocol_hash:
        raise ValueError(f"{stem}: inputs or inference settings differ")
    candidates = json.loads((directory / "candidates.json").read_text())
    if len(candidates) != 25 or {r["rank"] for r in candidates} != set(range(25)):
        raise ValueError("Incomplete Boltz-2 sample budget")
    for row in candidates:
        path = directory / row["file"]
        confidence = json.loads(path.with_name(f"confidence_{path.stem}.json").read_text())
        if (hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]
                or row["confidence"] != confidence["confidence_score"]
                or not math.isfinite(row["confidence"])):
            raise ValueError("Candidate coordinates or confidence differ from manifest")
    best = max(candidates, key=lambda r: r["confidence"])
    path = directory / "selected.cif"
    if (best["rank"] != 0 or marker["selected"] != best["file"]
            or marker["selection_confidence"] != best["confidence"]
            or path.read_bytes() != (directory / best["file"]).read_bytes()):
        raise ValueError("Selected Boltz-2 output differs from confidence argmax")
    gt_path = Path.home() / ".cache/helico/data/benchmarks/FoldBench/examples/ground_truths" / f"{target['pdb_id']}.cif.gz"
    structural = structure_scores(path, gt_path, record["sequence"])
    contacts = compute_contacts(path, target["prompt_sequence"], stem=stem)
    if contacts.alignment_identity < 0.9:
        raise ValueError(f"{stem}: contact alignment identity {contacts.alignment_identity}")
    score = degree_matrix(truth["L"], contacts.contacts)
    metrics = metric_rows(score, true_matrix(truth["L"], truth["contacts"]),
                          *resolved_pairs(np.asarray(truth["resolved"])), truth["L"], with_precision=True)
    return dict(stem=stem, method="boltz2", eval_set=record["eval_set"],
                selected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                ground_truth_sha256=hashlib.sha256(gt_path.read_bytes()).hexdigest(),
                selected_file=str(path.relative_to(ROOT)), selection_confidence=marker["selection_confidence"],
                contact_alignment_identity=contacts.alignment_identity, **structural,
                contact_metrics=metrics, contacts=contacts.contacts, timing=marker)


def main() -> None:
    """Score complete production outputs or explicitly isolated smoke results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HELICO, text=True).strip() != HELICO_SHA:
        raise ValueError("Helico metric implementation changed")
    protocol_path = ROOT / "data/boltz2_inputs.json"
    protocol_hash = hashlib.sha256(json.dumps(json.loads(protocol_path.read_text()), sort_keys=True).encode()).hexdigest()
    records_path = ROOT / "scratch/alphafold/inputs/targets.json"
    records = json.loads(records_path.read_text())
    truth_path = ROOT / "scratch/contacts/inputs/gt_universe_scored.jsonl"
    truth_sha = hashlib.sha256(truth_path.read_bytes()).hexdigest()
    if truth_sha != "f30c23e3d2fbab245755fc01548388b41730ddfa45da87325539698cadb153e5":
        raise ValueError("Frozen contact universe changed")
    truth = {r["stem"]: r for r in map(json.loads, truth_path.read_text().splitlines())}
    targets = pd.read_csv(HELICO / "experiments/exp14_foldbench_held_out_monomers/data/targets.csv").set_index("stem")
    prompts = pd.read_csv(EXPERIMENTS / "exp245_evals_foldbench_held_out_monomers/data/eval_sets.csv").set_index("stem")
    present = {p.parent.name for p in (ROOT / "scratch/boltz2/results/boltz2").glob("*/complete.json")}
    if not present or (not args.smoke and present != {r["stem"] for r in records}):
        raise ValueError(f"Incomplete Boltz-2: {len(present)}/333")
    tasks = [(r, truth[r["stem"]], {**targets.loc[r["stem"]].to_dict(),
              "prompt_sequence": prompts.loc[r["stem"], "sequence"]}, protocol_hash)
             for r in records if r["stem"] in present]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(score_one, tasks))
    output = ROOT / ("scratch/boltz2/smoke_scores" if args.smoke else "data")
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{k: v for k, v in r.items() if k not in {"contacts", "contact_metrics", "timing"}}
                  for r in results]).to_csv(output / "boltz2_structure_metrics.csv", index=False)
    pd.DataFrame([dict(stem=r["stem"], method="boltz2", eval_set=r["eval_set"], **m)
                  for r in results for m in r["contact_metrics"]]).to_csv(output / "boltz2_contact_metrics.csv", index=False)
    pd.DataFrame([r["timing"] for r in results]).to_csv(output / "boltz2_timings.csv", index=False)
    (ROOT / "scratch/boltz2/boltz2_contacts.json").write_text(json.dumps({r["stem"]: r["contacts"] for r in results}))
    (output / "boltz2_run.json").write_text(json.dumps({
        "variant": "boltz2", "n_targets": len(results), "n_candidates": 25 * len(results),
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "input_manifest_sha256": hashlib.sha256(records_path.read_bytes()).hexdigest(),
        "scorer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "shared_scorer_sha256": hashlib.sha256((ROOT / "generation/score_alphafold.py").read_bytes()).hexdigest(),
        "helico_metric_source": HELICO_SHA, "contact_truth_sha256": truth_sha,
        "contact_source": "exp78 pyconfind_contacts; exp89 compute_metrics; exp245 frozen truth",
        "scoring_packages": {name: importlib.metadata.version(name)
                             for name in ("numpy", "pandas", "gemmi", "pyconfind", "tmtools", "scikit-learn")},
        "inference_packages": results[0]["timing"]["packages"],
        "runner": "Modal H100 requests, us-east, up to sixteen resident workers; actual GPU recorded per target",
        "selection": "Verified 25 candidates, source confidence JSON, confidence argmax and selected coordinate bytes",
        "timing_scope": "Synchronized prediction batch; excludes input preparation and output writing",
        "selected_sha256": {r["stem"]: r["selected_sha256"] for r in results},
    }, indent=2) + "\n")
    print(f"Scored Boltz-2: {len(results)} proteins")


if __name__ == "__main__":
    main()
