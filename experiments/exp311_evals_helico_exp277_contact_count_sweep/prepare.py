"""Build the exp277 contact lists in verified Helico token coordinates.

Run from this directory with ``uv run python prepare.py --helico-exp14 DIR``.
DIR is Helico's ``experiments/exp14_foldbench_held_out_monomers`` after its
``build_eval_sets.py`` and ``build_index_map.py`` have been run. CoreWeave S3
credentials are read from the ``cw`` profile in ``~/.aws/credentials``, the
same source as exp277's launcher.
"""

import argparse
import configparser
import csv
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import boto3
import botocore
import numpy as np


HERE = Path(__file__).resolve().parent
SCRATCH = HERE / "scratch" / "targets"
MODEL = "marinfold-exp277-full-epoch-m2-p06-step266344"
BUCKET = "marin-us-east-02a"
PREFIX = (
    "MarinFold/exp277_models_single_mpnn_pilot/evals/rollout-v2/"
    "2026-09-13/v2-01/dense_scores/exp277_full_epoch_m2_p06_step266344"
)
GT_URI = (
    "hf://buckets/open-athena/MarinFold/data/"
    "contacts-v1-foldbench-monomers-exp245/gt_universe_scored.jsonl"
)
REFERENCE = (
    HERE.parent / "exp277_models_single_mpnn_pilot" / "data" /
    "eval_rollout_v2" / "contact_precision_all.csv"
)


def sha256(path: Path) -> str:
    """Return the content digest of a local input."""
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def score_client():
    """Build a CoreWeave client with required virtual-hosted addressing."""
    credentials = configparser.ConfigParser()
    credentials.read(Path.home() / ".aws" / "credentials")
    if "cw" not in credentials:
        raise RuntimeError("missing cw profile in ~/.aws/credentials")
    cw = credentials["cw"]
    return boto3.client(
        "s3", endpoint_url="https://cwobject.com",
        aws_access_key_id=cw["aws_access_key_id"],
        aws_secret_access_key=cw["aws_secret_access_key"],
        config=botocore.config.Config(s3={"addressing_style": "virtual"}),
    )


def ground_truth(path: Path) -> dict[str, dict]:
    """Load exp245's scored universe, downloading its public file if needed."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["hf", "buckets", "cp", GT_URI, str(path)], check=True)
    return {r["stem"]: r for r in (json.loads(line) for line in path.read_text().splitlines())}


def ranked_pairs(score: np.ndarray, resolved: list[int]) -> list[tuple[int, int]]:
    """Use exp245's resolved-pair universe and stable score tie order."""
    resolved_array = np.asarray(resolved, dtype=np.int64)
    a, b = np.triu_indices(len(resolved_array), k=1)
    i, j = resolved_array[a], resolved_array[b]
    keep = j - i >= 6
    i, j = i[keep], j[keep]
    order = np.argsort(-score[i, j], kind="mergesort")
    return [(int(i[x]), int(j[x])) for x in order]


def truth_matrix(length: int, contacts: list) -> np.ndarray:
    """Use the same all-range truth definition as exp245."""
    truth = np.zeros((length, length), dtype=bool)
    for i, j, degree in contacts:
        i, j = int(i), int(j)
        if degree >= 0.001 and j - i >= 6 and i < j < length:
            truth[i, j] = True
    return truth


def precision_reference() -> dict[tuple[str, str], float]:
    """Load the exp277 published precision used to validate pair ranking."""
    out = {}
    with REFERENCE.open() as stream:
        for row in csv.DictReader(stream):
            if row["model"] == MODEL and row["range"] == "all" and row["cut"] in ("L", "L/2", "L/5"):
                out[(row["stem"], row["cut"])] = float(row["precision"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--helico-exp14", type=Path, required=True)
    args = parser.parse_args()
    exp14 = args.helico_exp14.resolve()
    source = exp14 / "data"
    universe = ground_truth(HERE / "scratch" / "gt_universe_scored.jsonl")
    token_map = json.loads((source / "token_map.json").read_text())
    reference = precision_reference()
    with (source / "targets.csv").open() as stream:
        targets = [row for row in csv.DictReader(stream) if row["eval_set"] in ("eval-val", "eval-denovo")]
    if len(targets) != 116:
        raise ValueError(f"expected 116 eval-val and eval-denovo targets, found {len(targets)}")

    client = score_client()
    SCRATCH.mkdir(parents=True, exist_ok=True)
    (SCRATCH / "gt").mkdir(exist_ok=True)
    (SCRATCH / "scores").mkdir(exist_ok=True)
    pairs_by_target = {}
    checked = []
    selected = []
    excluded = []
    pins = {}
    for target in targets:
        stem = target["target_id"]
        mapping = token_map.get(stem)
        if not mapping:
            excluded.append({"stem": stem, "eval_set": target["eval_set"], "reason": "unverified prompt-to-token map"})
            continue
        record = universe[stem]
        length = int(record["L"])
        score_path = SCRATCH / "scores" / f"foldbench_monomer__{stem}.npz"
        if not score_path.exists():
            client.download_file(BUCKET, f"{PREFIX}/{score_path.name}", str(score_path))
        with np.load(score_path) as loaded:
            score = loaded["score"].astype(np.float64)
        if score.shape != (length, length):
            raise ValueError(f"{stem}: score shape {score.shape} != {(length, length)}")
        ranked = ranked_pairs(score, record["resolved"])
        truth = truth_matrix(length, record["contacts"])
        for label, count in (("L", length), ("L/2", max(1, length // 2)), ("L/5", max(1, length // 5))):
            top = ranked[:count]
            actual = float(np.mean([truth[i, j] for i, j in top]))
            expected = reference.get((stem, label))
            if expected is None or abs(actual - expected) > 1e-9:
                raise ValueError(f"{stem} {label}: precision {actual} != reference {expected}")
        checked.append(stem)
        mapped = {int(p): int(t) for p, t in mapping.items()}
        pairs = []
        for i, j in ranked[:length]:
            a, b = mapped.get(i), mapped.get(j)
            if a is None or b is None or a == b:
                continue
            pairs.append([min(a, b), max(a, b)])
        if len(pairs) != length:
            raise ValueError(f"{stem}: {len(pairs)} mapped pairs at top-L, expected {length}")
        pairs_by_target[stem] = pairs
        gt_path = source / "gt" / f"{stem}.cif.gz"
        dest = SCRATCH / "gt" / gt_path.name
        shutil.copyfile(gt_path, dest)
        pins[score_path.name] = sha256(score_path)
        pins[gt_path.name] = sha256(dest)
        selected.append(target)

    if len(selected) != 115 or len(excluded) != 1 or excluded[0]["stem"] != "7pv5_A":
        raise ValueError(f"unexpected target coverage: {len(selected)} selected, {excluded}")
    with (SCRATCH / "targets.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(targets[0]))
        writer.writeheader()
        writer.writerows(selected)
    (SCRATCH / "ranked_pairs.json").write_text(json.dumps(pairs_by_target, separators=(",", ":")))
    manifest = {
        "model": MODEL,
        "score_uri": f"s3://{BUCKET}/{PREFIX}/",
        "exp245_ground_truth_uri": GT_URI,
        "exp245_ground_truth_sha256": sha256(HERE / "scratch" / "gt_universe_scored.jsonl"),
        "helico_exp14_git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=exp14, text=True
        ).strip(),
        "helico_targets_sha256": sha256(source / "targets.csv"),
        "helico_token_map_sha256": sha256(source / "token_map.json"),
        "exp277_precision_sha256": sha256(REFERENCE),
        "prepared_targets_sha256": sha256(SCRATCH / "targets.csv"),
        "ranked_pairs_sha256": sha256(SCRATCH / "ranked_pairs.json"),
        "n_eval_val": sum(t["eval_set"] == "eval-val" for t in selected),
        "n_eval_denovo": sum(t["eval_set"] == "eval-denovo" for t in selected),
        "n_reference_precision_checked": len(checked) * 3,
        "excluded": excluded,
        "files_sha256": pins,
    }
    (HERE / "data" / "input_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"prepared {len(selected)} targets; verified {len(checked)*3} precision cuts; excluded {excluded}")


if __name__ == "__main__":
    main()
