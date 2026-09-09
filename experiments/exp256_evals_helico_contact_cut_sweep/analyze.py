# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Reproduce the Helico contact-cut sweep from portable eval-val-only inputs.

The historical curve uses 95 targets shared with all reference arms. A second
curve uses all 96 targets available in every contact-cut arm, so missing
Protenix inputs cannot silently determine the primary comparison's population.
Precision and recall are recomputed from the exact contact lists delivered to
Helico, on the same targets as each folding comparison. No inference is run.

    uv run python analyze.py --inputs /path/to/published/inputs --out data
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

EVAL_SET = "eval-val"
CUT_ORDER = ("mf_L5", "mf_L2", "mf_L", "mf_1p5L", "mf_2L", "mf_3L", "mf_5L", "mf_union")
REFERENCE_ARMS = ("off", "v2ss", "oracle", "v2msa")
REFERENCE_LABEL = {
    "off": "Helico, no contacts", "v2ss": "Helico + Protenix-v2 single-sequence contacts",
    "oracle": "Helico + oracle contacts", "v2msa": "Helico + Protenix-v2 MSA contacts",
}
BASELINE = "mf_L"
BOOTSTRAP_DRAWS = 10_000
SEED = 256


def load_arm(results: Path, tag: str, expected_ids: set[str]) -> pd.DataFrame:
    """Load a complete, unique eval-val result ledger, including failures."""
    frame = pd.read_csv(results / f"{tag}.csv")
    if not frame.dataset.eq(EVAL_SET).all():
        raise ValueError(f"{tag}: input includes a dataset other than {EVAL_SET}")
    if frame.target_id.duplicated().any() or set(frame.target_id) != expected_ids:
        raise ValueError(f"{tag}: duplicate, missing, or unexpected target IDs")
    valid = frame.status.eq("ok")
    if not np.isfinite(frame.loc[valid, "lddt"]).all():
        raise ValueError(f"{tag}: successful rows must have finite lDDT")
    if not frame.loc[valid, "lddt"].between(0, 1).all():
        raise ValueError(f"{tag}: lDDT outside [0, 1]")
    frame = frame.copy()
    frame["arm"] = tag
    return frame


def paired(wide: pd.DataFrame, a: str, b: str) -> dict:
    """Return target-paired mean difference and percentile bootstrap interval."""
    delta = (wide[a] - wide[b]).to_numpy()
    if not len(delta) or not np.isfinite(delta).all():
        raise ValueError("paired comparison needs nonempty, complete observations")
    generator = np.random.default_rng(SEED)
    index = generator.integers(0, len(delta), size=(BOOTSTRAP_DRAWS, len(delta)))
    means = delta[index].mean(axis=1)
    return dict(mean_delta=float(delta.mean()), lo=float(np.percentile(means, 2.5)),
                hi=float(np.percentile(means, 97.5)), targets_better=float((delta > 0).mean()))


def contact_metrics(record: dict, mapping: dict, pairs: list) -> dict:
    """Score actual Helico-token contacts against exp245's resolved universe."""
    inverse = {int(token): int(prompt) for prompt, token in mapping.items()}
    if len(inverse) != len(mapping):
        raise ValueError("token map is not one-to-one")
    resolved = set(record["resolved"])
    truth = {(int(i), int(j)) for i, j, degree in record["contacts"]
             if degree >= 0.001 and j - i >= 6 and i in resolved and j in resolved
             and 0 <= i < j < record["L"]}
    selected = [tuple(sorted((inverse[int(a)], inverse[int(b)]))) for a, b in pairs]
    if len(set(selected)) != len(selected):
        raise ValueError("contact list contains duplicate pairs")
    if any(i not in resolved or j not in resolved or j - i < 6 for i, j in selected):
        raise ValueError("contact list includes a pair outside the scoring universe")
    if not selected or not truth:
        raise ValueError("contact precision and recall require nonempty lists and truth")
    hits = len(set(selected) & truth)
    return dict(n_contacts=len(selected), n_true=len(truth), hits=hits,
                precision=hits / len(selected), recall=hits / len(truth),
                pairs_over_L=len(selected) / record["L"])



def verify_ranked_lists(inputs: Path) -> int:
    """Check exact delivered lists against the pinned eval-val dense matrices."""
    records = {r["stem"]: r for r in map(json.loads, (inputs / "gt_universe_scored.jsonl").read_text().splitlines())}
    maps = json.loads((inputs / "token_map.json").read_text())
    arms = {tag: json.loads((inputs / "arms" / f"{tag}.json").read_text()) for tag in CUT_ORDER}
    checks = 0
    for stem, mapping in maps.items():
        record = records[stem]
        length = record["L"]
        with np.load(inputs / "dense" / f"foldbench_monomer__{stem}.npz") as arrays:
            score = arrays["score"].astype(np.float64)
        if score.shape != (length, length) or not np.isfinite(score).all():
            raise ValueError(f"{stem}: invalid dense score matrix")
        resolved = np.asarray(record["resolved"])
        a, b = np.triu_indices(len(resolved), k=1)
        i, j = resolved[a], resolved[b]
        keep = j - i >= 6
        i, j = i[keep], j[keep]
        order = np.argsort(-score[i, j], kind="mergesort")
        ranked = list(zip(i[order].tolist(), j[order].tolist()))
        voted = int(np.sum(score[i, j] > 0))
        cuts = {"mf_L5": max(1, length // 5), "mf_L2": max(1, length // 2), "mf_L": length,
                "mf_1p5L": min(3 * length // 2, voted), "mf_2L": min(2 * length, voted),
                "mf_3L": min(3 * length, voted), "mf_5L": min(5 * length, voted), "mf_union": voted}
        for tag, count in cuts.items():
            expected = [sorted((mapping[str(a)], mapping[str(b)])) for a, b in ranked[:count]]
            if expected != arms[tag][stem]:
                raise ValueError(f"{tag}/{stem}: delivered contact list differs from pinned ranking")
            checks += 1
    return checks

def accuracy_table(inputs: Path, ledgers: pd.DataFrame) -> pd.DataFrame:
    """Compute contact accuracy only for successful eval-val contact predictions."""
    gt = {r["stem"]: r for r in map(json.loads, (inputs / "gt_universe_scored.jsonl").read_text().splitlines())}
    maps = json.loads((inputs / "token_map.json").read_text())
    rows = []
    for tag in CUT_ORDER:
        arm = json.loads((inputs / "arms" / f"{tag}.json").read_text())
        selected = ledgers[ledgers.arm.eq(tag) & ledgers.status.eq("ok")]
        for row in selected.itertuples():
            metrics = contact_metrics(gt[row.target_id], maps[row.target_id], arm[row.target_id])
            if metrics["n_contacts"] != row.n_contacts:
                raise ValueError(f"{tag}/{row.target_id}: arm differs from inference contact count")
            rows.append(dict(target_id=row.target_id, arm=tag, **metrics))
    return pd.DataFrame(rows)


def curve_table(wide: pd.DataFrame, accuracy: pd.DataFrame, provenance: dict) -> pd.DataFrame:
    """Aggregate contact and folding metrics on exactly the same targets."""
    rows = []
    for tag in CUT_ORDER:
        contacts = accuracy[accuracy.arm.eq(tag)].set_index("target_id")
        contacts = contacts.loc[wide.index]
        stats = contacts[["precision", "recall", "pairs_over_L"]].mean().to_dict()
        row = dict(arm=tag, source=provenance["files"][f"results/{tag}.csv"]["run"],
                   n=len(wide), lddt=float(wide[tag].mean()), **stats)
        if tag != BASELINE:
            row.update({f"vs_{BASELINE}_{key}": value for key, value in paired(wide, tag, BASELINE).items()})
        rows.append(row)
    return pd.DataFrame(rows)


def analyze(inputs: Path, out: Path) -> None:
    """Validate published inputs and reproduce both complete-case populations."""
    provenance = json.loads((inputs / "provenance.json").read_text())
    for name, record in provenance["files"].items():
        if hashlib.sha256((inputs / name).read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"input hash mismatch: {name}")
    targets = pd.read_csv(inputs / "targets.csv")
    if not targets.eval_set.eq(EVAL_SET).all() or targets.target_id.duplicated().any():
        raise ValueError("target universe must contain unique eval-val IDs")
    ids = set(targets.target_id)
    gt_records = [json.loads(line) for line in (inputs / "gt_universe_scored.jsonl").read_text().splitlines()]
    if len(gt_records) != len(ids) or {row["stem"] for row in gt_records} != ids:
        raise ValueError("ground truth must cover exactly the eval-val target universe")
    baseline_manifest = json.loads((inputs / "results/mf_L.manifest.json").read_text())
    for tag in CUT_ORDER:
        manifest = json.loads((inputs / "results" / f"{tag}.manifest.json").read_text())
        if any(manifest[key] != baseline_manifest[key] for key in ("sampling", "model")):
            raise ValueError(f"{tag}: sampling or model settings differ from top-L")
    ledger = pd.concat([load_arm(inputs / "results", tag, ids)
                        for tag in CUT_ORDER + REFERENCE_ARMS], ignore_index=True)
    successful = ledger[ledger.status.eq("ok")][["target_id", "lddt", "arm"]]
    wide = successful.pivot(index="target_id", columns="arm", values="lddt")
    common = wide.dropna()
    contact_common = wide[list(CUT_ORDER)].dropna()
    if common.empty or contact_common.empty:
        raise ValueError("no complete target pairs")
    ranking_checks = verify_ranked_lists(inputs)
    accuracy = accuracy_table(inputs, ledger)
    out.mkdir(parents=True, exist_ok=True)
    curve = curve_table(common, accuracy, provenance)
    curve.to_csv(out / "cut_sweep_curve.csv", index=False)
    curve_table(contact_common, accuracy, provenance).to_csv(out / "cut_sweep_contact_targets.csv", index=False)
    references = [dict(arm=tag, label=REFERENCE_LABEL[tag],
                       source=provenance["files"][f"results/{tag}.csv"]["run"],
                       n=len(common), lddt=float(common[tag].mean())) for tag in REFERENCE_ARMS]
    pd.DataFrame(references).to_csv(out / "reference_arms.csv", index=False)
    successful.to_csv(out / "per_target_lddt.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    accuracy.to_csv(out / "per_target_contact_accuracy.csv", index=False)
    ledger[~ledger.status.eq("ok")][["target_id", "arm", "status", "error"]].to_csv(out / "excluded_targets.csv", index=False)
    gain = float((common.mf_L - common.off).mean())
    loss = float((common.mf_L - common.mf_union).mean())
    provenance.update(eval_set=EVAL_SET, n_common=len(common), n_contact_common=len(contact_common),
                      verified_ranked_lists=ranking_checks,
                      bootstrap_draws=BOOTSTRAP_DRAWS, bootstrap_seed=SEED,
                      excluded_from_contact_curve=sorted(ids - set(contact_common.index)),
                      excluded_only_by_references=sorted(set(contact_common.index) - set(common.index)),
                      contact_gain_top_L=gain, union_loss=loss,
                      union_fraction_of_contact_gain_lost=loss / gain,
                      union_fraction_of_contact_gain_retained=1 - loss / gain)
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(curve.round(6).to_string(index=False))
    print(f"Contact-only population: {len(contact_common)}; all-arm population: {len(common)}")
    print(f"Union retains {1 - loss / gain:.6%} of the top-L gain above no contacts")


def main() -> None:
    """Run the analysis from a local extracted public-input directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("data"))
    args = parser.parse_args()
    analyze(args.inputs, args.out)


if __name__ == "__main__":
    main()
