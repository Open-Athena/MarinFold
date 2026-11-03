"""Freeze a secondary farthest-point arm without changing the primary experiment."""

import datetime as dt
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import HfFileSystem

from freeze_diversity import digest
from select_diversity import cached, jaccard

HERE = Path(__file__).resolve().parent


def farthest_points(distance: np.ndarray, k: int) -> np.ndarray:
    """Greedily maximize nearest-selected distance, starting at the global medoid."""
    chosen = [int(np.argmin(distance.sum(axis=1)))]
    while len(chosen) < k:
        nearest = distance[:, chosen].min(axis=1)
        nearest[chosen] = -1
        index = int(np.argmax(nearest))
        if nearest[index] <= 0:
            raise ValueError("Fewer than k distinct contact sets")
        chosen.append(index)
    return np.sort(chosen)


def main() -> None:
    """Save a separate secondary protocol and its selected inputs before folding."""
    base = json.loads((HERE / "data/diversity_plan.json").read_text())
    plan = {
        **base,
        "primary": "Unchanged original primary; maxmin10 versus random10 is secondary",
        "supplement_protocol_sha256": digest(HERE / "DIVERSITY_MAXMIN_PLAN.md"),
        "original_selection_sha256": digest(
            HERE / "data/diversity_selection_manifest.json"
        ),
        "selector": "greedy max-min Jaccard; start global medoid; ties lowest source index",
        "helico_arms": ["maxmin10"],
        "n_helico_predictions": 200,
    }
    plan_path = HERE / "data/maxmin_plan.json"
    plan["frozen_utc"] = (
        json.loads(plan_path.read_text())["frozen_utc"]
        if plan_path.exists()
        else dt.datetime.now(dt.timezone.utc).isoformat()
    )
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Secondary plan changed")
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    targets = json.loads((HERE / "inputs/diversity/targets.json").read_text())
    inventory = pd.read_csv(HERE / "data/diversity_raw_inventory.csv")
    fs = HfFileSystem(token=False)
    tasks, diagnostics = [], []
    for target in targets:
        stem = target["stem"]
        mapping = {i: j for j, i in enumerate(target["resolved"])}
        for model in base["models"]:
            samples = []
            sources = inventory[
                (inventory.stem == stem) & (inventory.model == model)
            ].sort_values("start")
            for source in sources.to_dict("records"):
                path = (
                    HERE
                    / "scratch/diversity/maxmin_sources"
                    / model
                    / f"{stem}-{source['start']}.json.gz"
                )
                raw = json.loads(
                    gzip.decompress(cached(fs, source["uri"], path, source["sha256"]))
                )
                samples.extend(
                    {**s, "raw_uri": source["uri"], "raw_sha256": source["sha256"]}
                    for s in raw["samples"]
                )
            samples.sort(key=lambda s: s["rollout"])
            assert [s["rollout"] for s in samples] == list(range(1000))
            mapped = [
                [
                    (mapping[i], mapping[j])
                    for i, j in s["contacts"]
                    if i in mapping and j in mapping
                ]
                for s in samples
            ]
            effective = [
                sorted(set((i, j) for i, j in pairs if j - i >= 6)) for pairs in mapped
            ]
            distance = jaccard(effective)
            chosen = farthest_points(distance, 10)
            labels = distance[:, chosen].argmin(axis=1)
            labels[chosen] = np.arange(10)
            sub = distance[np.ix_(chosen, chosen)]
            diagnostics.append(
                dict(
                    model=model,
                    stem=stem,
                    arm="maxmin10",
                    pairwise_jaccard=float(sub[np.triu_indices(10, 1)].mean()),
                    mean_nearest_distance=float(distance[:, chosen].min(axis=1).mean()),
                    mean_contacts=float(np.mean([len(effective[i]) for i in chosen])),
                    unique_contact_sets=len({tuple(effective[i]) for i in chosen}),
                    pool_unique_contact_sets=len({tuple(p) for p in effective}),
                    cluster_sizes=json.dumps(
                        np.bincount(labels, minlength=10).tolist()
                    ),
                )
            )
            for slot, index in enumerate(chosen):
                sample = samples[index]
                task = dict(
                    model=model,
                    stem=stem,
                    arm="maxmin10",
                    rollout=slot,
                    source_rollout=int(index),
                    task_id=f"{stem}__{model}__maxmin10__r{slot:02d}",
                    contacts=effective[index],
                    raw_sha256=sample["raw_sha256"],
                    raw_uri=sample["raw_uri"],
                    prompt_id=sample["prompt_id"],
                    marinfold_seed=sample["seed"],
                    helico_seed=int.from_bytes(
                        hashlib.sha256(
                            f"exp381-helico:{stem}:{slot}".encode()
                        ).digest()[:8],
                        "big",
                    )
                    % (2**63 - 1),
                    n_canonical_contacts=len(sample["contacts"]),
                    n_unresolved_removed=len(sample["contacts"]) - len(mapped[index]),
                    n_mapped_near_removed=len(mapped[index]) - len(effective[index]),
                    n_effective_contacts=len(effective[index]),
                )
                task["input_sha256"] = hashlib.sha256(
                    json.dumps(task, sort_keys=True).encode()
                ).hexdigest()
                tasks.append(task)
    task_path = HERE / "inputs/diversity/maxmin_tasks.json"
    payload = json.dumps(tasks, indent=2) + "\n"
    if task_path.exists() and task_path.read_text() != payload:
        raise ValueError("Secondary selections changed")
    task_path.write_text(payload)
    pd.DataFrame(tasks).drop(columns="contacts").to_csv(
        HERE / "data/maxmin_contact_inputs.csv", index=False
    )
    pd.DataFrame(diagnostics).to_csv(
        HERE / "data/maxmin_contact_diagnostics.csv", index=False
    )
    manifest = dict(
        plan_sha256=digest(plan_path),
        tasks_sha256=digest(task_path),
        selector_source_sha256=digest(Path(__file__)),
        n_helico_tasks=200,
    )
    (HERE / "data/maxmin_selection_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print("Frozen 200 secondary farthest-point Helico inputs")


if __name__ == "__main__":
    main()
