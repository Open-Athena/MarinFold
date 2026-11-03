"""Choose contact medoids and a matched random control without native scores."""

import datetime as dt
import gzip
import hashlib
import json
import platform
import socket
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import HfFileSystem
from scipy.sparse import csr_matrix

from freeze_diversity import digest

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "exp373_evals_mpnn_diversity_neff"
sys.path.insert(0, str(PREVIOUS))
from contacts import parse_contacts, sample_seed  # noqa: E402


def seed_for(label: str, stem: str) -> int:
    """Return the frozen independent selector/control seed."""
    return (
        int.from_bytes(
            hashlib.sha256(f"exp381-{label}:{stem}".encode()).digest()[:8], "big"
        )
        % 2**32
    )


def jaccard(contacts: list[list[tuple[int, int]]]) -> np.ndarray:
    """Compute binary Jaccard distances, retaining repeated observations."""
    vocabulary = {
        p: i for i, p in enumerate(sorted({p for pairs in contacts for p in pairs}))
    }
    rows, columns = [], []
    for row, pairs in enumerate(contacts):
        for pair in set(pairs):
            rows.append(row)
            columns.append(vocabulary[pair])
    matrix = csr_matrix(
        (np.ones(len(rows), dtype=np.int32), (rows, columns)),
        shape=(len(contacts), len(vocabulary)),
    )
    intersections = (matrix @ matrix.T).toarray().astype(np.float64)
    counts = np.asarray(matrix.sum(axis=1)).ravel()
    union = counts[:, None] + counts[None, :] - intersections
    similarity = np.ones_like(intersections)
    np.divide(intersections, union, out=similarity, where=union > 0)
    return 1 - similarity


def medoids(
    distance: np.ndarray, k: int, seed: int, starts: int = 10
) -> tuple[np.ndarray, dict]:
    """Fit repeated k-medoids++/alternate updates with deterministic ties."""
    rng = np.random.default_rng(seed)
    best = None
    trials = []
    for start in range(starts):
        chosen = [
            int(np.argmin(distance.sum(axis=1)))
            if start == 0
            else int(rng.integers(len(distance)))
        ]
        while len(chosen) < k:
            weights = distance[:, chosen].min(axis=1) ** 2
            if weights.sum() == 0:
                raise ValueError("Fewer than k distinct contact sets")
            chosen.append(int(rng.choice(len(distance), p=weights / weights.sum())))
        selected = np.sort(chosen)
        converged = False
        for iteration in range(100):
            labels = distance[:, selected].argmin(axis=1)
            # Medoids own themselves in tied assignments, so no cluster empties.
            labels[selected] = np.arange(k)
            updated = []
            for cluster in range(k):
                members = np.flatnonzero(labels == cluster)
                sums = distance[np.ix_(members, members)].sum(axis=1)
                updated.append(int(members[np.argmin(sums)]))
            updated = np.sort(updated)
            if np.array_equal(updated, selected):
                converged = True
                break
            selected = updated
        objective = float(distance[:, selected].min(axis=1).mean())
        candidate = (objective, tuple(int(i) for i in selected))
        if best is None or candidate < best:
            best = candidate
        trials.append(
            dict(
                start=start,
                iterations=iteration + 1,
                converged=converged,
                objective=objective,
            )
        )
    return np.array(best[1]), dict(objective=best[0], trials=trials)


def cached(
    fs: HfFileSystem, uri: str, path: Path, expected: str | None = None
) -> bytes:
    """Download one public source, requiring its identity where already pinned."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(fs.read_bytes(uri.removeprefix("hf://")))
    if expected is not None and digest(path) != expected:
        raise ValueError(f"Source checksum mismatch: {path}")
    return path.read_bytes()


def main() -> None:
    """Validate all 20,000 samples, select once, then freeze Helico inputs."""
    plan = json.loads((HERE / "data/diversity_plan.json").read_text())
    if digest(HERE / "DIVERSITY_PLAN.md") != plan["protocol_sha256"]:
        raise ValueError("Protocol changed")
    inputs = HERE / "inputs/diversity"
    if digest(inputs / "targets.json") != plan["targets_sha256"]:
        raise ValueError("Targets changed")
    targets = json.loads((inputs / "targets.json").read_text())
    fs = HfFileSystem(token=False)
    tasks, diagnostics, inventory, timing, fits, selection_timings = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for target in targets:
        stem = target["stem"]
        mapping = {i: j for j, i in enumerate(target["resolved"])}
        paired = {}
        for model in plan["models"]:
            source = next(
                r
                for r in plan["prefix_sources"]
                if r["stem"] == stem and r["model"] == model
            )
            prefix_root = "raw" if model == "default" else "raw_two_epoch"
            key = f"{stem}-t1.0-randomized"
            prefix = PREVIOUS / "scratch" / prefix_root / model / f"{key}.json.gz"
            raw_sources = [
                (source["uri"], prefix, source["sha256"], range(100)),
                (
                    f"{plan['artifact_root']}/raw/{model}/{key}.json.gz",
                    HERE / "scratch/diversity/raw" / model / f"{key}.json.gz",
                    None,
                    range(100, 1000),
                ),
            ]
            samples, identities = [], []
            for uri, path, expected, indices in raw_sources:
                data = cached(fs, uri, path, expected)
                raw = json.loads(gzip.decompress(data))
                assert (
                    raw["target"]["input_seq"] == target["input_seq"]
                    and raw["target"]["eval_set"] == "eval-val"
                )
                assert (
                    raw["model"] == model
                    and raw["temperature"] == 1
                    and raw["mode"] == "randomized"
                )
                ordered = sorted(raw["samples"], key=lambda r: r["rollout"])
                assert [r["rollout"] for r in ordered] == list(indices)
                sha = hashlib.sha256(data).hexdigest()
                inventory.append(
                    dict(
                        model=model,
                        stem=stem,
                        start=indices.start,
                        stop=indices.stop,
                        uri=uri,
                        sha256=sha,
                        bytes=len(data),
                    )
                )
                for sample in ordered:
                    ri = sample["rollout"]
                    assert (
                        sample["finish_reason"] == "stop"
                        and sample["prompt_id"] == f"{stem}:r{ri}"
                    )
                    parsed = parse_contacts(
                        sample["text"], sample["offset"], target["L"]
                    )
                    assert [list(p) for p in parsed["contacts"]] == sample["contacts"]
                    assert sample["seed"] == sample_seed(
                        stem, "randomized", ri // 100, ri % 100
                    )
                    identities.append(
                        (
                            sample["seed"],
                            sample["offset"],
                            sample["prompt_token_ids"],
                            raw["prompts"][sample["prompt_id"]],
                        )
                    )
                    samples.append({**sample, "raw_uri": uri, "raw_sha256": sha})
                if indices.start == 100:
                    marker_uri = uri.replace(".json.gz", ".complete.json")
                    marker = json.loads(
                        cached(fs, marker_uri, path.with_suffix(".complete.json"))
                    )
                    assert marker["count"] == 900 and marker["unfinished"] == 0
                    timing.append(dict(model=model, **marker["timing"]))
            if model == "default":
                paired = identities
            else:
                assert paired == identities
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
            started = time.monotonic()
            distance = jaccard(effective)
            distance_seconds = time.monotonic() - started
            started = time.monotonic()
            representatives, fit = medoids(distance, 10, seed_for("diversity", stem))
            selection_timings.append(
                dict(
                    model=model,
                    stem=stem,
                    n_rollouts=1000,
                    distance_seconds=distance_seconds,
                    fit_seconds=time.monotonic() - started,
                    hostname=socket.gethostname(),
                    platform=platform.platform(),
                )
            )
            fits.append(dict(model=model, stem=stem, **fit))
            random = np.sort(
                np.random.default_rng(seed_for("random10", stem)).choice(
                    1000, 10, replace=False
                )
            )
            for arm, chosen in [
                ("medoids10", representatives),
                ("random10", random),
                ("first10", np.arange(10)),
            ]:
                labels = distance[:, chosen].argmin(axis=1)
                labels[chosen] = np.arange(10)
                sub = distance[np.ix_(chosen, chosen)]
                diagnostics.append(
                    dict(
                        model=model,
                        stem=stem,
                        arm=arm,
                        pairwise_jaccard=float(sub[np.triu_indices(10, 1)].mean()),
                        mean_nearest_distance=float(
                            distance[:, chosen].min(axis=1).mean()
                        ),
                        mean_contacts=float(
                            np.mean([len(effective[i]) for i in chosen])
                        ),
                        unique_contact_sets=len({tuple(effective[i]) for i in chosen}),
                        pool_unique_contact_sets=len({tuple(p) for p in effective}),
                        cluster_sizes=json.dumps(
                            np.bincount(labels, minlength=10).tolist()
                        ),
                    )
                )
                if arm == "first10":
                    continue
                for slot, index in enumerate(chosen):
                    sample = samples[index]
                    task = dict(
                        model=model,
                        stem=stem,
                        arm=arm,
                        rollout=slot,
                        source_rollout=int(index),
                        task_id=f"{stem}__{model}__{arm}__r{slot:02d}",
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
                        n_unresolved_removed=len(sample["contacts"])
                        - len(mapped[index]),
                        n_mapped_near_removed=len(mapped[index])
                        - len(effective[index]),
                        n_effective_contacts=len(effective[index]),
                    )
                    task["input_sha256"] = hashlib.sha256(
                        json.dumps(task, sort_keys=True).encode()
                    ).hexdigest()
                    tasks.append(task)
    task_bytes = (json.dumps(tasks, indent=2) + "\n").encode()
    task_path = inputs / "tasks.json"
    if task_path.exists() and task_path.read_bytes() != task_bytes:
        raise ValueError("Refusing to change frozen selections")
    task_path.write_bytes(task_bytes)
    pd.DataFrame(tasks).drop(columns="contacts").to_csv(
        HERE / "data/diversity_contact_inputs.csv", index=False
    )
    pd.DataFrame(diagnostics).to_csv(
        HERE / "data/diversity_contact_diagnostics.csv", index=False
    )
    pd.DataFrame(inventory).to_csv(
        HERE / "data/diversity_raw_inventory.csv", index=False
    )
    pd.DataFrame(timing).to_csv(
        HERE / "data/diversity_marinfold_timings.csv", index=False
    )
    selection_timing_path = HERE / "data/diversity_clustering_timings.csv"
    if not selection_timing_path.exists():
        pd.DataFrame(selection_timings).to_csv(selection_timing_path, index=False)
    manifest = dict(
        plan_sha256=digest(HERE / "data/diversity_plan.json"),
        tasks_sha256=digest(task_path),
        selector_source_sha256=digest(Path(__file__)),
        inventory_sha256=digest(HERE / "data/diversity_raw_inventory.csv"),
        n_rollouts=20000,
        n_helico_tasks=len(tasks),
        fits=fits,
    )
    manifest_path = HERE / "data/diversity_selection_manifest.json"
    manifest["frozen_utc"] = (
        json.loads(manifest_path.read_text())["frozen_utc"]
        if manifest_path.exists()
        else dt.datetime.now(dt.timezone.utc).isoformat()
    )
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Frozen selection manifest differs")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print("Validated 20,000 contact samples; froze", len(tasks), "Helico inputs")


if __name__ == "__main__":
    main()
