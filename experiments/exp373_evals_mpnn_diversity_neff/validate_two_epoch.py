"""Audit every new sample, regional checkpoint and immutable original result."""

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from analyze import candidate_sets, load
from contacts import sample_seed
from prepare_extension import digest

HERE = Path(__file__).resolve().parent


def main() -> None:
    """Require complete coverage, exact paired prompts/seeds, and verified files."""
    plan = json.loads((HERE / "data/two_epoch_plan.json").read_text())
    metric_source = (
        HERE.parent / "exp89_evals_contacts_v1_model_on_eval_set/compute_metrics.py"
    )
    assert digest(metric_source) == plan["metric_source_sha256"]
    reference = json.loads(
        (HERE / "data/two_epoch_reference_protocol.json").read_text()
    )
    assert {Path(k).name: v for k, v in plan["source_inputs"].items()} == reference[
        "source_hashes"
    ]
    for name, sha in plan["input_hashes"].items():
        assert digest(HERE / name) == sha, name
    original_files = 0
    for inventory_name, directory in [
        ("raw_inventory.csv", "raw"),
        ("raw_inventory_1000.csv", "raw_1000"),
    ]:
        for row in pd.read_csv(HERE / "data" / inventory_name).itertuples():
            name = row.uri.rsplit("/", 1)[1]
            assert digest(HERE / "scratch" / directory / row.model / name) == row.sha256
            original_files += 1
    targets = {
        r["stem"]: r for r in json.loads((HERE / "inputs/targets.json").read_text())
    }
    truths = {
        r["stem"]: r
        for r in json.loads((HERE / "inputs/ground_truth.json").read_text())
    }
    inventory, diagnostics, timings, workers = [], [], [], []
    root = HERE / "scratch/raw_two_epoch"
    for stem, target in targets.items():
        for temp in plan["temperatures"]:
            for mode in ["fixed", "randomized"]:
                name = f"{stem}-t{temp:.1f}-{mode}.json.gz"
                reference = load(HERE / "scratch/raw/default" / name)
                expected = (1000 if temp == 1 else 300) if mode == "fixed" else 100
                for model in plan["models"]:
                    path = root / model / name
                    record = load(path)
                    assert record["model"] == model and record["target"] == target
                    assert record["temperature"] == temp and record["mode"] == mode
                    assert record["prompts"] == reference["prompts"]
                    assert len(record["samples"]) == expected
                    assert len({s["seed"] for s in record["samples"]}) == expected
                    counts = dict(
                        unfinished=0,
                        extended=0,
                        empty=0,
                        malformed_statements=0,
                        invalid_positions=0,
                        unexpected_text=0,
                        retractions=0,
                    )
                    for k, sample in enumerate(record["samples"]):
                        ref = reference["samples"][0 if mode == "fixed" else k]
                        assert sample["rollout"] == k and sample["block"] == k // 100
                        assert sample["seed"] == sample_seed(
                            stem, mode, k // 100, k % 100
                        )
                        assert (
                            sample["prompt_id"]
                            == f"{stem}:r{0 if mode == 'fixed' else k}"
                        )
                        assert (
                            sample["offset"] == ref["offset"]
                            and sample["prompt_token_ids"] == ref["prompt_token_ids"]
                        )
                        prompt_n = len(sample["prompt_token_ids"])
                        assert sample["initial_budget"] == min(
                            8192 - prompt_n, 6 * target["L"] + 128
                        )
                        assert prompt_n + len(sample["token_ids"]) <= 8192
                        assert sample["finish_reason"] in ["stop", "length"]
                        if sample["continuation_seed"] is not None:
                            a = hashlib.sha256(
                                json.dumps(
                                    sample["prompt_token_ids"], separators=(",", ":")
                                ).encode()
                            ).hexdigest()
                            b = hashlib.sha256(
                                json.dumps(
                                    sample["token_ids"][: sample["initial_tokens"]],
                                    separators=(",", ":"),
                                ).encode()
                            ).hexdigest()
                            identity = (
                                f"exp364-continuation-v1:{sample['seed']}:{a}:{b}"
                            )
                            assert sample["continuation_seed"] == int.from_bytes(
                                hashlib.sha256(identity.encode()).digest()[:8], "big"
                            ) % (2**63 - 1)
                        counts["unfinished"] += sample["finish_reason"] != "stop"
                        counts["extended"] += sample["continuation_seed"] is not None
                        counts["empty"] += not sample["contacts"]
                        for key in [
                            "malformed_statements",
                            "invalid_positions",
                            "unexpected_text",
                            "retractions",
                        ]:
                            counts[key] += sample[key] > 0
                    candidate_sets(record, truths[stem])
                    marker = json.loads(
                        path.with_name(
                            name.replace(".json.gz", ".complete.json")
                        ).read_text()
                    )
                    assert marker["count"] == expected
                    assert (
                        marker["unfinished"] == counts["unfinished"]
                        and marker["extended"] == counts["extended"]
                    )
                    uri = plan["artifact_root"] + f"/raw/{model}/{name}"
                    assert marker["raw_uri"] == uri
                    timing = marker["timing"]
                    assert (
                        timing["checkpoint_step"] == 373768
                        and timing["n_rollouts"] == expected
                    )
                    assert (
                        timing["rollout_start"] == 0
                        and timing["rollout_stop"] == expected
                    )
                    for key in [
                        "elapsed_seconds",
                        "model_load_seconds",
                        "model_stage_seconds",
                        "total_seconds",
                    ]:
                        assert np.isfinite(timing[key]) and timing[key] >= 0
                    assert timing["total_seconds"] >= timing["elapsed_seconds"]
                    base = dict(
                        model=model,
                        stem=stem,
                        temperature=temp,
                        mode=mode,
                        eval_set=target["eval_set"],
                    )
                    inventory.append(
                        {
                            **base,
                            "rollouts": expected,
                            "bytes": path.stat().st_size,
                            "sha256": digest(path),
                            "uri": uri,
                        }
                    )
                    diagnostics.append({**base, "n_rollouts": expected, **counts})
                    timings.append({**base, **timing})
        print("VALIDATED", stem, flush=True)
    assert (
        len(inventory)
        == plan["expected_conditions"]
        == len(list(root.glob("*/*.json.gz")))
    )
    assert sum(r["rollouts"] for r in inventory) == plan["expected_samples"]
    original_versions = json.loads((HERE / "data/runtime_versions.json").read_text())[
        "versions"
    ]["default"]
    for path in sorted(root.glob("*/worker-*.json")):
        record = json.loads(path.read_text())
        model = path.parent.name
        assert record["model"] == plan["models"][model]
        assert record["verified_files"] == plan["models"][model]["files"]
        assert all(
            version == original_versions[name]
            for name, version in record["runtime_versions"].items()
        )
        workers.append(
            {
                "model": model,
                "worker": path.name,
                **record["hardware"],
                **record["runtime_versions"],
            }
        )
    assert len(workers) == 12
    launches = json.loads((HERE / "data/launches_two_epoch.json").read_text())
    bundles = sorted({r["source_sha256"] for r in launches})
    for sha in bundles:
        path = HERE / "scratch" / (sha + ".zip")
        assert digest(path) == sha
        with zipfile.ZipFile(path) as bundle:
            for name in [
                "worker.py",
                "contacts.py",
                "publish_file.py",
                "data/two_epoch_plan.json",
                "inputs/targets.json",
            ]:
                assert bundle.read(name) == (HERE / name).read_bytes()
    for name, rows in [
        ("raw_inventory", inventory),
        ("diagnostics", diagnostics),
        ("timings", timings),
        ("workers", workers),
    ]:
        pd.DataFrame(rows).to_csv(HERE / "data" / f"{name}_two_epoch.csv", index=False)
    result = dict(
        conditions=len(inventory),
        samples=plan["expected_samples"],
        original_files_unchanged=original_files,
        raw_bytes=sum(r["bytes"] for r in inventory),
        paired_prompts_and_seeds=True,
        all_checkpoint_sha256_verified=True,
        source_bundles=bundles,
        inference_gpu_hours=sum(r["elapsed_seconds"] for r in timings) / 3600,
        counts={k: sum(r[k] for r in diagnostics) for k in counts},
    )
    (HERE / "data/validation_two_epoch.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
