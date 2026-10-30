"""Audit added samples and prove that the original 300-sample prefixes are intact."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import HfFileSystem

from analyze import candidate_sets, load
from contacts import sample_seed
from prepare_extension import digest

HERE = Path(__file__).resolve().parent


def validate_condition(record: dict, original: dict, stop: int) -> dict:
    """Check new indices, exact prompt identity, seeds, continuation and counts."""
    assert record["target"] == original["target"]
    assert record["prompts"] == original["prompts"]
    assert record["model"] == original["model"]
    assert record["mode"] == "fixed" and record["temperature"] == 1
    stem, length = record["target"]["stem"], record["target"]["L"]
    samples = record["samples"]
    assert len(samples) == stop - 300
    assert len({s["seed"] for s in original["samples"] + samples}) == stop
    reference = original["samples"][0]
    counts = {"unfinished": 0, "extended": 0, "empty_contacts": 0}
    diagnostic_names = [
        "malformed_statements",
        "invalid_positions",
        "unexpected_text",
        "retractions",
    ]
    counts.update({name + "_samples": 0 for name in diagnostic_names})
    for k, sample in enumerate(samples, 300):
        assert sample["rollout"] == k and sample["block"] == k // 100
        assert sample["seed"] == sample_seed(stem, "fixed", k // 100, k % 100)
        assert sample["prompt_id"] == f"{stem}:r0"
        assert sample["offset"] == reference["offset"]
        assert sample["prompt_token_ids"] == reference["prompt_token_ids"]
        prompt_n = len(sample["prompt_token_ids"])
        assert sample["initial_budget"] == min(8192 - prompt_n, 6 * length + 128)
        assert prompt_n + len(sample["token_ids"]) <= 8192
        assert sample["finish_reason"] in ["stop", "length"]
        if sample["continuation_seed"] is not None:
            a = hashlib.sha256(
                json.dumps(sample["prompt_token_ids"], separators=(",", ":")).encode()
            ).hexdigest()
            b = hashlib.sha256(
                json.dumps(
                    sample["token_ids"][: sample["initial_tokens"]],
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
            identity = f"exp364-continuation-v1:{sample['seed']}:{a}:{b}"
            assert sample["continuation_seed"] == int.from_bytes(
                hashlib.sha256(identity.encode()).digest()[:8], "big"
            ) % (2**63 - 1)
        counts["unfinished"] += sample["finish_reason"] != "stop"
        counts["extended"] += sample["continuation_seed"] is not None
        counts["empty_contacts"] += not sample["contacts"]
        for name in diagnostic_names:
            counts[name + "_samples"] += sample[name] > 0
    return counts


def main() -> None:
    """Verify anonymous downloads and save a machine-readable acceptance record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    plan = json.loads((HERE / "data/extension_plan.json").read_text())
    for relative, key in [
        ("data/plan.json", "original_plan_sha256"),
        ("data/extension_prefix_inventory.csv", "prefix_inventory_sha256"),
        ("EXTENSION_PLAN.md", "protocol_sha256"),
        ("inputs/targets.json", "targets_sha256"),
        ("inputs/ground_truth.json", "truth_sha256"),
    ]:
        assert digest(HERE / relative) == plan[key]
    truth = {
        r["stem"]: r
        for r in json.loads((HERE / "inputs/ground_truth.json").read_text())
    }
    targets = json.loads((HERE / "inputs/targets.json").read_text())
    stems = {r["stem"] for r in targets}
    if args.smoke:
        stems = {
            min(
                (r for r in targets if r["eval_set"] == "eval-val"),
                key=lambda r: abs(r["L"] - 200),
            )["stem"]
        }
    stop = 400 if args.smoke else 1000
    phase = "smoke" if args.smoke else "raw"
    directory = HERE / "scratch" / (phase + "_1000")
    prefix_hashes = (
        pd.read_csv(HERE / "data/extension_prefix_inventory.csv")
        .set_index(["model", "stem"])
        .sha256
    )
    records, inventory, timings = [], [], []
    for stem in sorted(stems):
        paired = {}
        for model in ["default", "mpnn"]:
            name = f"{stem}-t1.0-fixed.json.gz"
            original_path = HERE / "scratch/raw" / model / name
            assert digest(original_path) == prefix_hashes.loc[(model, stem)]
            original = load(original_path)
            path = directory / model / name
            record = load(path)
            counts = validate_condition(record, original, stop)
            paired[model] = record
            candidate_sets(record, truth[stem])
            marker = json.loads(
                path.with_name(name.replace(".json.gz", ".complete.json")).read_text()
            )
            assert marker["count"] == stop - 300
            assert all(marker[key] == counts[key] for key in ["unfinished", "extended"])
            timing = marker["timing"]
            assert timing["rollout_start"] == 300 and timing["rollout_stop"] == stop
            for key in [
                "elapsed_seconds",
                "model_load_seconds",
                "model_stage_seconds",
                "total_seconds",
            ]:
                assert np.isfinite(timing[key]) and timing[key] >= 0
            assert timing["total_seconds"] >= timing["elapsed_seconds"]
            uri = plan["artifact_root"] + f"/{phase}/{model}/{name}"
            assert marker["raw_uri"] == uri
            records.append(
                {
                    "stem": stem,
                    "model": model,
                    "n_rollouts": len(record["samples"]),
                    **counts,
                }
            )
            timings.append(
                {"model": model, "eval_set": record["target"]["eval_set"], **timing}
            )
            inventory.append(
                {
                    "model": model,
                    "stem": stem,
                    "rollout_start": 300,
                    "rollout_stop": stop,
                    "bytes": path.stat().st_size,
                    "sha256": digest(path),
                    "uri": uri,
                }
            )
        assert paired["default"]["prompts"] == paired["mpnn"]["prompts"]
        assert [s["seed"] for s in paired["default"]["samples"]] == [
            s["seed"] for s in paired["mpnn"]["samples"]
        ]
        assert [s["prompt_token_ids"] for s in paired["default"]["samples"]] == [
            s["prompt_token_ids"] for s in paired["mpnn"]["samples"]
        ]
        print("VALIDATED", stem, flush=True)
    checkpoints = json.loads((HERE / "data/validation.json").read_text())["checkpoints"]
    workers = []
    for path in sorted(directory.glob("*/worker-*.json")):
        worker = json.loads(path.read_text())
        for entry in worker["verified_files"]:
            assert entry["sha256"] == checkpoints[f"{path.parent.name}/{entry['name']}"]
        assert worker["runtime_versions"]["vllm"] == "0.9.2"
        assert worker["runtime_versions"]["torch"] == "2.7.0+cu128"
        workers.append(
            {
                "model": path.parent.name,
                "worker": path.name,
                **worker["hardware"],
                **worker["runtime_versions"],
            }
        )
    assert len(workers) == (2 if args.smoke else 12)
    if args.smoke:
        assert all(r["unfinished"] == 0 and r["empty_contacts"] == 0 for r in records)
        output = {r["model"]: r for r in records}
        (HERE / "data/smoke_validation_1000.json").write_text(
            json.dumps(output, indent=2) + "\n"
        )
        print(json.dumps(output, indent=2))
        return
    assert len(records) == 232 and sum(r["n_rollouts"] for r in records) == 162400
    assert len(list(directory.glob("*/*.json.gz"))) == 232
    source_hashes = []
    launches = json.loads((HERE / "data/launches_1000.json").read_text())
    for sha in sorted({r["source_sha256"] for r in launches if not r["smoke"]}):
        archive = HERE / "scratch" / f"{sha}.zip"
        if not archive.exists():
            uri = next(r["source_uri"] for r in launches if r["source_sha256"] == sha)
            archive.write_bytes(
                HfFileSystem(token=False).read_bytes(uri.removeprefix("hf://"))
            )
        assert digest(archive) == sha
        with zipfile.ZipFile(archive) as bundle:
            assert (
                bundle.read("data/extension_plan.json")
                == (HERE / "data/extension_plan.json").read_bytes()
            )
            assert (
                bundle.read("data/plan.json") == (HERE / "data/plan.json").read_bytes()
            )
            assert bundle.read("contacts.py") == (HERE / "contacts.py").read_bytes()
            source_hashes.append(
                {
                    "bundle": sha,
                    "worker_sha256": hashlib.sha256(
                        bundle.read("worker.py")
                    ).hexdigest(),
                }
            )
    assert len({r["worker_sha256"] for r in source_hashes}) == 1
    for name, rows in [
        ("raw_inventory_1000", inventory),
        ("timings_1000", timings),
        ("workers_1000", workers),
        ("diagnostics_1000", records),
    ]:
        pd.DataFrame(rows).to_csv(HERE / "data" / f"{name}.csv", index=False)
    validation = {
        "conditions": len(records),
        "new_samples": 162400,
        "combined_fixed_samples": 232000,
        "original_prefixes_unchanged": True,
        "paired_prompts_and_seeds": True,
        "checkpoint_files_match_original": True,
        "raw_bytes": sum(r["bytes"] for r in inventory),
        "inference_gpu_hours": sum(r["elapsed_seconds"] for r in timings) / 3600,
        "source_bundles": source_hashes,
        "totals": {
            key: sum(r[key] for r in records)
            for key in records[0]
            if key not in ["stem", "model"]
        },
    }
    (HERE / "data/validation_1000.json").write_text(
        json.dumps(validation, indent=2) + "\n"
    )
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
