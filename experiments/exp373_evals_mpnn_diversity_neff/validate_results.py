"""Independently audit complete raw samples, source identity and candidate scores."""

import ast
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from huggingface_hub import HfFileSystem

from analyze import load
from contacts import sample_seed

HERE = Path(__file__).resolve().parent


def main() -> None:
    """Require exact coverage and preserve a public raw-file checksum inventory."""
    plan = json.loads((HERE / "data/plan.json").read_text())
    inputs = json.loads((HERE / "inputs/targets.json").read_text())
    targets = {r["stem"]: r for r in inputs}
    expected = {
        (model, stem, temp, mode)
        for model in plan["models"]
        for stem in targets
        for temp in plan["temperatures"]
        for mode in ["fixed", "randomized"]
    }
    seen = set()
    inventory = []
    examples = []
    for path in sorted((HERE / "scratch/raw").glob("*/*.json.gz")):
        record = load(path)
        model = record["model"]
        stem = record["target"]["stem"]
        temp = record["temperature"]
        mode = record["mode"]
        key = (model, stem, temp, mode)
        assert key in expected and key not in seen
        seen.add(key)
        assert record["target"] == targets[stem]
        samples = record["samples"]
        assert len(samples) == (300 if mode == "fixed" else 100)
        marker = json.loads(
            path.with_name(path.name.replace(".json.gz", ".complete.json")).read_text()
        )
        for k, sample in enumerate(samples):
            assert sample["rollout"] == k and sample["block"] == k // 100
            assert sample["seed"] == sample_seed(stem, mode, k // 100, k % 100)
            assert sample["prompt_id"] == f"{stem}:r{0 if mode == 'fixed' else k}"
            prompt_n = len(sample["prompt_token_ids"])
            assert sample["initial_budget"] == min(
                8192 - prompt_n, 6 * targets[stem]["L"] + 128
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
                identity = f"exp364-continuation-v1:{sample['seed']}:{a}:{b}"
                assert sample["continuation_seed"] == int.from_bytes(
                    hashlib.sha256(identity.encode()).digest()[:8], "big"
                ) % (2**63 - 1)
            if (
                sample["finish_reason"] != "stop"
                or sample["continuation_seed"] is not None
            ):
                examples.append(
                    {
                        "model": model,
                        "stem": stem,
                        "temperature": temp,
                        "mode": mode,
                        "rollout": k,
                        "initial_tokens": sample["initial_tokens"],
                        "final_tokens": len(sample["token_ids"]),
                        "prompt_tokens": prompt_n,
                        "finish_reason": sample["finish_reason"],
                        "extended": sample["continuation_seed"] is not None,
                    }
                )
        timing = marker["timing"]
        for name in ["elapsed_seconds", "model_load_seconds", "total_seconds"]:
            assert np.isfinite(timing[name]) and timing[name] >= 0
        assert timing["total_seconds"] >= timing["elapsed_seconds"]
        assert timing["checkpoint_step"] == plan["models"][model]["step"]
        inventory.append(
            {
                "model": model,
                "stem": stem,
                "temperature": temp,
                "mode": mode,
                "rollouts": len(samples),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "uri": marker["raw_uri"],
            }
        )
    assert seen == expected, expected - seen
    assert sum(r["rollouts"] for r in inventory) == 278400
    identities = defaultdict(set)
    workers = []
    for path in sorted((HERE / "scratch/raw").glob("*/worker-*.json")):
        record = json.loads(path.read_text())
        model = path.parent.name
        for file in record["verified_files"]:
            identities[(model, file["name"])].add(file["sha256"])
        workers.append({"model": model, "worker": path.name, **record["hardware"]})
    assert len(workers) == 12 and all(len(v) == 1 for v in identities.values())
    launches = json.loads((HERE / "data/launches.json").read_text())
    source_hashes = []
    for digest in sorted({x["source_sha256"] for x in launches if not x["smoke"]}):
        archive = HERE / "scratch" / f"{digest}.zip"
        if not archive.exists():
            uri = next(
                x["source_uri"] for x in launches if x["source_sha256"] == digest
            )
            content = HfFileSystem(token=False).read_bytes(uri.removeprefix("hf://"))
            assert hashlib.sha256(content).hexdigest() == digest
            archive.write_bytes(content)
        assert hashlib.sha256(archive.read_bytes()).hexdigest() == digest
        with zipfile.ZipFile(archive) as bundle:
            worker = ast.dump(
                ast.parse(bundle.read("worker.py").decode()), include_attributes=False
            )
            contacts = ast.dump(
                ast.parse(bundle.read("contacts.py").decode()), include_attributes=False
            )
            source_hashes.append(
                {
                    "bundle_sha256": digest,
                    "worker_ast_sha256": hashlib.sha256(worker.encode()).hexdigest(),
                    "contacts_ast_sha256": hashlib.sha256(
                        contacts.encode()
                    ).hexdigest(),
                    "plan_sha256": hashlib.sha256(
                        bundle.read("data/plan.json")
                    ).hexdigest(),
                }
            )
    # Recovery changed only formatting/publication; the scientific worker AST is identical.
    assert len({r["worker_ast_sha256"] for r in source_hashes}) == 1
    assert len({r["contacts_ast_sha256"] for r in source_hashes}) == 1
    assert len({r["plan_sha256"] for r in source_hashes}) == 1
    metrics = pd.read_csv(HERE / "data/metrics.csv")
    assert len(metrics) == 1392 * 20
    assert metrics.groupby(["model", "stem", "temperature", "mode"]).size().eq(20).all()
    pd.DataFrame(inventory).to_csv(HERE / "data/raw_inventory.csv", index=False)
    pd.DataFrame(examples).to_csv(
        HERE / "data/continuation_diagnostics.csv", index=False
    )
    pd.DataFrame(workers).to_csv(HERE / "data/workers.csv", index=False)
    validation = {
        "conditions": len(seen),
        "samples": 278400,
        "raw_bytes": sum(r["bytes"] for r in inventory),
        "metric_rows": len(metrics),
        "checkpoint_files_identical_across_shards": True,
        "source_programs": source_hashes,
        "checkpoints": {
            f"{m}/{name}": next(iter(v)) for (m, name), v in identities.items()
        },
    }
    (HERE / "data/validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
