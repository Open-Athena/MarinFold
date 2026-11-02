"""Check both real-weight smoke outputs before the 12-worker production run."""

import json
from pathlib import Path

from analyze import analyze_one, candidate_sets, load

HERE = Path(__file__).resolve().parent


def main() -> None:
    """Enforce paired prompt/tokenizer identity, completion and valid scoring."""
    truth = {
        r["stem"]: r
        for r in json.loads((HERE / "inputs/ground_truth.json").read_text())
    }
    results = {}
    records = []
    for model in ["default", "mpnn"]:
        paths = list((HERE / "scratch/smoke" / model).glob("*.json.gz"))
        assert len(paths) == 1
        record = load(paths[0])
        records.append(record)
        assert len(record["samples"]) == 100
        assert all(s["finish_reason"] == "stop" for s in record["samples"])
        gt = truth[record["target"]["stem"]]
        sets, true, ids, labels = candidate_sets(record, gt)
        assert all(len(s) > 0 for s in sets)
        _, _, _, metrics = analyze_one(record, gt, sets, true, ids, labels)
        results[model] = {
            "stem": gt["stem"],
            "n_rollouts": 100,
            "unfinished": 0,
            "metrics": metrics,
            "worker": json.loads((paths[0].parent / "worker-00.json").read_text()),
        }
    assert records[0]["prompts"] == records[1]["prompts"]
    for a, b in zip(records[0]["samples"], records[1]["samples"], strict=True):
        assert a["seed"] == b["seed"] and a["prompt_token_ids"] == b["prompt_token_ids"]
    for filename in ["tokenizer.json", "tokenizer_config.json"]:
        hashes = [
            next(
                f["sha256"]
                for f in results[m]["worker"]["verified_files"]
                if f["name"] == filename
            )
            for m in results
        ]
        assert len(set(hashes)) == 1
    (HERE / "data/smoke_validation.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )
    print(
        "Both checkpoint smokes passed: 200/200 terminated, matched prompts, tokenizer hashes and seeds, canonical scoring valid."
    )


if __name__ == "__main__":
    main()
