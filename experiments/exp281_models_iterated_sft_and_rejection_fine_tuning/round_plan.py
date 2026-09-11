"""Render the explicit DAG for one offline generate/select/train round.

This planner never launches jobs or selects checkpoints. Finish each group before
the next. Each refresh names its own immutable output prefix and the checkpoint
chosen from the previous round; fixed-corpus controls reuse the corpus manifest.
"""

import argparse
import json

from common import ROOT


def plan(config: dict) -> list[dict]:
    """Return generation fan-out, corpus construction, and one training stage."""
    required = ("round", "phase", "generator", "initial_model", "tokenizer", "targets", "steps", "run_name")
    if any(key not in config for key in required):
        raise ValueError(f"round config requires {required}")
    if config["phase"] not in ("bootstrap", "synthesis", "rejection"):
        raise ValueError("invalid phase")
    if config["phase"] != "bootstrap" and config["generator"] != config["initial_model"]:
        raise ValueError("on-policy refresh must sample the checkpoint being fine-tuned")
    root = f"{config.get('root', ROOT)}/{config['round']}"
    workers = config.get("generation_workers", 4)
    if workers < 1:
        raise ValueError("generation worker count must be positive")
    stages = []
    for split in ("train", "validation"):
        split_workers = config.get(f"{split}_generation_workers", workers)
        if split_workers < 1:
            raise ValueError("generation worker count must be positive")
        for shard in range(split_workers):
            stages.append({"group": 0, "stage": "generate", "gpus": 1,
                           "name": f"exp281-{config['round']}-{split}-gen-{shard}",
                           "arguments": ["--model", config["generator"], "--targets", config["targets"],
                                         "--split", split, "--output", f"{root}/{split}/candidates",
                                         "--phase", "bootstrap" if config["phase"] == "bootstrap" else "synthesis",
                                         "--candidates", str(config.get("candidates", 1 if config["phase"] == "bootstrap" else 4)),
                                         "--bootstrap-hypotheses", str(config.get("bootstrap_hypotheses", 16)),
                                         "--shard-count", str(split_workers), "--shard-index", str(shard),
                                         "--seed", str(config.get("seed", 281))]})
        stages.append({"group": 1, "stage": "corpus", "gpus": 0,
                       "name": f"exp281-{config['round']}-{split}-corpus",
                       "arguments": ["--candidates", f"{root}/{split}/candidates/*.parquet",
                                     "--tokenizer", config["tokenizer"], "--output", f"{root}/{split}/corpus",
                                     "--selection", "best" if split == "train" and config["phase"] == "rejection" else "random",
                                     "--hypothesis-weight", str(1.0 if config["phase"] == "bootstrap" else config.get("hypothesis_weight", 0.1)),
                                     "--rehearsal-fraction", str(config.get("rehearsal_fraction", 0.5 if config["phase"] == "bootstrap" else 0.1))]})
    stages.append({"group": 2, "stage": "train", "gpus": 8, "name": config["run_name"],
                   "arguments": ["--model", config["initial_model"], "--train-manifest", f"{root}/train/corpus/manifest.json",
                                 "--validation-manifest", f"{root}/validation/corpus/manifest.json",
                                 "--output", root, "--run-name", config["run_name"], "--steps", str(config["steps"]),
                                 "--warmup", str(min(100, config["steps"] // 10))]})
    for key in ("global_batch", "microbatch", "lr", "save_every", "eval_every", "eval_documents"):
        if key in config:
            stages[-1]["arguments"].extend(["--" + key.replace("_", "-"), str(config[key])])
    return sorted(stages, key=lambda stage: stage["group"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    args = parser.parse_args()
    with open(args.config) as handle:
        print(json.dumps(plan(json.load(handle)), indent=2))


if __name__ == "__main__":
    main()
