# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Record the finished exp343 export as `data/exp343_checkpoint.json`.

`checkpoint_specs.load_exp343_checkpoint` reads this file, and the eval driver
verifies the bytes each worker downloads against the digests in it. So the digests
have to come from the export itself -- a hand-written or placeholder value would
not fail, it would quietly turn the driver's byte check into a no-op.

Sizes and ETags are read from CoreWeave object storage; the training and
validation losses come from the W&B run, so the recorded spec says which point in
training was evaluated and at what loss.

Writes `exp343_checkpoint.json` beside `checkpoint_specs.py`, because
`submit_coreweave.py` ships this directory as the job workspace and nothing
outside it reaches the pod.

Reading CoreWeave object storage from a workstation needs the credentials the
pods get injected:

    export FSSPEC_S3="$(python3 -c "import configparser,json;c=configparser.ConfigParser();\
c.read('$HOME/.aws/credentials');w=c['cw'];print(json.dumps({'key':w['aws_access_key_id'],\
'secret':w['aws_secret_access_key'],'endpoint_url':'https://cwobject.com',\
'config_kwargs':{'s3':{'addressing_style':'virtual'}}}))")"
    uv run python make_checkpoint_spec.py --step 280154
"""

import argparse
import json
from pathlib import Path

import fsspec

from checkpoint_specs import CHECKPOINT_SPEC_PATH, MARIN_PREFIX

RUN_NAME = "contacts-v1-exp343-m2-p06-complex-1.5B"
EXPERIMENT = "exp343_models_complex_corpus_training"
#: Files a contacts-v1 export must carry. The tokenizer is not optional: a model
#: without it is unloadable for eval, serving and reproduction.
REQUIRED = (
    "config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
WEIGHT_PREFIX = "model-"
WEIGHT_SUFFIX = ".safetensors"


def export_uri(step: int) -> str:
    return f"{MARIN_PREFIX}/{EXPERIMENT}/runs/{RUN_NAME}/hf/step-{step}"


def read_export(uri: str) -> list[dict]:
    """Every file in the export with its size and CoreWeave ETag."""
    fs, root = fsspec.core.url_to_fs(uri)
    entries = []
    for info in fs.ls(root, detail=True):
        name = info["name"].rsplit("/", 1)[-1]
        if info.get("type") == "directory" or not name:
            continue
        etag = (info.get("ETag") or info.get("etag") or "").strip('"')
        if not etag:
            raise ValueError(f"{name}: object storage returned no ETag")
        entries.append(
            {
                "name": name,
                "size": int(info["size"]),
                "digest": etag,
                "digest_kind": "s3-etag",
            }
        )
    entries.sort(key=lambda entry: entry["name"])
    missing = [name for name in REQUIRED if name not in {e["name"] for e in entries}]
    if missing:
        raise ValueError(f"{uri} is missing {missing}")
    return entries


def weight_shards(entries: list[dict]) -> tuple[str, ...]:
    shards = [
        entry
        for entry in entries
        if entry["name"].startswith(WEIGHT_PREFIX)
        and entry["name"].endswith(WEIGHT_SUFFIX)
    ]
    if not shards:
        raise ValueError("the export has no safetensors weight shards")
    return tuple(entry["digest"] for entry in shards)


def wandb_losses(step: int) -> dict:
    """The run's final train loss and its best full validation loss."""
    import wandb

    run = wandb.Api().run(f"open-athena/MarinFold/{RUN_NAME}")
    summary = dict(run.summary)
    history = run.history(keys=["eval/input/validation/loss"], pandas=False)
    best = None
    best_step = None
    for row in history:
        value = row.get("eval/input/validation/loss")
        if value is None:
            continue
        if best is None or value < best:
            best, best_step = value, row.get("_step")
    return {
        "train_loss": summary.get("train/loss"),
        "eval_loss": best if best is not None else summary.get("eval/input/validation/loss"),
        "eval_loss_step": best_step,
        "wandb_state": run.state,
        "wandb_last_step": summary.get("_step"),
        "complex_validation_loss": summary.get("eval/input/validation-complex/loss"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--out", type=Path, default=CHECKPOINT_SPEC_PATH)
    arguments = parser.parse_args()
    uri = export_uri(arguments.step)
    entries = read_export(uri)
    spec = {
        # Root AGENTS.md: "When referring to a checkpoint as a string identifier
        # anywhere ... the `<wandb-run-name>-step-<N>` format applies." This label
        # propagates into timing rows, metric model names, output prefixes and the
        # run manifest, so an abbreviation there makes every downstream artifact
        # harder to trace back to the exact run.
        "label": f"{RUN_NAME}-step-{arguments.step}",
        "job_label": "exp343",
        "run_name": RUN_NAME,
        "step": arguments.step,
        "coreweave_uri": uri,
        "source_dtype": "float32",
        "checkpoint_files": entries,
        "weight_shard_digests": list(weight_shards(entries)),
        "total_bytes": sum(entry["size"] for entry in entries),
    }
    if not arguments.no_wandb:
        spec.update(wandb_losses(arguments.step))
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(spec, indent=2) + "\n")
    print(json.dumps(spec, indent=2))
    print(f"-> {arguments.out}")


if __name__ == "__main__":
    main()
