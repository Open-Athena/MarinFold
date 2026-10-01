"""Freeze the eval-val inputs and verify the epoch-2 checkpoint in place."""

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import s3fs

HERE = Path(__file__).resolve().parent
REFERENCE = HERE.parent / "exp277_models_single_mpnn_pilot/evals/2026-09-13_rollout_v2"
sys.path.insert(0, str(REFERENCE))
from checkpoint_specs import EXP277_EPOCH2_CHECKPOINT, MARINFOLD_REVISION
from hf_to_s3 import expected_manifest


def main() -> None:
    """Prepare sequences only; the inference worker never receives ground truth."""
    targets = pd.read_csv(HERE.parent / "exp245_evals_foldbench_held_out_monomers/data/eval_sets.csv")
    targets = targets[targets.eval_set.eq("eval-val")].sort_values(["seq_len", "stem"])
    assert len(targets) == targets.stem.nunique() == 97
    records = [{"dataset": "foldbench_monomer", "stem": row.stem, "L": int(row.seq_len),
                "input_seq": row.sequence, "ordinal": n}
               for n, row in enumerate(targets.itertuples())]
    for row in records:
        assert len(row["input_seq"]) == row["L"]
    target_path = HERE / "data/whole_map_targets.json"
    target_path.write_text(json.dumps(records, indent=2) + "\n")
    manifest = expected_manifest(EXP277_EPOCH2_CHECKPOINT)
    fs = s3fs.S3FileSystem(profile="cw", endpoint_url="https://cwobject.com",
                         config_kwargs={"s3": {"addressing_style": "virtual"}})
    objects = fs.ls(EXP277_EPOCH2_CHECKPOINT.coreweave_uri, detail=True)
    by_name = {Path(o["name"]).name: o for o in objects}
    assert set(by_name) == {entry["name"] for entry in manifest["files"]}
    for entry in manifest["files"]:
        actual = by_name[entry["name"]]
        assert actual["size"] == entry["size"]
        assert actual["ETag"].strip('"') == entry["digest"]
    manifest.update({"marinfold_revision": MARINFOLD_REVISION,
                     "reference_worker_sha256": hashlib.sha256((REFERENCE / "score_rollout_worker.py").read_bytes()).hexdigest(),
                     "targets_sha256": hashlib.sha256(target_path.read_bytes()).hexdigest(),
                     "n_proteins": 97, "n_rollouts": 200, "n_pools": 2,
                     "temperature": 1.0, "top_p": 0.95, "top_k": -1,
                     "seed": 345, "token_budget": "min(8192-prompt_tokens,6L+128)",
                     "output": "s3://marin-us-east-02a/MarinFold/exp345/whole-map-v1"})
    (HERE / "data/whole_map_plan.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Verified {len(by_name)} checkpoint objects in place; froze {len(records)} eval-val sequences")


if __name__ == "__main__":
    main()
