"""Commit a complete externally dispatched evaluation for the periodic logger."""

import argparse
import json
from pathlib import Path

from aggregate_eval import aggregate, read_json, write_json
from common import ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dispatch", type=Path, required=True)
    parser.add_argument("--targets", type=Path, default=Path("eval_val.jsonl"))
    args = parser.parse_args()
    dispatch = json.loads(args.dispatch.read_text())
    request = read_json(dispatch["checkpoint"] + "/export.json")["request"]
    if request["format"] != dispatch["format"]:
        raise ValueError("Export and evaluation document formats disagree")
    result = {
        **aggregate(
            dispatch["out"], args.targets, request["checkpoint"], request["format"]
        ),
        "step": request["step"],
        "tokens": request["tokens"],
        "training_run": request["run_id"],
        "final": request["final"],
    }
    write_json(dispatch["out"] + "/dispatch.json", dispatch)
    filename = f"{request['run_id']}/step-{request['step']}.json"
    # The persistent driver sees the request only once the complete result is
    # available, so it logs the external fan-out without dispatching it again.
    write_json(f"{ROOT}/eval_results/{filename}", result)
    write_json(f"{ROOT}/eval_requests/{filename}", request)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
