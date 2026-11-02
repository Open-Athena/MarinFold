"""Verify the two completed parent checkpoints on a regional CoreWeave CPU."""

import argparse
import json
from datetime import UTC, datetime

import fsspec
import tensorstore as ts

from verify_final import verify_export

PREFIX = "s3://marin-us-east-02a/MarinFold/exp364_models_mpnn_only_lr_sweep"
FINAL_STEP = 373768


def main() -> None:
    """Check exact native counters and every exported parameter before completion."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-uri", required=True)
    args = parser.parse_args()
    fs = fsspec.filesystem("s3", use_listings_cache=False)
    records = {}
    for trial in ("lr5e-4", "lr1e-3"):
        run_id = f"contacts-v1-exp364-mpnn-only-2ep-{trial}-1.5B"
        output = f"{PREFIX}/runs/{run_id}"
        checkpoint = f"{output}/checkpoints/step-{FINAL_STEP}"
        with fs.open(checkpoint + "/metadata.json") as stream:
            metadata = json.load(stream)
        if metadata["step"] != FINAL_STEP or metadata["is_temporary"]:
            raise ValueError(f"Unexpected final native metadata: {metadata}")
        scalars = {}
        for key in (
            "step",
            "opt_state/count",
            "opt_state/hyperparams_states/learning_rate/count",
            "opt_state/inner_state/1/count",
        ):
            array = ts.open(
                {
                    "driver": "zarr3",
                    "kvstore": {
                        "driver": "ocdbt",
                        "base": {
                            "driver": "s3",
                            "path": checkpoint.split("/", 3)[3] + "/",
                            "endpoint": "http://marin-us-east-02a.cwlota.com",
                            "aws_region": "us-east-1",
                            "aws_credentials": {"type": "environment"},
                        },
                        "path": key,
                    },
                }
            ).result()
            scalars[key] = array.read().result().item()
            if scalars[key] != FINAL_STEP + 1:
                raise ValueError(
                    f"Incomplete native state: {trial} {key}={scalars[key]}"
                )
        export = verify_export(f"{output}/hf/step-{FINAL_STEP}")
        records[trial] = {
            "run_id": run_id,
            "checkpoint": checkpoint,
            "native_metadata": metadata,
            "scalar_counters": scalars,
            "export_verification": export,
        }
        print("VERIFIED", trial, json.dumps(records[trial]), flush=True)
    report = {"verified_at_utc": datetime.now(UTC).isoformat(), "trials": records}
    with fs.open(args.output_uri, "wt") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print("COMPLETE", args.output_uri, flush=True)


if __name__ == "__main__":
    main()
