"""Independently verify regional ingress metadata and restored scalar state."""

import json
import math
from datetime import UTC, datetime
from pathlib import Path

import fsspec
import tensorstore as ts
from levanter.tensorstore_serialization import build_kvstore_spec

from experiments.exp364_models_mpnn_only_lr_sweep.stage_tpu_inputs import (
    PLAN_PATH,
    check_source,
)
from experiments.exp364_models_mpnn_only_lr_sweep.tpu_cooldown import (
    SOURCE_ROOT,
    SOURCE_STEP,
    validate_inputs,
)


def main() -> None:
    """Check all object checksums without downloading the corpus again."""
    plan = json.loads(PLAN_PATH.read_text())
    validate_inputs()
    fs = fsspec.filesystem("gcs", cache_timeout=0)
    verified = []
    for artifact in plan["artifacts"]:
        root = fs._strip_protocol(artifact["destination"]).rstrip("/") + "/"
        objects = []
        for path, info in sorted(fs.find(artifact["destination"], detail=True).items()):
            if info["type"] != "file":
                continue
            relative = path.removeprefix(root)
            metadata = info.get("metadata", {})
            if (
                not info.get("md5Hash")
                or metadata.get("verified_md5") != info["md5Hash"]
                or metadata.get("source_uri") != f"{artifact['source']}/{relative}"
            ):
                raise ValueError(f"Unverified or altered destination object: {path}")
            objects.append(
                {
                    "path": relative,
                    "size": info["size"],
                    "etag": metadata["source_etag"],
                }
            )
        check_source(objects, artifact)
        verified.append(artifact)
    checkpoint = f"{SOURCE_ROOT}/checkpoints/step-{SOURCE_STEP}/"
    scalars = {}
    expected = {
        "step": SOURCE_STEP + 1,
        "opt_state/count": SOURCE_STEP + 1,
        "opt_state/hyperparams/learning_rate": 5e-4,
        "opt_state/hyperparams_states/learning_rate/count": SOURCE_STEP + 1,
        "opt_state/inner_state/1/count": SOURCE_STEP + 1,
    }
    for key, value in expected.items():
        array = (
            ts.open(
                {
                    "driver": "zarr3",
                    "kvstore": {
                        "driver": "ocdbt",
                        "base": build_kvstore_spec(checkpoint),
                        "path": key,
                    },
                }
            )
            .result()
            .read()
            .result()
        )
        scalars[key] = array.item()
        if not math.isclose(scalars[key], value, rel_tol=1e-6):
            raise ValueError(f"Source scalar {key} changed: {scalars[key]} != {value}")
    result = {
        "verified_at_utc": datetime.now(UTC).isoformat(),
        "artifacts": verified,
        "total_bytes": sum(artifact["bytes"] for artifact in verified),
        "objects": sum(artifact["objects"] for artifact in verified),
        "checkpoint_scalars": scalars,
        "validation": plan["existing_validation"],
    }
    output = Path(__file__).with_name("data") / "tpu_input_verification.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
