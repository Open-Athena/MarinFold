# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish and regionally restore exp199 continuation seed checkpoints.

``upload-hf`` is an idempotent one-time GCS -> Hugging Face transfer. It exits
without reading checkpoint payloads from GCS when every expected HF object is
already present at the correct size.

``restore-region`` downloads through hf-xet and concurrently uploads completed
files into the selected region's GCS prefix. It never reads another region's
GCS bucket and only transfers missing or wrong-sized destination objects.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

HF_REPO = "open-athena/marinfold-exp199"
SOURCE_STEP = 72_599
SOURCE_SUBTREES = (
    f"checkpoints/step-{SOURCE_STEP}",
    f"hf/step-{SOURCE_STEP}",
)

DESTINATION_NAMESPACE = "checkpoints/protein/exp199-continuation-init"
DESTINATION_VERSION = "2026.08.09.1"
SUPPORTED_REGIONS = ("europe-west4", "us-east1", "us-east5", "us-west4")


@dataclass(frozen=True)
class SourceSpec:
    key: str
    run_id: str
    source_region: str
    source_gcs_base: str
    final_eval_loss: float


SOURCES = {
    "aug": SourceSpec(
        key="aug",
        run_id="prot-exp199-cv1-s01-m1-p03-aug-us-east1",
        source_region="us-east1",
        source_gcs_base=(
            "gs://marin-us-east1/protein-structure/MarinFold/"
            "exp199_optimize_contacts_v1_afdb_esm/checkpoints/protein/"
            "prot-exp199-cv1-s01-m1-p03-aug/2026.08.07.1"
        ),
        final_eval_loss=3.011530637741089,
    ),
    "base": SourceSpec(
        key="base",
        run_id="prot-exp199-cv1-s01-m1-p03-base-us-east5",
        source_region="us-east5",
        source_gcs_base=(
            "gs://marin-us-east5/protein-structure/MarinFold/"
            "exp199_optimize_contacts_v1_afdb_esm/checkpoints/protein/"
            "prot-exp199-cv1-s01-m1-p03-base/2026.08.07.1"
        ),
        final_eval_loss=3.00742244720459,
    ),
}


def _readme() -> str:
    entries = []
    for source in SOURCES.values():
        entries.append(
            f"- `{source.run_id}`: final validation loss "
            f"`{source.final_eval_loss:.10f}`"
        )
    source_entries = "\n".join(entries)
    return f"""---
library_name: transformers
tags:
  - biology
  - protein-language-model
  - marinfold
---

# MarinFold exp199 checkpoints

Checkpoint artifacts from the MarinFold contacts-v1 AFDB/ESM optimization
experiment tracked in
[Open-Athena/MarinFold#199](https://github.com/Open-Athena/MarinFold/issues/199).

The following completed m1-p03 source runs are retained:

{source_entries}

Each run contains both final checkpoint formats:

- `hf/step-{SOURCE_STEP}` is a `Qwen3ForCausalLM` Hugging Face safetensors
  export with the 2,845-token contacts-v1 tokenizer colocated with the weights.
- `checkpoints/step-{SOURCE_STEP}` is the original Levanter OCDBT full-state
  checkpoint, including the AdamW state required for training continuation.

The training target was 72,600 steps; Levanter's zero-indexed final checkpoint
is `step-{SOURCE_STEP}`.
"""


README = _readme()


def _strip_gs(path: str) -> str:
    if not path.startswith("gs://"):
        raise ValueError(f"expected gs:// path, got {path!r}")
    return path.removeprefix("gs://").rstrip("/")


def _required_paths(source: SourceSpec) -> tuple[str, ...]:
    root = f"{source.run_id}"
    return (
        f"{root}/checkpoints/step-{SOURCE_STEP}/metadata.json",
        f"{root}/checkpoints/step-{SOURCE_STEP}/manifest.ocdbt",
        f"{root}/hf/step-{SOURCE_STEP}/config.json",
        f"{root}/hf/step-{SOURCE_STEP}/model.safetensors.index.json",
        f"{root}/hf/step-{SOURCE_STEP}/tokenizer.json",
    )


def _validate_inventory(
    inventory: dict[str, tuple[str, int]] | dict[str, int],
    source: SourceSpec,
) -> None:
    missing = [path for path in _required_paths(source) if path not in inventory]
    if missing:
        raise RuntimeError(f"checkpoint inventory is incomplete: {missing}")


def _source_inventory(gcs, source: SourceSpec) -> dict[str, tuple[str, int]]:
    root = _strip_gs(source.source_gcs_base)
    inventory: dict[str, tuple[str, int]] = {}
    for subtree in SOURCE_SUBTREES:
        for path, info in gcs.find(f"{root}/{subtree}", detail=True).items():
            if info.get("type") != "file":
                continue
            relative = path.removeprefix(f"{root}/")
            inventory[f"{source.run_id}/{relative}"] = (path, int(info["size"]))
    _validate_inventory(inventory, source)
    return inventory


def _hf_inventory(api) -> dict[str, int]:
    from huggingface_hub.errors import RepositoryNotFoundError

    try:
        tree = api.list_repo_tree(
            HF_REPO,
            repo_type="model",
            recursive=True,
            expand=True,
        )
        return {
            item.path: int(item.size)
            for item in tree
            if getattr(item, "size", None) is not None
        }
    except RepositoryNotFoundError:
        return {}


def _missing_or_wrong_size(
    expected: dict[str, tuple[str, int]],
    observed: dict[str, int],
) -> list[str]:
    return sorted(
        path for path, (_, size) in expected.items() if observed.get(path) != size
    )


def _download_gcs_files(
    gcs,
    expected: dict[str, tuple[str, int]],
    paths: list[str],
    root: Path,
    workers: int,
) -> None:
    def download(path: str) -> tuple[str, int]:
        source, size = expected[path]
        destination = root / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        gcs.get_file(source, str(destination))
        actual = destination.stat().st_size
        if actual != size:
            raise OSError(f"short GCS download for {path}: {actual} != {size}")
        return path, size

    completed = 0
    total = sum(expected[path][1] for path in paths)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(download, path) for path in paths]
        for future in as_completed(futures):
            path, size = future.result()
            completed += size
            print(
                f"[upload-hf] staged {path} "
                f"({completed / 2**30:.2f}/{total / 2**30:.2f} GiB)",
                flush=True,
            )


def upload_hf(*, source: SourceSpec, workers: int) -> None:
    import gcsfs
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise SystemExit("upload-hf requires HF_TOKEN")

    api = HfApi(token=token)
    gcs = gcsfs.GCSFileSystem()
    expected = _source_inventory(gcs, source)
    observed = _hf_inventory(api)
    missing = _missing_or_wrong_size(expected, observed)
    readme_matches = observed.get("README.md") == len(README.encode())

    if not missing and readme_matches:
        print(
            f"[upload-hf] NO-OP: {HF_REPO} already has all "
            f"{len(expected)} {source.key} checkpoint objects at expected sizes",
            flush=True,
        )
        return

    api.create_repo(HF_REPO, repo_type="model", private=False, exist_ok=True)
    print(
        f"[upload-hf] source={source.run_id} region={source.source_region}; "
        f"{len(missing)}/{len(expected)} objects require upload "
        f"({sum(expected[path][1] for path in missing) / 2**30:.2f} GiB)",
        flush=True,
    )
    with tempfile.TemporaryDirectory(prefix="exp199-hf-", dir=Path.cwd()) as temp:
        root = Path(temp)
        if missing:
            _download_gcs_files(gcs, expected, missing, root, workers)
        if not readme_matches:
            (root / "README.md").write_text(README)

        # huggingface_hub's hf-xet upload pipeline performs a single read pass,
        # handles large folders as resumable multi-commits, and deduplicates
        # objects already committed by a prior attempt.
        api.upload_folder(
            repo_id=HF_REPO,
            folder_path=root,
            repo_type="model",
            revision="main",
            commit_message=f"Add {source.run_id} final checkpoints",
        )

    observed = _hf_inventory(api)
    remaining = _missing_or_wrong_size(expected, observed)
    if remaining:
        raise RuntimeError(
            f"HF upload incomplete for {len(remaining)} objects: {remaining[:5]}"
        )
    print(
        f"[upload-hf] COMPLETE: {len(expected)} objects, "
        f"{sum(size for _, size in expected.values()) / 2**30:.2f} GiB -> {HF_REPO}",
        flush=True,
    )


def _destination(region: str, source: SourceSpec) -> str:
    from marin.rl.placement import marin_prefix_for_region
    from rigging.filesystem import prefix_join

    return prefix_join(
        marin_prefix_for_region(region),
        f"{DESTINATION_NAMESPACE}/{source.run_id}/{DESTINATION_VERSION}",
    )


def _remote_checkpoint_inventory(api, source: SourceSpec) -> dict[str, int]:
    observed = _hf_inventory(api)
    prefixes = (
        f"{source.run_id}/checkpoints/step-{SOURCE_STEP}/",
        f"{source.run_id}/hf/step-{SOURCE_STEP}/",
    )
    checkpoint = {
        path: size for path, size in observed.items() if path.startswith(prefixes)
    }
    _validate_inventory(checkpoint, source)
    return checkpoint


def _gcs_destination_inventory(gcs, destination: str) -> dict[str, int]:
    root = _strip_gs(destination)
    if not gcs.exists(root):
        return {}
    return {
        path.removeprefix(f"{root}/"): int(info["size"])
        for path, info in gcs.find(root, detail=True).items()
        if info.get("type") == "file"
    }


def restore_region(*, source: SourceSpec, region: str, workers: int) -> None:
    import gcsfs
    from huggingface_hub import HfApi, hf_hub_download

    if region not in SUPPORTED_REGIONS:
        raise SystemExit(f"region must be one of: {', '.join(SUPPORTED_REGIONS)}")

    api = HfApi()
    gcs = gcsfs.GCSFileSystem()
    remote = _remote_checkpoint_inventory(api, source)
    destination = _destination(region, source)
    destination_root = _strip_gs(destination)
    observed = _gcs_destination_inventory(gcs, destination)
    expected = {
        path.removeprefix(f"{source.run_id}/"): size for path, size in remote.items()
    }
    missing = sorted(
        path for path, size in expected.items() if observed.get(path) != size
    )
    if not missing:
        print(
            f"[restore-region] NO-OP: {region} already has all {len(expected)} "
            f"{source.key} objects at {destination}",
            flush=True,
        )
        return

    print(
        f"[restore-region] {region}: pipelining {len(missing)}/{len(expected)} "
        f"objects ({sum(expected[path] for path in missing) / 2**30:.2f} GiB) "
        f"from {HF_REPO} to {destination}",
        flush=True,
    )

    with tempfile.TemporaryDirectory(prefix="exp199-restore-", dir=Path.cwd()) as temp:
        root = Path(temp)

        def copy(path: str) -> tuple[str, int]:
            expected_size = expected[path]
            hf_path = f"{source.run_id}/{path}"
            target = f"{destination_root}/{path}"
            for attempt in range(1, 4):
                try:
                    local = Path(
                        hf_hub_download(
                            repo_id=HF_REPO,
                            filename=hf_path,
                            repo_type="model",
                            local_dir=root,
                            token=False,
                        )
                    )
                    actual = local.stat().st_size
                    if actual != expected_size:
                        raise OSError(
                            f"short HF download for {path}: {actual} != {expected_size}"
                        )
                    gcs.put_file(
                        str(local),
                        target,
                        chunksize=64 * 2**20,
                    )
                    actual = int(gcs.info(target)["size"])
                    if actual != expected_size:
                        raise OSError(
                            f"short regional upload for {path}: {actual} != {expected_size}"
                        )
                    local.unlink(missing_ok=True)
                    return path, actual
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(2**attempt)
            raise AssertionError("unreachable")

        completed = 0
        total = sum(expected[path] for path in missing)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(copy, path) for path in missing]
            for future in as_completed(futures):
                path, size = future.result()
                completed += size
                print(
                    f"[restore-region] {region}: {path} "
                    f"({completed / 2**30:.2f}/{total / 2**30:.2f} GiB)",
                    flush=True,
                )

    observed = _gcs_destination_inventory(gcs, destination)
    remaining = sorted(
        path for path, size in expected.items() if observed.get(path) != size
    )
    if remaining:
        raise RuntimeError(
            f"regional restore incomplete for {len(remaining)} objects: {remaining[:5]}"
        )

    marker = {
        "source": HF_REPO,
        "source_run": source.run_id,
        "source_step": SOURCE_STEP,
        "objects": len(expected),
        "bytes": sum(expected.values()),
    }
    gcs.pipe(
        f"{destination_root}/_HF_RESTORE_COMPLETE.json",
        json.dumps(marker, indent=2, sort_keys=True).encode(),
    )
    print(
        f"[restore-region] COMPLETE: {region}, {len(expected)} objects -> {destination}",
        flush=True,
    )


def show_plan(*, source: SourceSpec) -> None:
    plan = {
        "hf_repo": HF_REPO,
        "source": source.key,
        "source_gcs_base": source.source_gcs_base,
        "source_region": source.source_region,
        "source_run": source.run_id,
        "source_step": SOURCE_STEP,
        "regional_destinations": {
            region: _destination(region, source) for region in SUPPORTED_REGIONS
        },
    }
    print(json.dumps(plan, indent=2, sort_keys=True))


def _source(value: str) -> SourceSpec:
    return SOURCES[value]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan")
    plan.add_argument("--source", choices=SOURCES, required=True)

    upload = subparsers.add_parser("upload-hf")
    upload.add_argument("--source", choices=SOURCES, required=True)
    upload.add_argument("--workers", type=int, default=16)

    restore = subparsers.add_parser("restore-region")
    restore.add_argument("--source", choices=SOURCES, required=True)
    restore.add_argument("--region", choices=SUPPORTED_REGIONS, required=True)
    restore.add_argument("--workers", type=int, default=16)

    args = parser.parse_args()
    if getattr(args, "workers", 1) < 1:
        raise SystemExit("--workers must be positive")

    # Keep hf-xet's cache on the Iris workspace disk and allow it to saturate
    # the remote VM's network. These must be set before importing the Hub.
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    os.environ.setdefault("HF_HOME", str(Path.cwd() / ".exp199-hf-cache"))
    os.environ.setdefault("HF_XET_CACHE", str(Path.cwd() / ".exp199-hf-cache" / "xet"))

    source = _source(args.source)
    if args.command == "plan":
        show_plan(source=source)
    elif args.command == "upload-hf":
        upload_hf(source=source, workers=args.workers)
    else:
        restore_region(source=source, region=args.region, workers=args.workers)


if __name__ == "__main__":
    main()
