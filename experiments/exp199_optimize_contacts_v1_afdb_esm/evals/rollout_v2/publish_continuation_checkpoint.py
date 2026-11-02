# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish exp199's final srcbase continuation checkpoints from GCS to HF."""

import argparse
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import gcsfs
from huggingface_hub import HfApi, hf_hub_download

HF_REPO = "open-athena/marinfold-exp199"
RUN_NAME = "prot-exp199-cv1-cont-s03-m1-p03-srcbase-aug100-us-east1"
STEP = 145_199
SOURCE_ROOT = (
    "gs://marin-us-east1/protein-structure/MarinFold/"
    "exp199_continue_contacts_v1/checkpoints/protein/"
    "prot-exp199-cv1-cont-s03-m1-p03-srcbase-aug100/2026.08.10.3"
)
SUBTREES = (f"checkpoints/step-{STEP}", f"hf/step-{STEP}")
EXPECTED_OBJECTS = 178
EXPECTED_BYTES = 23_545_272_337

README = f"""---
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

## Runs

| Run | Final <code>eval/tokenized/contacts-v1-val/loss</code> | Levanter checkpoints | Hugging Face export |
| --- | ---: | --- | --- |
| <code>prot-exp199-cv1-s01-m1-p03-aug-us-east1</code> | <code>3.0115306377</code> | [step-72599](https://huggingface.co/open-athena/marinfold-exp199/tree/main/prot-exp199-cv1-s01-m1-p03-aug-us-east1/checkpoints/step-72599) | [step-72599](https://huggingface.co/open-athena/marinfold-exp199/tree/main/prot-exp199-cv1-s01-m1-p03-aug-us-east1/hf/step-72599) |
| <code>prot-exp199-cv1-s01-m1-p03-base-us-east5</code> | <code>3.0074224472</code> | [step-72599](https://huggingface.co/open-athena/marinfold-exp199/tree/main/prot-exp199-cv1-s01-m1-p03-base-us-east5/checkpoints/step-72599) | [step-72599](https://huggingface.co/open-athena/marinfold-exp199/tree/main/prot-exp199-cv1-s01-m1-p03-base-us-east5/hf/step-72599) |
| <code>prot-exp199-cw-cv1-s02-m1-p06-aug</code> | <code>2.9712009430</code> | [step-116160](https://huggingface.co/open-athena/marinfold-exp199/tree/main/prot-exp199-cw-cv1-s02-m1-p06-aug/checkpoints/step-116160), [step-145199](https://huggingface.co/open-athena/marinfold-exp199/tree/main/prot-exp199-cw-cv1-s02-m1-p06-aug/checkpoints/step-145199) | [step-145199](https://huggingface.co/open-athena/marinfold-exp199/tree/main/prot-exp199-cw-cv1-s02-m1-p06-aug/hf/step-145199) |
| <code>{RUN_NAME}</code> | <code>2.9638261795</code> | [step-{STEP}](https://huggingface.co/open-athena/marinfold-exp199/tree/main/{RUN_NAME}/checkpoints/step-{STEP}) | [step-{STEP}](https://huggingface.co/open-athena/marinfold-exp199/tree/main/{RUN_NAME}/hf/step-{STEP}) |

## Formats

The <code>checkpoints/</code> paths contain the original Levanter OCDBT full
state, including AdamW state for continued training. The <code>hf/</code>
paths contain <code>Qwen3ForCausalLM</code> safetensors exports and the
2,845-token contacts-v1 tokenizer.

The two original <code>m1-p03</code> runs trained for 72,600 steps, making
<code>step-72599</code> their zero-indexed final checkpoint. The CoreWeave
<code>m1-p06</code> and continued <code>srcbase-aug100</code> runs trained
through <code>step-145199</code>. Only final checkpoint formats are retained
for the continuation run.
"""


def strip_gs(path: str) -> str:
    """Remove the GCS protocol from an absolute path."""

    if not path.startswith("gs://"):
        raise ValueError(f"expected a gs:// path, got {path!r}")
    return path.removeprefix("gs://").rstrip("/")


def required_paths() -> tuple[str, ...]:
    """Return the structural files required in a complete final checkpoint."""

    return (
        f"{RUN_NAME}/checkpoints/step-{STEP}/metadata.json",
        f"{RUN_NAME}/checkpoints/step-{STEP}/manifest.ocdbt",
        f"{RUN_NAME}/hf/step-{STEP}/config.json",
        f"{RUN_NAME}/hf/step-{STEP}/model.safetensors.index.json",
        f"{RUN_NAME}/hf/step-{STEP}/tokenizer.json",
    )


def source_inventory(
    filesystem: gcsfs.GCSFileSystem,
) -> dict[str, tuple[str, int]]:
    """Build and validate the exact final-checkpoint GCS inventory."""

    root = strip_gs(SOURCE_ROOT)
    inventory: dict[str, tuple[str, int]] = {}
    for subtree in SUBTREES:
        for path, info in filesystem.find(f"{root}/{subtree}", detail=True).items():
            if info.get("type") != "file":
                continue
            relative = path.removeprefix(f"{root}/")
            inventory[f"{RUN_NAME}/{relative}"] = (path, int(info["size"]))

    missing = [path for path in required_paths() if path not in inventory]
    if missing:
        raise RuntimeError(f"source checkpoint is incomplete: {missing}")
    total_bytes = sum(size for _, size in inventory.values())
    if len(inventory) != EXPECTED_OBJECTS or total_bytes != EXPECTED_BYTES:
        raise RuntimeError(
            "source checkpoint inventory changed: "
            f"objects={len(inventory)} bytes={total_bytes}"
        )
    return inventory


def hf_inventory(api: HfApi) -> tuple[str, dict[str, int]]:
    """Return the pinned current HF revision and its full file inventory."""

    revision = api.repo_info(HF_REPO, repo_type="model").sha
    tree = api.list_repo_tree(
        HF_REPO,
        repo_type="model",
        revision=revision,
        recursive=True,
        expand=True,
    )
    inventory = {
        item.path: int(item.size)
        for item in tree
        if getattr(item, "size", None) is not None
    }
    return revision, inventory


def readme_matches(revision: str, token: str) -> bool:
    """Return whether the pinned HF README already has the desired content."""

    path = hf_hub_download(
        HF_REPO,
        "README.md",
        repo_type="model",
        revision=revision,
        token=token,
    )
    return Path(path).read_text() == README


def download_files(
    *,
    filesystem: gcsfs.GCSFileSystem,
    inventory: dict[str, tuple[str, int]],
    paths: list[str],
    destination: Path,
    workers: int,
) -> None:
    """Download missing source objects concurrently onto regional scratch disk."""

    def download(path: str) -> tuple[str, int]:
        source, expected_size = inventory[path]
        local = destination / path
        local.parent.mkdir(parents=True, exist_ok=True)
        filesystem.get_file(source, str(local))
        actual_size = local.stat().st_size
        if actual_size != expected_size:
            raise OSError(f"short GCS download for {path}: {actual_size}")
        return path, actual_size

    completed = 0
    total = sum(inventory[path][1] for path in paths)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(download, path) for path in paths]
        for future in as_completed(futures):
            path, size = future.result()
            completed += size
            print(
                f"[publish] staged {path}: "
                f"{completed / 2**30:.2f}/{total / 2**30:.2f} GiB",
                flush=True,
            )


def publish(*, workers: int) -> None:
    """Upload missing final-checkpoint objects and verify the resulting HF tree."""

    token = os.environ.get("MARINFOLD_HF_RW_TOKEN", "").strip()
    if not token:
        raise SystemExit("MARINFOLD_HF_RW_TOKEN is required")
    api = HfApi(token=token)
    identity = api.whoami()
    token_role = (
        identity.get("auth", {}).get("accessToken", {}).get("role")
    )
    if token_role != "write":
        raise SystemExit(
            "MARINFOLD_HF_RW_TOKEN must be write-scoped; "
            f"observed role {token_role!r}"
        )
    print(
        f"[publish] authenticated as {identity.get('name')} with write role",
        flush=True,
    )
    filesystem = gcsfs.GCSFileSystem()
    expected = source_inventory(filesystem)
    before_revision, observed = hf_inventory(api)
    missing = sorted(
        path for path, (_, size) in expected.items() if observed.get(path) != size
    )
    update_readme = not readme_matches(before_revision, token)
    if not missing and not update_readme:
        print(
            f"[publish] NO-OP: {RUN_NAME} is complete at {before_revision}",
            flush=True,
        )
        return

    print(
        f"[publish] source={SOURCE_ROOT}; destination={HF_REPO}/{RUN_NAME}; "
        f"objects={len(missing)}/{len(expected)} "
        f"bytes={sum(expected[path][1] for path in missing)}",
        flush=True,
    )
    with tempfile.TemporaryDirectory(prefix="exp199-cont-hf-", dir=Path.cwd()) as temp:
        root = Path(temp)
        download_files(
            filesystem=filesystem,
            inventory=expected,
            paths=missing,
            destination=root,
            workers=workers,
        )
        if update_readme:
            (root / "README.md").write_text(README)
        commit = api.upload_folder(
            repo_id=HF_REPO,
            folder_path=root,
            repo_type="model",
            revision="main",
            commit_message=f"Add {RUN_NAME} final checkpoints",
        )

    after_revision, observed = hf_inventory(api)
    remaining = sorted(
        path for path, (_, size) in expected.items() if observed.get(path) != size
    )
    if remaining:
        raise RuntimeError(f"HF upload is incomplete: {remaining[:5]}")
    if not readme_matches(after_revision, token):
        raise RuntimeError("HF README did not update to the expected content")
    print(
        f"[publish] COMPLETE: revision={after_revision} objects={len(expected)} "
        f"bytes={sum(size for _, size in expected.values())} url={commit.commit_url}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    publish(workers=args.workers)


if __name__ == "__main__":
    main()
