# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish #199's CoreWeave cooldown export to the public MarinFold bucket — issue #238.

The winning checkpoint of [#234] (`prot-exp199-cw-cv1-p06-cool-s01`, step
290,400) exists in exactly one place: CoreWeave S3, in the account that ran the
continuation. Nothing outside that cluster can read it, so it cannot be a
`MODELS.yaml` entry until it is somewhere public.

WHY THIS RUNS ON A POD. The workstation has no credentials for
`marin-us-east-02a` at all (`~/.config/marin/cw-rno2a.env` is a different
CoreWeave region and 403s there), and even with them, moving 5.9 GiB through a
~2.5 MB/s uplink twice is an hour of wall clock against about two minutes
cloud-to-cloud. A `cw-us-east-02a` pod has the S3 credentials injected and sits
beside the bytes.

THREE THINGS ARE CHECKED BEFORE ANYTHING IS UPLOADED, because each is silent
when wrong:

* **The source is the checkpoint #234 evaluated**, not something that happens to
  live at that path now. `SOURCE_FILES` is #234's own recorded manifest
  (`data/cooldown_checkpoint_verification.json` in exp199), and every object
  must match it on size *and* S3 ETag.
* **The rope block must be repaired.** levanter's transformers-5 export states
  rope only as `rope_parameters`; 4.x ignores that and silently loads the
  architecture-default theta 10000 in place of the trained 500000 (#180, #184,
  #197). The repair runs the *checked-in* `marinfold/inference/_config.py`,
  shipped into the job by value (`EXP238_ROPE_REPAIR_B64`) with its sha256
  recorded, because marinfold itself is not installable here — it pins
  `transformers`, which pins `huggingface_hub<1`, and bucket writes need
  `huggingface_hub>=1.5`.
* **The tokenizer travels with the weights and its contact ids have not
  drifted.** A checkpoint published with the wrong tokenizer is unloadable;
  one published with a *shifted* contact vocab is worse, because it decodes to
  plausible nonsense. Checked by reading `tokenizer.json` directly rather than
  through `transformers`, for the dependency reason above.

`PROVENANCE.md` is deliberately NOT written here. It carries a per-checkpoint
measurement of what the rope defect costs *this* checkpoint on real documents,
which needs a forward pass; `measure_rope_cost.py` does that against the
published copy and uploads the file afterwards.

    uv run python publish_cooldown.py --submit    # from the workstation
    uv run python publish_cooldown.py             # on the pod
"""

import argparse
import base64
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]

RUN_NAME = "prot-exp199-cw-cv1-p06-cool-s01"
STEP = 290_400
SOURCE_URI = (
    "s3://marin-us-east-02a/marin/protein-structure/MarinFold/"
    "exp199_continue_contacts_v1_cw/checkpoints/protein/"
    f"{RUN_NAME}/2026.08.14.1/hf/step-{STEP}"
)
BUCKET_ID = "open-athena/MarinFold"
BUCKET_PATH = f"checkpoints/{RUN_NAME}/hf/step-{STEP}"

# #234's recorded manifest of the evaluated export: name -> (size, S3 ETag).
# Source: experiments/exp199_optimize_contacts_v1_afdb_esm/evals/rollout_v2/
# data/cooldown_checkpoint_verification.json (verified 2026-08-15T12:41:53Z).
SOURCE_FILES = {
    "config.json": (1557, "d8e904f8170ddf00d74c864f31d258a4"),
    "model-00001-of-00002.safetensors": (
        4_979_485_528, "c4685b3b45694c66418a6f1ff779af91-95"),
    "model-00002-of-00002.safetensors": (
        906_042_048, "3788cd21299125acfe3e2d04e91e84e0-18"),
    "model.safetensors.index.json": (20882, "bc0a5fd2c9aae096abae4caf9040c79c"),
    "tokenizer.json": (64407, "c4b3a16978e30eb150cca4fd8934b6ae"),
    "tokenizer_config.json": (290, "336f4e2ca951fa13a20cb1c4b68b2040"),
}

# The rope base this model line was trained with. 10000 is the Qwen3
# architecture default and the value a 4.x reader lands on when the repair is
# missing, so it is the one number worth asserting rather than merely printing.
EXPECTED_ROPE_THETA = 500_000

# Fully determined by marinfold's build_tokenizer. A drift here means the
# published weights and the published tokenizer disagree about what a contact
# statement is.
EXPECTED_TOKEN_IDS = {
    "<contacts-v1>": 2, "<contact>": 5, "<begin_statements>": 9, "<end>": 10,
    "<p0>": 143, "<p1999>": 2142,
}

# The repair module, shipped by value into the job.
ROPE_REPAIR_SOURCE = REPO_ROOT / "marinfold/marinfold/inference/_config.py"

# HF rate-limits the bucket write-token endpoint; exp139 saw sustained 429s.
_MAX_RETRIES = 8
_BASE_BACKOFF = 5.0

TARGET_CLUSTER = "cw-us-east-02a"
DEFAULT_IRIS = "/home/bizon/git/marin/.venv/bin/iris"


def log(message: str) -> None:
    print(f"[exp238] {message}", file=sys.stderr, flush=True)


def s3_filesystem():
    """An fsspec S3 handle for CoreWeave object storage.

    CoreWeave rejects path-style addressing, and the pod's injected credentials
    name the in-cluster endpoint through ``AWS_ENDPOINT_URL``.
    """
    import fsspec

    return fsspec.filesystem(
        "s3",
        endpoint_url=os.environ.get("AWS_ENDPOINT_URL"),
        config_kwargs={"s3": {"addressing_style": "virtual"}},
    )


def verify_source(filesystem) -> None:
    """Fail unless the S3 prefix is exactly the export #234 evaluated."""
    root = SOURCE_URI.removeprefix("s3://")
    listing = {
        Path(entry["Key"]).name: entry
        for entry in filesystem.ls(root, detail=True)
        if entry["type"] == "file"
    }
    if set(listing) != set(SOURCE_FILES):
        raise SystemExit(
            f"FATAL: {SOURCE_URI} holds {sorted(listing)}, "
            f"#234 recorded {sorted(SOURCE_FILES)}"
        )
    for name, (size, etag) in SOURCE_FILES.items():
        entry = listing[name]
        found = entry["ETag"].strip('"')
        if entry["size"] != size or found != etag:
            raise SystemExit(
                f"FATAL: {name} is {entry['size']} B / ETag {found}; #234 "
                f"evaluated {size} B / ETag {etag}. This is not that checkpoint."
            )
    log(f"verified {len(SOURCE_FILES)} objects against #234's manifest")


def rope_repair_module():
    """Import the checked-in rope repair, from the repo or from the job env.

    On the pod the repo is not present, so the module arrives base64-encoded in
    ``EXP238_ROPE_REPAIR_B64`` and its sha256 is checked against
    ``EXP238_ROPE_REPAIR_SHA256`` before exec — the point of shipping it by
    value is that the published config is repaired by code someone can read in
    git, not by a paraphrase of it.
    """
    import types

    encoded = os.environ.get("EXP238_ROPE_REPAIR_B64")
    if encoded:
        payload = base64.b64decode(encoded)
        digest = hashlib.sha256(payload).hexdigest()
        expected = os.environ["EXP238_ROPE_REPAIR_SHA256"]
        if digest != expected:
            raise SystemExit(f"FATAL: rope repair payload is {digest}, expected {expected}")
    else:
        payload = ROPE_REPAIR_SOURCE.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()

    module = types.ModuleType("exp238_rope_repair")
    exec(compile(payload, "marinfold/inference/_config.py", "exec"), module.__dict__)
    log(f"rope repair source sha256 {digest}")
    return module


def repair_config(local: Path) -> dict:
    """Restate the config's rope in transformers-4.x terms, in place."""
    repair = rope_repair_module()
    path = local / "config.json"
    raw = json.loads(path.read_text())
    if not repair.needs_rope_repair(raw):
        log("config.json already states rope in 4.x terms; leaving it alone")
        fixed = raw
    else:
        fixed = repair.repair_rope(raw)
        path.write_text(json.dumps(fixed, indent=2) + "\n")
        log(f"repaired config.json: rope_theta={fixed.get('rope_theta')} "
            f"rope_scaling={(fixed.get('rope_scaling') or {}).get('rope_type')}")
    if fixed.get("rope_theta") != EXPECTED_ROPE_THETA:
        raise SystemExit(
            f"FATAL: rope_theta is {fixed.get('rope_theta')!r}, expected "
            f"{EXPECTED_ROPE_THETA}. A 4.x reader would use the architecture "
            f"default and silently lose ~0.5 nats/token."
        )
    return fixed


def verify_tokenizer(local: Path) -> None:
    """Refuse to publish weights whose contact vocabulary has moved."""
    vocab = json.loads((local / "tokenizer.json").read_text())["model"]["vocab"]
    drift = {
        token: vocab.get(token)
        for token, want in EXPECTED_TOKEN_IDS.items()
        if vocab.get(token) != want
    }
    if drift:
        raise SystemExit(
            f"FATAL: contacts-v1 vocab drift {drift} (expected "
            f"{EXPECTED_TOKEN_IDS}). Publishing this would pair the weights "
            f"with a tokenizer that disagrees with them."
        )
    log(f"tokenizer verified: {len(vocab)} tokens, contact ids unmoved")


def stage(filesystem, local: Path) -> dict[str, str]:
    """Download the export and return each file's sha256."""
    root = SOURCE_URI.removeprefix("s3://")
    digests = {}
    for name in sorted(SOURCE_FILES):
        started = time.time()
        filesystem.get_file(f"{root}/{name}", str(local / name))
        digest = hashlib.sha256((local / name).read_bytes()).hexdigest()
        digests[name] = digest
        size = (local / name).stat().st_size
        log(f"staged {name} ({size / 2**20:.1f} MiB, {time.time() - started:.0f}s) "
            f"sha256 {digest}")
    return digests


def upload(local: Path, token: str) -> None:
    """Copy the staged directory into the public bucket, with backoff."""
    from huggingface_hub import HfFileSystem

    bucket = HfFileSystem(token=token)
    for name in sorted(SOURCE_FILES):
        destination = f"buckets/{BUCKET_ID}/{BUCKET_PATH}/{name}"
        source = local / name
        for attempt in range(_MAX_RETRIES):
            try:
                started = time.time()
                with open(source, "rb") as reader, bucket.open(destination, "wb") as writer:
                    while chunk := reader.read(32 << 20):
                        writer.write(chunk)
                log(f"put {name} ({source.stat().st_size / 2**20:.1f} MiB, "
                    f"{time.time() - started:.0f}s)")
                break
            except Exception as exc:  # noqa: BLE001 — HF 429s and transient 5xx
                if attempt == _MAX_RETRIES - 1:
                    raise
                delay = min(_BASE_BACKOFF * 2**attempt, 120.0)
                log(f"retry {name} {attempt + 1}/{_MAX_RETRIES} after {delay:.0f}s: {exc!r}")
                time.sleep(delay)


def run() -> int:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN with open-athena write scope is required")
    started = time.time()
    filesystem = s3_filesystem()
    verify_source(filesystem)
    with tempfile.TemporaryDirectory(prefix="exp238-") as tmp:
        local = Path(tmp)
        digests = stage(filesystem, local)
        source_config_sha256 = digests["config.json"]
        config = repair_config(local)
        verify_tokenizer(local)
        digests["config.json"] = hashlib.sha256((local / "config.json").read_bytes()).hexdigest()
        upload(local, token)
    manifest = {
        "run_name": RUN_NAME,
        "step": STEP,
        "source_uri": SOURCE_URI,
        "bucket_uri": f"hf://buckets/{BUCKET_ID}/{BUCKET_PATH}",
        "sha256": digests,
        "source_config_sha256": source_config_sha256,
        "rope_theta": config.get("rope_theta"),
        "vocab_size": config.get("vocab_size"),
    }
    log(f"DONE in {(time.time() - started) / 60:.1f} min")
    # Printed on stdout so the checked-in manifest can be lifted from the job log.
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


def hf_token() -> str:
    """The workstation's open-athena-scoped token.

    The active token writes the bucket; ``write2`` is the one that can also
    create repos. Read from the stored-token file so a submit is reproducible
    rather than depending on which token happens to be active.
    """
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    import configparser

    path = Path.home() / ".cache/huggingface/stored_tokens"
    parser = configparser.ConfigParser()
    parser.read(path)
    for name in ("write2", "DEFAULT"):
        if parser.has_option(name, "hf_token"):
            return parser.get(name, "hf_token")
    raise SystemExit(f"no HF_TOKEN and no usable entry in {path}")


def submit(iris_bin: str, dry_run: bool) -> int:
    """Dispatch this script to a CoreWeave pod through Iris federation.

    The workspace bundle is built from ``git ls-files`` of this directory, so
    uncommitted edits are not uploaded — the job would run HEAD and look like it
    worked.
    """
    dirty = [
        line for line in subprocess.run(
            ["git", "status", "--porcelain", "--", "."],
            cwd=HERE, capture_output=True, text=True, check=True,
        ).stdout.splitlines() if not line.startswith("??")
    ]
    if dirty:
        raise SystemExit("refusing to submit with uncommitted changes here:\n  "
                         + "\n  ".join(dirty))

    payload = ROPE_REPAIR_SOURCE.read_bytes()
    argv = [
        iris_bin, "--cluster=marin", "job", "run",
        "--target-cluster", TARGET_CLUSTER,
        "--job-name", f"exp238-publish-cooldown-step{STEP}",
        "--priority", "batch", "--enable-extra-resources", "--no-wait",
        "--cpu", "4", "--memory", "16GB", "--disk", "32GB",
        "--max-retries", "3", "--timeout", "7200",
        "-e", "HF_TOKEN", hf_token(),
        "-e", "EXP238_ROPE_REPAIR_B64", base64.b64encode(payload).decode(),
        "-e", "EXP238_ROPE_REPAIR_SHA256", hashlib.sha256(payload).hexdigest(),
        "--", "python", "publish_cooldown.py",
    ]
    if dry_run:
        # Redact rather than truncate: the token is short enough to survive a
        # length filter, and a dry run is exactly when someone pastes the output
        # into an issue.
        secrets = {hf_token(), base64.b64encode(payload).decode()}
        log("DRY RUN " + " ".join("<redacted>" if a in secrets else a for a in argv))
        return 0
    subprocess.run(argv, cwd=HERE, check=True)
    log(f"submitted; logs: {iris_bin} --cluster=marin job logs "
        f"/bizon/exp238-publish-cooldown-step{STEP}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", action="store_true",
                        help="run on a CoreWeave pod instead of here")
    parser.add_argument("--iris-bin", default=os.environ.get("IRIS_BIN", DEFAULT_IRIS))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.submit:
        return submit(args.iris_bin, args.dry_run)
    return run()


if __name__ == "__main__":
    sys.exit(main())
