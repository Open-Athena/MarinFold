# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish #232's best decontaminated checkpoint to the public MarinFold bucket — issue #250.

Two exports qualify, and `--checkpoint` chooses:

* `training` (the default) — `prot-exp232-trc-cv1-decontam-train-s01-m2-p06-srcpeak-augcont-lr005-us-east1`
  step 363,000, the `m2-p06` point trained on past the sweep. #232's
  2026-08-24 evaluation scores it at 0.6051 R-precision on the legacy 554 and
  0.5517 on eval-val, against the sweep final's 0.5916 / 0.5203.
* `sweep` — `prot-exp232-cw-cv1-decontam-s02-m2-p06-aug` step 145,199, the
  better of the two finals [#244] selected and [#245] scored. Published first,
  and still what the figures used before the training checkpoint existed.

Both are trained from scratch on [#225]'s corpora with every FoldBench
protein's homologs removed at the 30 % / 50 %-coverage rule, so their accuracy
on these proteins is not open to a leakage objection — and each existed in
exactly one place, CoreWeave S3 in the account that trained it.

This copies one to the bucket so it can be a `MODELS.yaml` entry.

Adapted from `experiments/exp238_models_promote_exp199_cooldown_to_default/publish_cooldown.py`,
which did the same for #199's cooldown; the mechanism and its three pre-upload
checks are that script's, not new here.

WHY THIS RUNS ON A POD. The workstation has no credentials for
`marin-us-east-02a` (`aws s3 ls` on the source returns InvalidAccessKeyId), and
moving 5.9 GiB through a ~2.5 MB/s uplink twice is hours against about two
minutes cloud-to-cloud. A `cw-us-east-02a` pod has the S3 credentials injected
and sits beside the bytes.

THREE THINGS ARE CHECKED BEFORE ANYTHING IS UPLOADED, each silent when wrong:

* **The source is the checkpoint that was evaluated**, not whatever lives at
  that path now. Each `Checkpoint.files` is the manifest the evaluation named in
  its `pinned_by` recorded; every object must match on size *and* S3 ETag.
  `test_publish_specs.py` asserts both copies against those specs.
* **The rope block must be repaired.** levanter's transformers-5 export states
  rope only as `rope_parameters`; 4.x ignores that and silently loads the Qwen3
  architecture default theta 10000 where the model was trained with 500000
  (#180, #184, #197). The repair runs the *checked-in*
  `marinfold/inference/_config.py`, shipped into the job by value with its
  sha256 recorded, because marinfold is not installable here — it pins
  `transformers`, which pins `huggingface_hub<1`, and bucket writes need
  `huggingface_hub>=1.5`.
* **The tokenizer travels with the weights and its contact ids have not
  drifted.** Publishing weights with a *shifted* contact vocabulary is worse
  than publishing none: it decodes to plausible nonsense. Checked by reading
  `tokenizer.json` directly rather than through `transformers`, for the
  dependency reason above.

    uv run python publish_exp232_m2_p06.py --submit    # from the workstation
    uv run python publish_exp232_m2_p06.py             # on the pod
    uv run python publish_exp232_m2_p06.py --verify    # after it lands
"""

import argparse
import base64
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
# On the pod this file lands at /app/publish_exp232_m2_p06.py with no repo above
# it, so the repo root is resolved lazily and only on the paths that need it.
REPO_ROOT = HERE.parents[1] if len(HERE.parents) > 1 else None

@dataclasses.dataclass(frozen=True)
class Checkpoint:
    """One publishable export: its identity, its source, and its pinned manifest."""

    key: str
    run_name: str
    step: int
    source_uri: str
    #: name -> (size, S3 ETag)
    files: dict
    #: where that manifest was pinned, named in every failure message
    pinned_by: str
    note: str


#: The two #232 finals worth publishing. Both are the `m2-p06` point; they differ in
#: how far training ran. `test_publish_specs.py` asserts each manifest still equals
#: the pinned identity of the evaluation that produced the number we quote.
CHECKPOINTS = {
    "sweep": Checkpoint(
        key="sweep",
        run_name="prot-exp232-cw-cv1-decontam-s02-m2-p06-aug",
        step=145_199,
        source_uri=(
            "s3://marin-us-east-02a/MarinFold/exp232_sweep_cv1_decontam/checkpoints/"
            "protein/prot-exp232-cw-cv1-decontam-s02-m2-p06-aug/2026.08.14.2/hf/step-145199"
        ),
        files={
            "config.json": (1_557, "d8e904f8170ddf00d74c864f31d258a4"),
            "model-00001-of-00002.safetensors": (
                4_979_485_528, "2e38a75033f4df3a73a4be9bc2ceeefe-95"),
            "model-00002-of-00002.safetensors": (
                906_042_048, "f444bd62152329ef71c7c46e7ee1c3cd-18"),
            "model.safetensors.index.json": (20_882, "bc0a5fd2c9aae096abae4caf9040c79c"),
            "tokenizer.json": (64_407, "c4b3a16978e30eb150cca4fd8934b6ae"),
            "tokenizer_config.json": (290, "336f4e2ca951fa13a20cb1c4b68b2040"),
        },
        pinned_by="#245's rollout/checkpoint_specs.py",
        note="the sweep final: legacy-554 R-precision 0.5916",
    ),
    "training": Checkpoint(
        key="training",
        run_name=(
            "prot-exp232-trc-cv1-decontam-train-s01-m2-p06-srcpeak-augcont-lr005-us-east1"
        ),
        step=363_000,
        source_uri=(
            "s3://marin-us-east-02a/marin/protein-structure/MarinFold/"
            "exp232_sweep_cv1_decontam/evals/rollout-v2/2026-08-24/v2-01/models/"
            "exp232-decontam-train-m2-p06-step363000/hf/step-363000"
        ),
        files={
            "config.json": (1_557, "d8e904f8170ddf00d74c864f31d258a4"),
            "model-00001-of-00002.safetensors": (
                4_979_485_528, "9a11736b507565aa2be00a5753f51b12-95"),
            "model-00002-of-00002.safetensors": (
                906_042_048, "1a042bf9b4acde490f0c0ffee76306dd-18"),
            "model.safetensors.index.json": (20_882, "bc0a5fd2c9aae096abae4caf9040c79c"),
            "tokenizer.json": (64_407, "c4b3a16978e30eb150cca4fd8934b6ae"),
            "tokenizer_config.json": (290, "336f4e2ca951fa13a20cb1c4b68b2040"),
        },
        pinned_by="#232's evals/2026-08-24_rollout_v2/checkpoint_specs.py",
        note="training continued to step 363,000: legacy-554 R-precision 0.6051",
    ),
}
#: The better of the two, and the one every #250 figure is drawn from.
DEFAULT_CHECKPOINT = "training"
#: Set by `main`; every function below reads this rather than a bare constant.
SPEC = CHECKPOINTS[DEFAULT_CHECKPOINT]

BUCKET_ID = "open-athena/MarinFold"


def bucket_path(spec: "Checkpoint" = None) -> str:
    """Where a checkpoint lands on the bucket — the repo-wide `<run>/hf/step-<N>` layout."""
    spec = spec or SPEC
    return f"checkpoints/{spec.run_name}/hf/step-{spec.step}"


def bucket_resolve(spec: "Checkpoint" = None) -> str:
    return f"https://huggingface.co/buckets/{BUCKET_ID}/resolve/{bucket_path(spec)}"


# The rope base this model line was trained with. 10000 is the Qwen3 architecture
# default and the value a 4.x reader lands on when the repair is missing, so it is
# the one number worth asserting rather than merely printing.
EXPECTED_ROPE_THETA = 500_000

# Fully determined by marinfold's build_tokenizer. Drift here means the published
# weights and the published tokenizer disagree about what a contact statement is.
EXPECTED_TOKEN_IDS = {
    "<contacts-v1>": 2, "<contact>": 5, "<begin_statements>": 9, "<end>": 10,
    "<p0>": 143, "<p1999>": 2142,
}

# The repair module, shipped by value into the job. Absent on the pod, where it
# arrives through EXP250_ROPE_REPAIR_B64 instead.
ROPE_REPAIR_SOURCE = (
    REPO_ROOT / "marinfold/marinfold/inference/_config.py" if REPO_ROOT else None
)

# HF rate-limits the bucket write-token endpoint; exp139 saw sustained 429s.
_MAX_RETRIES = 8
_BASE_BACKOFF = 5.0

TARGET_CLUSTER = "cw-us-east-02a"
# NOT /home/bizon/git/marin — that checkout is old enough for the controller's
# 14-day client-freshness gate to reject it, and its CLI predates
# `--target-cluster` entirely.
DEFAULT_IRIS = "/home/bizon/git/marin-freshiris/.venv/bin/iris"


def log(message: str) -> None:
    print(f"[exp250] {message}", file=sys.stderr, flush=True)


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
    """Fail unless the S3 prefix is exactly the export #244/#245 evaluated."""
    root = SPEC.source_uri.removeprefix("s3://")
    listing = {
        Path(entry["Key"]).name: entry
        for entry in filesystem.ls(root, detail=True)
        if entry["type"] == "file"
    }
    if set(listing) != set(SPEC.files):
        raise SystemExit(
            f"FATAL: {SPEC.source_uri} holds {sorted(listing)}, "
            f"{SPEC.pinned_by} recorded {sorted(SPEC.files)}"
        )
    for name, (size, etag) in SPEC.files.items():
        entry = listing[name]
        found = entry["ETag"].strip('"')
        if entry["size"] != size or found != etag:
            raise SystemExit(
                f"FATAL: {name} is {entry['size']} B / ETag {found}; {SPEC.pinned_by} "
                f"pinned {size} B / ETag {etag}. This is not that checkpoint."
            )
    log(f"verified {len(SPEC.files)} objects against {SPEC.pinned_by}")


def rope_repair_module():
    """Import the checked-in rope repair, from the repo or from the job env.

    On the pod the repo is not present, so the module arrives base64-encoded in
    ``EXP250_ROPE_REPAIR_B64`` and its sha256 is checked against
    ``EXP250_ROPE_REPAIR_SHA256`` before exec — the point of shipping it by value
    is that the published config is repaired by code someone can read in git.
    """
    import types

    encoded = os.environ.get("EXP250_ROPE_REPAIR_B64")
    if encoded:
        payload = base64.b64decode(encoded)
        digest = hashlib.sha256(payload).hexdigest()
        expected = os.environ["EXP250_ROPE_REPAIR_SHA256"]
        if digest != expected:
            raise SystemExit(f"FATAL: rope repair payload is {digest}, expected {expected}")
    else:
        payload = ROPE_REPAIR_SOURCE.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()

    module = types.ModuleType("exp250_rope_repair")
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
            f"default and silently lose accuracy on every document."
        )
    return fixed


def verify_tokenizer_bytes(payload: bytes) -> int:
    """Refuse to publish weights whose contact vocabulary has moved."""
    vocab = json.loads(payload)["model"]["vocab"]
    drift = {
        token: vocab.get(token)
        for token, want in EXPECTED_TOKEN_IDS.items()
        if vocab.get(token) != want
    }
    if drift:
        raise SystemExit(
            f"FATAL: contacts-v1 vocab drift {drift} (expected {EXPECTED_TOKEN_IDS}). "
            f"Publishing this would pair the weights with a tokenizer that "
            f"disagrees with them."
        )
    return len(vocab)


def stage(filesystem, local: Path) -> dict[str, str]:
    """Download the export and return each file's sha256."""
    root = SPEC.source_uri.removeprefix("s3://")
    digests = {}
    for name in sorted(SPEC.files):
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
    for name in sorted(SPEC.files):
        destination = f"buckets/{BUCKET_ID}/{bucket_path()}/{name}"
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
    with tempfile.TemporaryDirectory(prefix="exp250-") as tmp:
        local = Path(tmp)
        digests = stage(filesystem, local)
        source_config_sha256 = digests["config.json"]
        config = repair_config(local)
        vocab_size = verify_tokenizer_bytes((local / "tokenizer.json").read_bytes())
        log(f"tokenizer verified: {vocab_size} tokens, contact ids unmoved")
        digests["config.json"] = hashlib.sha256((local / "config.json").read_bytes()).hexdigest()
        upload(local, token)
    manifest = {
        "checkpoint": SPEC.key,
        "run_name": SPEC.run_name,
        "step": SPEC.step,
        "source_uri": SPEC.source_uri,
        "bucket_uri": f"hf://buckets/{BUCKET_ID}/{bucket_path()}",
        "sha256": digests,
        "source_config_sha256": source_config_sha256,
        "rope_theta": config.get("rope_theta"),
        "vocab_size": config.get("vocab_size"),
    }
    log(f"DONE in {(time.time() - started) / 60:.1f} min")
    # Printed on stdout so the checked-in manifest can be lifted from the job log.
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


def verify_published(out: Path | None) -> int:
    """Check the bucket copy from anywhere: sizes, the rope repair, the vocabulary.

    Reads the two small files in full and takes the weight shards by size alone —
    re-hashing 5.9 GiB over the workstation's downlink to learn what the job
    already recorded is not worth the wall clock.
    """
    from huggingface_hub import list_bucket_tree

    entries = {
        Path(entry.path).name: entry
        for entry in list_bucket_tree(BUCKET_ID, prefix=bucket_path(), recursive=True, token=False)
    }
    problems = []
    for name, (size, _) in SPEC.files.items():
        entry = entries.get(name)
        if entry is None:
            problems.append(f"{name}: missing from the bucket")
        elif name != "config.json" and entry.size != size:
            # config.json is the one file this publish rewrites, so its size moves.
            problems.append(f"{name}: {entry.size} B on the bucket, {size} B at the source")
    if problems:
        for problem in problems:
            log(f"FAIL {problem}")
        return 1

    config = json.loads(urllib.request.urlopen(f"{bucket_resolve()}/config.json").read())
    if config.get("rope_theta") != EXPECTED_ROPE_THETA:
        log(f"FAIL config.json states rope_theta={config.get('rope_theta')!r}")
        return 1
    vocab_size = verify_tokenizer_bytes(
        urllib.request.urlopen(f"{bucket_resolve()}/tokenizer.json").read())
    log(f"published copy ok: {len(SPEC.files)} files, rope_theta "
        f"{config['rope_theta']}, rope_type "
        f"{(config.get('rope_scaling') or {}).get('rope_type')}, {vocab_size} tokens")
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "bucket_uri": f"hf://buckets/{BUCKET_ID}/{bucket_path()}",
            "files": {name: entries[name].size for name in sorted(SPEC.files)},
            "rope_theta": config.get("rope_theta"),
            "rope_scaling": config.get("rope_scaling"),
            "vocab_size": vocab_size,
        }, indent=2) + "\n")
        log(f"wrote {out}")
    return 0


def hf_token() -> str:
    """The org-scoped token that can write the open-athena bucket.

    Bucket writes and repo creation want *different* tokens on this workstation,
    and picking the wrong one fails late: the write shows up as a 403 from
    `.../buckets/open-athena/MarinFold/xet-write-token` after the whole
    checkpoint has been staged. ``write2`` can create model repos and cannot
    write this bucket; ``oa-marinfold`` is the reverse, and is the one wanted
    here. Named explicitly so a submit does not depend on `hf auth switch`.
    """
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    import configparser

    path = Path.home() / ".cache/huggingface/stored_tokens"
    parser = configparser.ConfigParser()
    parser.read(path)
    if parser.has_option("oa-marinfold", "hf_token"):
        return parser.get("oa-marinfold", "hf_token")
    raise SystemExit(f"no HF_TOKEN and no 'oa-marinfold' entry in {path}")


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
        "--job-name", f"exp250-publish-m2p06-{SPEC.key}-step{SPEC.step}",
        "--priority", "batch", "--enable-extra-resources", "--no-wait",
        "--cpu", "4", "--memory", "16GB", "--disk", "32GB",
        "--max-retries", "3", "--timeout", "7200",
        "-e", "HF_TOKEN", hf_token(),
        "-e", "EXP250_ROPE_REPAIR_B64", base64.b64encode(payload).decode(),
        "-e", "EXP250_ROPE_REPAIR_SHA256", hashlib.sha256(payload).hexdigest(),
        "--", "python", "publish_exp232_m2_p06.py", "--checkpoint", SPEC.key,
    ]
    if dry_run:
        # Redact rather than truncate: the token is short enough to survive a
        # length filter, and a dry run is exactly when someone pastes the output
        # into an issue.
        secrets = {hf_token(), base64.b64encode(payload).decode()}
        log("DRY RUN " + " ".join("<redacted>" if a in secrets else a for a in argv))
        return 0
    # Not check=True: CalledProcessError puts the whole argv — token included —
    # into a traceback that ends up in logs and pasted terminal output.
    if subprocess.run(argv, cwd=HERE).returncode != 0:
        raise SystemExit("iris job run failed; rerun with --dry-run to see the "
                         "(redacted) command line")
    log(f"submitted; logs: {iris_bin} --cluster=marin job logs "
        f"/bizon/exp250-publish-m2p06-{SPEC.key}-step{SPEC.step}")
    return 0


def main() -> int:
    global SPEC
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", choices=sorted(CHECKPOINTS),
                        default=DEFAULT_CHECKPOINT,
                        help="which #232 final to publish (default: %(default)s)")
    parser.add_argument("--submit", action="store_true",
                        help="run on a CoreWeave pod instead of here")
    parser.add_argument("--verify", action="store_true",
                        help="check the already-published bucket copy")
    parser.add_argument("--out", type=Path, default=None,
                        help="where --verify writes its record")
    parser.add_argument("--iris-bin", default=os.environ.get("IRIS_BIN", DEFAULT_IRIS))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    SPEC = CHECKPOINTS[args.checkpoint]
    log(f"checkpoint {SPEC.key}: {SPEC.run_name} step {SPEC.step} — {SPEC.note}")
    if args.verify:
        return verify_published(args.out or Path(f"data/m2_p06_{SPEC.key}_publish_check.json"))
    if args.submit:
        return submit(args.iris_bin, args.dry_run)
    return run()


if __name__ == "__main__":
    sys.exit(main())
