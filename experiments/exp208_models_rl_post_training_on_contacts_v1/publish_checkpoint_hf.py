# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish the exp199 starting checkpoint to an HF model repo — issue #208.

WHY A REPO AND NOT THE BUCKET, AND NOT GCS. The checkpoint is already public in
the open-athena *bucket* and already staged to GCS in bf16 by
``stage_model_gcs.sh``, and neither is usable here.
``vLLMInferenceContext.__init__`` resolves its tokenizer with
``levanter.tokenizers.load_tokenizer(VLLMEngineConfig.model_name)``, which
accepts a local directory, a ``mirror://`` ref, or an HF Hub **repo id** — a
bucket path is none of those, and a ``gs://`` URL raises ``HFValidationError``.
vLLM streams weights from GCS happily, so the failure lands only on the
tokenizer, inside a rollout worker, after the gang has scheduled.

SOURCE IS THE GCS bf16 STAGE, not the bucket. That copy has already been cast to
bf16 (TPU parameters are bf16 and vLLM shards the checkpoint as it loads; fp32
weights are a known failure rather than a silent cast) and has already had its
config and tokenizer verified. Re-deriving from the bucket would repeat a 5.5 GiB
download to produce the same bytes.

Runs cloud-side by default: the workstation uplink is ~2.5 MB/s and this moves
~2.7 GiB, which is ~20 minutes from here against about one on a pod.

Three things are checked before anything is uploaded, because all three are
silent when wrong and each has already cost this project results:

* **rope_theta** must be top-level. levanter writes Llama3 rope under
  ``rope_parameters`` and leaves the top-level key null; readers older than
  transformers 5 then fall back to default rope — a 50x wrong base frequency
  (#163, #180, #198).
* **the tokenizer must travel with the weights** (a hard rule in this repo), and
* **the contact vocab ids must not have drifted**: ``<contact>``=5,
  ``<begin_statements>``=9, ``<end>``=10, ``<p0>``=143, contiguous to
  ``<p1999>``. The dense reward walks token ids, so a drift would place rewards
  on the wrong positions and produce a plausible-looking run with no signal.

    uv run python publish_checkpoint_hf.py --submit      # from the workstation
    uv run python publish_checkpoint_hf.py               # on the pod
"""

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

DEFAULT_SRC = "gs://marin-us-central1/protein-structure/MarinFold/exp208/models/exp199"
DEFAULT_REPO = "timodonnell/marinfold-contacts-v1-exp199-1_5b-step145199"

# Fully determined by marinfold's build_tokenizer; contact_rewards bakes the same
# values in and refuses to run on drift.
WANT_IDS = {"<contact>": 5, "<begin_statements>": 9, "<end>": 10, "<p0>": 143,
            "<contacts-v1>": 2}
NUM_POSITIONS = 2000

CARD = """---
license: apache-2.0
tags:
- protein
- contact-prediction
- marinfold
---

# contacts-v1 1.5B (exp199) — exp208 RL starting checkpoint

Republished copy of MarinFold's default contacts-v1 model
(`prot-exp199-cw-cv1-s02-m1-p06-aug`, step 145199), cast to bf16, used as the
warm start for the [#208](https://github.com/Open-Athena/MarinFold/issues/208) RL
post-training runs.

It exists as a *repo* only because marin's RL rollout worker resolves its
tokenizer through `load_tokenizer`, which cannot read the open-athena bucket path
or a `gs://` URL. The canonical copy, with provenance, is the bucket:
`hf://buckets/open-athena/MarinFold/checkpoints/prot-exp199-cw-cv1-s02-m1-p06-aug/hf/step-145199`.

Consensus R-precision on the 554-protein eval set (#82 rollout recipe, n=100):
**0.5873** all / **0.5422** long.

`rope_theta` is stated top-level (500000). The contacts-v1 tokenizer is
co-located; do not load this model with a different one.
"""


def verify(local: Path) -> None:
    """Refuse to publish a checkpoint that would fail silently downstream."""
    cfg = json.loads((local / "config.json").read_text())
    if cfg.get("rope_theta") is None:
        raise SystemExit(
            "FATAL: no top-level rope_theta — readers older than transformers 5 would "
            "silently use default rope. Repair before publishing."
        )
    if not (local / "tokenizer.json").exists():
        raise SystemExit("FATAL: no tokenizer beside the weights (repo hard rule)")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(local))
    drift = {t: tok.convert_tokens_to_ids(t) for t, want in WANT_IDS.items()
             if tok.convert_tokens_to_ids(t) != want}
    last = tok.convert_tokens_to_ids(f"<p{NUM_POSITIONS - 1}>")
    if drift or last != WANT_IDS["<p0>"] + NUM_POSITIONS - 1:
        raise SystemExit(
            f"FATAL: contact vocab drift {drift}, <p{NUM_POSITIONS - 1}>={last}. The dense "
            "reward walks token ids; publishing this would place rewards on the wrong "
            "positions and train on noise that looks like signal."
        )
    print(f"[publish] verified: rope_theta={cfg['rope_theta']} vocab_size={cfg.get('vocab_size')} "
          f"ids OK", flush=True)


def run(src: str, repo: str, private: bool) -> int:
    import fsspec
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is required (the write2 token — the default "
                         "oa-marinfold one cannot create repos)")
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="exp208-publish-") as tmp:
        local = Path(tmp)
        fs, root = fsspec.core.url_to_fs(src)
        files = [f for f in fs.ls(root, detail=True) if f["type"] == "file"]
        if not files:
            raise SystemExit(f"nothing under {src}")
        for f in files:
            fs.get_file(f["name"], str(local / os.path.basename(f["name"])))
        size = sum(f["size"] for f in files)
        print(f"[publish] staged {len(files)} files, {size / 2**30:.2f} GiB, "
              f"{time.time() - t0:.0f}s", flush=True)

        verify(local)
        (local / "README.md").write_text(CARD)

        api = HfApi(token=token)
        api.create_repo(repo, repo_type="model", private=private, exist_ok=True)
        api.upload_folder(repo_id=repo, folder_path=str(local), repo_type="model",
                          commit_message="exp208: exp199 warm-start checkpoint (bf16)")
    print(f"[publish] DONE in {(time.time() - t0) / 60:.1f} min -> https://huggingface.co/{repo}",
          flush=True)
    return 0


def hf_token() -> str:
    """The write2 token, which is the one that can create repos.

    The workstation's *active* token is scoped to timodonnell-only and 403s on
    repo creation, so reading the stored-token file is what makes this
    reproducible rather than a documented manual step.
    """
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    import configparser
    path = Path.home() / ".cache/huggingface/stored_tokens"
    if path.exists():
        parser = configparser.ConfigParser()
        parser.read(path)
        for name in ("write2", "DEFAULT"):
            if parser.has_option(name, "hf_token"):
                return parser.get(name, "hf_token")
    raise SystemExit("no HF_TOKEN and no `write2` entry in ~/.cache/huggingface/stored_tokens")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--submit", action="store_true", help="run cloud-side instead of here")
    a = ap.parse_args()

    if a.submit:
        from _submit import check_clean, submit
        check_clean()
        name = submit(
            job_name="exp208-publish-exp199",
            command=["python", "publish_checkpoint_hf.py", "--src", a.src, "--repo", a.repo]
                    + (["--private"] if a.private else []),
            extras=("cpu",), cpu=4, memory="32GB", disk="32GB",
            region="us-central1", priority="batch",
            env={"HF_TOKEN": hf_token()},
        )
        print(f"[publish] submitted /bizon/{name}")
        return 0
    return run(a.src, a.repo, a.private)


if __name__ == "__main__":
    sys.exit(main())
