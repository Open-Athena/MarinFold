# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Trained checkpoint -> a CoreWeave-S3 model the eval workers can load (#175).

#160 did these steps by hand; they are scripted here because they are the same
four every time and the ordering matters. Waits for the run's own
``hf/step-<N>/`` export, then:

1. **Fixes three small JSON files.** levanter exports under transformers 5.x,
   which writes ``rope_parameters`` (a 4.x config silently loses the llama3 rope
   and degrades *worse as the protein gets longer* — the failure most easily
   mistaken for a real finding) and ``tokenizer_class: TokenizersBackend``
   (``AutoTokenizer`` cannot resolve it). It also writes no
   ``special_tokens_map.json``.
2. **Skips the bf16 recast.** That was only ever a TPU ragged-paged-attention
   requirement; CUDA vLLM casts fp32 at load. So model preparation is three
   JSON files, not a 5.5 GB round trip.
3. **Stages cloud-side.** GCS -> CoreWeave S3 from a GCS-local marin pod
   (~40 s for 5.5 GB) rather than ~30 min over the workstation uplink.
4. **Uploads the fixed JSONs over the staged copy**, last, so a partially
   staged model is never left looking complete.

    set -a; source ~/.config/marin/cw-rno2a.env; set +a
    uv run --no-project --with s3fs --with gcsfs python finalize_model.py
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

EXP160 = Path(__file__).resolve().parent.parent / "exp160_models_backtracking_training"
sys.path.insert(0, str(EXP160))

RUN = ("gs://marin-us-east5/protein-structure/MarinFold/exp175_backtracking_mode/"
       "runs/exp175-cv1-1_5b-mode50-lr3e-4-e1-cos")
S3_MODELS = ("s3://marin-us-east-02a/protein-structure/MarinFold/"
             "exp175_backtracking_mode/eval/models")
LABEL = "exp175-mode-step2058"
VOCAB = 3850
META = ("config.json", "tokenizer_config.json", "special_tokens_map.json")


def wait_for_export(export: str, expect: int = 6) -> None:
    import gcsfs

    fs = gcsfs.GCSFileSystem()
    while True:
        files = [f for f in fs.ls(export[len("gs://"):]) if not f.endswith("/")] \
            if fs.exists(export[len("gs://"):]) else []
        if len(files) >= expect:
            print(f"[finalize] export present: {len(files)} objects", flush=True)
            return
        print(f"[finalize] waiting for {export} ({len(files)}/{expect})", flush=True)
        time.sleep(120)


def build_fixed_meta(export: str, out: Path) -> None:
    """Download the small files and rewrite them into transformers-4.x shape."""
    import gcsfs
    from prepare_eval_model import copy_tokenizer, downgrade_config

    fs = gcsfs.GCSFileSystem()
    src = out / "src"
    src.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json"):
        fs.get(f"{export[len('gs://'):]}/{name}", str(src / name))
    fixed = out / "fixed"
    fixed.mkdir(parents=True, exist_ok=True)
    downgrade_config(src, fixed, vocab_size=VOCAB)
    copy_tokenizer(src, fixed)
    # The whole point of #175 is this token; refuse to ship a model whose
    # tokenizer cannot resolve it.
    vocab = json.loads((fixed / "tokenizer.json").read_text())["model"]["vocab"]
    tid = vocab.get("<contacts-v1.backtracking>")
    if tid != VOCAB - 1:
        raise SystemExit(f"<contacts-v1.backtracking> is {tid}, expected {VOCAB - 1}")
    print(f"[finalize] <contacts-v1.backtracking> = {tid}", flush=True)


def stage(export: str, step_dir: str, label: str = LABEL,
          job: str = "exp175-stage-model") -> None:
    """Submit the cloud-side copy AND WAIT for it.

    ``stage_to_cw.py`` submits with ``--no-wait`` and exits 0 as soon as the job
    is accepted, so treating its return as "staged" is wrong: the first version
    of this script did exactly that, uploaded the fixed JSONs against an empty
    prefix, and then the staging job copied the export's *unfixed* config.json
    and tokenizer_config.json straight over them. The result loaded with a
    defaulted rope -- the failure that degrades worse the longer the protein is,
    i.e. the one most easily mistaken for a finding. Hence: wait, then fix.
    """
    subprocess.run(
        [sys.executable, str(EXP160 / "stage_to_cw.py"),
         "--job-name", job,
         "--gcs", export.rsplit("/", 1)[0],
         "--s3", S3_MODELS,
         "--assets", f"{step_dir}={label}"],
        check=True,
    )
    iris = "/home/bizon/git/marin-freshiris/.venv/bin/iris"
    while True:
        out = subprocess.run([iris, "--cluster=marin", "job", "summary", f"/bizon/{job}"],
                             capture_output=True, text=True).stdout
        state = next((ln for ln in out.splitlines() if ln.startswith("State:")), "")
        if "succeeded" in state:
            print(f"[finalize] staging {state.strip()}", flush=True)
            return
        if "failed" in state or "killed" in state:
            raise SystemExit(f"[finalize] staging did not succeed: {state.strip()}")
        print(f"[finalize] staging in progress ({state.strip() or 'pending'})", flush=True)
        time.sleep(30)


def upload_meta(fixed: Path, label: str = LABEL) -> None:
    import os

    import fsspec

    fs = fsspec.filesystem("s3", endpoint_url="https://cwobject.com",
                           key=os.environ["CW_KEY_ID"],
                           secret=os.environ["CW_KEY_SECRET"],
                           config_kwargs={"s3": {"addressing_style": "virtual"}})
    base = f"{S3_MODELS[len('s3://'):]}/{label}"
    for name in META:
        fs.put_file(str(fixed / name), f"{base}/{name}")
        print(f"[finalize] uploaded {name}", flush=True)
    print("[finalize] files: " + ", ".join(sorted(p.split("/")[-1] for p in fs.ls(base))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", default="step-2058")
    ap.add_argument("--run", default=RUN, help="run dir holding hf/<step>/")
    ap.add_argument("--label", default=LABEL, help="model dir name under the S3 prefix")
    ap.add_argument("--work", type=Path, default=Path("/home/bizon/exp175_eval/meta"))
    a = ap.parse_args()

    export = f"{a.run}/hf/{a.step}"
    wait_for_export(export)
    build_fixed_meta(export, a.work)
    stage(export, a.step, label=a.label, job=f"exp175-stage-{a.label}")
    upload_meta(a.work / "fixed", label=a.label)
    print(f"[finalize] MODEL READY  {S3_MODELS}/{a.label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
