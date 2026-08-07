# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Stamp the retraction-mode marker onto #160's 50:50 mix, cloud-side (#175).

#160's corpus is correct in every respect except that its two halves are
indistinguishable to the model: both begin ``<contacts-v1> <begin_sequence>``.
This rewrites token 0 of the backtracking half to
``<contacts-v1.backtracking>`` and leaves everything else — statements,
sections, ordering, the clean half — byte-identical.

**Mark by generator, not by content.** Every document in the backtracking half
came from #159's model-in-the-loop engine, including the **20.1% where the
trigger never fired and the document contains no ``<retract>`` at all**. Those
keep the marker: they teach the honest conditional *"in this mode, sometimes
nothing needs retracting"*. Marking only documents that happen to contain a
retraction would instead teach *"this token implies a retraction follows"* —
a different target, and one that destroys the token's use as a mode switch,
because at inference we want to be able to enter the mode without promising
the model that it must use it.

Runs on a **marin CPU pod**: the read and the write are both GCS, so this
never touches the workstation uplink (3.4 GB would be ~20 minutes there).
Same launcher pattern as #160's ``stage_to_cw.py``.

    uv run --no-project python mark_corpus.py
    uv run --no-project python mark_corpus.py --dry-run   # print the pod script
"""
from __future__ import annotations

import argparse
import base64
import os
import subprocess
from pathlib import Path

MARIN = Path(os.environ.get("MARIN_CHECKOUT", "/home/bizon/git/marin-freshiris"))
IRIS = os.environ.get("IRIS_BIN", str(MARIN / ".venv/bin/iris"))

SRC = ("gs://marin-us-east5/protein-structure/MarinFold/"
       "exp160_backtracking_training/corpus/train")
DST = ("gs://marin-us-east5/protein-structure/MarinFold/"
       "exp175_backtracking_mode/corpus/train")

PLAIN = "<contacts-v1>"
MARKED = "<contacts-v1.backtracking>"

POD_SCRIPT = '''
import sys
import gcsfs
import pandas as pd

src, dst, plain, marked = sys.argv[1:5]
fs = gcsfs.GCSFileSystem()
shards = sorted(fs.glob(f"{src}/*.parquet"))
if not shards:
    raise SystemExit(f"nothing under {src}")
print(f"[mark] {len(shards)} shards", flush=True)

totals = {"docs": 0, "marked": 0, "clean": 0, "silent_marked": 0}
for i, key in enumerate(shards):
    with fs.open(key, "rb") as fh:
        df = pd.read_parquet(fh)
    is_bt = df.kind == "backtracking"
    # Every backtracking document must start with the plain doc type, or the
    # corpus is not what we think it is -- fail rather than silently skip.
    bad = int((~df.document[is_bt].str.startswith(plain + " ")).sum())
    if bad:
        raise SystemExit(f"{key}: {bad} backtracking docs do not start with {plain}")
    n_silent = int((~df.document[is_bt].str.contains("<retract>")).sum())
    df.loc[is_bt, "document"] = (
        marked + " " + df.document[is_bt].str.slice(len(plain) + 1)
    )
    # The clean half must be untouched.
    assert (df.document[~is_bt].str.startswith(plain + " ")).all()
    totals["docs"] += len(df)
    totals["marked"] += int(is_bt.sum())
    totals["clean"] += int((~is_bt).sum())
    totals["silent_marked"] += n_silent
    out = f"{dst}/{key.rsplit('/', 1)[-1]}"
    with fs.open(out, "wb") as fh:
        df.to_parquet(fh, index=False, compression="zstd")
    print(f"[mark] {i + 1}/{len(shards)} {out} "
          f"({int(is_bt.sum()):,} marked, {n_silent:,} of them silent)", flush=True)

print(f"[mark] DONE {totals}", flush=True)
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--dst", default=DST)
    ap.add_argument("--job-name", default="exp175-mark-corpus")
    ap.add_argument("--zone", default="us-east5-a")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    script_b64 = base64.b64encode(POD_SCRIPT.encode()).decode()
    bootstrap = f"""
set -euo pipefail
echo "[mark] host=$(hostname)"
echo {script_b64} | base64 -d > /tmp/mark_corpus.py
exec uv run --no-project --with gcsfs --with pandas --with pyarrow \\
    python /tmp/mark_corpus.py '{a.src}' '{a.dst}' '{PLAIN}' '{MARKED}'
""".strip()

    command = [
        IRIS, "--cluster=marin", "job", "run",
        "--job-name", a.job_name, "--no-wait", "--enable-extra-resources",
        "--priority", "interactive", "--zone", a.zone,
        "--cpu", "4", "--memory", "16GB", "--disk", "64GB",
        "--", "bash", "-lc", bootstrap,
    ]
    print(f"[mark] {a.src}\n    -> {a.dst}")
    if a.dry_run:
        print(bootstrap)
        return 0
    subprocess.run(command, cwd=MARIN, check=True)
    print(f"[mark] submitted /bizon/{a.job_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
