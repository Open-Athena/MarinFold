# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare weight statistics across contacts-v1 exports — issue #208.

WHY. Two exp208 results point at the exp199 checkpoint being numerically unusual:
it reads +0.023 R-precision against its own published number on a different
inference stack while exp117 reproduces across the same two stacks to -0.0015
(see RPRECISION_STACK_DISCREPANCY.md), and it NaNs on the first levanter training
step where exp163 arm F trains cleanly through identical code.

The hypothesis is that exp199 sits closer to bf16's precision limits, which would
make it sensitive both to which bf16 kernels execute it and to a bf16 backward
pass. Testing that needs **exp117** in the comparison, because exp117 is the
checkpoint that actually serves as the reproducibility control — comparing exp199
only against exp163 arm F (as the first version of this script did) cannot
distinguish "exp199 is an outlier" from "exp163 arm F is unusually small".

SOURCES. Accepts HF model repos and open-athena **bucket** paths, which are
different namespaces: `snapshot_download` cannot see a bucket at all, so bucket
sources go through `list_bucket_tree` / `download_bucket_files`.

DTYPE. Bucket copies are fp32 as exported; the exp163 and exp199 model repos are
bf16. Everything is cast to f32 for the statistics, and bf16 rounding moves
max|w| by well under 1% — immaterial against the multiples this is looking for,
but the per-source dtype is reported so the reader can see which is which.

    uv run python compare_weights.py --submit
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

# label = source spec. "repo:<id>" or "bucket:<bucket_id>:<prefix>".
DEFAULT_SOURCES = [
    "exp199=repo:timodonnell/marinfold-contacts-v1-exp199-1_5b-step145199",
    "exp117=bucket:open-athena/MarinFold:checkpoints/prot-exp117-cv1-s02-1_5b-e16-lr3p162e-3-wd0p2-bs256-europe-west4/hf/step-35679",
    "exp163F=repo:timodonnell/plm-exp163-refine-cv1-1_5b-lr1e-4-e1-cos-tpuF-step404",
]
REFERENCE = "exp117"


def family(name: str) -> str:
    """Collapse ``model.layers.7.mlp.down_proj.weight`` -> ``mlp.down_proj.weight``."""
    return re.sub(r"\.\d+\.", ".N.", name).replace("model.layers.N.", "")


def fetch(spec: str) -> Path:
    """Materialise a source locally and return the directory holding safetensors."""
    kind, _, rest = spec.partition(":")
    if kind == "repo":
        from huggingface_hub import snapshot_download
        return Path(snapshot_download(rest, allow_patterns=["*.safetensors"], max_workers=8))
    if kind == "bucket":
        from huggingface_hub import download_bucket_files, list_bucket_tree
        bucket, _, prefix = rest.partition(":")
        entries = [e for e in list_bucket_tree(bucket, prefix, recursive=True, token=False)
                   if getattr(e, "size", None) is not None
                   and e.path.endswith(".safetensors")]
        if not entries:
            raise SystemExit(f"no safetensors under bucket {bucket}/{prefix}")
        dest = Path("/tmp") / re.sub(r"[^a-zA-Z0-9]", "_", prefix)[-60:]
        dest.mkdir(parents=True, exist_ok=True)
        download_bucket_files(
            bucket, [(e, str(dest / Path(e.path).name)) for e in entries], token=False)
        return dest
    raise SystemExit(f"unknown source kind {kind!r} in {spec!r}")


def stats_for(spec: str) -> tuple[dict[str, dict[str, float]], str]:
    import torch
    from safetensors.torch import safe_open

    local = fetch(spec)
    out: dict[str, dict[str, float]] = {}
    dtypes: set[str] = set()
    for shard in sorted(local.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt") as fh:
            for key in fh.keys():
                raw = fh.get_tensor(key)
                dtypes.add(str(raw.dtype).replace("torch.", ""))
                t = raw.to(torch.float32)
                out[key] = {
                    "max_abs": float(t.abs().max()),
                    "rms": float(t.pow(2).mean().sqrt()),
                    "n_nonfinite": int((~torch.isfinite(t)).sum()),
                }
    return out, "/".join(sorted(dtypes))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", action="append", default=[], metavar="LABEL=SPEC")
    ap.add_argument("--reference", default=REFERENCE)
    ap.add_argument("--submit", action="store_true")
    args = ap.parse_args()
    sources = args.source or DEFAULT_SOURCES

    if args.submit:
        from _submit import check_clean, submit
        check_clean()
        cmd = ["python", "compare_weights.py", "--reference", args.reference]
        for s in sources:
            cmd += ["--source", s]
        name = submit(
            job_name="exp208-compare-weights-3way",
            command=cmd,
            extras=("cpu",), cpu=8, memory="64GB", disk="80GB",
            region="us-central1", priority="batch",
        )
        print(f"[weights] submitted /bizon/{name}")
        return 0

    parsed = {}
    for spec in sources:
        label, _, rest = spec.partition("=")
        parsed[label] = rest
    if args.reference not in parsed:
        raise SystemExit(f"reference {args.reference!r} not among {sorted(parsed)}")

    stats, dtypes = {}, {}
    for label, spec in parsed.items():
        stats[label], dtypes[label] = stats_for(spec)
        n_bad = sum(v["n_nonfinite"] for v in stats[label].values())
        print(f"[weights] {label:8s} {len(stats[label])} tensors  dtype={dtypes[label]}  "
              f"non-finite={n_bad}  ({parsed[label]})", flush=True)

    labels = list(parsed)
    shared = set.intersection(*(set(s) for s in stats.values()))
    print(f"[weights] {len(shared)} tensors shared by all {len(labels)} sources")

    fam = defaultdict(lambda: defaultdict(list))
    for key in shared:
        f = family(key)
        for label in labels:
            fam[f][label].append(stats[label][key])

    ref = args.reference
    others = [l for l in labels if l != ref]
    head = f"{'family':32s}" + "".join(f"{l + ' max|w|':>17s}" for l in labels)
    head += "".join(f"{l + '/' + ref:>12s}" for l in others)
    print("\n" + head)
    ratios = defaultdict(list)
    for name in sorted(fam):
        row = f"{name:32s}"
        maxes = {l: max(v["max_abs"] for v in fam[name][l]) for l in labels}
        for l in labels:
            row += f"{maxes[l]:17.4f}"
        for l in others:
            r = maxes[l] / maxes[ref] if maxes[ref] else float("inf")
            ratios[l].append(r)
            row += f"{r:12.2f}"
        print(row)

    print()
    for l in others:
        rs = sorted(ratios[l])
        med = rs[len(rs) // 2]
        print(f"[weights] {l:8s} vs {ref}: median max|w| ratio {med:.2f}x, "
              f"range {rs[0]:.2f}x - {rs[-1]:.2f}x")
    print(f"\n[weights] VERDICT: if {ref} resembles exp163F and exp199 is the outlier, the")
    print( "[weights] numerical-sensitivity hypothesis survives. If exp117 is itself large,")
    print( "[weights] exp199 is not unusual and the hypothesis is dead.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
