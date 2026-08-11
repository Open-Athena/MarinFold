# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare weight statistics of two HF exports — issue #208.

WHY. exp208's code trains cleanly from exp163 arm F (train/ratio_mean 1.0000 +/-
0.0005 over 10 steps) and NaNs on the first step from exp199, with everything
else identical. Their configs are semantically the same, rope included, and vLLM
loads exp199's files and generates well — precision 0.22, consensus 0.43 — so the
weights are finite and correctly mapped for inference.

What is left is the model itself. levanter trains under
``mp="p=f32,c=bfloat16"``: parameters in f32, compute in bf16. bf16 carries the
same exponent range as f32 but only ~3 decimal digits of mantissa, so it does not
overflow where f32 would not — but a backward pass through unusually large
weights can still produce non-finite intermediates. If exp199 carries materially
larger per-tensor magnitudes than exp163 arm F, that names the mechanism and
points at a mixed-precision fix rather than a re-export.

This reads the safetensors headers and tensors from each repo and reports, per
tensor and per tensor-family, max|w|, RMS, and any non-finite entries. Runs
cloud-side: the two exports are ~2.7 and ~2.9 GB and the workstation uplink is
~2.5 MB/s.

    uv run python compare_weights.py --submit
    uv run python compare_weights.py            # on the pod
"""

import argparse
import re
import sys
from collections import defaultdict

A_REPO = "timodonnell/marinfold-contacts-v1-exp199-1_5b-step145199"
B_REPO = "timodonnell/plm-exp163-refine-cv1-1_5b-lr1e-4-e1-cos-tpuF-step404"


def family(name: str) -> str:
    """Collapse ``model.layers.7.mlp.down_proj.weight`` -> ``mlp.down_proj``."""
    return re.sub(r"\.\d+\.", ".N.", name).replace("model.layers.N.", "")


def stats_for(repo: str) -> dict[str, dict[str, float]]:
    import torch
    from huggingface_hub import snapshot_download
    from safetensors.torch import safe_open
    from pathlib import Path

    local = Path(snapshot_download(repo, allow_patterns=["*.safetensors"], max_workers=8))
    out: dict[str, dict[str, float]] = {}
    for shard in sorted(local.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt") as fh:
            for key in fh.keys():
                t = fh.get_tensor(key).to(torch.float32)
                finite = torch.isfinite(t)
                out[key] = {
                    "max_abs": float(t.abs().max()),
                    "rms": float(t.pow(2).mean().sqrt()),
                    "n_nonfinite": int((~finite).sum()),
                    "numel": int(t.numel()),
                }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default=A_REPO, help="the export that NaNs (exp199)")
    ap.add_argument("--b", default=B_REPO, help="the export that trains (exp163 arm F)")
    ap.add_argument("--submit", action="store_true")
    args = ap.parse_args()

    if args.submit:
        from _submit import check_clean, submit
        check_clean()
        name = submit(
            job_name="exp208-compare-weights",
            command=["python", "compare_weights.py", "--a", args.a, "--b", args.b],
            extras=("cpu",), cpu=8, memory="48GB", disk="48GB",
            region="us-central1", priority="batch",
        )
        print(f"[weights] submitted /bizon/{name}")
        return 0

    a, b = stats_for(args.a), stats_for(args.b)
    shared = sorted(set(a) & set(b))
    print(f"[weights] A={args.a}\n[weights] B={args.b}")
    print(f"[weights] {len(a)} vs {len(b)} tensors, {len(shared)} shared")
    only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
    if only_a or only_b:
        print(f"[weights] ONLY IN A: {only_a[:5]}\n[weights] ONLY IN B: {only_b[:5]}")

    bad = [(k, v) for k, v in a.items() if v["n_nonfinite"]]
    print(f"\n[weights] non-finite entries in A: {len(bad)} tensor(s)"
          + (f" e.g. {bad[:3]}" if bad else " (none)"))
    bad_b = [(k, v) for k, v in b.items() if v["n_nonfinite"]]
    print(f"[weights] non-finite entries in B: {len(bad_b)} tensor(s)"
          + (f" e.g. {bad_b[:3]}" if bad_b else " (none)"))

    fam = defaultdict(lambda: {"a_max": 0.0, "b_max": 0.0, "a_rms": [], "b_rms": []})
    for k in shared:
        f = fam[family(k)]
        f["a_max"] = max(f["a_max"], a[k]["max_abs"])
        f["b_max"] = max(f["b_max"], b[k]["max_abs"])
        f["a_rms"].append(a[k]["rms"])
        f["b_rms"].append(b[k]["rms"])

    print(f"\n{'family':34s} {'A max|w|':>10s} {'B max|w|':>10s} {'ratio':>7s} "
          f"{'A rms':>9s} {'B rms':>9s} {'ratio':>7s}")
    worst = []
    for name in sorted(fam):
        f = fam[name]
        a_rms = sum(f["a_rms"]) / len(f["a_rms"])
        b_rms = sum(f["b_rms"]) / len(f["b_rms"])
        mr = f["a_max"] / f["b_max"] if f["b_max"] else float("inf")
        rr = a_rms / b_rms if b_rms else float("inf")
        worst.append((mr, name))
        print(f"{name:34s} {f['a_max']:10.4f} {f['b_max']:10.4f} {mr:7.2f} "
              f"{a_rms:9.5f} {b_rms:9.5f} {rr:7.2f}")

    worst.sort(reverse=True)
    print(f"\n[weights] largest max|w| ratios A/B: "
          + ", ".join(f"{n} {r:.2f}x" for r, n in worst[:4]))
    print("[weights] a ratio near 1 everywhere means magnitude is NOT the mechanism.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
