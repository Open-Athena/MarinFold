# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Run MarinFold 1B zero-shot on the 100 FoldBench monomers.

For every protein in ``protenix_data/manifest.csv``:
  1. Read the canonical 1..N residue sequence from the matching
     ``protenix_data/gt/<stem>.cif`` (Protenix's GT for that protein).
  2. Build the v1 prompt with no seeded contacts.
  3. Query vLLM at every (i, j) with i < j, asking for the CB-CB
     distance bin (CA for GLY on either side, matching Protenix's
     distogram representative-atom convention).
  4. Capture the full 64-bin probability vector per pair, build a
     symmetric ``[N, N, 64]`` array, and save it to
     ``outputs/<stem>/distogram.npz`` with key ``probs``.

Idempotent: re-running skips proteins whose ``distogram.npz`` already
exists and has the expected shape. ``--limit N`` keeps the run small
during smoke tests.
"""

import argparse
import json
import math
import os
import platform
import socket
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# exp1 path dep: we use its parse + vocab modules to stay byte-identical
# with the training-time tokenization. The exp1 dir is a script dir, not
# a python package, so we add it to sys.path before importing.
_THIS = Path(__file__).resolve().parent
_EXP1 = _THIS.parent / "exp1_document_structures_contacts_and_distances_v1"
if str(_EXP1) not in sys.path:
    sys.path.insert(0, str(_EXP1))

from vocab import NAME  # noqa: E402 — needs sys.path insertion above

from canonical_sequence import read_canonical_sequence  # noqa: E402


# Distance-bin constants — must match exp1's tokenizer. Bin k (1..64)
# covers ((k-1)*0.5, k*0.5] Å, so its midpoint is k*0.5 - 0.25.
_DISTANCE_BIN_WIDTH_A = 0.5
_NUM_DISTANCE_BINS = 64
_DISTANCE_MAX_A = _NUM_DISTANCE_BINS * _DISTANCE_BIN_WIDTH_A  # 32.0 Å
_BIN_MIDPOINTS = np.array(
    [(k + 1) * _DISTANCE_BIN_WIDTH_A - _DISTANCE_BIN_WIDTH_A / 2
     for k in range(_NUM_DISTANCE_BINS)],
    dtype=np.float32,
)


def _resolve_model_path(models_yaml_path: Path, nickname: str) -> str:
    """Return the local HF snapshot dir for the named model entry.

    Reads ``MODELS.yaml``, finds the entry with ``nickname == nickname``,
    snapshot-downloads only the matching subdir of the HF repo (the
    repo can contain many sibling models — exp9's resolver uses the
    same trick), and returns the local path that vLLM should load from.

    Re-runnable: ``snapshot_download`` is content-addressed; unchanged
    files don't redownload.
    """
    from huggingface_hub import snapshot_download

    entries = yaml.safe_load(models_yaml_path.read_text())
    matched = [e for e in entries if e.get("nickname") == nickname]
    if not matched:
        raise ValueError(f"no entry for {nickname!r} in {models_yaml_path}")
    entry = matched[0]
    url = entry["url"]
    # URL format: https://huggingface.co/<owner>/<repo>/tree/<ref>/<subdir>
    prefix = "https://huggingface.co/"
    if not url.startswith(prefix):
        raise ValueError(f"unexpected model URL {url!r}")
    rest = url[len(prefix):].split("/")
    if len(rest) < 2:
        raise ValueError(f"unexpected model URL {url!r}")
    repo_id = "/".join(rest[:2])
    subdir = rest[4] if len(rest) > 4 and rest[2] == "tree" else None
    # Use the default HF cache to share with other experiments
    # (exp9 etc.) instead of re-downloading into a local-only dir.
    local = snapshot_download(
        repo_id=repo_id,
        allow_patterns=[f"{subdir}/*"] if subdir else None,
    )
    return str(Path(local) / subdir) if subdir else local


def _load_vllm(model_local_path: str):
    """Load the model into vLLM with our standard config + return tokenizer.

    Mirrors exp9's settings (max_logprobs=128 so we can recover all 64
    distance-bin probabilities even when they're not in the natural
    top-K of the next-token distribution).
    """
    from vllm import LLM
    llm = LLM(
        model=str(model_local_path),
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        enforce_eager=True,
        trust_remote_code=True,
        max_logprobs=128,
        max_model_len=8192,
    )
    return llm, llm.get_tokenizer()


def _resolve_distance_token_ids(tokenizer) -> list[int]:
    """Return the 64 distance-bin token IDs in bin order (k=0 → <d0.5>)."""
    ids = []
    for k in range(_NUM_DISTANCE_BINS):
        tok = f"<d{(k+1)*_DISTANCE_BIN_WIDTH_A:.1f}>"
        enc = tokenizer.encode(tok, add_special_tokens=False)
        if len(enc) != 1:
            raise ValueError(f"bad encoding for {tok}: {enc!r}")
        ids.append(int(enc[0]))
    if len(set(ids)) != _NUM_DISTANCE_BINS:
        raise ValueError("distance bins collapsed in tokenizer")
    return ids


def _encode_tokens(tokenizer, token_strs: list[str]) -> list[int]:
    """1:1 WordLevel encoding for the v1 ``<...>`` tokens."""
    ids = tokenizer.encode(" ".join(token_strs), add_special_tokens=False)
    if len(ids) != len(token_strs):
        raise ValueError(
            f"tokenizer 1:1 broke: {token_strs[:5]!r} -> {ids[:5]!r}"
        )
    return [int(x) for x in ids]


def _build_base_prompt(residue_names: Iterable[str]) -> list[str]:
    """``<contacts-and-distances-v1> <begin_sequence> <AAs…> <begin_statements>``."""
    toks = [f"<{NAME}>", "<begin_sequence>"]
    toks.extend(f"<{name}>" for name in residue_names)
    toks.append("<begin_statements>")
    return toks


def _query_atom_for(name: str) -> str:
    """CB for non-GLY, CA for GLY / UNK / anything without a CB.

    UNK is generated by canonical_sequence for non-canonical residues;
    those have undefined CB conventions in the training data so we
    fall back to CA to keep the prompt well-formed.
    """
    if name in ("GLY", "UNK"):
        return "CA"
    return "CB"


def _predict_distogram(
    *,
    llm,
    tokenizer,
    residue_names: tuple[str, ...],
    distance_token_ids: list[int],
    batch_size: int = 128,
    top_k_logprobs: int = 128,
) -> np.ndarray:
    """Run the full N×N pair sweep and return the [N, N, 64] prob matrix.

    Per-pair atom = CB-CB with CA fallback for GLY/UNK on either side.
    Diagonal is left as zero (matches Protenix's distogram .npz layout —
    self-pair is meaningless). Upper and lower triangles get the same
    value via symmetry.
    """
    from vllm import SamplingParams, TokensPrompt

    n = len(residue_names)
    base_tokens = _build_base_prompt(residue_names)
    base_ids = _encode_tokens(tokenizer, base_tokens)

    distance_id_set = set(distance_token_ids)
    bin_of = {tid: k for k, tid in enumerate(distance_token_ids)}

    # All upper-triangle pairs (i, j) with i < j, 1-indexed for the
    # v1 position tokens <p1>..<pN>.
    pairs: list[tuple[int, int]] = [
        (i, j) for i in range(1, n + 1) for j in range(i + 1, n + 1)
    ]
    atoms: list[tuple[str, str]] = [
        (_query_atom_for(residue_names[i - 1]),
         _query_atom_for(residue_names[j - 1]))
        for i, j in pairs
    ]

    prompts = []
    for (i, j), (a_i, a_j) in zip(pairs, atoms, strict=True):
        tail = _encode_tokens(tokenizer, [
            "<distance>", f"<p{i}>", f"<p{j}>", f"<{a_i}>", f"<{a_j}>",
        ])
        prompts.append(TokensPrompt(prompt_token_ids=base_ids + tail))

    sampling = SamplingParams(
        temperature=1.0, top_p=1.0, top_k=-1,
        max_tokens=1, logprobs=top_k_logprobs, n=1,
    )

    probs = np.zeros((n, n, _NUM_DISTANCE_BINS), dtype=np.float32)
    for chunk_start in range(0, len(prompts), batch_size):
        chunk_prompts = prompts[chunk_start : chunk_start + batch_size]
        outputs = llm.generate(chunk_prompts, sampling, use_tqdm=False)
        for offset, gen in enumerate(outputs):
            lp_dict = gen.outputs[0].logprobs[0] if gen.outputs[0].logprobs else {}
            row = np.zeros(_NUM_DISTANCE_BINS, dtype=np.float32)
            for tok_id, lp in lp_dict.items():
                tid = int(tok_id)
                if tid in distance_id_set:
                    row[bin_of[tid]] = math.exp(float(lp.logprob))
            total = float(row.sum())
            if total > 0:
                row /= total
            i, j = pairs[chunk_start + offset]
            probs[i - 1, j - 1, :] = row
            probs[j - 1, i - 1, :] = row  # symmetric
    return probs


def _read_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    """Manifest CSV from exp12 — one row per protein."""
    import csv
    with manifest_path.open() as f:
        return list(csv.DictReader(f))


def _hardware_info() -> dict[str, Any]:
    """Snapshot of the current host + GPU. Recorded once per run.

    We capture this in every provenance.json so that the
    ``length-vs-runtime`` plot can group by GPU type. Both the local
    A5000 runner and the Modal H100 runner end up writing the same
    schema; the plot script then doesn't care where the timing came
    from.
    """
    info: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "runner_tag": os.environ.get("MARINFOLD_RUNNER_TAG", "local"),
    }
    try:
        import torch  # local import — torch is heavy
        info["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["gpu_name"] = props.name
            info["gpu_total_memory_gb"] = round(props.total_memory / 1e9, 2)
            info["gpu_compute_capability"] = f"{props.major}.{props.minor}"
        else:
            info["gpu_name"] = None
            info["gpu_total_memory_gb"] = None
            info["gpu_compute_capability"] = None
    except Exception as exc:  # noqa: BLE001 — best-effort hardware capture
        info["torch_version"] = None
        info["gpu_name"] = f"unavailable: {exc!r}"
    return info


def _output_complete(out_path: Path, n: int, *, model_nickname: str) -> bool:
    """Return True iff the .npz has probs ``[n, n, 64]`` AND was generated by the same model.

    Without the model check, switching ``MODELS.yaml`` to a new
    checkpoint would silently leave the old outputs in place. We
    keep the file iff:
      - ``distogram.npz`` parses and has the expected shape, AND
      - the sibling ``provenance.json`` records the same
        ``model_nickname``.
    Older runs without ``provenance.json`` get treated as
    incomplete (forces a re-run when the model changes; safe
    fallback).
    """
    if not out_path.exists():
        return False
    try:
        with np.load(out_path) as data:
            shape_ok = (
                "probs" in data.files
                and data["probs"].shape == (n, n, _NUM_DISTANCE_BINS)
            )
    except (OSError, ValueError, KeyError):
        return False
    if not shape_ok:
        return False
    prov_path = out_path.parent / "provenance.json"
    if not prov_path.exists():
        return False
    try:
        prov = json.loads(prov_path.read_text())
    except (OSError, ValueError):
        return False
    return prov.get("model_nickname") == model_nickname


def run(
    *,
    protenix_dir: Path,
    out_dir: Path,
    model_nickname: str,
    models_yaml: Path,
    limit: int | None = None,
    batch_size: int = 128,
) -> int:
    """Drive the full sweep. Returns the number of NEW proteins scored.

    Idempotent: existing distogram.npz files with the right shape are
    left alone. ``limit`` truncates the manifest after sorting (most
    proteins are listed in PDB order).
    """
    manifest = _read_manifest(protenix_dir / "manifest.csv")
    if limit is not None:
        manifest = manifest[:limit]

    # Quick pre-flight: figure out which proteins still need work. If
    # everything is already complete, skip the vLLM load entirely
    # (saves ~20s + ~6 GB of GPU memory).
    pending: list[dict[str, Any]] = []
    for entry in manifest:
        stem = entry["stem"]
        n = int(entry["n_residues"])
        out_path = out_dir / stem / "distogram.npz"
        if _output_complete(out_path, n, model_nickname=model_nickname):
            print(f"skip {stem}: distogram already complete (N={n})")
            continue
        pending.append(entry)
    if not pending:
        print(f"All {len(manifest)} proteins already have complete distograms.")
        return 0

    model_path = _resolve_model_path(models_yaml, model_nickname)
    print(f"Loading vLLM with model={model_path} ...")
    t_load_start = time.time()
    llm, tokenizer = _load_vllm(model_path)
    model_load_seconds = time.time() - t_load_start
    distance_token_ids = _resolve_distance_token_ids(tokenizer)
    hw = _hardware_info()
    print(
        f"vLLM ready ({model_load_seconds:.1f} s). "
        f"GPU={hw.get('gpu_name')} ({hw.get('gpu_total_memory_gb')} GB). "
        f"Driving {len(pending)} of {len(manifest)} proteins."
    )

    n_written = 0
    for entry in pending:
        stem = entry["stem"]
        n_expected = int(entry["n_residues"])
        gt_cif = protenix_dir / "gt" / f"{stem}.cif"
        seq = read_canonical_sequence(gt_cif)
        if seq.n_residues != n_expected:
            print(
                f"WARN: {stem} manifest n_residues={n_expected} but "
                f"GT sequence is {seq.n_residues}; using GT length."
            )
        start = time.time()
        probs = _predict_distogram(
            llm=llm, tokenizer=tokenizer,
            residue_names=seq.residue_names,
            distance_token_ids=distance_token_ids,
            batch_size=batch_size,
        )
        elapsed = time.time() - start
        out_path = out_dir / stem / "distogram.npz"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, probs=probs)
        # Provenance: tiny JSON sidecar lets future readers know
        # exactly which model nickname + atom convention + hardware
        # produced this. ``elapsed_seconds`` + ``n_pairs`` + ``hardware``
        # are what the timing plots (sequence-length vs runtime) join on.
        n_pairs = seq.n_residues * (seq.n_residues - 1) // 2
        (out_path.parent / "provenance.json").write_text(json.dumps({
            "stem": stem,
            "n_residues": seq.n_residues,
            "n_pairs": n_pairs,
            "model_nickname": model_nickname,
            "model_path": model_path,
            "atom_convention": "CB-CB (CA for GLY/UNK)",
            "bin_scheme": {
                "min_A": 0.0,
                "max_A": _DISTANCE_MAX_A,
                "n_bins": _NUM_DISTANCE_BINS,
                "midpoints_A": _BIN_MIDPOINTS.tolist(),
            },
            "elapsed_seconds": round(elapsed, 3),
            "model_load_seconds": round(model_load_seconds, 3),
            "batch_size": batch_size,
            "hardware": hw,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }, indent=2) + "\n")
        n_written += 1
        print(
            f"wrote {out_path} ({seq.n_residues} residues, {n_pairs} pairs, "
            f"{elapsed:.1f} s; {n_pairs / max(elapsed, 1e-6):.0f} pairs/s)"
        )
    return n_written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protenix-dir",
        type=Path,
        default=_THIS / "protenix_data" / "data" / "protenix-foldbench-monomers",
        help="Local mirror of the exp12 HF bucket (default: ./protenix_data/...).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_THIS / "outputs",
        help="Where to save per-protein distogram.npz files (default: ./outputs/).",
    )
    parser.add_argument(
        "--model",
        default="1B",
        help="MODELS.yaml nickname (default: 1B).",
    )
    parser.add_argument(
        "--models-yaml",
        type=Path,
        default=_THIS.parent.parent / "MODELS.yaml",
        help="Path to MODELS.yaml (default: repo root).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Smoke-test: only process the first N manifest entries.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="vLLM per-call batch size for pair queries (default: 128).",
    )
    args = parser.parse_args()
    n = run(
        protenix_dir=args.protenix_dir,
        out_dir=args.out,
        model_nickname=args.model,
        models_yaml=args.models_yaml,
        limit=args.limit,
        batch_size=args.batch_size,
    )
    print(f"Done. Wrote {n} new distograms.")


if __name__ == "__main__":
    main()
