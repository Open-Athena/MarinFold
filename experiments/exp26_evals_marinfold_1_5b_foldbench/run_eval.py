# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Run MarinFold zero-shot on the 100 FoldBench monomers.

For every protein in ``protenix_data/manifest.csv``:
  1. Read the canonical 1..N residue sequence from the matching
     ``protenix_data/gt/<stem>.cif`` (Protenix's GT for that protein).
  2. Build the v1 prompt with no seeded contacts.
  3. Query vLLM at every (i, j) with i < j, asking for the CB-CB
     distance bin (CA for GLY on either side, matching Protenix's
     distogram representative-atom convention).
  4. Capture the full 64-bin probability vector per pair, build a
     symmetric ``[N, N, 64]`` array, and save it to
     ``<out>/<stem>/distogram.npz`` with key ``probs``.

``--out`` accepts either a local directory or a ``gs://`` URI; in the
GCS case, per-protein outputs are written to a temp dir and then
``gsutil cp``-ed to GCS so the iris worker doesn't need gcsfs.

Idempotent: re-running skips proteins whose ``provenance.json``
already records the requested ``model_nickname``.
"""

import argparse
import json
import math
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

_THIS = Path(__file__).resolve().parent
# vocab.py and canonical_sequence.py live next to this file (vendored
# for iris shipping — the worker only has the experiment dir, not the
# wider repo, so sys.path hacks back to a sibling exp dir don't work).
sys.path.insert(0, str(_THIS))

from vocab import NAME  # noqa: E402

from canonical_sequence import (  # noqa: E402
    read_canonical_sequence,
    representative_atom_name,
)


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


# --- HuggingFace URL resolution (vendored, see marinfold.registry) ---
#
# Inlined into this file so the iris worker doesn't need the wider
# MarinFold repo. Handles both shapes seen in MODELS.yaml today:
#   - https://huggingface.co/<owner>/<repo>/tree/<rev>[/<subdir>]
#   - https://huggingface.co/buckets/<org>/<bucket>/tree/<prefix>
# The bucket shape is what the 1.5B entry uses (added by PR #25).

_HF_URL_PATTERN = re.compile(
    r"^https?://(?:www\.)?huggingface\.co/"
    r"(?P<repo>(?!buckets/)[^/]+/[^/]+)"
    r"(?:/tree/(?P<rev>[^/]+)(?:/(?P<subfolder>.+?))?)?/?$"
)
_HF_BUCKET_URL_PATTERN = re.compile(
    r"^https?://(?:www\.)?huggingface\.co/buckets/"
    r"(?P<repo>[^/]+/[^/]+)"
    r"(?:/tree/(?P<prefix>.+?))?/?$"
)


@dataclass(frozen=True)
class _HFLocation:
    repo_id: str
    revision: str
    subfolder: str | None
    is_bucket: bool = False


def _parse_hf_url(url: str) -> _HFLocation:
    m = _HF_BUCKET_URL_PATTERN.match(url.strip())
    if m is not None:
        prefix = m.group("prefix")
        if prefix is not None:
            prefix = prefix.rstrip("/") or None
        return _HFLocation(
            repo_id=m.group("repo"), revision="",
            subfolder=prefix, is_bucket=True,
        )
    m = _HF_URL_PATTERN.match(url.strip())
    if m is None:
        raise ValueError(f"could not parse HuggingFace URL {url!r}")
    return _HFLocation(
        repo_id=m.group("repo"),
        revision=m.group("rev") or "main",
        subfolder=m.group("subfolder"),
    )


def _download_bucket_prefix(location: _HFLocation) -> Path:
    """Mirror an HF storage-bucket prefix into the local HF cache.

    Faithful port of ``marinfold.registry._download_bucket_prefix``
    — including ``?recursive=true`` on the tree listing and URL
    encoding on the resolve URL.
    """
    from urllib.parse import quote
    from huggingface_hub import constants
    from huggingface_hub.constants import HF_HUB_CACHE
    from huggingface_hub.file_download import http_get
    from huggingface_hub.utils import (
        build_hf_headers, get_session, hf_raise_for_status,
    )

    endpoint = constants.ENDPOINT
    prefix = location.subfolder
    encoded_prefix = f"/{quote(prefix, safe='')}" if prefix else ""
    tree_url = f"{endpoint}/api/buckets/{location.repo_id}/tree{encoded_prefix}"
    resp = get_session().get(
        tree_url, params={"recursive": "true"}, headers=build_hf_headers(),
    )
    hf_raise_for_status(resp)
    entries = resp.json()
    files = [
        (e["path"], int(e["size"]))
        for e in entries
        if isinstance(e, dict) and e.get("type") == "file"
    ]
    if not files:
        raise FileNotFoundError(
            f"no files under bucket {location.repo_id!r} prefix {prefix!r}"
        )

    local_root = Path(HF_HUB_CACHE) / "buckets" / location.repo_id
    for path, size in files:
        dest = local_root / path
        if dest.is_file() and dest.stat().st_size == size:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(f"{dest.name}.incomplete")
        try:
            with tmp.open("wb") as fh:
                http_get(
                    f"{endpoint}/buckets/{location.repo_id}/resolve/"
                    f"{quote(path, safe='/')}",
                    fh,
                    headers=build_hf_headers(),
                    expected_size=size,
                    displayed_filename=dest.name,
                )
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
        tmp.replace(dest)
    result = (
        local_root / location.subfolder
        if location.subfolder is not None
        else local_root
    )
    if not result.is_dir():
        raise FileNotFoundError(
            f"expected bucket prefix at {result} but missing"
        )
    return result.resolve()


def _resolve_model_path(models_yaml_path: Path, nickname: str) -> str:
    """Return the local HF snapshot dir for the named MODELS.yaml entry.

    Supports both the regular HF tree URLs and the storage-bucket URLs
    (``huggingface.co/buckets/<org>/<bucket>/tree/<prefix>``) used by
    the 1.5B entry. Re-runnable: HF Hub's ``snapshot_download`` is
    content-addressed, and the bucket mirror keeps files keyed by
    ``(path, size)`` — unchanged files don't redownload.
    """
    entries = yaml.safe_load(models_yaml_path.read_text())
    matched = [e for e in entries if e.get("nickname") == nickname]
    if not matched:
        raise ValueError(f"no entry for {nickname!r} in {models_yaml_path}")
    location = _parse_hf_url(matched[0]["url"])
    if location.is_bucket:
        return str(_download_bucket_prefix(location))

    from huggingface_hub import snapshot_download
    allow = [f"{location.subfolder}/*"] if location.subfolder else None
    local = snapshot_download(
        repo_id=location.repo_id,
        revision=location.revision,
        allow_patterns=allow,
    )
    return str(Path(local) / location.subfolder) if location.subfolder else local


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
        (representative_atom_name(residue_names[i - 1]),
         representative_atom_name(residue_names[j - 1]))
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


# --- Output sink: local dir OR gs:// URI -------------------------------
#
# iris workers don't write back to bizon's filesystem, so the eval has
# to push results to GCS as it goes. To avoid pulling in gcsfs/fsspec
# as deps, we write each protein's outputs to a tempdir locally and
# then ``gsutil cp`` them to the configured prefix. Provenance lookups
# (idempotency check) use ``gsutil cat`` over the prov.json sidecar.

def _is_gcs(uri: str) -> bool:
    return uri.startswith("gs://")


def _gsutil(*args: str, capture: bool = False) -> subprocess.CompletedProcess:
    cmd = ["gsutil", "-q", *args]
    return subprocess.run(
        cmd, check=True, capture_output=capture, text=True,
    )


def _read_remote_provenance(uri: str) -> dict[str, Any] | None:
    """Return the parsed prov dict at ``uri`` (gs:// or local), or None."""
    if _is_gcs(uri):
        try:
            proc = subprocess.run(
                ["gsutil", "-q", "cat", uri],
                check=True, capture_output=True, text=True,
            )
        except subprocess.CalledProcessError:
            return None
        try:
            return json.loads(proc.stdout)
        except ValueError:
            return None
    p = Path(uri)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def _output_complete(out_base: str, stem: str, n: int, *, model_nickname: str) -> bool:
    """Idempotency check: is ``<out_base>/<stem>/provenance.json`` for this nickname?

    We key only on the provenance file (not the distogram.npz shape) for
    GCS sinks — fetching the .npz over gsutil just to check a shape is
    wasteful. Local sinks use the same convention for symmetry.
    """
    sep = "/" if _is_gcs(out_base) else os.sep
    prov_uri = f"{out_base.rstrip('/')}{sep}{stem}{sep}provenance.json"
    prov = _read_remote_provenance(prov_uri)
    if prov is None:
        return False
    return (
        prov.get("model_nickname") == model_nickname
        and int(prov.get("n_residues", -1)) == n
    )


def _write_output(
    out_base: str, stem: str, *,
    probs: np.ndarray, provenance: dict[str, Any],
) -> str:
    """Save distogram.npz + provenance.json under ``<out_base>/<stem>/``.

    Returns the destination URI (gs://... or local path).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        npz_path = tmp / "distogram.npz"
        np.savez_compressed(npz_path, probs=probs)
        prov_path = tmp / "provenance.json"
        prov_path.write_text(json.dumps(provenance, indent=2) + "\n")
        if _is_gcs(out_base):
            dest = f"{out_base.rstrip('/')}/{stem}/"
            _gsutil("cp", str(npz_path), dest)
            _gsutil("cp", str(prov_path), dest)
            return dest
        dest_dir = Path(out_base) / stem
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(npz_path, dest_dir / "distogram.npz")
        shutil.copy2(prov_path, dest_dir / "provenance.json")
        return str(dest_dir)


_DEFAULT_OUT_GCS = (
    "gs://marin-us-east5/protein-structure/MarinFold/exp26/"
    "protein-contacts-1_5b-distance-masked-70f8f5-step-49999-foldbench-monomers"
)


def run(
    *,
    protenix_dir: Path,
    out_base: str,
    model_url: str,
    model_nickname: str,
    limit: int | None = None,
    batch_size: int = 128,
) -> int:
    """Drive the full sweep. Returns the number of NEW proteins scored.

    Idempotent: skips proteins whose ``<out_base>/<stem>/provenance.json``
    already records the requested ``model_nickname``.
    """
    manifest = _read_manifest(protenix_dir / "manifest.csv")
    if limit is not None:
        manifest = manifest[:limit]

    # Pre-flight: skip vLLM load entirely if everything is already done.
    pending: list[dict[str, Any]] = []
    for entry in manifest:
        stem = entry["stem"]
        n = int(entry["n_residues"])
        if _output_complete(out_base, stem, n, model_nickname=model_nickname):
            print(f"skip {stem}: provenance.json already records {model_nickname} (N={n})")
            continue
        pending.append(entry)
    if not pending:
        print(f"All {len(manifest)} proteins already complete for {model_nickname}.")
        return 0

    location = _parse_hf_url(model_url)
    print(
        f"Resolving model: nickname={model_nickname} url={model_url} -> "
        f"{'bucket' if location.is_bucket else 'tree'}({location.repo_id})"
    )
    if location.is_bucket:
        model_path = str(_download_bucket_prefix(location))
    else:
        from huggingface_hub import snapshot_download
        allow = [f"{location.subfolder}/*"] if location.subfolder else None
        local = snapshot_download(
            repo_id=location.repo_id,
            revision=location.revision,
            allow_patterns=allow,
        )
        model_path = (
            str(Path(local) / location.subfolder)
            if location.subfolder else local
        )

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
        n_pairs = seq.n_residues * (seq.n_residues - 1) // 2
        provenance = {
            "stem": stem,
            "n_residues": seq.n_residues,
            "n_pairs": n_pairs,
            "model_nickname": model_nickname,
            "model_url": model_url,
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
        }
        dest = _write_output(out_base, stem, probs=probs, provenance=provenance)
        n_written += 1
        print(
            f"wrote {dest} ({seq.n_residues} residues, {n_pairs} pairs, "
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
        default=_DEFAULT_OUT_GCS,
        help=(
            "Where to save per-protein distogram.npz + provenance.json. "
            "Either a local directory path or a gs:// URI. Default points "
            "at this experiment's canonical GCS prefix."
        ),
    )
    parser.add_argument(
        "--model-url",
        required=True,
        help=(
            "Full HuggingFace URL for the checkpoint (regular tree URL or "
            "bucket URL). Pass-through from MODELS.yaml; this script does "
            "not read MODELS.yaml so the iris worker is self-contained."
        ),
    )
    parser.add_argument(
        "--model-nickname",
        required=True,
        help=(
            "Short identifier for the model (e.g. '1.5B'). Used only for "
            "provenance.json idempotency and the timing CSV's "
            "model_nickname column. Pass-through from MODELS.yaml."
        ),
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
        out_base=args.out,
        model_url=args.model_url,
        model_nickname=args.model_nickname,
        limit=args.limit,
        batch_size=args.batch_size,
    )
    print(f"Done. Wrote {n} new distograms.")


if __name__ == "__main__":
    main()
