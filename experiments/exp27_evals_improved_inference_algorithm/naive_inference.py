# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Single-process MarinFold 1B naive distogram readout — the baseline algorithm.

This is a **library module + CLI** wrapping the exact same algorithm
exp20's ``run_1b_eval.py`` uses: read out the full 64-bin distance
distribution from the model at every (i, j) pair in parallel, no
seeded contacts, no rollouts, no inference-time search.

We copy-fork it here (rather than path-deping exp20) so the experiment
dir is self-contained per the AGENTS.md convention. The two files
should stay byte-equivalent in inference logic — if you fix a bug in
one, fix it in the other.

The library surface (used by ``run_baseline.py`` and future
algorithm scripts):

- :func:`load_runtime` — one-time vLLM load + tokenizer setup on the
  GPU bound by ``CUDA_VISIBLE_DEVICES``. Returns a frozen ``Runtime``
  that subsequent ``predict_one`` calls reuse.
- :func:`predict_one` — run the naive algorithm on one stem; write
  ``outputs/<stem>/{distogram.npz, provenance.json}``. Returns
  per-protein elapsed seconds.

The CLI invocation (single stem) is mostly for debugging — the main
driver is ``run_baseline.py`` which keeps one ``Runtime`` per GPU
alive across many ``predict_one`` calls.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# exp1 path dep: we use its NAME constant (the document-structure
# header token) to stay byte-identical with the training-time
# tokenization. The exp1 dir is a script dir, not a python package,
# so we add it to sys.path before importing.
_THIS = Path(__file__).resolve().parent
_EXP1 = _THIS.parent / "exp1_document_structures_contacts_and_distances_v1"
if str(_EXP1) not in sys.path:
    sys.path.insert(0, str(_EXP1))

from vocab import NAME  # noqa: E402 — needs sys.path insertion above

from canonical_sequence import (  # noqa: E402
    read_canonical_sequence,
    representative_atom_name,
)


# Distance-bin constants — must match exp1's tokenizer.
DISTANCE_BIN_WIDTH_A = 0.5
NUM_DISTANCE_BINS = 64
DISTANCE_MAX_A = NUM_DISTANCE_BINS * DISTANCE_BIN_WIDTH_A  # 32.0 Å
BIN_MIDPOINTS = np.array(
    [(k + 1) * DISTANCE_BIN_WIDTH_A - DISTANCE_BIN_WIDTH_A / 2
     for k in range(NUM_DISTANCE_BINS)],
    dtype=np.float32,
)


@dataclass(frozen=True)
class Runtime:
    """One vLLM instance + everything keyed off the tokenizer.

    Created once per worker process by :func:`load_runtime` and then
    passed into :func:`predict_one` for each protein.
    """

    llm: Any              # vllm.LLM
    tokenizer: Any
    distance_token_ids: list[int]
    model_nickname: str
    model_path: str
    model_load_seconds: float
    hardware: dict[str, Any]


def resolve_model_path(models_yaml_path: Path, nickname: str) -> str:
    """Return the local HF snapshot dir for the named model entry."""
    from huggingface_hub import snapshot_download

    entries = yaml.safe_load(models_yaml_path.read_text())
    matched = [e for e in entries if e.get("nickname") == nickname]
    if not matched:
        raise ValueError(f"no entry for {nickname!r} in {models_yaml_path}")
    entry = matched[0]
    url = entry["url"]
    prefix = "https://huggingface.co/"
    if not url.startswith(prefix):
        raise ValueError(f"unexpected model URL {url!r}")
    rest = url[len(prefix):].split("/")
    if len(rest) < 2:
        raise ValueError(f"unexpected model URL {url!r}")
    repo_id = "/".join(rest[:2])
    subdir = rest[4] if len(rest) > 4 and rest[2] == "tree" else None
    local = snapshot_download(
        repo_id=repo_id,
        allow_patterns=[f"{subdir}/*"] if subdir else None,
    )
    return str(Path(local) / subdir) if subdir else local


def _hardware_info() -> dict[str, Any]:
    """Snapshot of host + GPU. Same schema exp20 uses."""
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


def _resolve_distance_token_ids(tokenizer) -> list[int]:
    """The 64 distance-bin token IDs in bin order (k=0 → <d0.5>)."""
    ids = []
    for k in range(NUM_DISTANCE_BINS):
        tok = f"<d{(k+1)*DISTANCE_BIN_WIDTH_A:.1f}>"
        enc = tokenizer.encode(tok, add_special_tokens=False)
        if len(enc) != 1:
            raise ValueError(f"bad encoding for {tok}: {enc!r}")
        ids.append(int(enc[0]))
    if len(set(ids)) != NUM_DISTANCE_BINS:
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


def load_runtime(
    *,
    model_nickname: str,
    models_yaml: Path,
    dtype: str = "auto",
    gpu_memory_utilization: float = 0.85,
    max_logprobs: int = 128,
    max_model_len: int = 8192,
) -> Runtime:
    """Load vLLM on the current GPU (set by ``CUDA_VISIBLE_DEVICES``).

    Caller is responsible for setting ``CUDA_VISIBLE_DEVICES`` BEFORE
    this function imports vLLM — vLLM bakes the visible device set
    into its worker config at startup, so changing it later doesn't
    help.
    """
    from vllm import LLM
    model_path = resolve_model_path(models_yaml, model_nickname)
    t_start = time.time()
    llm = LLM(
        model=str(model_path),
        dtype=dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=True,
        trust_remote_code=True,
        max_logprobs=max_logprobs,
        max_model_len=max_model_len,
    )
    tokenizer = llm.get_tokenizer()
    distance_token_ids = _resolve_distance_token_ids(tokenizer)
    model_load_seconds = time.time() - t_start
    return Runtime(
        llm=llm,
        tokenizer=tokenizer,
        distance_token_ids=distance_token_ids,
        model_nickname=model_nickname,
        model_path=model_path,
        model_load_seconds=model_load_seconds,
        hardware=_hardware_info(),
    )


def predict_distogram(
    *,
    rt: Runtime,
    residue_names: tuple[str, ...],
    batch_size: int = 128,
    top_k_logprobs: int = 128,
) -> np.ndarray:
    """Naive distogram readout — the baseline algorithm.

    Returns a symmetric ``[N, N, 64]`` probability matrix. One forward
    pass per (i, j) pair via vLLM's prefix cache; the trunk
    ``<contacts-and-distances-v1> ... <begin_statements>`` is shared
    across all pairs, and only the 5-token tail changes per pair.

    Identical to exp20's ``_predict_distogram`` — kept here so the
    baseline row is reproducibly the same algorithm.
    """
    from vllm import SamplingParams, TokensPrompt

    n = len(residue_names)
    base_tokens = _build_base_prompt(residue_names)
    base_ids = _encode_tokens(rt.tokenizer, base_tokens)

    distance_id_set = set(rt.distance_token_ids)
    bin_of = {tid: k for k, tid in enumerate(rt.distance_token_ids)}

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
        tail = _encode_tokens(rt.tokenizer, [
            "<distance>", f"<p{i}>", f"<p{j}>", f"<{a_i}>", f"<{a_j}>",
        ])
        prompts.append(TokensPrompt(prompt_token_ids=base_ids + tail))

    sampling = SamplingParams(
        temperature=1.0, top_p=1.0, top_k=-1,
        max_tokens=1, logprobs=top_k_logprobs, n=1,
    )

    probs = np.zeros((n, n, NUM_DISTANCE_BINS), dtype=np.float32)
    for chunk_start in range(0, len(prompts), batch_size):
        chunk_prompts = prompts[chunk_start : chunk_start + batch_size]
        outputs = rt.llm.generate(chunk_prompts, sampling, use_tqdm=False)
        for offset, gen in enumerate(outputs):
            lp_dict = gen.outputs[0].logprobs[0] if gen.outputs[0].logprobs else {}
            row = np.zeros(NUM_DISTANCE_BINS, dtype=np.float32)
            for tok_id, lp in lp_dict.items():
                tid = int(tok_id)
                if tid in distance_id_set:
                    row[bin_of[tid]] = math.exp(float(lp.logprob))
            total = float(row.sum())
            if total > 0:
                row /= total
            i, j = pairs[chunk_start + offset]
            probs[i - 1, j - 1, :] = row
            probs[j - 1, i - 1, :] = row
    return probs


def predict_one(
    *,
    rt: Runtime,
    stem: str,
    protenix_dir: Path,
    out_dir: Path,
    batch_size: int = 128,
    algorithm: str = "baseline_naive",
) -> float:
    """Predict the distogram for one stem, save to ``out_dir/<stem>/``.

    Returns the per-protein inference elapsed seconds (excludes model
    load, which is amortized over many proteins by the worker pool).
    """
    gt_cif = protenix_dir / "gt" / f"{stem}.cif"
    seq = read_canonical_sequence(gt_cif)
    t_start = time.time()
    probs = predict_distogram(
        rt=rt, residue_names=seq.residue_names, batch_size=batch_size,
    )
    elapsed = time.time() - t_start

    out_path = out_dir / stem / "distogram.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, probs=probs)

    n_pairs = seq.n_residues * (seq.n_residues - 1) // 2
    (out_path.parent / "provenance.json").write_text(json.dumps({
        "stem": stem,
        "n_residues": seq.n_residues,
        "n_pairs": n_pairs,
        "algorithm": algorithm,
        "model_nickname": rt.model_nickname,
        "model_path": rt.model_path,
        "atom_convention": "CB-CB (CA for GLY/UNK)",
        "bin_scheme": {
            "min_A": 0.0,
            "max_A": DISTANCE_MAX_A,
            "n_bins": NUM_DISTANCE_BINS,
            "midpoints_A": BIN_MIDPOINTS.tolist(),
        },
        "elapsed_seconds": round(elapsed, 3),
        "model_load_seconds": round(rt.model_load_seconds, 3),
        "batch_size": batch_size,
        "hardware": rt.hardware,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, indent=2) + "\n")
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stem", required=True, help="Protein stem (e.g. 8b6e_A).")
    parser.add_argument(
        "--protenix-dir",
        type=Path,
        default=_THIS / "protenix_data" / "data" / "protenix-foldbench-monomers",
    )
    parser.add_argument(
        "--out", type=Path, default=_THIS / "outputs",
    )
    parser.add_argument("--model", default="1B")
    parser.add_argument(
        "--models-yaml",
        type=Path,
        default=_THIS.parent.parent / "MODELS.yaml",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--dtype",
        default="auto",
        help=(
            "vLLM dtype. 'auto' = upstream default (bf16 where available). "
            "V100 (compute 7.0) doesn't support bf16 — use 'half' (fp16) "
            "or 'float32'."
        ),
    )
    args = parser.parse_args()
    rt = load_runtime(
        model_nickname=args.model, models_yaml=args.models_yaml,
        dtype=args.dtype,
    )
    elapsed = predict_one(
        rt=rt,
        stem=args.stem,
        protenix_dir=args.protenix_dir,
        out_dir=args.out,
        batch_size=args.batch_size,
    )
    print(f"{args.stem}: {elapsed:.1f} s")


if __name__ == "__main__":
    main()
