"""Shared vLLM-backed inference helpers for both eval notebooks.

Both `eval_notebook.ipynb` (zero-shot heatmaps) and
`contact_seeding_search.ipynb` (greedy seeded-contact search)
need the same primitives:

- load vLLM with our standard config
- resolve the 64 `<d_X.X>` token IDs in the tokenizer
- build the base prompt (`<task> <begin_sequence> <AAs>
  <begin_statements> [seeded contacts]`)
- query the model at a list of (i, j) pairs and recover an expected
  CA-CA distance per pair from the renormalized top-K logprobs

The split is here so the two notebooks stay short and consistent.
The functions are stateless: caller passes the loaded `llm` /
`tokenizer` / `distance_token_ids` in, gets numpy back.
"""

import math
import sys
import time
from pathlib import Path

import numpy as np

# Distance-bin constants (mirrored from exp1's inference.py).
DISTANCE_BIN_WIDTH_A = 0.5
NUM_DISTANCE_BINS = 64
DISTANCE_MAX_A = NUM_DISTANCE_BINS * DISTANCE_BIN_WIDTH_A  # 32.0
BIN_MIDPOINTS = np.array(
    [(k + 1) * DISTANCE_BIN_WIDTH_A - DISTANCE_BIN_WIDTH_A / 2
     for k in range(NUM_DISTANCE_BINS)],
    dtype=np.float32,
)


def add_exp1_to_path():
    """Add the exp1 directory to sys.path so `parse`, `vocab` can be imported."""
    here = Path(__file__).resolve().parent
    exp1 = here.parent / "exp1_document_structures_contacts_and_distances_v1"
    if str(exp1) not in sys.path:
        sys.path.insert(0, str(exp1))


def load_vllm(model_local_path):
    """Load the model into vLLM with our standard config + return tokenizer."""
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


def resolve_distance_token_ids(tokenizer):
    """Return the 64 distance-bin token IDs in bin order (k=0 → `<d0.5>`)."""
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


def encode_token_strs(tokenizer, token_strs):
    """1:1 WordLevel encoding for `<...>` tokens."""
    ids = tokenizer.encode(" ".join(token_strs), add_special_tokens=False)
    if len(ids) != len(token_strs):
        raise ValueError(
            f"tokenizer 1:1 broke: {token_strs[:5]!r} -> {ids[:5]!r}"
        )
    return [int(x) for x in ids]


def build_base_prompt_tokens(parsed, seeded_contacts):
    """`<task> <begin_sequence> <AAs> <begin_statements> [seeded contacts]`.

    Args:
        parsed: a `parse.ParsedStructure`.
        seeded_contacts: list of contact entries. Each entry is
            either ``(i, j)`` (2-tuple — assumed long-range, kept
            for back-compat) or ``(type_token, i, j)`` where
            ``type_token`` is one of ``"<long-range-contact>"``,
            ``"<medium-range-contact>"``, ``"<short-range-contact>"``.
    """
    from vocab import NAME  # noqa: F401 — needs add_exp1_to_path()

    toks = [f"<{NAME}>", "<begin_sequence>"]
    toks.extend(f"<{r.name}>" for r in parsed.residues)
    toks.append("<begin_statements>")
    for entry in seeded_contacts:
        if len(entry) == 2:
            i, j = entry
            type_tok = "<long-range-contact>"
        elif len(entry) == 3:
            type_tok, i, j = entry
        else:
            raise ValueError(f"seeded_contacts entry must be (i,j) or (type,i,j); got {entry!r}")
        toks.extend([type_tok, f"<p{i}>", f"<p{j}>"])
    return toks


def gt_long_range_contacts(parsed, cutoff_angstrom=8.0, sep_min=24):
    """1-indexed (i, j) pairs with CB-CB (or CA for GLY) ≤ cutoff, sep ≥ sep_min.

    Matches the training-data `<long-range-contact>` definition.
    """
    from parse import cb_or_ca_position, euclidean  # noqa: F401

    cb = {}
    for r in parsed.residues:
        p = cb_or_ca_position(r)
        if p is not None:
            cb[r.index] = p
    out = []
    indices = sorted(cb)
    for ii in range(len(indices)):
        for jj in range(ii + 1, len(indices)):
            i, j = indices[ii], indices[jj]
            if j - i < sep_min:
                continue
            if euclidean(cb[i], cb[j]) <= cutoff_angstrom:
                out.append((i, j))
    out.sort()
    return out


# Contact-range definitions matching the protein-docs training data:
# long-range: sep >= 24, medium: 12 <= sep < 24, short: 6 <= sep < 12.
# All use the CB-CB ≤ 8 Å cutoff (CA for GLY / missing CB).
_CONTACT_TYPE_RANGES = [
    ("<long-range-contact>", 24, None),
    ("<medium-range-contact>", 12, 24),
    ("<short-range-contact>", 6, 12),
]


def gt_contacts_all_ranges(parsed, cutoff_angstrom=8.0):
    """All GT contacts (long + medium + short) as (type_token, i, j).

    Same CB-CB ≤ 8 Å cutoff as the training data, partitioned by
    sequence separation. Returned in a deterministic order:
    long-range first (matching the training convention that 100% of
    long-range contacts come before medium/short in rank), then
    medium, then short, each block sorted by (i, j).
    """
    from parse import cb_or_ca_position, euclidean  # noqa: F401

    cb = {}
    for r in parsed.residues:
        p = cb_or_ca_position(r)
        if p is not None:
            cb[r.index] = p
    indices = sorted(cb)

    buckets: dict[str, list[tuple[int, int]]] = {tok: [] for tok, _, _ in _CONTACT_TYPE_RANGES}
    for ii in range(len(indices)):
        for jj in range(ii + 1, len(indices)):
            i, j = indices[ii], indices[jj]
            sep = j - i
            if euclidean(cb[i], cb[j]) > cutoff_angstrom:
                continue
            for tok, lo, hi in _CONTACT_TYPE_RANGES:
                if sep >= lo and (hi is None or sep < hi):
                    buckets[tok].append((i, j))
                    break
    out: list[tuple[str, int, int]] = []
    for tok, _, _ in _CONTACT_TYPE_RANGES:
        for i, j in sorted(buckets[tok]):
            out.append((tok, i, j))
    return out


def predict_at_pairs(
    *,
    llm,
    tokenizer,
    parsed,
    pairs,
    seeded_contacts,
    distance_token_ids,
    query_atom="CA",
    batch_size=128,
    top_k_logprobs=128,
):
    """Return expected distance (Å) for each (i, j) in `pairs`.

    Shared prefix per call: vLLM's prefix cache reuses the
    `base_ids` KV-cache across the whole batch, so per-pair cost
    is dominated by the 5-token tail forward pass.
    """
    from vllm import SamplingParams, TokensPrompt

    base_tokens = build_base_prompt_tokens(parsed, seeded_contacts)
    base_ids = encode_token_strs(tokenizer, base_tokens)
    distance_id_set = set(distance_token_ids)
    bin_of = {tid: k for k, tid in enumerate(distance_token_ids)}

    prompts = []
    for i, j in pairs:
        tail = encode_token_strs(tokenizer, [
            "<distance>", f"<p{i}>", f"<p{j}>",
            f"<{query_atom}>", f"<{query_atom}>",
        ])
        prompts.append(TokensPrompt(prompt_token_ids=base_ids + tail))

    sampling = SamplingParams(
        temperature=1.0, top_p=1.0, top_k=-1,
        max_tokens=1, logprobs=top_k_logprobs, n=1,
    )

    out = np.full(len(pairs), np.nan, dtype=np.float32)
    for chunk_start in range(0, len(prompts), batch_size):
        chunk_prompts = prompts[chunk_start : chunk_start + batch_size]
        outputs = llm.generate(chunk_prompts, sampling, use_tqdm=False)
        for offset, gen in enumerate(outputs):
            lp_dict = gen.outputs[0].logprobs[0] if gen.outputs[0].logprobs else {}
            row = np.zeros(NUM_DISTANCE_BINS, dtype=np.float32)
            for tok_id, lp in lp_dict.items():
                tid = int(tok_id)
                if tid in distance_id_set:
                    row[bin_of[tid]] = math.exp(float(lp.logprob))
            total = float(row.sum())
            if total <= 0:
                continue
            row /= total
            out[chunk_start + offset] = float((row * BIN_MIDPOINTS).sum())
    return out


def predict_distance_matrix(
    *,
    llm,
    tokenizer,
    parsed,
    seeded_contacts,
    distance_token_ids,
    query_atom="CA",
    batch_size=128,
    top_k_logprobs=128,
):
    """Full N×N expected-distance matrix for a single protein.

    Queries pairs (i, j) with i < j and mirrors. Diagonal is 0,
    unqueried entries are NaN.
    """
    n = len(parsed.residues)
    pairs = [(i, j) for i in range(1, n + 1) for j in range(i + 1, n + 1)]
    flat = predict_at_pairs(
        llm=llm,
        tokenizer=tokenizer,
        parsed=parsed,
        pairs=pairs,
        seeded_contacts=seeded_contacts,
        distance_token_ids=distance_token_ids,
        query_atom=query_atom,
        batch_size=batch_size,
        top_k_logprobs=top_k_logprobs,
    )
    out = np.full((n, n), np.nan, dtype=np.float32)
    np.fill_diagonal(out, 0.0)
    for (i, j), val in zip(pairs, flat, strict=True):
        out[i - 1, j - 1] = val
        out[j - 1, i - 1] = val
    return out


def sample_ca_pairs(parsed, n_pairs, seed):
    """Deterministically sample up to `n_pairs` unique CA-CA residue pairs (i<j).

    Returns list of (i, j) 1-indexed.
    """
    from parse import atom_position  # noqa: F401

    valid = sorted(r.index for r in parsed.residues
                   if atom_position(r, "CA") is not None)
    n_valid = len(valid)
    if n_valid < 2:
        return []
    max_pairs = n_valid * (n_valid - 1) // 2
    target = min(n_pairs, max_pairs)

    rng = np.random.default_rng(seed)
    seen = set()
    out = []
    attempts = 0
    while len(out) < target and attempts < target * 50:
        attempts += 1
        a, b = rng.choice(valid, size=2, replace=False)
        i, j = (int(a), int(b)) if a < b else (int(b), int(a))
        key = (i, j)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def ca_distance_matrix(parsed):
    """N×N CA-CA distance matrix in Å. NaN at any residue missing CA."""
    from parse import atom_position  # noqa: F401

    n = len(parsed.residues)
    pts = np.array(
        [atom_position(r, "CA") or (np.nan,) * 3 for r in parsed.residues],
        dtype=np.float32,
    )
    return np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)


def mae_on_pairs(parsed, pairs, predicted, query_atom="CA"):
    """MAE on `pairs` between `predicted` (1D in pair order) and GT coords.

    GT is computed from the parsed structure on the fly. Pairs where
    GT is non-finite or > DISTANCE_MAX_A are skipped (model can't
    say anything past the saturated bin).
    """
    from parse import atom_position  # noqa: F401

    n = len(parsed.residues)
    coords = [atom_position(r, query_atom) for r in parsed.residues]
    abs_errs = []
    for (i, j), pred in zip(pairs, predicted, strict=True):
        if not math.isfinite(pred):
            continue
        ci, cj = coords[i - 1], coords[j - 1]
        if ci is None or cj is None:
            continue
        gt = math.sqrt(sum((a - b) ** 2 for a, b in zip(ci, cj)))
        if gt > DISTANCE_MAX_A:
            continue
        abs_errs.append(abs(pred - gt))
    if not abs_errs:
        return float("nan"), 0
    return float(np.mean(abs_errs)), len(abs_errs)
