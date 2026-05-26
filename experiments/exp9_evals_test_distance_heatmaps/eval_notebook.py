# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # exp9 — zero-shot distance heatmap eval (10 random AFDB test proteins, `1B` model)
#
# What this notebook does:
#
# 1. Deterministically picks 10 AFDB entries from
#    `timodonnell/protein-docs` (`contacts-and-distances-v1-5x`,
#    `split=test`) — that split is held out by structural cluster
#    so the `1B` model has never seen these proteins.
# 2. Downloads each entry's `model_v6.cif` from the AlphaFold
#    Database and parses it with the exp1 toolkit.
# 3. Loads the `1B` model via vLLM (entry from top-level
#    [`MODELS.yaml`](../../MODELS.yaml)).
# 4. Runs zero-shot CA-CA inference for every pair (i, j), i < j,
#    batched with a shared prefix per protein (matches the marin
#    `eval_protein_distogram.py` pattern).
# 5. Plots GT vs. predicted vs. residual heatmaps per protein,
#    saved to `plots/`. Per-protein and macro MAE land in `data/`.
#
# Re-run top-to-bottom with `Restart & Run All`. Sampling is seeded,
# so the same 10 entries come back every time.

# %%
import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

EXP_DIR = Path.cwd()
REPO_ROOT = EXP_DIR.parents[1]
EXP1_DIR = REPO_ROOT / "experiments" / "exp1_document_structures_contacts_and_distances_v1"
sys.path.insert(0, str(EXP1_DIR))

from parse import parse_structure, atom_position
from vocab import NAME

from select_test_proteins import select_test_proteins, download_cif
from build_summary import save_plot_with_meta

print("repo root:", REPO_ROOT)
print("exp1 dir:", EXP1_DIR)

# %% [markdown]
# ## Resolve the `1B` model from `MODELS.yaml`
#
# `MODELS.yaml` is the source of truth for what high-quality
# trained models exist and what doc structures they support.

# %%
with (REPO_ROOT / "MODELS.yaml").open() as fh:
    models = yaml.safe_load(fh)

MODEL_NICK = "1B"
model_entry = next(m for m in models if m["nickname"] == MODEL_NICK)
assert "contacts-and-distances-v1" in model_entry["document_structures"], model_entry
MODEL_HF_URL = model_entry["url"]
# Convert HF tree URL → repo_id + revision for snapshot_download.
# tree/main/<subfolder> → repo_id is owner/repo, subfolder picked
# from the path tail.
_, _, tail = MODEL_HF_URL.partition("huggingface.co/")
parts = tail.split("/")
MODEL_REPO = "/".join(parts[:2])
MODEL_SUBFOLDER = parts[4] if len(parts) > 4 and parts[2] == "tree" else None
print(f"using model: {MODEL_NICK} = {MODEL_REPO}/{MODEL_SUBFOLDER}")

# %% [markdown]
# ## Select 10 test proteins (deterministic)

# %%
SEED = 0
N_PROTEINS = 10
MAX_SEQ_LEN = 150   # keeps the run snappy on a single 24GB GPU

specs = select_test_proteins(n=N_PROTEINS, seed=SEED, max_seq_len=MAX_SEQ_LEN)
for s in specs:
    print(f"  {s.entry_id:<24} len={s.seq_len:<4} -> {s.cif_url}")

# %% [markdown]
# ## Download + parse each AFDB structure
#
# Cached on disk so re-runs are free.

# %%
CACHE_DIR = EXP_DIR / "data" / "afdb_cache"

structures = []
for spec in specs:
    cif_path = download_cif(spec, CACHE_DIR)
    parsed = parse_structure(cif_path)
    structures.append((spec, parsed))
    print(f"  {spec.entry_id}: {len(parsed.residues)} residues parsed ({cif_path.stat().st_size//1024} KB)")

# %% [markdown]
# ## Compute ground-truth CA-CA distance matrices

# %%
def ca_distance_matrix(parsed) -> np.ndarray:
    """N×N CA-CA distance matrix in Å. NaN at any residue missing CA."""
    n = len(parsed.residues)
    coords = [atom_position(r, "CA") for r in parsed.residues]
    pts = np.array([c if c is not None else (np.nan,)*3 for c in coords], dtype=np.float32)
    diff = pts[:, None, :] - pts[None, :, :]
    return np.linalg.norm(diff, axis=-1)


gt_matrices = {}
for spec, parsed in structures:
    m = ca_distance_matrix(parsed)
    gt_matrices[spec.entry_id] = m
    print(f"  {spec.entry_id}: GT median CA-CA = {np.nanmedian(m):.2f} Å, max = {np.nanmax(m):.2f} Å")

# %% [markdown]
# ## Stage the `1B` model locally + load vLLM

# %%
import os
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

from huggingface_hub import snapshot_download

# allow_patterns keeps us from downloading every sibling model in the repo.
MODEL_LOCAL = Path(snapshot_download(
    repo_id=MODEL_REPO,
    allow_patterns=[f"{MODEL_SUBFOLDER}/*"] if MODEL_SUBFOLDER else None,
))
if MODEL_SUBFOLDER:
    MODEL_LOCAL = MODEL_LOCAL / MODEL_SUBFOLDER
print("model local path:", MODEL_LOCAL)
assert (MODEL_LOCAL / "config.json").exists(), MODEL_LOCAL

# %%
from vllm import LLM, SamplingParams, TokensPrompt

llm = LLM(
    model=str(MODEL_LOCAL),
    dtype="bfloat16",
    gpu_memory_utilization=0.85,
    enforce_eager=True,
    trust_remote_code=True,
    max_logprobs=128,
    max_model_len=8192,
)
tokenizer = llm.get_tokenizer()
print("vocab size:", len(tokenizer))

# %% [markdown]
# ## Resolve the 64 distance-bin token IDs
#
# Bin k (0..63) maps to token `<d_{(k+1)*0.5}>`. At inference we
# read the top-K next-token logprobs, mask to these 64 ids,
# renormalize, and compute an expected distance from the bin
# midpoints.

# %%
DISTANCE_BIN_WIDTH_A = 0.5
NUM_DISTANCE_BINS = 64
DISTANCE_MAX_A = NUM_DISTANCE_BINS * DISTANCE_BIN_WIDTH_A
BIN_MIDPOINTS = np.array(
    [(k + 1) * DISTANCE_BIN_WIDTH_A - DISTANCE_BIN_WIDTH_A / 2
     for k in range(NUM_DISTANCE_BINS)],
    dtype=np.float32,
)

def resolve_distance_token_ids(tok):
    ids = []
    for k in range(NUM_DISTANCE_BINS):
        s = f"<d{(k+1)*DISTANCE_BIN_WIDTH_A:.1f}>"
        enc = tok.encode(s, add_special_tokens=False)
        if len(enc) != 1:
            raise ValueError(f"bad encoding for {s}: {enc!r}")
        ids.append(int(enc[0]))
    if len(set(ids)) != NUM_DISTANCE_BINS:
        raise ValueError("distance bins collapsed in tokenizer")
    return ids

DISTANCE_TOKEN_IDS = resolve_distance_token_ids(tokenizer)
DISTANCE_ID_SET = set(DISTANCE_TOKEN_IDS)
BIN_OF = {tid: k for k, tid in enumerate(DISTANCE_TOKEN_IDS)}
print(f"resolved {len(DISTANCE_TOKEN_IDS)} distance-bin tokens (ids: {DISTANCE_TOKEN_IDS[:4]}..{DISTANCE_TOKEN_IDS[-2:]})")

# %% [markdown]
# ## Inference: per-protein, shared-prefix batched
#
# For each protein we build the base prompt once:
#
# ```
# <contacts-and-distances-v1> <begin_sequence> <AA_1> ... <AA_n>
# <begin_statements>
# ```
#
# Then we batch one `TokensPrompt` per (i, j) pair with the tail
# `<distance> <p_i> <p_j> <CA> <CA>`. vLLM's prefix-cache reuses
# the shared `base_ids` KV-cache across the whole batch, so a
# 100-residue protein's ~5000 pair queries only pay the prefix
# compute once.

# %%
def encode_token_strs(tok, token_strs):
    ids = tok.encode(" ".join(token_strs), add_special_tokens=False)
    if len(ids) != len(token_strs):
        raise ValueError(f"tokenizer 1:1 broke: {token_strs[:5]!r} -> {ids[:5]!r}")
    return [int(x) for x in ids]


def predict_distance_matrix(parsed, *, batch_size=128):
    """Return (expected (n,n), bin_probs (n,n,64)) for all i<j; mirrored for i>j."""
    n = len(parsed.residues)

    base_tokens = [f"<{NAME}>", "<begin_sequence>"]
    base_tokens.extend(f"<{r.name}>" for r in parsed.residues)
    base_tokens.append("<begin_statements>")
    base_ids = encode_token_strs(tokenizer, base_tokens)

    pair_keys: list[tuple[int, int]] = []
    pair_prompts: list = []
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            tail_ids = encode_token_strs(tokenizer, [
                "<distance>", f"<p{i}>", f"<p{j}>", "<CA>", "<CA>",
            ])
            pair_prompts.append(TokensPrompt(prompt_token_ids=base_ids + tail_ids))
            pair_keys.append((i, j))

    sampling = SamplingParams(
        temperature=1.0, top_p=1.0, top_k=-1,
        max_tokens=1, logprobs=128, n=1,
    )

    expected = np.full((n, n), np.nan, dtype=np.float32)
    bin_probs = np.zeros((n, n, NUM_DISTANCE_BINS), dtype=np.float32)
    np.fill_diagonal(expected, 0.0)

    t0 = time.time()
    for chunk_start in range(0, len(pair_prompts), batch_size):
        chunk_prompts = pair_prompts[chunk_start:chunk_start + batch_size]
        chunk_keys = pair_keys[chunk_start:chunk_start + batch_size]
        outputs = llm.generate(chunk_prompts, sampling, use_tqdm=False)
        for (i, j), out in zip(chunk_keys, outputs, strict=True):
            lp_dict = out.outputs[0].logprobs[0] if out.outputs[0].logprobs else {}
            row = np.zeros(NUM_DISTANCE_BINS, dtype=np.float32)
            for tok_id, lp in lp_dict.items():
                tid = int(tok_id)
                if tid in DISTANCE_ID_SET:
                    row[BIN_OF[tid]] = math.exp(float(lp.logprob))
            total = float(row.sum())
            if total <= 0:
                continue
            row /= total
            e = float((row * BIN_MIDPOINTS).sum())
            expected[i - 1, j - 1] = e
            expected[j - 1, i - 1] = e
            bin_probs[i - 1, j - 1] = row
            bin_probs[j - 1, i - 1] = row
    elapsed = time.time() - t0
    print(f"    {len(pair_prompts)} pairs in {elapsed:.1f}s ({1000*elapsed/max(1,len(pair_prompts)):.1f} ms/pair)")
    return expected, bin_probs


predicted_matrices: dict[str, np.ndarray] = {}
bin_probs_by_entry: dict[str, np.ndarray] = {}
for spec, parsed in structures:
    print(f"  {spec.entry_id} ({len(parsed.residues)} res):")
    pred, probs = predict_distance_matrix(parsed)
    predicted_matrices[spec.entry_id] = pred
    bin_probs_by_entry[spec.entry_id] = probs

# %% [markdown]
# ## Per-protein heatmaps + MAE
#
# Three panels per protein: GT, prediction, |residual|. We mask
# out pairs whose GT exceeds the saturated bin (32 Å) — the model
# can't say anything finer than "≥ 32 Å" there, so they belong
# at the ceiling, not in the MAE.

# %%
PLOTS_DIR = EXP_DIR / "plots"
DATA_DIR = EXP_DIR / "data"
PLOTS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

per_protein_mae: list[dict] = []

for spec, parsed in structures:
    gt = gt_matrices[spec.entry_id]
    pred = predicted_matrices[spec.entry_id]
    n = gt.shape[0]
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    valid = (ii != jj) & np.isfinite(gt) & (gt <= DISTANCE_MAX_A)
    abs_err = np.abs(pred - gt)
    mae = float(abs_err[valid & np.isfinite(pred)].mean())
    per_protein_mae.append({
        "entry_id": spec.entry_id,
        "uniprot_accession": spec.uniprot_accession,
        "n_residues": n,
        "n_valid_pairs": int(valid.sum()),
        "mae_ca_ca_angstrom": mae,
    })

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    vmax = DISTANCE_MAX_A
    im0 = axes[0].imshow(gt, vmin=0, vmax=vmax, cmap="viridis")
    axes[0].set_title(f"GT CA-CA (Å)")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)
    im1 = axes[1].imshow(pred, vmin=0, vmax=vmax, cmap="viridis")
    axes[1].set_title("Predicted CA-CA (Å)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    im2 = axes[2].imshow(abs_err, vmin=0, vmax=10.0, cmap="magma")
    axes[2].set_title(f"|residual|  MAE = {mae:.2f} Å")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)
    fig.suptitle(f"{spec.entry_id}  ({n} residues)")
    for ax in axes:
        ax.set_xlabel("residue j")
        ax.set_ylabel("residue i")
    fig.tight_layout()
    out_path = PLOTS_DIR / f"{spec.entry_id}.png"
    save_plot_with_meta(
        fig, out_path,
        caption=(
            f"V1 (zero-shot) — CA-CA distance heatmap for {spec.entry_id} "
            f"({n} aa, MAE {mae:.2f} Å). Panels: GT, predicted, |residual|. "
            f"Model: 1B, RTX A5000."
        ),
        script="eval_notebook.ipynb",
        dpi=110,
    )
    plt.show()
    print(f"    saved {out_path.relative_to(REPO_ROOT)}; MAE = {mae:.2f} Å")

# %% [markdown]
# ## All proteins at a glance — 10 × 3 grid
#
# One row per protein, columns = GT, predicted, |residual|. Color
# scales are shared per column so brightness is comparable across
# rows.

# %%
n_proteins = len(structures)
fig, axes = plt.subplots(
    n_proteins, 3,
    figsize=(11, 3.2 * n_proteins),
    squeeze=False,
)
for row, (spec, parsed) in enumerate(structures):
    gt = gt_matrices[spec.entry_id]
    pred = predicted_matrices[spec.entry_id]
    abs_err = np.abs(pred - gt)
    n = gt.shape[0]
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    valid = (ii != jj) & np.isfinite(gt) & (gt <= DISTANCE_MAX_A)
    mae = float(abs_err[valid & np.isfinite(pred)].mean())

    im0 = axes[row, 0].imshow(gt, vmin=0, vmax=DISTANCE_MAX_A, cmap="viridis")
    im1 = axes[row, 1].imshow(pred, vmin=0, vmax=DISTANCE_MAX_A, cmap="viridis")
    im2 = axes[row, 2].imshow(abs_err, vmin=0, vmax=10.0, cmap="magma")

    axes[row, 0].set_ylabel(f"{spec.entry_id}\n({n} res)\nMAE={mae:.2f} Å",
                            fontsize=9)
    if row == 0:
        axes[row, 0].set_title("GT CA-CA (Å)")
        axes[row, 1].set_title("Predicted CA-CA (Å)")
        axes[row, 2].set_title("|residual| (Å)")
    for col in range(3):
        axes[row, col].set_xticks([])
        axes[row, col].set_yticks([])

# Shared colorbars on the right edge.
cbar_ax_dist = fig.add_axes([0.93, 0.55, 0.012, 0.32])
fig.colorbar(im1, cax=cbar_ax_dist, label="distance (Å)")
cbar_ax_err = fig.add_axes([0.93, 0.13, 0.012, 0.32])
fig.colorbar(im2, cax=cbar_ax_err, label="|residual| (Å)")

fig.suptitle(
    f"All {n_proteins} test proteins — 1B model, zero-shot CA-CA "
    f"(macro MAE = {float(np.mean([r['mae_ca_ca_angstrom'] for r in per_protein_mae])):.2f} Å)",
    fontsize=12,
)
fig.subplots_adjust(left=0.13, right=0.91, top=0.97, bottom=0.02, hspace=0.18, wspace=0.05)
_grid_macro_mae = float(np.mean([r["mae_ca_ca_angstrom"] for r in per_protein_mae]))
save_plot_with_meta(
    fig, PLOTS_DIR / "all_proteins_grid.png",
    caption=(
        f"V1 (zero-shot) — 10×3 grid of CA-CA heatmaps (GT / predicted / "
        f"|residual|) across {n_proteins} AFDB test proteins. "
        f"Macro MAE = {_grid_macro_mae:.2f} Å. Model: 1B."
    ),
    script="eval_notebook.ipynb",
    dpi=110,
)
plt.show()
print(f"saved {(PLOTS_DIR / 'all_proteins_grid.png').relative_to(REPO_ROOT)}")

# %% [markdown]
# ## Aggregate
#
# Macro-mean MAE across the 10 proteins, plus a scatter of
# expected vs. GT distance pooled over all evaluable pairs.

# %%
import csv

macro_mae = float(np.mean([r["mae_ca_ca_angstrom"] for r in per_protein_mae]))
print(f"macro CA-CA MAE across {len(per_protein_mae)} proteins: {macro_mae:.3f} Å")
print()
print(f"  {'entry_id':<24} {'n_res':>5} {'n_pairs':>8} {'mae (Å)':>9}")
for r in per_protein_mae:
    print(f"  {r['entry_id']:<24} {r['n_residues']:>5} {r['n_valid_pairs']:>8} {r['mae_ca_ca_angstrom']:>9.3f}")

with (DATA_DIR / "per_protein_mae.csv").open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(per_protein_mae[0]))
    w.writeheader()
    w.writerows(per_protein_mae)
print(f"\nwrote {(DATA_DIR / 'per_protein_mae.csv').relative_to(REPO_ROOT)}")

# %%
all_gt = []
all_pred = []
for spec, parsed in structures:
    gt = gt_matrices[spec.entry_id]
    pred = predicted_matrices[spec.entry_id]
    n = gt.shape[0]
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    mask = (ii < jj) & np.isfinite(gt) & (gt <= DISTANCE_MAX_A) & np.isfinite(pred)
    all_gt.append(gt[mask])
    all_pred.append(pred[mask])
all_gt = np.concatenate(all_gt)
all_pred = np.concatenate(all_pred)

fig, ax = plt.subplots(figsize=(5.5, 5.0))
ax.hexbin(all_gt, all_pred, gridsize=60, mincnt=1, cmap="cividis")
lim = (0, DISTANCE_MAX_A)
ax.plot(lim, lim, "r--", lw=1, alpha=0.7)
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_xlabel("GT CA-CA distance (Å)")
ax.set_ylabel("predicted expected distance (Å)")
ax.set_title(f"pooled across {len(per_protein_mae)} proteins, {len(all_gt):,} pairs\nmacro MAE = {macro_mae:.2f} Å")
fig.tight_layout()
save_plot_with_meta(
    fig, PLOTS_DIR / "pooled_scatter.png",
    caption=(
        f"V1 (zero-shot) — pooled expected vs GT CA-CA distance across "
        f"{len(per_protein_mae)} proteins ({len(all_gt):,} evaluable pairs). "
        f"Macro CA-CA MAE = {macro_mae:.2f} Å. Red dashed line is y=x."
    ),
    script="eval_notebook.ipynb",
    dpi=110,
)
plt.show()

# %% [markdown]
# ## Summary
#
# - 10 unseen AFDB structures from the held-out test split
# - Zero-shot CA-CA inference with the `1B` model
# - Heatmaps + per-protein MAEs above, source data in `data/`
#
# To re-run on a different model: add the new model to
# `MODELS.yaml` (with `contacts-and-distances-v1` in its
# `document_structures`), set `MODEL_NICK` above, and
# `Restart & Run All`.
