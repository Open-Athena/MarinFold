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
# # exp9 — minimal seeded-contact search (1B model, same 10 proteins)
#
# Companion to `eval_notebook.ipynb`. There we saw zero-shot CA-CA
# MAE around 3.3 Å. The training recipe also exposes the model to
# "seeded" GT long-range contacts at the start of the statements
# section, and the in-training benchmark shows MAE drops sharply as
# more seeded contacts are added.
#
# This notebook asks: **for each of the same 10 test proteins, what
# is the smallest set of true long-range contacts (up to 5) that
# brings full-matrix CA-CA MAE below 1.0 Å?**
#
# Strategy per protein:
# 1. Identify the candidate set = GT long-range contacts
#    (`<long-range-contact>` = CB-CB ≤ 8 Å, sep ≥ 24).
# 2. Greedy search: at each round 1..5, try every remaining
#    candidate as the next seeded contact, measure MAE on a
#    deterministic 500-pair CA-CA sample, pick the contact that
#    minimizes MAE.
# 3. Stop early when the sample MAE drops under 1.0 Å.
# 4. After the search finishes, re-run the model with the chosen
#    contacts on the **full N×N matrix** and report both the
#    sample-MAE trace and the final full-matrix MAE.
#
# Greedy isn't guaranteed to find the absolute minimal subset, but
# at this k it's the right complexity tradeoff and it consistently
# tracks the best-contacts-first ordering you'd want anyway.

# %%
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

EXP_DIR = Path.cwd()
REPO_ROOT = EXP_DIR.parents[1]
sys.path.insert(0, str(EXP_DIR))

import inference_helpers as IH
from build_summary import save_plot_with_meta
IH.add_exp1_to_path()

from parse import parse_structure
from select_test_proteins import select_test_proteins, download_cif

print("repo root:", REPO_ROOT)

# %% [markdown]
# ## Resolve the `1B` model from `MODELS.yaml`
#
# Identical setup to `eval_notebook.ipynb` — bump the nickname or
# the YAML entry to swap models.

# %%
with (REPO_ROOT / "MODELS.yaml").open() as fh:
    models = yaml.safe_load(fh)

MODEL_NICK = "1B"
model_entry = next(m for m in models if m["nickname"] == MODEL_NICK)
assert "contacts-and-distances-v1" in model_entry["document_structures"], model_entry
MODEL_HF_URL = model_entry["url"]
_, _, tail = MODEL_HF_URL.partition("huggingface.co/")
parts = tail.split("/")
MODEL_REPO = "/".join(parts[:2])
MODEL_SUBFOLDER = parts[4] if len(parts) > 4 and parts[2] == "tree" else None
print(f"using model: {MODEL_NICK} = {MODEL_REPO}/{MODEL_SUBFOLDER}")

# %% [markdown]
# ## Same 10 test proteins (seed=0, max_seq_len=150)
#
# Selection is deterministic; the AFDB cifs are already cached from
# the zero-shot notebook.

# %%
SEED = 0
N_PROTEINS = 10
MAX_SEQ_LEN = 150

specs = select_test_proteins(n=N_PROTEINS, seed=SEED, max_seq_len=MAX_SEQ_LEN)
CACHE_DIR = EXP_DIR / "data" / "afdb_cache"
structures = []
for spec in specs:
    cif_path = download_cif(spec, CACHE_DIR)
    parsed = parse_structure(cif_path)
    structures.append((spec, parsed))
    print(f"  {spec.entry_id}: {len(parsed.residues)} residues")

# %% [markdown]
# ## Load the model
#
# vLLM with prefix caching makes the search practical: the base
# prompt (sequence + `<begin_statements>` + 0..K seeded contacts)
# gets re-used across the per-pair tails, and across search trials
# that share a prefix the KV cache is also reused.

# %%
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

from huggingface_hub import snapshot_download

MODEL_LOCAL = Path(snapshot_download(
    repo_id=MODEL_REPO,
    allow_patterns=[f"{MODEL_SUBFOLDER}/*"] if MODEL_SUBFOLDER else None,
))
if MODEL_SUBFOLDER:
    MODEL_LOCAL = MODEL_LOCAL / MODEL_SUBFOLDER
print("model local path:", MODEL_LOCAL)

llm, tokenizer = IH.load_vllm(MODEL_LOCAL)
DISTANCE_TOKEN_IDS = IH.resolve_distance_token_ids(tokenizer)
print(f"resolved {len(DISTANCE_TOKEN_IDS)} distance tokens")

# %% [markdown]
# ## Beam search per protein
#
# Pure greedy (the first cut) repeatedly hit the local-optimum
# trap: an early "best" contact would lead the search down a
# branch that couldn't be recovered from later. Here we use
# **beam search with width 2** — at each round we expand the top
# 2 subsets so far, evaluate every (subset × candidate) pair, and
# keep the top 2 of all resulting sets. That doubles per-round
# cost but routinely escapes early misses.
#
# Tuning knobs:
#
# - `SAMPLE_PAIRS = 300` — fewer pairs per MAE measurement than
#   the first cut (500), still stable enough to compare
#   candidates. ~40% wall-time speedup.
# - `CANDIDATES_PER_ROUND = 8` — per (beam state, round), this
#   many random remaining-candidate trials are evaluated. The
#   candidate pool is the GT long-range contacts (CB-CB ≤ 8 Å,
#   sep ≥ 24), some proteins have 100+ of them.
# - `BEAM_WIDTH = 2` — number of subsets carried between rounds.
# - `MAX_CONTACTS = 30` — bumped from 5; the previous run showed
#   5 is not enough for any protein in this set.
# - `TARGET_MAE = 1.0` — stop early if the best beam state has
#   sample MAE below this.

# %%
SAMPLE_PAIRS = 300
TARGET_MAE = 1.0
MAX_CONTACTS = 30
BEAM_WIDTH = 2
CANDIDATES_PER_ROUND = 8
PAIR_SAMPLE_SEED = 1
CAND_RNG_SEED = 2


def beam_search_for_protein(spec, parsed):
    """Beam-search up to MAX_CONTACTS long-range GT contacts; stop when sample MAE < TARGET_MAE.

    Beam state: tuple of selected contacts (canonicalised by
    sorted order so that {A, B} and {B, A} dedupe). Carried with
    its sample MAE. Each round expands every state by a random
    subset of remaining candidates; the top BEAM_WIDTH unique
    states across all expansions become the next beam.
    """
    pair_seed = hash((spec.entry_id, PAIR_SAMPLE_SEED)) & 0xFFFFFFFF
    sample_pairs = IH.sample_ca_pairs(parsed, SAMPLE_PAIRS, seed=pair_seed)
    candidates = IH.gt_long_range_contacts(parsed)
    if not candidates:
        return {
            "entry_id": spec.entry_id,
            "n_candidates": 0,
            "trace": [],
            "selected_contacts": [],
            "search_terminated": "no candidate long-range contacts",
        }
    cand_rng = np.random.default_rng(
        hash((spec.entry_id, CAND_RNG_SEED)) & 0xFFFFFFFF)

    def measure(contacts):
        pred = IH.predict_at_pairs(
            llm=llm,
            tokenizer=tokenizer,
            parsed=parsed,
            pairs=sample_pairs,
            seeded_contacts=list(contacts),
            distance_token_ids=DISTANCE_TOKEN_IDS,
        )
        return IH.mae_on_pairs(parsed, sample_pairs, pred)

    # k=0 baseline. One state in the initial beam.
    t0 = time.time()
    base_mae, base_n = measure(())
    print(f"  k=0 sample MAE = {base_mae:.3f} Å (n={base_n})")
    trace = [{
        "k": 0,
        "best_sample_mae_angstrom": base_mae,
        "beam_size": 1,
        "n_evals_this_round": 1,
        "elapsed_seconds": time.time() - t0,
    }]
    beam = [((), base_mae)]
    if base_mae < TARGET_MAE:
        return {
            "entry_id": spec.entry_id,
            "n_candidates": len(candidates),
            "trace": trace,
            "selected_contacts": [],
            "search_terminated": "target met at k=0",
        }

    seen_states: set[frozenset] = {frozenset()}
    k = 0
    while k < MAX_CONTACTS:
        k += 1
        round_t0 = time.time()
        expansions: list[tuple[tuple, float]] = []
        n_evals = 0
        for state, _state_mae in beam:
            remaining = [c for c in candidates if c not in state]
            if not remaining:
                continue
            if len(remaining) > CANDIDATES_PER_ROUND:
                idx = cand_rng.choice(len(remaining), size=CANDIDATES_PER_ROUND, replace=False)
                this_state_round = [remaining[i] for i in sorted(idx)]
            else:
                this_state_round = remaining
            for cand in this_state_round:
                new_state = tuple(sorted(state + (cand,)))
                if frozenset(new_state) in seen_states:
                    continue
                seen_states.add(frozenset(new_state))
                trial_mae, _ = measure(new_state)
                expansions.append((new_state, trial_mae))
                n_evals += 1
        if not expansions:
            return {
                "entry_id": spec.entry_id,
                "n_candidates": len(candidates),
                "trace": trace,
                "selected_contacts": list(beam[0][0]),
                "search_terminated": "exhausted candidates",
            }
        expansions.sort(key=lambda x: x[1])
        beam = expansions[:BEAM_WIDTH]
        best_state, best_mae = beam[0]
        elapsed = time.time() - round_t0
        trace.append({
            "k": k,
            "best_sample_mae_angstrom": best_mae,
            "best_state_size": len(best_state),
            "beam_size": len(beam),
            "n_evals_this_round": n_evals,
            "elapsed_seconds": elapsed,
        })
        beam_summary = "; ".join(f"{m:.2f}({len(s)})" for s, m in beam)
        print(f"  k={k} best sample MAE = {best_mae:.3f} Å "
              f"[beam: {beam_summary}] ({n_evals} evals, {elapsed:.1f}s)")
        if best_mae < TARGET_MAE:
            return {
                "entry_id": spec.entry_id,
                "n_candidates": len(candidates),
                "trace": trace,
                "selected_contacts": list(best_state),
                "search_terminated": f"target met at k={k}",
            }

    best_state, _ = min(beam, key=lambda x: x[1])
    return {
        "entry_id": spec.entry_id,
        "n_candidates": len(candidates),
        "trace": trace,
        "selected_contacts": list(best_state),
        "search_terminated": f"reached MAX_CONTACTS={MAX_CONTACTS}",
    }


results = []
for spec, parsed in structures:
    print(f"\n{spec.entry_id} ({len(parsed.residues)} residues):")
    res = beam_search_for_protein(spec, parsed)
    print(f"  -> {res['search_terminated']}; selected {len(res['selected_contacts'])} contacts")
    results.append(res)

# %% [markdown]
# ## Final full-matrix MAE with the selected contacts
#
# The greedy search runs on a 500-pair sample for speed. Now we
# re-run inference with the selected contacts on the full N×N
# matrix, which is what the zero-shot notebook reports.

# %%
final_predicted = {}
final_gt = {}
for (spec, parsed), res in zip(structures, results, strict=True):
    seeded = res["selected_contacts"]
    print(f"  {spec.entry_id}: running full matrix with {len(seeded)} seeded contacts")
    t0 = time.time()
    pred = IH.predict_distance_matrix(
        llm=llm,
        tokenizer=tokenizer,
        parsed=parsed,
        seeded_contacts=seeded,
        distance_token_ids=DISTANCE_TOKEN_IDS,
    )
    gt = IH.ca_distance_matrix(parsed)
    n = gt.shape[0]
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    valid = (ii != jj) & np.isfinite(gt) & (gt <= IH.DISTANCE_MAX_A)
    full_mae = float(np.abs(pred - gt)[valid & np.isfinite(pred)].mean())
    res["full_matrix_mae_angstrom"] = full_mae
    res["full_matrix_n_valid_pairs"] = int(valid.sum())
    final_predicted[spec.entry_id] = pred
    final_gt[spec.entry_id] = gt
    print(f"    full-matrix MAE = {full_mae:.3f} Å (n={int(valid.sum())} pairs, {time.time()-t0:.1f}s)")

# %% [markdown]
# ## Per-protein summary table

# %%
import csv

PLOTS_DIR = EXP_DIR / "plots"
DATA_DIR = EXP_DIR / "data"
PLOTS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

print(f"{'entry_id':<24} {'n_res':>5} {'cands':>6} {'k_chosen':>9} {'sample_MAE':>11} {'full_MAE':>10}  status")
rows = []
for (spec, parsed), res in zip(structures, results, strict=True):
    k = len(res["selected_contacts"])
    sample_mae = res["trace"][-1]["best_sample_mae_angstrom"] if res["trace"] else float("nan")
    full_mae = res.get("full_matrix_mae_angstrom", float("nan"))
    rows.append({
        "entry_id": spec.entry_id,
        "uniprot_accession": spec.uniprot_accession,
        "n_residues": len(parsed.residues),
        "n_candidate_contacts": res["n_candidates"],
        "k_selected": k,
        "selected_contacts": "; ".join(f"{i}-{j}" for i, j in res["selected_contacts"]),
        "final_sample_mae_angstrom": sample_mae,
        "full_matrix_mae_angstrom": full_mae,
        "search_terminated": res["search_terminated"],
    })
    print(f"{spec.entry_id:<24} {len(parsed.residues):>5} {res['n_candidates']:>6} "
          f"{k:>9} {sample_mae:>11.3f} {full_mae:>10.3f}  {res['search_terminated']}")

with (DATA_DIR / "contact_search_summary.csv").open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f"\nwrote {(DATA_DIR / 'contact_search_summary.csv').relative_to(REPO_ROOT)}")

# %% [markdown]
# ## Trace plot — sample MAE vs. number of seeded contacts
#
# One line per protein. Horizontal red dash = the 1.0 Å target.

# %%
fig, ax = plt.subplots(figsize=(9.5, 5.5))
for (spec, _), res in zip(structures, results, strict=True):
    if not res["trace"]:
        continue
    ks = [step["k"] for step in res["trace"]]
    maes = [step["best_sample_mae_angstrom"] for step in res["trace"]]
    ax.plot(ks, maes, "-o", markersize=3, label=spec.entry_id, alpha=0.85)
ax.axhline(TARGET_MAE, color="red", linestyle="--", alpha=0.7, label=f"target = {TARGET_MAE} Å")
ax.set_xlabel("k = number of seeded long-range contacts")
ax.set_ylabel(f"best sample MAE (Å), {SAMPLE_PAIRS} CA-CA pairs")
ax.set_title(f"Beam search (width {BEAM_WIDTH}, ≤{CANDIDATES_PER_ROUND} cands/round) — MAE trace per protein")
ax.legend(loc="upper right", fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)
fig.tight_layout()
save_plot_with_meta(
    fig, PLOTS_DIR / "contact_search_trace.png",
    caption=(
        f"V2 (beam-{BEAM_WIDTH} long-range search) — sample-MAE-vs-k curves "
        f"on {SAMPLE_PAIRS} deterministic CA-CA pairs. One line per protein; "
        f"red dashed line is target = {TARGET_MAE} Å."
    ),
    script="contact_seeding_search.ipynb",
    dpi=110,
)
plt.show()

# %% [markdown]
# ## Final heatmaps with selected contacts
#
# 10 × 3 grid: GT, predicted-with-selected-contacts, |residual|.
# Compare to `plots/all_proteins_grid.png` from the zero-shot
# notebook to see how the seeded contacts sharpen the prediction.

# %%
n_proteins = len(structures)
fig, axes = plt.subplots(n_proteins, 3, figsize=(11, 3.2 * n_proteins), squeeze=False)
for row, ((spec, parsed), res) in enumerate(zip(structures, results, strict=True)):
    gt = final_gt[spec.entry_id]
    pred = final_predicted[spec.entry_id]
    abs_err = np.abs(pred - gt)
    n = gt.shape[0]
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    valid = (ii != jj) & np.isfinite(gt) & (gt <= IH.DISTANCE_MAX_A)
    mae = res["full_matrix_mae_angstrom"]
    k = len(res["selected_contacts"])

    im0 = axes[row, 0].imshow(gt, vmin=0, vmax=IH.DISTANCE_MAX_A, cmap="viridis")
    im1 = axes[row, 1].imshow(pred, vmin=0, vmax=IH.DISTANCE_MAX_A, cmap="viridis")
    im2 = axes[row, 2].imshow(abs_err, vmin=0, vmax=10.0, cmap="magma")

    axes[row, 0].set_ylabel(
        f"{spec.entry_id}\n({n} res)\nk={k}, MAE={mae:.2f} Å",
        fontsize=9,
    )
    if row == 0:
        axes[row, 0].set_title("GT CA-CA (Å)")
        axes[row, 1].set_title("Predicted (with seeded) (Å)")
        axes[row, 2].set_title("|residual| (Å)")
    for col in range(3):
        axes[row, col].set_xticks([])
        axes[row, col].set_yticks([])

cbar_dist = fig.add_axes([0.93, 0.55, 0.012, 0.32])
fig.colorbar(im1, cax=cbar_dist, label="distance (Å)")
cbar_err = fig.add_axes([0.93, 0.13, 0.012, 0.32])
fig.colorbar(im2, cax=cbar_err, label="|residual| (Å)")
fig.suptitle(
    f"Final predictions with greedy-selected contacts (target sample MAE < {TARGET_MAE} Å, k ≤ {MAX_CONTACTS})",
    fontsize=12,
)
fig.subplots_adjust(left=0.13, right=0.91, top=0.97, bottom=0.02, hspace=0.18, wspace=0.05)
save_plot_with_meta(
    fig, PLOTS_DIR / "contact_search_grid.png",
    caption=(
        f"V2 (beam-{BEAM_WIDTH} long-range search) — 10×3 CA-CA heatmap grid "
        f"(GT / predicted-with-seeded / |residual|) at each protein's chosen "
        f"k. Target sample MAE < {TARGET_MAE} Å, k ≤ {MAX_CONTACTS}."
    ),
    script="contact_seeding_search.ipynb",
    dpi=110,
)
plt.show()
print(f"saved {(PLOTS_DIR / 'contact_search_grid.png').relative_to(REPO_ROOT)}")

# %% [markdown]
# ## Summary
#
# Records the smallest k (≤5) that brings the sample MAE under 1
# Å for each of the 10 test proteins, the resulting full-matrix
# MAE with those k seeded contacts, and a comparison heatmap. See
# `data/contact_search_summary.csv` for the per-protein record;
# `plots/contact_search_trace.png` for the MAE-vs-k curves;
# `plots/contact_search_grid.png` for the final heatmaps.
