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
# # exp9 — seeded-contact search over ALL contact ranges (greedy)
#
# Companion to `contact_seeding_search.ipynb` (which restricted
# the candidate set to **long-range** contacts only, sep ≥ 24).
# Here we let greedy search choose freely from all three contact
# ranges used in training:
#
# - `<long-range-contact>` — sep ≥ 24
# - `<medium-range-contact>` — sep 12..23
# - `<short-range-contact>` — sep 6..11
#
# All three use the CB-CB ≤ 8 Å cutoff (CA for GLY / missing CB).
# Adding medium + short range substantially expands the candidate
# pool (90 → 240 contacts for some proteins) and also gives the
# two "extended" proteins (AF-A0A1C5BRX1-F1, AF-A0A1N7G8C0-F1) a
# candidate set for the first time — they have no long-range
# contacts in their AFDB structure but do have short-range ones.
#
# Algorithm: **pure greedy** (no beam) — at each round, try
# `CANDIDATES_PER_ROUND` random remaining-candidate contacts,
# pick the one that minimizes sample MAE. The extra candidates
# from medium + short range slow the search per round, so we keep
# the algorithm simple to make the run tractable.

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
IH.add_exp1_to_path()

from parse import parse_structure
from select_test_proteins import select_test_proteins, download_cif

print("repo root:", REPO_ROOT)

# %% [markdown]
# ## Resolve `1B` model + same 10 test proteins

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
# ## Candidate counts by range
#
# Quick sanity check before the search — confirms there's
# something to pick for every protein.

# %%
candidate_pools = {}
for spec, parsed in structures:
    all_c = IH.gt_contacts_all_ranges(parsed)
    n_long = sum(1 for c in all_c if c[0] == "<long-range-contact>")
    n_med = sum(1 for c in all_c if c[0] == "<medium-range-contact>")
    n_short = sum(1 for c in all_c if c[0] == "<short-range-contact>")
    candidate_pools[spec.entry_id] = all_c
    print(f"  {spec.entry_id}: long={n_long:>3}  medium={n_med:>3}  short={n_short:>3}  total={len(all_c):>4}")

# %% [markdown]
# ## Pure-greedy search per protein
#
# Per round we try `CANDIDATES_PER_ROUND` random remaining
# candidates (across all three ranges) and pick the one
# minimizing the 300-pair sample MAE. Stop early at the target
# MAE or when `MAX_CONTACTS` is reached.

# %%
SAMPLE_PAIRS = 300
TARGET_MAE = 1.0
MAX_CONTACTS = 30
CANDIDATES_PER_ROUND = 10
PAIR_SAMPLE_SEED = 1
CAND_RNG_SEED = 2


def greedy_search_all_ranges(spec, parsed):
    """Greedy seeded-contact search over ALL ranges (long+medium+short).

    Selected contact entries are 3-tuples ``(type_token, i, j)`` so
    the prompt builder emits the right range-specific contact
    statement (`<long-range-contact>`, `<medium-range-contact>`,
    `<short-range-contact>`).
    """
    pair_seed = hash((spec.entry_id, PAIR_SAMPLE_SEED)) & 0xFFFFFFFF
    sample_pairs = IH.sample_ca_pairs(parsed, SAMPLE_PAIRS, seed=pair_seed)
    candidates = candidate_pools[spec.entry_id]
    if not candidates:
        return {
            "entry_id": spec.entry_id,
            "n_candidates": 0,
            "trace": [],
            "selected_contacts": [],
            "search_terminated": "no candidate contacts at any range",
        }
    cand_rng = np.random.default_rng(
        hash((spec.entry_id, CAND_RNG_SEED)) & 0xFFFFFFFF)

    selected: list[tuple[str, int, int]] = []
    trace = []

    def measure(contacts):
        pred = IH.predict_at_pairs(
            llm=llm,
            tokenizer=tokenizer,
            parsed=parsed,
            pairs=sample_pairs,
            seeded_contacts=contacts,
            distance_token_ids=DISTANCE_TOKEN_IDS,
        )
        return IH.mae_on_pairs(parsed, sample_pairs, pred)

    t0 = time.time()
    base_mae, base_n = measure([])
    print(f"  k=0 sample MAE = {base_mae:.3f} Å (n={base_n})")
    trace.append({
        "k": 0,
        "added_contact": None,
        "sample_mae_angstrom": base_mae,
        "elapsed_seconds": time.time() - t0,
    })
    if base_mae < TARGET_MAE:
        return {
            "entry_id": spec.entry_id,
            "n_candidates": len(candidates),
            "trace": trace,
            "selected_contacts": [],
            "search_terminated": "target met at k=0",
        }

    while len(selected) < MAX_CONTACTS:
        remaining = [c for c in candidates if c not in selected]
        if not remaining:
            return {
                "entry_id": spec.entry_id,
                "n_candidates": len(candidates),
                "trace": trace,
                "selected_contacts": selected,
                "search_terminated": "exhausted candidates",
            }
        if len(remaining) > CANDIDATES_PER_ROUND:
            idx = cand_rng.choice(len(remaining), size=CANDIDATES_PER_ROUND, replace=False)
            this_round = [remaining[i] for i in sorted(idx)]
        else:
            this_round = remaining
        t0 = time.time()
        best_mae = float("inf")
        best_contact = None
        for cand in this_round:
            trial = selected + [cand]
            mae, _ = measure(trial)
            if mae < best_mae:
                best_mae = mae
                best_contact = cand
        elapsed = time.time() - t0
        selected.append(best_contact)
        type_short = best_contact[0].strip("<>").replace("-range-contact", "")
        trace.append({
            "k": len(selected),
            "added_contact": best_contact,
            "added_contact_type": type_short,
            "sample_mae_angstrom": best_mae,
            "n_tried_this_round": len(this_round),
            "n_remaining_before_cap": len(remaining),
            "elapsed_seconds": elapsed,
        })
        print(f"  k={len(selected)} +{type_short}({best_contact[1]},{best_contact[2]}) "
              f"-> sample MAE = {best_mae:.3f} Å "
              f"(tried {len(this_round)}/{len(remaining)} cands in {elapsed:.1f}s)")
        if best_mae < TARGET_MAE:
            return {
                "entry_id": spec.entry_id,
                "n_candidates": len(candidates),
                "trace": trace,
                "selected_contacts": selected,
                "search_terminated": f"target met at k={len(selected)}",
            }

    return {
        "entry_id": spec.entry_id,
        "n_candidates": len(candidates),
        "trace": trace,
        "selected_contacts": selected,
        "search_terminated": f"reached MAX_CONTACTS={MAX_CONTACTS}",
    }


results = []
for spec, parsed in structures:
    print(f"\n{spec.entry_id} ({len(parsed.residues)} residues, {len(candidate_pools[spec.entry_id])} candidates):")
    res = greedy_search_all_ranges(spec, parsed)
    print(f"  -> {res['search_terminated']}; selected {len(res['selected_contacts'])} contacts")
    results.append(res)

# %% [markdown]
# ## Final full-matrix MAE with the selected contacts

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
    print(f"    full-matrix MAE = {full_mae:.3f} Å ({time.time()-t0:.1f}s)")

# %% [markdown]
# ## Per-protein summary

# %%
import csv

PLOTS_DIR = EXP_DIR / "plots"
DATA_DIR = EXP_DIR / "data"
PLOTS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

print(f"{'entry_id':<24} {'n_res':>5} {'cands':>6} {'k':>4} {'long':>4} {'med':>4} {'short':>5} {'sample_MAE':>11} {'full_MAE':>10}  status")
rows = []
for (spec, parsed), res in zip(structures, results, strict=True):
    k = len(res["selected_contacts"])
    sel = res["selected_contacts"]
    n_long = sum(1 for c in sel if c[0] == "<long-range-contact>")
    n_med = sum(1 for c in sel if c[0] == "<medium-range-contact>")
    n_short = sum(1 for c in sel if c[0] == "<short-range-contact>")
    sample_mae = res["trace"][-1]["sample_mae_angstrom"] if res["trace"] else float("nan")
    full_mae = res.get("full_matrix_mae_angstrom", float("nan"))
    rows.append({
        "entry_id": spec.entry_id,
        "uniprot_accession": spec.uniprot_accession,
        "n_residues": len(parsed.residues),
        "n_candidate_contacts": res["n_candidates"],
        "k_selected": k,
        "n_long_selected": n_long,
        "n_medium_selected": n_med,
        "n_short_selected": n_short,
        "selected_contacts": "; ".join(
            f"{c[0].strip('<>').replace('-range-contact','')}:{c[1]}-{c[2]}" for c in sel),
        "final_sample_mae_angstrom": sample_mae,
        "full_matrix_mae_angstrom": full_mae,
        "search_terminated": res["search_terminated"],
    })
    print(f"{spec.entry_id:<24} {len(parsed.residues):>5} {res['n_candidates']:>6} "
          f"{k:>4} {n_long:>4} {n_med:>4} {n_short:>5} "
          f"{sample_mae:>11.3f} {full_mae:>10.3f}  {res['search_terminated']}")

with (DATA_DIR / "contact_search_all_ranges_summary.csv").open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f"\nwrote {(DATA_DIR / 'contact_search_all_ranges_summary.csv').relative_to(REPO_ROOT)}")

# %% [markdown]
# ## Trace plot
#
# One line per protein. Horizontal red dash = 1.0 Å target.

# %%
fig, ax = plt.subplots(figsize=(9.5, 5.5))
for (spec, _), res in zip(structures, results, strict=True):
    if not res["trace"]:
        continue
    ks = [step["k"] for step in res["trace"]]
    maes = [step["sample_mae_angstrom"] for step in res["trace"]]
    ax.plot(ks, maes, "-o", markersize=3, label=spec.entry_id, alpha=0.85)
ax.axhline(TARGET_MAE, color="red", linestyle="--", alpha=0.7, label=f"target = {TARGET_MAE} Å")
ax.set_xlabel("k = number of seeded contacts (any range)")
ax.set_ylabel(f"sample MAE (Å), {SAMPLE_PAIRS} CA-CA pairs")
ax.set_title(f"Greedy search over all contact ranges — MAE trace per protein "
             f"(≤{CANDIDATES_PER_ROUND} cands/round)")
ax.legend(loc="upper right", fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(PLOTS_DIR / "contact_search_all_ranges_trace.png", dpi=110)
plt.show()

# %% [markdown]
# ## Final heatmaps with selected contacts (10 × 3 grid)

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

    axes[row, 0].set_ylabel(f"{spec.entry_id}\n({n} res)\nk={k}, MAE={mae:.2f} Å", fontsize=9)
    if row == 0:
        axes[row, 0].set_title("GT CA-CA (Å)")
        axes[row, 1].set_title("Predicted (with seeded, any range) (Å)")
        axes[row, 2].set_title("|residual| (Å)")
    for col in range(3):
        axes[row, col].set_xticks([])
        axes[row, col].set_yticks([])

cbar_dist = fig.add_axes([0.93, 0.55, 0.012, 0.32])
fig.colorbar(im1, cax=cbar_dist, label="distance (Å)")
cbar_err = fig.add_axes([0.93, 0.13, 0.012, 0.32])
fig.colorbar(im2, cax=cbar_err, label="|residual| (Å)")
fig.suptitle(
    f"Greedy contact search over ALL ranges (target sample MAE < {TARGET_MAE} Å, k ≤ {MAX_CONTACTS})",
    fontsize=12,
)
fig.subplots_adjust(left=0.13, right=0.91, top=0.97, bottom=0.02, hspace=0.18, wspace=0.05)
fig.savefig(PLOTS_DIR / "contact_search_all_ranges_grid.png", dpi=110, bbox_inches="tight")
plt.show()
print(f"saved {(PLOTS_DIR / 'contact_search_all_ranges_grid.png').relative_to(REPO_ROOT)}")

# %% [markdown]
# ## Summary
#
# Adds the medium- and short-range contact types to the candidate
# pool, runs pure greedy. Compare to
# `contact_seeding_search.ipynb` (beam-2, long-range only) to see
# whether the extra short / medium contacts let more proteins
# reach 1 Å MAE.
