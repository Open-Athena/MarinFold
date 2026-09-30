#!/usr/bin/env python
# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate ``notebooks/fold_switching_exp301.ipynb`` (issue #301).

Written here rather than by hand so the notebook and the experiment stay one
source of truth: same discriminative universe, same exp82 sampling recipe, same
fold score.

Deliberately **self-contained** — it installs ``marinfold`` from the public
GitHub repo and reads exp301's published tables and the checkpoint from the
public ``open-athena/MarinFold`` bucket anonymously. No credentials, so the
Colab link works for anyone.

The last section is the point: pick any fold switcher, choose how many fold2
contacts to put in the prompt, and watch the model move.

    uv run python build_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "notebooks" / "fold_switching_exp301.ipynb"
BUCKET = "hf://buckets/open-athena/MarinFold/data/contacts-v1-fold-switching-exp301"
MODEL = ("hf://buckets/open-athena/MarinFold/checkpoints/"
         "contacts-v1-exp277-m2-p06-full-epoch-1.5B/hf/step-266344")

BADGE = ("[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
         "(https://colab.research.google.com/github/Open-Athena/MarinFold/blob/main/"
         "notebooks/fold_switching_exp301.ipynb)")

CELLS: list[tuple[str, str]] = []


def md(src: str) -> None:
    CELLS.append(("markdown", src.strip("\n")))


def code(src: str) -> None:
    CELLS.append(("code", src.strip("\n")))


md(f"""
# Does MarinFold represent both folds of a fold-switching protein?

{BADGE}

Fold-switching proteins adopt two distinct stable folds from one sequence.
AlphaFold2 predicts only one conformation for
[94% of them](https://doi.org/10.1002/pro.4353), and its successes are
[driven by training-set memorization](https://doi.org/10.1038/s41467-024-51801-z).

MarinFold is a different kind of model — `contacts-v1` is **generative over
contact sets**, so it samples, and it is **promptable**, so it can be steered.
This notebook reproduces [issue #301](https://github.com/Open-Athena/MarinFold/issues/301):

1. the model reaches one fold (Fold1) and sampling harder does not find the other;
2. but teacher forcing shows the other fold is **not** improbable;
3. and putting a handful of its contacts in the prompt moves the model to it.

Everything reads from public buckets. **Runtime → Change runtime type → GPU**
for the last section.
""")

code("""
%pip -q install "marinfold[contacts-v1] @ git+https://github.com/Open-Athena/MarinFold.git#subdirectory=marinfold"
%pip -q install "huggingface_hub>=1.5" pandas matplotlib pyarrow
print("ready")
""")

md("## 1. The eval set and its premise")

code(f"""
import io, json
import fsspec, pandas as pd, numpy as np
import matplotlib.pyplot as plt

BUCKET = "{BUCKET}"

def table(name, **kw):
    with fsspec.open(f"{{BUCKET}}/{{name}}", "rb") as fh:
        return pd.read_csv(fh, **kw)

def universe():
    with fsspec.open(f"{{BUCKET}}/foldswitch_universe.jsonl", "rb") as fh:
        return {{r["pair_id"]: r for r in map(json.loads, io.TextIOWrapper(fh))}}

U = universe()
gate = table("premise_gate.csv")
floor = table("noise_floor.csv")
passing = gate[gate.passes_gate == True]
print(f"{{len(gate)}} pairs built from the ncbi/AF2_benchmark TableS1; {{len(passing)}} pass the premise gate")
print(f"  tiers: {{passing.tier.value_counts().to_dict()}}")
""")

md("""
### The premise, against its own noise floor

The two folds of a pair share a median Jaccard of ~0.52. That only means
something next to how much two contact maps differ when *nothing* has switched —
measured here from two chains of the same sequence in the same fold **inside one
crystal**.
""")

code("""
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(floor.jaccard, bins=20, alpha=0.75, label=f"same fold, same crystal (n={len(floor)})", color="#2ca02c")
ax.hist(passing.jaccard, bins=20, alpha=0.75, label=f"fold-switch pairs (n={len(passing)})", color="#d62728")
ax.axvline(floor.jaccard.median(), color="#2ca02c", ls="--", lw=1.5)
ax.axvline(passing.jaccard.median(), color="#d62728", ls="--", lw=1.5)
ax.set_xlabel("Jaccard(contacts of the two structures)"); ax.set_ylabel("count")
ax.legend(frameon=False); ax.set_title("The maps really do differ")
plt.show()
print(f"replicate median {floor.jaccard.median():.3f} vs fold-switch median {passing.jaccard.median():.3f}")
""")

md("## 2. Which fold does the model reach?")

code("""
pref = table("fold_preference.csv")
clean = pref[pref.frac_finished >= 0.9]
n1 = int((clean.phi > 0).sum())
print(f"prefers Fold1 in {n1}/{len(clean)} pairs; mean phi {clean.phi.mean():+.4f}")
print(f"R-precision  fold1 {clean.fold1_R_all.mean():.4f}  vs  fold2 {clean.fold2_R_all.mean():.4f}")

order = clean.sort_values("phi")
fig, ax = plt.subplots(figsize=(7.5, 5))
ax.barh(np.arange(len(order)), order.phi,
        color=["#1f77b4" if v > 0 else "#d62728" for v in order.phi])
ax.axvline(0, color="k", lw=1); ax.set_yticks([])
ax.set_xlabel(r"fold score $\\varphi$   (>0 = Fold1)")
ax.set_ylabel("fold-switching pair")
ax.set_title("The preference is lopsided: Fold1 reaches +0.6, Fold2 stops at -0.2")
plt.show()
""")

md("""
## 3. Sampling does not find the other fold — but likelihood almost does

A generative model's native route to a second conformation is simply to sample
more. The per-rollout spread of φ is compared against a **binomial null** built
from each pair's own recalls, because a one-fold model still spreads φ.
""")

code("""
bim = table("bimodality.csv")
dn = table("delta_nll.csv"); matched = dn[dn.variant == "matched"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].scatter(bim.null_sd, bim.phi_sd, s=22, color="#7f7f7f", alpha=0.8)
lim = max(bim.null_sd.max(), bim.phi_sd.max()) * 1.1
axes[0].plot([0, lim], [0, lim], "k--", lw=1, label="independent-sampling null")
axes[0].set_xlabel("null sd"); axes[0].set_ylabel("observed sd"); axes[0].legend(frameon=False)
axes[0].set_title(f"one mode (median dispersion {bim.dispersion.median():.2f})")
axes[1].hist(matched.delta_nll_tok, bins=22, color="#9467bd")
axes[1].axvline(0, color="k", lw=1)
axes[1].set_xlabel(r"$\\Delta$NLL/token (fold1 - fold2)")
axes[1].set_title(f"near coin-flip: mean {matched.delta_nll_tok.mean():+.3f}")
plt.show()
print("Fold2 is improbable to SAMPLE, not improbable to SCORE.")
""")

md("""
## 4. Force the other fold

This is the part with no AlphaFold analogue. `contacts-v1` conditions by simply
extending the prompt past `<begin_statements>` with `<contact> <pi> <pj>`
statements. Put *k* of fold2's contacts there and the model follows.

Below: the published dose-response, then a live cell you can drive yourself.
""")

code("""
curve = table("conditioning_curve.csv")
fig, ax = plt.subplots(figsize=(7, 4.5))
for arm, color, label in (("seed_b", "#d62728", "seeded with fold2 contacts"),
                          ("seed_a", "#1f77b4", "seeded with fold1 contacts (control)")):
    sub = curve[curve.arm == arm].sort_values("k")
    ax.plot(sub.k, sub.phi, "o-", color=color, label=label)
    ax.fill_between(sub.k, sub.phi_lo, sub.phi_hi, color=color, alpha=0.15, lw=0)
ax.axhline(0, color="k", lw=1); ax.set_xscale("symlog", linthresh=1); ax.set_xlim(left=0)
ax.set_xlabel("k contacts placed in the prompt"); ax.set_ylabel(r"$\\varphi$ on the REMAINING sets")
ax.legend(frameon=False); ax.set_title("Conditioning steers the model")
plt.show()
""")

md("""
### Drive it yourself  *(needs a GPU runtime)*

Pick a pair, pick `K`, and run. The given contacts are removed from the scored
sets, so what you see is recovery of the contacts the model was **not** told.
""")

code(f"""
%pip -q install vllm
import fsspec, os
from pathlib import Path

MODEL = "{MODEL}"
dst = Path("/content/marinfold_model"); dst.mkdir(parents=True, exist_ok=True)
fs, root = fsspec.core.url_to_fs(MODEL)
for f in [f for f in fs.ls(root, detail=True) if f["type"] == "file"]:
    target = dst / os.path.basename(f["name"])
    if not target.exists():
        fs.get_file(f["name"], str(target))
print("checkpoint staged:", sorted(p.name for p in dst.iterdir()))
""")

code('''
import random, re
import numpy as np
from marinfold.document_structures.contacts_v1 import (
    GenerationConfig, build_document, residues_from_sequence)
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

BEGIN, NUM_POS, MIN_SEP = "<begin_statements>", 2000, 6
CONTACT_RE = re.compile(r"<contact>\\s+<p(\\d+)>\\s+<p(\\d+)>")
cfg = GenerationConfig()
tok = AutoTokenizer.from_pretrained("/content/marinfold_model")
llm = LLM(model="/content/marinfold_model", dtype="bfloat16", max_model_len=8192,
          enable_prefix_caching=False, generation_config="vllm", max_num_seqs=64)
END = tok.convert_tokens_to_ids("<end>")

def steer(pair_id, K=0, seed_from="fold2", n_rollouts=50, seed=0):
    rec = U[pair_id]
    residues = residues_from_sequence(rec["sequence"])
    A = {(i, j) for i, j in rec["contacts_fold1"]}
    B = {(i, j) for i, j in rec["contacts_fold2"]}
    only_a, only_b = A - B, B - A
    common = set(rec["common_positions"])
    source = only_b if seed_from == "fold2" else only_a
    rng = random.Random(f"{pair_id}:{seed_from}:{K}:{seed}")
    given = set(rng.sample(sorted(source), min(K, len(source))))
    a_rem, b_rem = only_a - given, only_b - given

    prompts, maps = [], []
    for r in range(n_rollouts):
        doc = build_document(f"{pair_id}:demo{seed}:r{r}", residues, [], config=cfg)
        prefix = doc.document[: doc.document.index(BEGIN) + len(BEGIN)]
        pos = {t: (doc.n_term_index + t) % NUM_POS for t in range(doc.seq_len)}
        order = sorted(given); rng.shuffle(order)
        for i, j in order:
            x, y = (i, j) if rng.random() < 0.5 else (j, i)
            prefix += f" <contact> <p{pos[x]}> <p{pos[y]}>"
        prompts.append(prefix); maps.append({pos[t]: t for t in range(doc.seq_len)})

    plen = len(tok(prompts[0], add_special_tokens=False).input_ids)
    sp = SamplingParams(temperature=1.0, top_p=0.95, top_k=-1, stop_token_ids=[END],
                        skip_special_tokens=False,
                        max_tokens=min(8192 - plen, 6 * rec["L"] + 128))
    outs = llm.generate(prompts, [sp] * len(prompts), use_tqdm=False)

    ra = rb = 0.0
    for out, m in zip(outs, maps):
        pred = set()
        for x, y in CONTACT_RE.findall(out.outputs[0].text):
            a, b = m.get(int(x)), m.get(int(y))
            if a is None or b is None or abs(a - b) < MIN_SEP:
                continue
            p = (min(a, b), max(a, b))
            if p[0] in common and p[1] in common:
                pred.add(p)
        ra += len(pred & a_rem) / max(len(a_rem), 1)
        rb += len(pred & b_rem) / max(len(b_rem), 1)
    ra /= len(outs); rb /= len(outs)
    print(f"{pair_id}  K={K} from {seed_from}:  recall(fold1 remaining)={ra:.3f}  "
          f"recall(fold2 remaining)={rb:.3f}   phi={ra - rb:+.3f}")
    return ra - rb

# KaiB: the protein AF-Cluster was built for.
for K in (0, 5, 10, 20, 40):
    steer("5jyta_2qkee", K=K, seed_from="fold2")
''')

md("""
Try `seed_from="fold1"` for the symmetric control — the same dose pushes φ the
other way, and the gap between the two arms is how much harder it is to move the
model *off* its preferred fold than further onto it.

Other pairs worth a look: `list(U)` for all of them; `6c6sd_2ougc` is RfaH.
""")


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    notebook = {
        "cells": [
            {"cell_type": kind, "metadata": {},
             **({"source": src.splitlines(keepends=True)} if kind == "markdown" else
                {"source": src.splitlines(keepends=True), "outputs": [], "execution_count": None})}
            for kind, src in CELLS
        ],
        "metadata": {
            "colab": {"provenance": [], "toc_visible": True},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    OUT.write_text(json.dumps(notebook, indent=1) + "\n")
    print(f"wrote {OUT} ({len(CELLS)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
