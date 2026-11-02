#!/usr/bin/env python
# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate ``notebooks/circular_permutation_1un2.ipynb`` (issue #224).

The notebook is written here rather than by hand so the analysis in it stays a
single source of truth with ``prepare_inputs.py`` / ``analyze.py``: same contact
definition, same CP<->WT decomposition, same exp82 sampling recipe.

It is deliberately **self-contained** — it installs ``marinfold`` from the public
GitHub repo, pulls structures from RCSB and the checkpoint from the public
``open-athena/MarinFold`` bucket, and recomputes ground truth with pyconfind. No
exp224 artifacts and no credentials are needed, so the link works for anyone.

    uv run python build_notebook.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "notebooks" / "circular_permutation_1un2.ipynb"

BADGE = ("[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
         "(https://colab.research.google.com/github/Open-Athena/MarinFold/blob/main/"
         "notebooks/circular_permutation_1un2.ipynb)")

CELLS: list[tuple[str, str]] = []


def md(src: str) -> None:
    CELLS.append(("markdown", src.strip("\n")))


def code(src: str) -> None:
    CELLS.append(("code", src.strip("\n")))


md(f"""
{BADGE}

# Does MarinFold fold a *circularly permuted* protein? — 1UN2 vs wild-type DsbA

[Issue #224](https://github.com/Open-Athena/MarinFold/issues/224)

A circular permutation is close to a perfect controlled experiment for a
sequence-to-contacts model. **1UN2** (`CPDsbA-Q100T99`) is *E. coli* DsbA with the
chain cut between T99 and Q100 and the two halves swapped:

| segment | wild-type residues | length |
|---|---|---|
| new N-terminus | **Q100 – K189** | 90 |
| linker | `GGGTG` | 5 |
| new C-terminus | **A1 – T99** | 99 |
| cloning tail | `LIK` | 3 |

Same residues, same fold, **same 3D contacts** — the crystal paper's title is
"Preserved Global Fold and Local Structural Adjustments". What changes is where
those contacts sit in *sequence* space: a pair that straddles the cut moves from
separation `s` to separation `194 − s`.

So if the model has learned a sequence-separation prior, or is retrieving the
parent protein it almost certainly saw in training (wild-type DsbA is in AFDB),
the permutant should break it. If the model is reasoning about the fold from
local sequence, the permutant should be fine.

**What this notebook does**

1. Rebuilds the ground truth from RCSB with pyconfind — the exact contact
   definition MarinFold's training documents use.
2. Runs the current default checkpoint with exp82's fixed rollout recipe.
3. Scores CP vs WT, split by whether the permutation moved each pair.
4. **Lets you cut the chain wherever you like** and watch the accuracy move.

Runtime → Change runtime type → **GPU** (a T4 is enough).
""")

code("""
#@title Install (~3 min)
!pip -q install 'marinfold[contacts-v1] @ git+https://github.com/Open-Athena/MarinFold.git#subdirectory=marinfold'
!pip -q install 'huggingface_hub>=1.5' accelerate safetensors scikit-learn
print('ready')
""")

code("""
#@title Fetch structures + rebuild ground-truth contacts
# pyconfind side-chain contact degree, run with contacts_v1's own geometry
# defaults. A "contact" is degree >= 0.001 at sequence separation >= 6 -- the
# same definition behind every published MarinFold contact number.
import difflib, json, requests
import gemmi
from marinfold.document_structures.contacts_v1 import analyze_structure

PYCONFIND_KWARGS = dict(native_only=True, contact_distance=3.0, dcut=25.0,
                        clash_distance=2.0, assembly=None)
MIN_DEG, MIN_SEP = 0.001, 6
THREE_TO_ONE = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E',
                'GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F',
                'PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}

def fetch_cif(pdb):
    p = f'/content/{pdb.lower()}.cif'
    open(p, 'w').write(requests.get(
        f'https://files.rcsb.org/download/{pdb.upper()}.cif', timeout=60).text)
    return p

def seqres(pdb):
    d = requests.get(
        f'https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb.upper()}/1',
        timeout=60).json()
    return d['entity_poly']['pdbx_seq_one_letter_code_can'].replace('\\n', '').strip()

def align(obs, ref):
    \"\"\"Map each resolved residue onto an input-sequence index (difflib, so it
    makes no assumption about crystallographic numbering).\"\"\"
    if obs == ref:
        return list(range(len(obs)))
    out = [None] * len(obs)
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=obs, b=ref,
                                                       autojunk=False).get_opcodes():
        if tag in ('equal', 'replace'):
            for k in range(min(i2 - i1, j2 - j1)):
                out[i1 + k] = j1 + k
    return out

def ground_truth(pdb, chain='A'):
    st = gemmi.read_structure(fetch_cif(pdb)); st.setup_entities()
    while len(st) > 1:
        del st[1]
    model = st[0]
    keep = chain if any(c.name == chain for c in model) else max(
        model, key=lambda c: len(c.get_polymer())).name
    for name in [c.name for c in list(model)]:
        if name != keep:
            model.remove_chain(name)
    st.remove_ligands_and_waters(); st.remove_empty_chains()
    seq = seqres(pdb)
    an = analyze_structure(st, entry_id=pdb.lower(), **PYCONFIND_KWARGS)
    obs = ''.join(THREE_TO_ONE.get(r.resname, 'X') for r in an.residues)
    m = align(obs, seq)
    contacts = set()
    for c in an.contacts:
        i, j = m[c.seq_i], m[c.seq_j]
        if i is None or j is None or i == j or c.degree < MIN_DEG:
            continue
        lo, hi = min(i, j), max(i, j)
        if hi - lo >= MIN_SEP:
            contacts.add((lo, hi))
    resolved = sorted({x for x in m if x is not None})
    return dict(pdb=pdb, seq=seq, L=len(seq), contacts=contacts, resolved=resolved)

CP = ground_truth('1UN2')   # circular permutant
WT = ground_truth('1FVK')   # wild-type DsbA, 1.7 A
for g in (CP, WT):
    print(f\"{g['pdb']}: L={g['L']}  resolved={len(g['resolved'])}  \"
          f\"true contacts (sep>=6) = {len(g['contacts'])}\")
""")

md("""
## The CP↔WT residue map

Every permutant residue except the linker and the cloning tail has a wild-type
counterpart. We *derive* the decomposition from the two sequences rather than
hard-coding it, so the same cell works if you swap in a different permutant.
""")

code("""
#@title Derive the permutation
def decompose(cp_seq, wt_seq):
    \"\"\"CP -> WT index map, by finding the two WT segments the construct is built from.\"\"\"
    a = next(n for n in range(len(cp_seq), 0, -1) if wt_seq.endswith(cp_seq[:n]))
    best_b, b_start = 0, None
    for start in range(a, len(cp_seq)):
        n = 0
        while start + n < len(cp_seq) and n < len(wt_seq) and cp_seq[start + n] == wt_seq[n]:
            n += 1
        if n > best_b:
            best_b, b_start = n, start
    cp_to_wt = [None] * len(cp_seq)
    off = len(wt_seq) - a
    for k in range(a):
        cp_to_wt[k] = off + k
    for k in range(best_b):
        cp_to_wt[b_start + k] = k
    return dict(cp_to_wt=cp_to_wt, seg_a=(0, a, off, len(wt_seq)),
                seg_b=(b_start, b_start + best_b, 0, best_b),
                linker=cp_seq[a:b_start], tail=cp_seq[b_start + best_b:])

M = decompose(CP['seq'], WT['seq'])
a0, a1, wa0, wa1 = M['seg_a']
b0, b1, wb0, wb1 = M['seg_b']
print(f\"segment A: CP[{a0}:{a1}] = WT {wa0+1}-{wa1}   (the new N-terminus)\")
print(f\"linker   : {M['linker']!r}\")
print(f\"segment B: CP[{b0}:{b1}] = WT {wb0+1}-{wb1}   (the new C-terminus)\")
print(f\"tail     : {M['tail']!r}\")
print(f\"\\nSo the construct is CPDsbA-Q{wa0+1}T{wb1} -- which is exactly what the PDB calls it.\")
""")

code("""
#@title How much does the permutation actually move? (no model yet)
import numpy as np
c2w = M['cp_to_wt']
w2c = {w: c for c, w in enumerate(c2w) if w is not None}
in_a = lambda w: wa0 <= w < wa1

cp_in_wt = {(min(c2w[i], c2w[j]), max(c2w[i], c2w[j]))
            for i, j in CP['contacts'] if c2w[i] is not None and c2w[j] is not None}
both = {c2w[p] for p in CP['resolved'] if c2w[p] is not None} & set(WT['resolved'])
rest = lambda S: {(i, j) for i, j in S if i in both and j in both}
A, B = rest(cp_in_wt), rest(WT['contacts'])
print(f'CP and WT share {len(A & B)} contacts; {len(A & B) / len(B):.1%} of WT contacts '
      f'survive the permutation (Jaccard {len(A & B) / len(A | B):.3f}).')
print('-> the two molecules really are the same fold, so any model gap is about SEQUENCE, '
      'not structure.\\n')

cross = [(i, j) for i, j in B if in_a(i) != in_a(j)]
same = [(i, j) for i, j in B if in_a(i) == in_a(j)]
sw = np.array([j - i for i, j in cross])
sc = np.array([abs(w2c[j] - w2c[i]) for i, j in cross])
print(f'{len(same)} WT contacts keep their separation; {len(cross)} cross the cut and change it.')
print(f'  cross-cut separation: WT median {np.median(sw):.0f} -> CP median {np.median(sc):.0f}')
# A residue in segment B sits at CP index b0 + w; one in segment A sits at
# w - wa0. So a pair straddling the cut has CP_sep = (b0 + wa0) - WT_sep.
K = b0 + wa0
print(f'  the transform is exactly CP_sep = {K} - WT_sep: {bool(np.all(sc == K - sw))}')
""")

md("""
## Load the model

Whatever `MODELS.yaml` marks `default: true` — today
`contacts-v1-exp232-m2-p06-train-1.5B`: Qwen3 1.47B trained from scratch on a
50/50 AFDB + ESM-Atlas mixture, on corpora decontaminated against the eval
proteins ([#225](https://github.com/Open-Athena/MarinFold/issues/225) /
[#232](https://github.com/Open-Athena/MarinFold/issues/232)). `load_backend`
resolves it from `MODELS.yaml`, downloads it from the public bucket, and applies
the config / tokenizer repairs the export needs. Because the notebook follows
the registry rather than pinning a checkpoint, re-running it after a new default
lands will not reproduce the numbers below exactly.
""")

code("""
#@title Load the default checkpoint (~5.9 GB download)
import torch
from marinfold.inference import load_backend

# Colab's free tier is a T4 (Turing), which has no bfloat16 support -- ask for
# float16 there. load_backend also applies the two repairs this export needs:
# transformers 4.x silently ignores its transformers-5 `rope_parameters` block,
# and its `tokenizer_class: TokenizersBackend` is unresolvable by AutoTokenizer.
DTYPE = 'bfloat16' if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else 'float16'
backend = load_backend('transformers', model=None, dtype=DTYPE)
tok = backend.tokenizer
print(f'loaded ({DTYPE}); vocab {tok.vocab_size}')
""")

code("""
#@title The exp82 rollout recipe
# Fixed, and worth leaving fixed: 100 rollouts per protein, each from a FRESH
# document realization (resampled N-terminus + shuffled <pX> <AA> statement
# order), temperature 1.0, top-p 0.95, top-k DISABLED, budget 6L+128. Top-k is
# the trap -- HF's default of 50 rides in from config.json, inflates <end>, and
# silently costs ~0.011 R-precision.
import re
from marinfold.document_structures.contacts_v1 import (
    GenerationConfig, build_document, residues_from_sequence)

BEGIN, NUM_POS = '<begin_statements>', 2000
CONTACT_RE = re.compile(r'<contact>\\s+<p(\\d+)>\\s+<p(\\d+)>')

def rollout_scores(seq, *, n_rollouts=100, tag='x', seed=0, batch_size=25):
    \"\"\"[L,L] matrix of per-pair occurrence frequency across rollouts.\"\"\"
    L = len(seq)
    residues = residues_from_sequence(seq)
    prefixes, maps = [], []
    for k in range(n_rollouts):
        r = build_document(f'{tag}:s{seed}r{k}', residues, [], config=GenerationConfig())
        prefixes.append(r.document[:r.document.index(BEGIN) + len(BEGIN)])
        maps.append({(r.n_term_index + i) % NUM_POS: i for i in range(r.seq_len)})
    ids = [tok(p)['input_ids'] for p in prefixes]
    out = backend.sample_completions(
        ids, max_new_tokens=6 * L + 128, temperature=1.0, top_p=0.95,
        top_k=0,                              # 0 = disabled in the HF path
        stop_token_id=tok.eos_token_id, seed=seed, batch_size=batch_size)
    M = np.zeros((L, L), np.float32)
    for gen, seqidx in zip(out, maps):
        seen = set()
        for a, b in CONTACT_RE.findall(tok.decode(gen)):
            ia, ib = seqidx.get(int(a)), seqidx.get(int(b))
            if ia is None or ib is None or abs(ia - ib) < MIN_SEP:
                continue
            key = (min(ia, ib), max(ia, ib))
            if key not in seen:
                seen.add(key)
                M[key[0], key[1]] += 1
                M[key[1], key[0]] += 1
    return M / n_rollouts

N_ROLLOUTS = 100  #@param {type:"integer"}
score_cp = rollout_scores(CP['seq'], n_rollouts=N_ROLLOUTS, tag='1un2')
score_wt = rollout_scores(WT['seq'], n_rollouts=N_ROLLOUTS, tag='1fvk')
print('done')
""")

code("""
#@title Score both arms (exp89 metrics)
from sklearn.metrics import roc_auc_score

def metrics(g, score, ranges=(('all', 6, 10**9), ('short', 6, 11),
                              ('medium', 12, 23), ('long', 24, 10**9))):
    res = np.array(g['resolved'])
    a, b = np.triu_indices(len(res), k=1)
    pi, pj = res[a], res[b]
    sep = pj - pi
    truth = np.array([(int(i), int(j)) in g['contacts'] for i, j in zip(pi, pj)], int)
    s = score[pi, pj]
    out = {}
    for name, lo, hi in ranges:
        m = (sep >= lo) & (sep <= hi)
        ss, gg = s[m], truth[m]
        nt = int(gg.sum())
        if not nt:
            continue
        gs = gg[np.argsort(-ss, kind='mergesort')]
        out[name] = dict(n_true=nt, r_precision=gs[:nt].sum() / nt,
                         auc=roc_auc_score(gg, ss))
    return out

mc, mw = metrics(CP, score_cp), metrics(WT, score_wt)
print(f\"{'range':8s} {'CP R-prec':>10s} {'WT R-prec':>10s} {'CP AUC':>8s} {'WT AUC':>8s}\")
for r in ('all', 'short', 'medium', 'long'):
    print(f\"{r:8s} {mc[r]['r_precision']:10.4f} {mw[r]['r_precision']:10.4f} \"
          f\"{mc[r]['auc']:8.4f} {mw[r]['auc']:8.4f}\")
""")

md("""
## The figures

Prediction above the diagonal, ground truth below it — so you read the two
against each other in one square instead of flipping between panels.
""")

code("""
#@title Contact maps
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

BLUE = LinearSegmentedColormap.from_list('b', ['#ffffff','#cde2fb','#9ec5f4','#5598e7','#2a78d6','#1c5cab','#0d366b'])
# One shared ramp for every contact map: the panels are facets of the same
# measure, so a second hue would make the CP-vs-WT confidence gap -- a real
# result -- read as a palette artefact.

def gt_matrix(g):
    m = np.zeros((g['L'], g['L']), bool)
    for i, j in g['contacts']:
        m[i, j] = m[j, i] = True
    return m

def split_map(ax, score, g, cmap, title, cuts=()):
    L = g['L']
    i, j = np.indices((L, L))
    ok = np.abs(i - j) >= MIN_SEP
    up = np.where(np.triu(np.ones((L, L), bool), 1) & ok, score, np.nan)
    ax.imshow(up, cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
    yy, xx = np.nonzero(np.tril(np.ones((L, L), bool), -1) & ok & gt_matrix(g))
    ax.scatter(xx, yy, s=0.7, c='#3f3e3b', marker='s', linewidths=0)
    ax.plot([0, L-1], [0, L-1], color='#52514e', lw=.5, alpha=.6)
    for c in cuts:
        ax.axhline(c-.5, color='#d03b3b', lw=.7, ls='--')
        ax.axvline(c-.5, color='#d03b3b', lw=.7, ls='--')
    ax.set_xlim(-.5, L-.5); ax.set_ylim(L-.5, -.5)
    ax.set_title(title, fontsize=10); ax.tick_params(labelsize=7)
    ax.set_xlabel('residue index', fontsize=8); ax.set_ylabel('residue index', fontsize=8)

fig, axes = plt.subplots(1, 2, figsize=(11, 5.4))
split_map(axes[0], score_cp, CP, BLUE,
          f\"CP — 1UN2 (L={CP['L']})  R-prec {mc['all']['r_precision']:.3f}\", cuts=(a1, b0, b1))
split_map(axes[1], score_wt, WT, BLUE,
          f\"WT — 1FVK (L={WT['L']})  R-prec {mw['all']['r_precision']:.3f}\")
fig.suptitle('MarinFold prediction (upper triangle) vs pyconfind truth (lower)', y=.99)
plt.tight_layout(); plt.show()
""")

code("""
#@title Both predictions in wild-type coordinates
# Re-index the CP prediction onto WT numbering. Now the two panels describe the
# same molecule in the same frame -- any difference is the permutation.
Lw = WT['L']
cp_in_wt_score = np.full((Lw, Lw), np.nan)
idx = [(c, w) for c, w in enumerate(c2w) if w is not None]
ci = np.array([c for c, _ in idx]); wi = np.array([w for _, w in idx])
cp_in_wt_score[np.ix_(wi, wi)] = score_cp[np.ix_(ci, ci)]

fig, axes = plt.subplots(1, 2, figsize=(11, 5.4))
for ax, (mat, ttl, cm) in zip(axes, [(cp_in_wt_score, 'CP prediction, re-indexed to WT', BLUE),
                                     (score_wt, 'WT prediction', BLUE)]):
    split_map(ax, np.nan_to_num(mat), WT, cm, ttl, cuts=(wb1,))
fig.suptitle('Same frame, same truth (lower triangle) — red dashes = the T99/Q100 cut', y=.99)
plt.tight_layout(); plt.show()
""")

code("""
#@title The contrast: pairs the permutation moved vs pairs it left alone
def contrast(seg_test=None):
    seg_test = seg_test or in_a
    idxs = np.array(sorted(both))
    a_, b_ = np.triu_indices(len(idxs), k=1)
    wi_, wj_ = idxs[a_], idxs[b_]
    keep = (wj_ - wi_) >= MIN_SEP
    wi_, wj_ = wi_[keep], wj_[keep]
    moved = np.array([seg_test(int(i)) != seg_test(int(j)) for i, j in zip(wi_, wj_)])
    ci_ = np.array([w2c[int(w)] for w in wi_]); cj_ = np.array([w2c[int(w)] for w in wj_])
    lo_, hi_ = np.minimum(ci_, cj_), np.maximum(ci_, cj_)
    g_wt = np.array([(int(i), int(j)) in WT['contacts'] for i, j in zip(wi_, wj_)], int)
    g_cp = np.array([(int(i), int(j)) in CP['contacts'] for i, j in zip(lo_, hi_)], int)
    rows = []
    for arm, s, g in (('WT', score_wt[wi_, wj_], g_wt), ('CP', score_cp[lo_, hi_], g_cp)):
        for cls, m in (('unchanged', ~moved), ('changed', moved)):
            ss, gg = s[m], g[m]
            nt = int(gg.sum())
            gs = gg[np.argsort(-ss, kind='mergesort')]
            rows.append(dict(arm=arm, pair_class=cls, n_true=nt,
                             r_precision=gs[:nt].sum()/nt, auc=roc_auc_score(gg, ss)))
    return rows

rows = contrast()
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
for ax, key, name in ((axes[0], 'r_precision', 'R-precision'), (axes[1], 'auc', 'AUC')):
    x = np.arange(2)
    for k, (arm, col) in enumerate([('WT', '#2a78d6'), ('CP', '#eb6834')]):
        v = [next(r[key] for r in rows if r['arm']==arm and r['pair_class']==c)
             for c in ('unchanged', 'changed')]
        ax.bar(x + (k-.5)*.34, v, .31, color=col, label=arm)
        for xi, vi in zip(x + (k-.5)*.34, v):
            ax.text(xi, vi+.015, f'{vi:.3f}', ha='center', fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(['separation\\nUNCHANGED', 'separation\\nCHANGED'], fontsize=8)
    ax.set_ylabel(name); ax.set_ylim(0, 1); ax.grid(axis='y', color='#e6e5e1', lw=.6)
    ax.set_axisbelow(True)
    for s_ in ('top','right'): ax.spines[s_].set_visible(False)
axes[0].legend(fontsize=8)
fig.suptitle('Same 3D contacts, same model — split by whether the permutation moved the pair', y=1.04)
plt.tight_layout(); plt.show()
for r in rows:
    print(f\"{r['arm']}  {r['pair_class']:10s} n_true={r['n_true']:3d}  \"
          f\"R-prec {r['r_precision']:.4f}  AUC {r['auc']:.4f}\")
""")

md("""
## Play: cut the chain wherever you like

This is the part worth poking at. Pick any cut point and the cell builds that
permutant, runs the model, and scores it against **wild-type ground truth mapped
through your permutation** — legitimate because the fold is preserved (the real
1UN2 crystal keeps 88% of WT's contacts).

Things worth trying:

* **`CUT = 99`** reproduces the real 1UN2 construct.
* **A cut inside the thioredoxin domain** (say 40–60) should be more damaging
  than one in a loop — it breaks a contiguous structural unit.
* **`CUT = 0`** is the identity permutation, and is the control: it must score
  the same as WT, because it *is* WT. If it doesn't, something is wrong.
* Set `LINKER = ''` to remove the GGGTG linker and see whether the model cares.
""")

code("""
#@title Build your own permutant { run: "auto" }
CUT = 99          #@param {type:"slider", min:0, max:188, step:1}
LINKER = 'GGGTG'  #@param {type:"string"}
N = 100           #@param {type:"integer"}

wt_seq = WT['seq']
my_seq = wt_seq[CUT:] + LINKER + wt_seq[:CUT]
# my index -> WT index
my_to_wt = list(range(CUT, len(wt_seq))) + [None]*len(LINKER) + list(range(CUT))
w2m = {w: m for m, w in enumerate(my_to_wt) if w is not None}

score_my = rollout_scores(my_seq, n_rollouts=N, tag=f'cut{CUT}')

# WT truth, expressed in the permuted frame.
res_my = [w2m[w] for w in WT['resolved']]
gt_my = {(min(w2m[i], w2m[j]), max(w2m[i], w2m[j])) for i, j in WT['contacts']}
gt_my = {(i, j) for i, j in gt_my if j - i >= MIN_SEP}
G = dict(pdb=f'cut{CUT}', seq=my_seq, L=len(my_seq), contacts=gt_my,
         resolved=sorted(res_my))
mm = metrics(G, score_my)
print(f'cut after WT residue {CUT} -> L={len(my_seq)}, {len(gt_my)} scorable true contacts')
print(f\"  R-precision (all)  {mm['all']['r_precision']:.4f}   \"
      f\"(wild-type reference: {mw['all']['r_precision']:.4f})\")
print(f\"  AUC (all)          {mm['all']['auc']:.4f}   \"
      f\"(wild-type reference: {mw['all']['auc']:.4f})\")

fig, ax = plt.subplots(figsize=(5.6, 5.4))
split_map(ax, score_my, G, BLUE,
          f\"permutant cut@{CUT}  R-prec {mm['all']['r_precision']:.3f}\",
          cuts=(len(wt_seq)-CUT, len(wt_seq)-CUT+len(LINKER)))
plt.tight_layout(); plt.show()
""")

code("""
#@title Sweep several cut points (slow — a few minutes)
CUTS = [0, 25, 50, 75, 99, 125, 150, 175]  #@param
SWEEP_N = 50  #@param {type:"integer"}

sweep = []
for cut in CUTS:
    s = wt_seq[cut:] + 'GGGTG' + wt_seq[:cut]
    m2w = list(range(cut, len(wt_seq))) + [None]*5 + list(range(cut))
    w2m_ = {w: m for m, w in enumerate(m2w) if w is not None}
    sc = rollout_scores(s, n_rollouts=SWEEP_N, tag=f'sw{cut}')
    gt = {(min(w2m_[i], w2m_[j]), max(w2m_[i], w2m_[j])) for i, j in WT['contacts']}
    gt = {(i, j) for i, j in gt if j - i >= MIN_SEP}
    g = dict(pdb=str(cut), seq=s, L=len(s), contacts=gt,
             resolved=sorted(w2m_[w] for w in WT['resolved']))
    r = metrics(g, sc)['all']
    sweep.append((cut, r['r_precision'], r['auc']))
    print(f'cut {cut:3d}: R-prec {r[\"r_precision\"]:.4f}  AUC {r[\"auc\"]:.4f}')

cuts, rp, au = zip(*sweep)
fig, ax = plt.subplots(figsize=(7, 3.4))
ax.plot(cuts, rp, marker='o', color='#eb6834', lw=2, label='R-precision')
ax.plot(cuts, au, marker='o', color='#2a78d6', lw=2, label='AUC')
ax.axvline(99, color='#d03b3b', ls='--', lw=.8)
ax.text(99, .02, ' the real 1UN2 cut', color='#d03b3b', fontsize=7.5)
ax.set_xlabel('cut point (wild-type residue)'); ax.set_ylabel('score')
ax.set_ylim(0, 1); ax.grid(color='#e6e5e1', lw=.6); ax.set_axisbelow(True)
for s_ in ('top','right'): ax.spines[s_].set_visible(False)
ax.legend(fontsize=8); plt.tight_layout(); plt.show()
""")

md("""
## Caveats worth keeping in mind

* **n = 1 protein per arm.** The permutation contrast is internally controlled
  (same molecule, same model, same 3D contacts), which is what makes it
  informative — but the CP-vs-WT *absolute* gap rests on a single pair of
  crystals. exp224 puts error bars on it by repeating the rollouts under 10
  seeds and by scoring two extra wild-type crystals (1DSB, 1A2J) as a
  ground-truth noise floor.
* **The cut-sweep uses wild-type truth**, which assumes every permutant folds
  like wild-type. That is true for 1UN2 and is *not* generally true — most cut
  points would not fold at all. Read the sweep as "how does the model's output
  change when the sequence is re-ordered", not "how accurate is it on a real
  protein".
* **Rollout scoring is stochastic.** With 100 rollouts, differences below
  roughly 0.01 R-precision are noise.
""")

nb = {
    "cells": [
        {"cell_type": t, "metadata": {}, "source": s.splitlines(keepends=True),
         **({"execution_count": None, "outputs": []} if t == "code" else {})}
        for t, s in CELLS
    ],
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1))
print(f"wrote {OUT} ({len(CELLS)} cells)")
