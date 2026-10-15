#!/usr/bin/env python
# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""8b · make — fold 8ubs_A once per contact, from none to the whole list.

Step two of three. Reads what :mod:`8_make_contact_ranking_data` wrote, puts the
contacts in an order, and folds the protein once per prefix of that order with
Helico: no contacts, then the first, the first two, and so on. Every prediction's
coordinates and scores are stored, so :mod:`8_plot_contact_titration` can be
rewritten as many times as it needs to be without folding anything again.

``--order`` picks the order, and each writes its own dataset:

``confidence`` (default)
    Vote count across the 100 rollouts, tie-broken by the pairwise log-prob —
    #82's ranking, and the order a deployment would actually cut its list in.

``rollout``
    The order **one** rollout wrote its contacts in. This is not a ranking: the
    model is never asked to emit its best guess first. Whether it does anyway is
    the question the second animation puts on screen. The rollout is the median
    one by F1 against the experimental contacts — a rule, not a hand-picked
    index, so it stays typical rather than flattering.

**This runs in Helico's environment, not MarinFold's.** Helico is a separate
repository with its own torch pin, and the MarinFold inference stack has no
business in it. Both write into this directory's ``data/``:

    cd ~/git/helico && .venv/bin/python \\
        ~/git/MarinFold/experiments/exp250_evals_exploration_notebook/figures/\\
8_make_titration_data.py

Roughly 11 s per fold on an RTX A5000 — about half an hour for the default 151.

**Two indexings, reconciled by assertion.** MarinFold is prompted with #245's
151-residue sequence; Helico folds the 150 residues the deposited structure
resolves, which is that sequence without its unresolved N-terminal serine. The
offset is found by locating Helico's sequence inside MarinFold's and checked to
be an exact substring match, so a target where the relationship is not a clean
offset fails here rather than producing a plausible animation of the wrong
contacts. Pairs that do not map are dropped and counted.

**The recipe is helico exp14's published one** — ``n_samples=3``, ``n_cycles=6``,
the ``contacts-msafree-01`` checkpoint — so the last frame is comparable with the
published ``mf_L`` arm. One fixed seed for every *k*: the diffusion noise is then
the same at every frame and the structure changes because the conditioning
changed, not because the sampler rolled differently.
"""

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from helico.bench import (
    _default_foldbench_dir,
    _find_gt_path,
    match_atoms,
    oracle_contact_state,
    predict_target,
    score_monomer,
    structure_to_chains,
)
from helico.contacts import load_rotamer_library
from helico.data import CONTACT_PRESENT, parse_ccd, parse_mmcif
from helico.inference import _token_positions, load_model

import figlib

# --- parameters ---------------------------------------------------------------------------------
#: Two orders, two datasets. `confidence` is the deployment order — vote count across 100
#: rollouts, which is how the published `mf_L` arm cuts its list. `rollout` is the order one
#: rollout actually wrote its contacts in, which is not a ranking at all: the model is not asked
#: to emit its best guess first, and whether it does anyway is the question the second animation
#: puts on screen.
DATASETS = {"confidence": "8_contact_titration", "rollout": "8_contact_titration_rollout"}
RANKING = "8_contact_ranking"    # the dataset 8_make_contact_ranking_data.py wrote
PDB_ID = "8ubs-assembly1"        # FoldBench's ground-truth entry for 8ubs_A
CHAIN = "A"

N_SAMPLES = 3                    # helico exp14's published sampling
N_CYCLES = 6                     # helico exp14's published recycling
SEED = 0                         # the same diffusion noise at every k
K_MAX = None                     # None -> the deposited chain's length (exp14's top-L cut)


def rollout_order_pairs(statements: pd.DataFrame, rollout: int, offset: int, n_helico: int,
                        min_separation: int):
    """One rollout's distinct in-band contacts, in the order it wrote them, re-seated.

    Repeats are dropped as they arrive — a restatement adds no contact — and so is anything the
    chain does not cover. The ``score`` carried along is the pair's vote count across all 100
    rollouts, which is not what orders this list but is worth keeping beside it: the second
    animation's whole question is whether emission order tracks confidence.
    """
    emitted = statements[statements.rollout == rollout].sort_values("order")
    pairs, dropped, seen = [], 0, set()
    for row in emitted.itertuples(index=False):
        if row.seq_i < 0 or row.seq_j - row.seq_i < min_separation:
            continue
        i, j = int(row.seq_i) - offset, int(row.seq_j) - offset
        if not (0 <= i < n_helico and 0 <= j < n_helico):
            dropped += 1
            continue
        if (i, j) in seen:
            continue
        seen.add((i, j))
        pairs.append((i, j, float(row.seq_i)))
    return pairs, dropped


def ranked_pairs(score: np.ndarray, offset: int, n_helico: int, min_separation: int):
    """MarinFold's candidate pairs, best first, re-seated onto Helico residue indices.

    Candidates are upper-triangle pairs at sequence separation >= ``min_separation``, ordered by
    a stable ``argsort(-score)`` so pairs tied on score fall in ``triu_indices`` order — exp245's
    ranking rule. The re-seating is applied after the ordering, not before: ranking in Helico
    space would silently promote whatever the dropped pairs were ranked above.
    """
    length = score.shape[0]
    rows, columns = np.triu_indices(length, k=min_separation)
    order = np.argsort(-score[rows, columns], kind="mergesort")
    pairs, dropped = [], 0
    for index in order:
        i, j = int(rows[index]) - offset, int(columns[index]) - offset
        if 0 <= i < n_helico and 0 <= j < n_helico:
            pairs.append((i, j, float(score[rows[index], columns[index]])))
        else:
            dropped += 1
    return pairs, dropped


def main() -> None:
    """Fold the protein once per prefix of the ranking and write the dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k-max", type=int, default=K_MAX,
                        help="highest contact count to fold (default: the chain's length)")
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    parser.add_argument("--n-cycles", type=int, default=N_CYCLES)
    parser.add_argument("--pdb-id", default=PDB_ID)
    parser.add_argument("--order", default="confidence", choices=sorted(DATASETS),
                        help="confidence: vote count across the 100 rollouts (the deployment "
                             "order). rollout: the order one median-accuracy rollout wrote them.")
    arguments = parser.parse_args()
    dataset = DATASETS[arguments.order]

    inputs = figlib.Inputs()
    ranking_dir = figlib.require(RANKING, "score.npy", "votes.npy", "target.json",
                                 "statements.csv")
    score = np.load(inputs.add_file(ranking_dir / "score.npy")).astype(np.float64)
    votes = np.load(inputs.add_file(ranking_dir / "votes.npy")).astype(np.int64)
    target = json.loads(inputs.add_file(ranking_dir / "target.json").read_text())
    ranking_metadata = figlib.load_metadata(RANKING)
    prompt = target["sequence"]

    gt_path = _find_gt_path(_default_foldbench_dir() / "examples/ground_truths", arguments.pdb_id)
    gt_structure = parse_mmcif(inputs.add_file(gt_path), max_resolution=float("inf"))
    if gt_structure is None:
        raise SystemExit(f"could not parse {gt_path}")
    chains = structure_to_chains(gt_structure)
    protein = [chain for chain in chains if chain.get("type") == "protein"]
    if len(protein) != 1 or protein[0]["id"] != CHAIN:
        raise SystemExit(f"expected one protein chain {CHAIN!r}, got "
                         f"{[(c['id'], c.get('type')) for c in chains]}")
    folded = protein[0]["sequence"]

    # The one relationship this whole step rests on. `find` rather than a hard-coded 1: the
    # assertion is that Helico's chain is a contiguous piece of MarinFold's prompt, and the
    # offset falls out of it.
    offset = prompt.find(folded)
    if offset < 0:
        raise SystemExit(f"the deposited chain is not a substring of the #245 prompt — the two "
                         f"indexings are not a clean offset and every contact would be re-seated "
                         f"onto the wrong residues\n  prompt  {prompt}\n  folded  {folded}")
    print(f"{target['stem']}: prompt {len(prompt)} residues, deposited chain {len(folded)}, "
          f"offset {offset}")

    ccd = parse_ccd()
    model = load_model()

    # The experimental contact map, by the same pyconfind rule MarinFold's documents use, so the
    # animation can colour each predicted contact by whether it is real. Token space -> the
    # chain's residue space, which is what everything else here is in. It comes first because
    # choosing the median rollout needs it.
    reference = predict_target(model, chains, ccd, target_name=target["stem"],
                               n_samples=1, n_cycles=1)
    if reference is None:
        raise SystemExit("the target did not tokenize")
    tokenized, _ = reference
    positions = _token_positions(tokenized)[CHAIN]
    oracle = oracle_contact_state(gt_structure, tokenized, load_rotamer_library())
    if oracle is None:
        raise SystemExit("no protein contacts could be derived from the deposited structure")
    true_contacts = (oracle[np.ix_(positions, positions)] == CONTACT_PRESENT).numpy()
    print(f"{len(positions)} protein tokens of {tokenized.n_tokens} · "
          f"{int(np.triu(true_contacts, figlib.MIN_SEPARATION).sum())} experimental contacts")

    featured = None
    if arguments.order == "rollout":
        statements = pd.read_csv(ranking_dir / "statements.csv")
        # `rollout_accuracy` scores against a matrix, so the statements have to be in that
        # matrix's indexing first. Anything the chain does not cover becomes -1, which is the
        # value it already drops.
        seated = statements.seq_i - offset, statements.seq_j - offset
        within = (seated[0] >= 0) & (seated[1] < len(folded)) & (statements.seq_i >= 0)
        mapped = statements.assign(seq_i=seated[0].where(within, -1),
                                   seq_j=seated[1].where(within, -1))
        accuracy = figlib.rollout_accuracy(mapped, true_contacts)
        featured = figlib.median_rollout(accuracy)
        row = accuracy.loc[featured]
        print(f"rollouts      precision {accuracy.precision.mean():.3f} mean, "
              f"{accuracy.precision.min():.3f}-{accuracy.precision.max():.3f} over "
              f"{len(accuracy)} · {accuracy.n.median():.0f} contacts median")
        print(f"featured      rollout {featured}: {int(row.n)} distinct contacts, "
              f"{int(row.hits)} in the structure (precision {row.precision:.3f}, "
              f"recall {row.recall:.3f}, F1 {row.f1:.3f} against a median F1 of "
              f"{accuracy.f1.median():.3f})")
        candidates, dropped = rollout_order_pairs(statements, featured, offset, len(folded),
                                                  figlib.MIN_SEPARATION)
        default_k = len(candidates)
    else:
        candidates, dropped = ranked_pairs(score, offset, len(folded), figlib.MIN_SEPARATION)
        default_k = len(folded)

    k_max = arguments.k_max if arguments.k_max is not None else default_k
    if k_max > len(candidates):
        raise SystemExit(f"asked for {k_max} contacts but only {len(candidates)} candidates map")
    print(f"{arguments.order} order: {len(candidates)} pairs map onto the chain, {dropped} "
          f"dropped; folding k = 0 .. {k_max}")

    pairs_frame = pd.DataFrame([
        dict(rank=rank, seq_i=i, seq_j=j, score=value, votes=int(votes[i + offset, j + offset]),
             separation=j - i, true_contact=bool(true_contacts[i, j]))
        for rank, (i, j, value) in enumerate(candidates[:k_max], start=1)])

    rows, coordinates, gt_coords = [], [], None
    for k in range(k_max + 1):
        contact_pairs = [(i, j) for i, j, _ in candidates[:k]]
        started = time.time()
        # Re-seeded per fold rather than once for the run: otherwise fold k's noise depends on how
        # many folds preceded it, and re-running a single k would not reproduce its frame.
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
        result = predict_target(model, chains, ccd, target_name=target["stem"],
                                n_samples=arguments.n_samples, n_cycles=arguments.n_cycles,
                                contact_pairs=contact_pairs if k else None)
        if result is None:
            raise SystemExit(f"prediction failed at k={k}")
        tokenized, output = result
        predicted = output["coords"][0].float().cpu().numpy()
        matched = match_atoms(tokenized, predicted, gt_structure)
        scores = score_monomer(matched)
        if gt_coords is None:
            gt_coords = matched.gt_coords.astype(np.float32)
            atom_index = pd.DataFrame({
                "chain_id": matched.chain_ids, "res_seq_id": matched.res_seq_ids,
                "atom_name": matched.atom_names, "element": matched.elements,
                "entity_type": matched.entity_types})
        elif len(matched.pred_coords) != len(gt_coords):
            # The matched set is a function of the tokenization and the deposited structure, both
            # fixed here. If it moves, frame k's atoms are not frame k-1's and neither the stored
            # array nor the overlay means anything.
            raise SystemExit(f"k={k} matched {len(matched.pred_coords)} atoms, "
                             f"k=0 matched {len(gt_coords)}")
        coordinates.append(matched.pred_coords.astype(np.float32))
        rows.append(dict(k=k, seconds=round(time.time() - started, 2),
                         mean_plddt=float(output["plddt"][0].float().mean()),
                         n_true=int(pairs_frame.true_contact[:k].sum()),
                         **{key: float(value) for key, value in scores.items()}))
        if k % 10 == 0 or k == k_max:
            print(f"  k={k:>4d}  lddt {scores['lddt']:.4f}  gdt_ts {scores['gdt_ts']:.4f}  "
                  f"rmsd {scores['rmsd']:6.2f}  {rows[-1]['seconds']:.1f}s")

    metrics = pd.DataFrame(rows)
    coordinates = np.stack(coordinates)
    print(f"\nlDDT {metrics.lddt.iloc[0]:.4f} at k=0 -> {metrics.lddt.iloc[-1]:.4f} at "
          f"k={k_max}; best {metrics.lddt.max():.4f} at k={int(metrics.lddt.idxmax())}")

    figlib.write_dataset(
        dataset,
        notebook="8_make_titration_data.py",
        parameters=dict(protein=target["stem"], pdb_id=arguments.pdb_id, chain=CHAIN,
                        order=arguments.order, featured_rollout=featured, k_max=k_max,
                        n_samples=arguments.n_samples, n_cycles=arguments.n_cycles,
                        seed=SEED, ranking_dataset=RANKING),
        inputs=inputs,
        files={
            # (k_max + 1, n_atoms, 3): one predicted structure per contact count, in the atom
            # order `atom_index.csv` lists, which is also `gt_coords.npy`'s order. Unaligned —
            # superposing is the plot's job and its method is something to iterate on.
            "pred_coords.npy": lambda path: np.save(path, coordinates),
            "gt_coords.npy": lambda path: np.save(path, gt_coords),
            "atom_index.csv": lambda path: atom_index.to_csv(path, index=False),
            "metrics.csv": lambda path: metrics.to_csv(path, index=False),
            # The ranking as folded: rank 1 is the first contact the animation adds.
            "ranked_contacts.csv": lambda path: pairs_frame.to_csv(path, index=False),
            "true_contacts.npy": lambda path: np.save(path, true_contacts),
            "sequence.txt": folded.encode(),
        },
        extra={
            "order": {"kind": arguments.order, "featured_rollout": featured,
                      "note": "confidence = vote count over the 100 rollouts (the deployment "
                              "order). rollout = the order the median-accuracy rollout wrote "
                              "its contacts in; the model is never asked to emit its best guess "
                              "first."},
            "model": {"nickname": "helico contacts-msafree-01",
                      "marinfold": ranking_metadata["model"],
                      "note": "Helico folds; the contacts it conditions on come from the "
                              "MarinFold checkpoint recorded under `marinfold`."},
            "recipe": {"n_samples": arguments.n_samples, "n_cycles": arguments.n_cycles,
                       "seed": SEED, "min_seq_separation": figlib.MIN_SEPARATION,
                       "note": "helico exp14's published sampling, one fixed seed per fold"},
            "protein": {"stem": target["stem"], "pdb_id": arguments.pdb_id, "chain": CHAIN,
                        "L_prompt": len(prompt), "L_folded": len(folded), "offset": offset,
                        "n_tokens": int(tokenized.n_tokens), "n_matched_atoms": len(gt_coords),
                        "n_true_contacts": int(np.triu(true_contacts,
                                                       figlib.MIN_SEPARATION).sum())},
            "titration": {"k_max": k_max, "n_candidates": len(candidates),
                          "n_dropped_off_chain": dropped,
                          "precision_at_k_max": float(pairs_frame.true_contact.mean()),
                          "lddt_k0": float(metrics.lddt.iloc[0]),
                          "lddt_kmax": float(metrics.lddt.iloc[-1]),
                          "lddt_best": float(metrics.lddt.max()),
                          "k_best": int(metrics.lddt.idxmax())},
        })


if __name__ == "__main__":
    main()
