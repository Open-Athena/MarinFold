"""Freeze exact top-L/2 maps from the archived 248B-model contact rankings.

Run in the pinned Helico environment. No MarinFold inference or correctness-
based contact selection occurs here. The existing top-L lists are long enough
to supply floor(L/2) pairs after Helico's sequence-separation filter.
"""

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "scratch/helico/marinfold_l2"
EXP311 = ROOT.parent / "exp311_evals_helico_exp277_contact_count_sweep"


def digest(path: Path) -> str:
    """Hash a frozen source or conditioning artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_pairs(ranked: list[list[int]], count: int) -> list[tuple[int, int, int]]:
    """Take the first count distinct, Helico-eligible pairs, retaining source ranks."""
    chosen, seen = [], set()
    for rank, (i, j) in enumerate(ranked, 1):
        if i < 0 or j < 0:
            raise ValueError("Negative residue index")
        pair = tuple(sorted((i, j)))
        if pair in seen:
            raise ValueError("Duplicate ranked pair")
        seen.add(pair)
        if abs(i - j) >= 6:
            chosen.append((rank, i, j))
        if len(chosen) == count:
            return chosen
    raise ValueError(f"Only {len(chosen)} eligible pairs for {count} requested contacts")


def main() -> None:
    """Recheck original rankings and build exact, source-traced conditioning maps."""
    # Helico is an optional dependency installed in the dedicated prep runtime.
    from helico.bench import structure_to_chains
    from helico.data import parse_ccd, parse_mmcif, tokenize_sequences
    from helico.inference import contacts_from_pairs

    targets_path = ROOT / "data/oracle_l2_targets.csv"
    targets = pd.read_csv(targets_path)
    if len(targets) != 305 or targets.stem.duplicated().any() or targets.designed.any():
        raise ValueError("Expected the same 305 natural figure proteins")
    figure = pd.read_csv(ROOT / "data/figure_rows.csv")
    baseline = figure[(figure.figure == "05_folding") & figure.method.isin(["marinfold_helico", "oracle_L2", "no_contacts"])]
    baseline.to_csv(ROOT / "data/marinfold_l2_baselines.csv", index=False)
    val_path = ROOT / "scratch/marinfold_l2/exp311_ranked_pairs.json"
    test_path = ROOT / "scratch/helico/folding/ranked_pairs.json"
    val_manifest = json.loads((EXP311 / "data/input_manifest.json").read_text())
    if digest(val_path) != val_manifest["ranked_pairs_sha256"]:
        raise ValueError("Published validation rankings changed")
    val, test = json.loads(val_path.read_text()), json.loads(test_path.read_text())
    if set(val) & set(test):
        raise ValueError("Overlapping validation/test contact rankings")
    rankings = {**val, **test}
    # Independently reproduce the test top-L lists from the original sparse votes.
    truth_path = ROOT / "scratch/contacts/inputs/gt_universe_scored.jsonl"
    if digest(truth_path) != val_manifest["exp245_ground_truth_sha256"]:
        raise ValueError("Resolved-residue universe changed")
    truth = {r["stem"]: r for r in map(json.loads, truth_path.read_text().splitlines())}
    mapping_path = ROOT / "scratch/helico/folding/token_map.json"
    mappings = json.loads(mapping_path.read_text())
    score_paths = sorted((ROOT / "scratch/contacts/results/scores").glob("*.parquet"))
    votes = pd.concat([pd.read_parquet(p) for p in score_paths])
    for target in targets[targets.eval_set == "eval-test"].itertuples():
        record = truth[target.stem]
        sparse = votes[votes.stem == target.stem]
        matrix = np.zeros((record["L"], record["L"]), dtype=np.int16)
        matrix[sparse.i.to_numpy(), sparse.j.to_numpy()] = sparse.votes.to_numpy()
        resolved = np.asarray(record["resolved"], dtype=int)
        a, b = np.triu_indices(len(resolved), 1)
        i, j = resolved[a], resolved[b]
        keep = j - i >= 6
        i, j = i[keep], j[keep]
        order = np.argsort(-matrix[i, j], kind="mergesort")[:record["L"]]
        mapping = mappings[target.stem]
        expected = [[mapping[str(i[k])], mapping[str(j[k])]] for k in order]
        if rankings[target.stem] != expected:
            raise ValueError(f"{target.stem}: archived test ranking differs from votes")
    oracle_root = ROOT / "scratch/helico/oracle_l2"
    oracle_protocol = json.loads((oracle_root / "protocol.json").read_text())
    (DEST / "maps").mkdir(parents=True, exist_ok=True)
    (DEST / "gt").mkdir(exist_ok=True)
    ccd = parse_ccd()
    rows, pairs, pins = [], [], {}
    for target in targets.sort_values("stem").itertuples():
        stem, length = target.stem, int(target.L_exp245)
        ranked = rankings[stem]
        if len(ranked) != length:
            raise ValueError(f"{stem}: original list is not top-L")
        source = oracle_root / "gt" / f"{stem}.cif.gz"
        if digest(source) != oracle_protocol["files_sha256"][stem]["gt_sha256"]:
            raise ValueError(f"{stem}: structural reference changed")
        gt = parse_mmcif(source, max_resolution=float("inf"))
        if gt is None:
            raise ValueError(f"Cannot parse {stem}")
        chains = structure_to_chains(gt)
        proteins = [c for c in chains if c["type"] == "protein"]
        if len(proteins) != 1 or proteins[0]["sequence"] != target.input_seq:
            raise ValueError(f"{stem}: sequence differs from original folding input")
        chain = proteins[0]["id"]
        tokenized = tokenize_sequences(chains, ccd)
        chosen = select_pairs(ranked, length // 2)
        state = contacts_from_pairs([(chain, i, chain, j) for _, i, j in chosen], tokenized=tokenized, strict=False).numpy()
        if int(np.triu(state == 2, 1).sum()) != length // 2 or np.any(state == 1):
            raise ValueError(f"{stem}: final conditioning is not exactly L/2 positives")
        map_path = DEST / "maps" / f"{stem}.npz"
        np.savez_compressed(map_path, **{"top_L2-0": state})
        shutil.copyfile(source, DEST / "gt" / source.name)
        pins[stem] = dict(maps_sha256=digest(map_path), gt_sha256=digest(source))
        rows.append(dict(stem=stem, arm="top_L2", map_seed=0, L=length, n_tokens=tokenized.n_tokens,
            requested_contacts=length // 2, n_present=length // 2, n_absent=0,
            n_unknown=int(np.triu(state == 0, 1).sum()), skipped_short_pairs=chosen[-1][0] - len(chosen),
            state_sha256=hashlib.sha256(state.tobytes()).hexdigest(), ranking_source="exp311" if stem in val else "exp325-test"))
        pairs.extend(dict(stem=stem, selection_rank=k, original_rank=rank, helico_residue_i=i, helico_residue_j=j)
                     for k, (rank, i, j) in enumerate(chosen, 1))
    targets.to_csv(DEST / "targets.csv", index=False)
    targets.to_csv(ROOT / "data/marinfold_l2_targets.csv", index=False)
    (DEST / "ranked_pairs.json").write_text(json.dumps({s: rankings[s] for s in targets.stem}) + "\n")
    for name, records in (("maps", rows), ("pairs", pairs)):
        pd.DataFrame(records).to_csv(ROOT / f"data/marinfold_l2_{name}.csv", index=False)
    protocol = dict(phase="marinfold_l2", n_targets=305, n_maps_per_target=1, map_keys=["top_L2-0"],
        model="contacts-v1-exp277-m2-p06-full-epoch-1.5B-step-266344", raw_training_tokens=248583762834,
        ranking="Reuse the exact 100-rollout consensus rankings underlying Figure 05 top-L: exp311 validation and exp325 publication test. No new MarinFold inference.",
        selection="First floor(L/2) pairs after filtering Helico residue separation <6; retain original vote order and ties. No ground-truth correctness or seed contacts.",
        length="L_exp245 frozen full input sequence length, not resolved token count",
        sparse_encoding="PRESENT=2 for selected pairs, UNKNOWN=0 everywhere else; no non-contacts",
        inference="MSA-free Helico step6000; three diffusion samples, six recycles, seed42 per map",
        selector="Highest ranking_score of three, matching Figure 05; pTM sensitivity saved separately",
        oracle_comparison="Existing two random subsets of floor(L/2) true contacts, averaged within protein; 8f8n_A oracle caps at 109 vs 110 requested",
        sources_sha256={str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): digest(p)
                        for p in [targets_path, val_path, test_path, truth_path, mapping_path, *score_paths]},
        files_sha256=pins)
    for path in (DEST / "protocol.json", ROOT / "data/marinfold_l2_protocol.json"):
        path.write_text(json.dumps(protocol, indent=2) + "\n")
    print(f"Prepared 305 exact L/2 maps; skipped {sum(row['skipped_short_pairs'] for row in rows)} short mapped pairs")


if __name__ == "__main__":
    main()
