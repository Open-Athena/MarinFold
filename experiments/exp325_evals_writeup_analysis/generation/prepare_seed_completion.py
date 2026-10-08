"""Freeze matched seed prompts, then build Helico maps from completed rollouts.

First run with the pinned Helico environment; run again with --maps after the
MarinFold Modal job. Ground truth is used for seed sampling and retrospective
precision, never for selecting the predicted additions.
"""

import argparse
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from seed_completion import CASES, complete_pairs, state_digest

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "scratch/seed_completion"
OLD = ROOT / "scratch/helico/oracle_budget_low_msa"
DEST = ROOT / "scratch/helico/seed_completion"
HELICO = Path("/home/bizon/git/helico/experiments/exp14_foldbench_held_out_monomers")


def digest(path: Path) -> str:
    """Hash a frozen input or result."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare() -> None:
    """Invert the audited residue map and reuse the exact previous oracle subsets."""
    spec = importlib.util.spec_from_file_location("index_mapping", HELICO / "build_index_map.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    truth_path = ROOT / "scratch/contacts/inputs/gt_universe_scored.jsonl"
    if digest(truth_path) != "f30c23e3d2fbab245755fc01548388b41730ddfa45da87325539698cadb153e5":
        raise ValueError("Ground-truth residue universe changed")
    truth = {row["stem"]: row for row in map(json.loads, truth_path.read_text().splitlines())}
    prompts = pd.read_csv(ROOT.parent / "exp245_evals_foldbench_held_out_monomers/data/eval_sets.csv").set_index("stem")
    targets = pd.read_csv(ROOT / "data/oracle_budget_targets.csv")
    report = pd.read_csv(HELICO / "data/index_map_report.csv").set_index("target_id")
    old_protocol = json.loads((OLD / "protocol.json").read_text())
    if len(targets) != 5 or not (targets.msa_depth < 10).all():
        raise ValueError("Unexpected low-depth cohort")
    (RAW / "inputs").mkdir(parents=True, exist_ok=True)
    records, pins = [], {}
    for target in targets.itertuples():
        stem = target.stem
        sequence = prompts.loc[stem, "sequence"]
        mapping, rule = module.build_map(target.input_seq, sequence, truth[stem]["resolved"])
        verified = report.loc[stem]
        if not verified.ok or rule != verified.rule or len(mapping) != verified.n_mapped:
            raise ValueError(f"{stem}: unverified mapping")
        inverse = {value: key for key, value in mapping.items()}
        if len(inverse) != len(mapping) or len(sequence) != target.L_exp245:
            raise ValueError(f"{stem}: non-bijective map or length mismatch")
        source = OLD / "maps" / f"{stem}.npz"
        if digest(source) != old_protocol["files_sha256"][stem]["maps_sha256"]:
            raise ValueError(f"{stem}: previous seed maps changed")
        if digest(OLD / "gt" / f"{stem}.cif.gz") != old_protocol["files_sha256"][stem]["gt_sha256"]:
            raise ValueError(f"{stem}: previous structural reference changed")
        cases = []
        with np.load(source) as maps:
            for budget, replicate in CASES:
                direct = "top_0" if budget == "0" else f"random_{budget}"
                state = maps[f"{direct}-{replicate}"]
                token_pairs = np.column_stack(np.where(np.triu(state == 2, 1))).tolist()
                seed_pairs = [[inverse[i], inverse[j]] for i, j in token_pairs]
                if any(j - i < 6 for i, j in seed_pairs):
                    raise ValueError("Invalid seed sequence separation")
                case = dict(budget=budget, replicate=replicate, direct_arm=direct,
                            arm=f"complete_{budget}", seed_pairs=seed_pairs,
                            seed_token_pairs=token_pairs, n_seed=len(seed_pairs),
                            final_contacts=target.L_exp245 // 2, direct_state_sha256=state_digest(state))
                cases.append(case)
                records.append(dict(stem=stem, **{k: v for k, v in case.items() if "pairs" not in k},
                                    mapping_rule=rule, n_mapped=len(mapping)))
        path = RAW / "inputs" / f"{stem}.json"
        path.write_text(json.dumps(dict(stem=stem, sequence=sequence, L=len(sequence),
            mapping=mapping, cases=cases), indent=2) + "\n")
        pins[stem] = digest(path)
    protocol = dict(cohort="Five natural FoldBench eval-test proteins with MSA depth <10",
        checkpoint="contacts-v1-exp277-m2-p06-full-epoch-1.5B-step-266344",
        raw_training_tokens=248583762834, budgets=["0", "5", "10", "L/5"],
        n_targets=5, n_cases=35, rollouts_per_case=100, seed_subsets=2,
        generation="100 randomized sequence serializations; append the same known contact pairs in random order/orientation; temperature=1, top_p=.95, top_k=-1, per-rollout seed=0..99",
        continuation="Generate full continuations (up to 6L+128 tokens); only normally terminated rollouts vote. Each new pair votes at most once per rollout.",
        completion="Keep all n seeds, add floor(L/2)-n highest-vote distinct pairs. Require separation>=6 in prompt and mapped Helico indices. Break ties by sequence (i,j). Never fill with zero-vote pairs.",
        truth_use="Ground truth supplies the seed pairs and evaluation labels only. Completion selection receives no truth labels.",
        direct="Reuse exact random subsets and Helico predictions from oracle_budget; n=0 has one map, other budgets two nested random subsets",
        helico="Same MSA-free step6000 model, seed42, six recycles, three diffusion samples/map; primary ranking_score, pTM sensitivity",
        length="L_exp245 frozen sequence length; completion count includes seed contacts",
        input_sha256=pins)
    pd.DataFrame(records).to_csv(ROOT / "data/seed_completion_cases.csv", index=False)
    (ROOT / "data/seed_completion_design.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(f"Frozen {len(records)} cases across {len(targets)} proteins")


def build_maps() -> None:
    """Rank predictions without oracle access, then annotate correctness for analysis."""
    protocol = json.loads((ROOT / "data/seed_completion_design.json").read_text())
    (DEST / "maps").mkdir(parents=True, exist_ok=True)
    (DEST / "gt").mkdir(exist_ok=True)
    records, pairs, timings, pins = [], [], [], {}
    for stem, expected in protocol["input_sha256"].items():
        source = RAW / "inputs" / f"{stem}.json"
        if digest(source) != expected:
            raise ValueError(f"{stem}: changed prompt inputs")
        inputs = json.loads(source.read_text())
        mapping = {int(k): int(v) for k, v in inputs["mapping"].items()}
        states = {}
        with np.load(OLD / "maps" / f"{stem}.npz") as previous:
            for case in inputs["cases"]:
                key = f"{case['arm']}-{case['replicate']}"
                result_path = RAW / "results" / stem / f"{key}.json"
                result = json.loads(result_path.read_text())
                if result["input_sha256"] != expected or len(result["rollouts"]) != 100:
                    raise ValueError(f"{stem}/{key}: changed or incomplete rollouts")
                additions = complete_pairs(case["seed_pairs"], result["votes"], mapping, case["final_contacts"])
                state = previous[f"{case['direct_arm']}-{case['replicate']}"].copy()
                if state_digest(state) != case["direct_state_sha256"]:
                    raise ValueError("Direct seed set changed")
                for i, j, _ in additions:
                    state[mapping[i], mapping[j]] = state[mapping[j], mapping[i]] = 2
                if int(np.triu(state == 2, 1).sum()) != case["final_contacts"]:
                    raise ValueError("Completion has wrong total count")
                states[key] = state
                # All choice is complete above; oracle labels only annotate output.
                oracle = previous["oracle-0"]
                correct = sum(oracle[mapping[i], mapping[j]] == 2 for i, j, _ in additions)
                for kind, selected in (("given", [(i, j, -1) for i, j in case["seed_pairs"]]), ("predicted", additions)):
                    for rank, (i, j, votes) in enumerate(selected, 1):
                        pairs.append(dict(stem=stem, arm=case["arm"], map_seed=case["replicate"],
                            kind=kind, rank=rank, i=i, j=j, helico_i=mapping[i], helico_j=mapping[j],
                            votes=votes, oracle_positive=int(oracle[mapping[i], mapping[j]] == 2)))
                records.append(dict(stem=stem, arm=case["arm"], map_seed=case["replicate"],
                    budget=case["budget"], n_seed=case["n_seed"], n_present=case["final_contacts"],
                    n_added=len(additions), added_correct=int(correct), added_precision=correct / len(additions),
                    total_precision=(correct + case["n_seed"]) / case["final_contacts"],
                    n_stopped=result["timing"]["stopped_rollouts"], state_sha256=state_digest(state),
                    direct_state_sha256=case["direct_state_sha256"], result_sha256=digest(result_path)))
                timings.append(result["timing"])
        map_path = DEST / "maps" / f"{stem}.npz"
        np.savez_compressed(map_path, **states)
        shutil.copyfile(OLD / "gt" / f"{stem}.cif.gz", DEST / "gt" / f"{stem}.cif.gz")
        pins[stem] = dict(maps_sha256=digest(map_path), gt_sha256=digest(DEST / "gt" / f"{stem}.cif.gz"))
    shutil.copyfile(OLD / "targets.csv", DEST / "targets.csv")
    (DEST / "ranked_pairs.json").write_text("{}\n")
    frozen = dict(**protocol, phase="seed_completion", files_sha256=pins,
                  map_keys=[f"complete_{budget}-{r}" for budget, r in CASES])
    (DEST / "protocol.json").write_text(json.dumps(frozen, indent=2) + "\n")
    (ROOT / "data/seed_completion_protocol.json").write_text(json.dumps(frozen, indent=2) + "\n")
    for name, rows in (("maps", records), ("pairs", pairs), ("timings", timings)):
        pd.DataFrame(rows).to_csv(ROOT / f"data/seed_completion_{name}.csv", index=False)
    print(f"Built {len(records)} exact L/2 maps; {len(pairs)} traceable contacts")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maps", action="store_true")
    arguments = parser.parse_args()
    build_maps() if arguments.maps else prepare()
